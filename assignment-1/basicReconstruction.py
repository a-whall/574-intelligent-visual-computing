import argparse
import numpy as np
from skimage import measure
from sklearn.neighbors import KDTree
import open3d as o3d;

def createGrid(points, resolution=64):
    """
    Constructs a 3D grid containing the point cloud.
    Each grid point will store the implicit function value.

    Args:
        points: 3D points of the point cloud
        resolution: grid resolution, i.e. grid will be NxNxN where N=resolution
            set N=16 for quick debugging, use *N=64* for reporting results

    Returns:
        X,Y,Z coordinates of grid vertices
        max and min dimensions of the bounding box of the point cloud
    """
    # Largest, Smallest xyz coordinates among all surface points, respectively.
    max_xyz, min_xyz = np.max(points, axis=0), np.min(points, axis=0)

    # Compute the bounding box dimensions of the point cloud.
    bounding_box_dimensions = max_xyz - min_xyz

    # Extend the bounding box to fit the surface.
    max_xyz = max_xyz + bounding_box_dimensions / 10
    min_xyz = min_xyz - bounding_box_dimensions / 10

    # Generate the grid points.
    X, Y, Z = np.meshgrid(
        np.linspace(min_xyz[0], max_xyz[0], resolution),
        np.linspace(min_xyz[1], max_xyz[1], resolution),
        np.linspace(min_xyz[2], max_xyz[2], resolution)
    )
    return X, Y, Z, max_xyz, min_xyz

def sphere(center, R, X, Y, Z):
    """
    Constructs an implicit function of a sphere sampled at grid coordinates X, Y, Z.

    Args:
        center: 3D location of the sphere center
        R     : Radius of the sphere
        X,Y,Z : coordinates of grid vertices

    Returns:
        IF    : implicit function of the sphere sampled at the grid points
    """
    # The implicit surface is a function that maps position to a distance from
    # the surface. Any point for which this function is 0 lies on the implicit
    # surface, any point at distance R from the center is on the surface.
    IF = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2 - R**2
    return IF

def showMeshReconstruction(IF):
    """
    Calls marching cubes on the implicit function passed as an argument
    sampled in the 3D grid and shows the reconstruction mesh.

    Args:
        IF    : implicit function sampled at the grid points
    """
    # Extract the surface mesh from the implicit-surface.
    verts, triangles, normals, values = measure.marching_cubes(IF, 0)

    # Create an empty triangle mesh.
    mesh = o3d.geometry.TriangleMesh()

    # Use mesh.vertex to access the vertices' attributes.
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # Use mesh.triangle to access the triangles' attributes.
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))

    # Compute normals for shading.
    mesh.compute_vertex_normals()

    # Show the result with Open3D renderer.
    o3d.visualization.draw_geometries([mesh])

def naiveReconstruction(points, normals, X, Y, Z):
    """
    Performs surface reconstruction with an implicit function f(x,y,z) representing the
    signed distance to the tangent plane of the surface point nearest to each point (x,y,z).

    Args:
        input: filename of a point cloud.
    Returns:
        IF: implicit function sampled at the grid points.
    """
    ################################################
    # ================ START CODE ================ #
    ################################################

    # Create a Vertex-array that contains every 3D grid point.
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()

    # Get the index of the closest surface point to each grid point.
    _, NN_indices = KDTree(points, metric='euclidean').query(Q, k=1)

    # The "naive" heuristic for reconstructing an implicit surface.
    def dist_to_tangent_plane(q_i, v_i):
        return normals[v_i].dot((Q[q_i] - points[v_i]).T)

    IF = np.empty(X.shape)
    for y in range(X.shape[0]):
        for x in range(X.shape[1]):
            for z in range(X.shape[2]):
                q_i = y * X.shape[1]**2 + x * X.shape[0] + z
                v_i = NN_indices[q_i][0]
                IF[x][y][z] = dist_to_tangent_plane(q_i, v_i)

    ################################################
    # ================ END CODE ================== #
    ################################################
    return IF 

def mlsReconstruction(points, normals, X, Y, Z):
    """
    Surface reconstruction with an implicit function f(x,y,z) representing
    Minimum-Least-Squares distance to the tangent plane of the input surface points.
    The returned implicit function will be used to show the reconstructed mesh.

    Args:
        input: Filename of a point cloud.
    Returns:
        IF: Implicit Function, sampled at regular grid points.
    """
    ################################################
    # ================ START CODE ================ #
    ################################################

    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()

    kd_tree = KDTree(points, metric='euclidean')

    # Get 50 nearest surface points to each grid point.
    NN_distances, NN_indices = kd_tree.query(Q, k=50)

    # Get distance of the 2 nearest neighboring surface points for each surface point.
    inter_surface_distances, _ = kd_tree.query(points, k=2)

    # Index 1 is used because the distance to itself is 0 and thus the NN is itself.
    sq_inv_ꞵ = 1 / (2*sum(d[1] for d in inter_surface_distances)/len(points)) ** 2

    def dist_to_tangent_plane(q_i, v_i):
        return normals[v_i].dot((Q[q_i] - points[v_i]).T)

    IF = np.empty(X.shape)
    for y in range(X.shape[0]):
        for x in range(X.shape[1]):
            for z in range(X.shape[2]):
                q_i = y * X.shape[1]**2 + x * X.shape[0] + z
                φ = np.exp(-NN_distances[q_i]**2 * sq_inv_ꞵ)
                d = [dist_to_tangent_plane(q_i, v_i) for v_i in NN_indices[q_i]]
                IF[x][y][z] = sum(d * φ) / sum(φ)

    ################################################
    # ================ END CODE ================== #
    ################################################
    return IF 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic Surface Reconstruction')
    parser.add_argument('--file', type=str, default="sphere.pts", help='Name of input file')
    parser.add_argument('--method', type=str, default="", help='Method of reconstruction: [mls (Moving Least Squares), naive]')
    args = parser.parse_args()

    data = np.loadtxt(args.file)
    points = data[:, :3]
    normals = data[:, 3:6]

    X, Y, Z, max_dimensions, min_dimensions = createGrid(points)

    if args.method == 'mls':
        print(f'Running Moving Least Squares reconstruction on {args.file}')
        IF = mlsReconstruction(points, normals, X, Y, Z)
    elif args.method == 'naive':
        print(f'Running naive reconstruction on {args.file}')
        IF = naiveReconstruction(points, normals, X, Y, Z)
    else:
        print(f'No method argument. Replacing point cloud {args.file} with a sphere!')
        center = (max_dimensions + min_dimensions) / 2
        R = max(max_dimensions - min_dimensions) / 4
        IF = sphere(center, R, X, Y, Z)

    showMeshReconstruction(IF)