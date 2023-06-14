import numpy

input_file = "cow.obj"
output_file = "../data/spot-sub-0.pts"

# Parse the OBJ

vertices = []
faces = []

with open(input_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == 'v':
            vertices.append([float(x) for x in parts[1:]])
        elif parts[0] == 'f':
            faces.append([int(x) for x in parts[1:]])

vertices = numpy.array(vertices)
faces = numpy.array(faces) - 1

# Compute normals

vertex_normals = numpy.zeros(vertices.shape)
face_normals = numpy.cross(
    vertices[faces[:,1]] - vertices[faces[:,0]],
    vertices[faces[:,2]] - vertices[faces[:,0]]
)

for i, f in enumerate(faces):
    for v in f:
        vertex_normals[v] += face_normals[i]

vertex_normals /= numpy.linalg.norm(vertex_normals, axis=1, keepdims=True)

# Save the pts file

point_cloud = numpy.hstack((vertices, vertex_normals))

numpy.savetxt(output_file, point_cloud, fmt='%.6f %.6f %.6f %.6f %.6f %.6f', delimiter=' ')