import bpy
import numpy as np

# Specify the path to your .pts file
pts_file = '../data/spot-sub-0.pts'

# Load point cloud data from the .pts file
data = np.loadtxt(pts_file)

# Separate vertices and normals (if available)
vertices = data[:, :3]
normals = data[:, 3:] if data.shape[1] > 3 else None

# Create a new mesh and link it to a new object
mesh = bpy.data.meshes.new('PointCloud')
obj = bpy.data.objects.new('PointCloud', mesh)

# Link the object to the current collection
bpy.context.collection.objects.link(obj)

# Create vertices in the mesh using the point cloud data
mesh.from_pydata(vertices, [], [])

# Update the mesh with the new data
mesh.update()