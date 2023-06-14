import bpy
import numpy

input_file = "cow.obj"
output_file = "../data/spot-sub-1.pts"
subdivisions = 1

bpy.ops.import_scene.obj(filepath=input_file)
obj = bpy.context.selected_objects[0]

subsurf_mod = obj.modifiers.new('Subdivision', 'SUBSURF')
subsurf_mod.levels = subdivisions
subsurf_mod.render_levels = subdivisions

bpy.ops.object.modifier_apply({'object':obj}, modifier=subsurf_mod.name)

point_cloud = numpy.array([(*v.co, *v.normal) for v in obj.data.vertices])

numpy.savetxt(output_file, point_cloud, fmt='%.6f %.6f %.6f %.6f %.6f %.6f', delimiter=' ')