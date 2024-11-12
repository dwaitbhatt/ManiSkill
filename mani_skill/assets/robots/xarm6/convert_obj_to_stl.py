import trimesh
import os

# Convert all xarm6 vhacd files
input_dir = '/home/dwait/ManiSkill/mani_skill/assets/robots/xarm6/collision'
output_dir = '/home/dwait/ManiSkill/mani_skill/assets/robots/xarm6/visual'

for filename in os.listdir(input_dir):
    if 'vhacd' in filename and filename.endswith('.obj'):
        mesh = trimesh.load(os.path.join(input_dir, filename))
        output_filename = filename.replace('_vhacd.obj', '.stl.convex.stl')
        mesh.export(os.path.join(output_dir, output_filename))
