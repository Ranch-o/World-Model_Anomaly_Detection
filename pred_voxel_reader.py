import sys
import open3d as o3d
import numpy as np

voxel_resolution = 0.5

def check_file_ending(path):
    file_ending = path.split(".")[-1]
    print("     ending", file_ending)
    if file_ending == "npy":
        return True
    return False


def open_ply(file_path):
    try:
        o3d.visualization.draw_geometries([o3d.io.read_voxel_grid(file_path)])
    except Exception as e:
        print("Could not open file:", file_path)
        print(e)

# def open_npy(file_path):
#     voxel_array = np.load(file_path)
#     voxel_coors = voxel_array[:, :3]
#     voxel_colors = voxel_array[:, 3:]
#     print(voxel_coors)
#     print(voxel_colors)

def open_npy(file_path):
    try:
        voxel_pcd = o3d.geometry.PointCloud()

        voxel_data = np.load(file_path)
        voxel_points = voxel_data[:,:3]
        voxel_colors = voxel_data[:,3:] * 1.7
        scores, counts = np.unique(voxel_colors, return_counts=True)        # voxel_colors = np.squeeze(Definitions.COLOR_PALETTE[voxel_colors]) / 255.0
        voxel_colors = np.c_[voxel_colors, np.zeros((voxel_colors.size,2))] 
        voxel_pcd.colors = o3d.utility.Vector3dVector(voxel_colors)
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
        voxel_world = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_pcd, voxel_resolution)
        o3d.visualization.draw_geometries([voxel_world])
    except Exception as e:
        print("Could not open file:", file_path)
        print(e)

if __name__ == "__main__":
    inputs = sys.argv[1:]
    for input in inputs:
        print(f"## Opening {input} ##")
        if check_file_ending(input):
            open_npy(input)
        else:
           print("     --> wrong filetype")
