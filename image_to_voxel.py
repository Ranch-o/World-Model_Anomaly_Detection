import numpy as np
import open3d as o3d
import sys
import argparse
from numba import jit
import os
from PIL import Image



bev_offset_forward = 0 # in px
bev_resolution = 0.2
offset_z = 0 # in px

COLOR_PALETTE = (
        np.array(
            [
                (0, 0, 0),          # unlabeled     =   0u
                (128, 64, 128),     # road          =   1u
                (244, 35, 232),     # sidewalk      =   2u
                (70, 70, 70),       # building      =   3u
                (102, 102, 156),    # wall          =   4u
                (190, 153, 153),    # fence         =   5u
                (153, 153, 153),    # pole          =   6u
                (250, 170, 30),     # traffic light =   7u
                (220, 220, 0),      # traffic sign  =   8u
                (107, 142, 35),     # vegetation    =   9u
                (152, 251, 152),    # terrain       =  10u
                (70, 130, 180),     # sky           =  11u
                (220, 20, 60),      # pedestrian    =  12u
                (255, 0, 0),        # rider         =  13u
                (0, 0, 142),        # Car           =  14u
                (0, 0, 70),         # truck         =  15u
                (0, 60, 100),       # bus           =  16u
                (0, 80, 100),       # train         =  17u
                (0, 0, 230),        # motorcycle    =  18u
                (119, 11, 32),      # bicycle       =  19u
                (110, 190, 160),    # static        =  20u
                (170, 120, 50),     # dynamic       =  21u
                (55, 90, 80),       # other         =  22u
                (45, 60, 150),      # water         =  23u
                (157, 234, 50),     # road line     =  24u
                (81, 0, 81),        # ground         = 25u
                (150, 100, 100),    # bridge        =  26u
                (230, 150, 140),    # rail track    =  27u
                (180, 165, 180),    # guard rail    =  28u
                (250, 128, 114),    # home          =  29u
                (255, 36, 0),       # animal        =  30u
                (224, 17, 95),      # nature        =  31u
                (184, 15, 10),      # special       =  32u
                (245, 0, 0),        # airplane      =  33u
                (245, 0, 0),        # falling       =  34u
            ]
        )
)

WORLD_SIZE = [1000,1000,64]


parser = argparse.ArgumentParser(description='OOD Evaluation')



parser.add_argument('--dataset_path', type=str, default='/disk/vanishing_data/du541/AnoVox(Sample)/Anovox',
                    help=""""path to depth images""")

parser.add_argument('--camera_fov', type=float, default=90.0,
                    help=""""Value for camera field of view""")

parser.add_argument('--voxel_resolution', type=float, default=0.5,
                    help=""""size for a single voxel""")

parser.add_argument('--depth_preds', default=False, action='store_true',
                    help="""""set true for separate depth prediction instead of carla ground truth""")



args = parser.parse_args()

def check_file_ending(path):
    file_ending = path.split(".")[-1]
    print("     ending", file_ending)
    return file_ending == "npy"

def collect_carla_depth_img():
    root = args.dataset_path
    depth_data = []

    for scenario in os.listdir(root):
        scenario_path = os.path.join(root, scenario)
        # Skip if it's not a directory or if it's a .DS_Store file
        if not os.path.isdir(scenario_path) or scenario == '.DS_Store':
            continue

        depth_dir = os.path.join(scenario_path, 'DEPTH_IMG')
        # Check if DEPTH_IMG is a valid directory
        if not os.path.isdir(depth_dir):
            print(f"Warning: Expected directory not found: {depth_dir}")
            continue

        for image in os.listdir(depth_dir):
            depth_data.append(os.path.join(depth_dir, image))        

        # if scenario == 'Scenario_Configuration_Files':
        #     continue
        # depth_dir = os.path.join(root, scenario, 'DEPTH_IMG')
        # for image in os.listdir(depth_dir):
        #     depth_data.append(os.path.join(depth_dir, image))
    
    def sorter(file_paths):
        try:
            identifier = (os.path.basename(file_paths).split('.')[0]).split('_')[-1]
            return int(identifier)
        except ValueError:
            return float('inf')  # In case the identifier is not an integer or is empty


        # identifier = (os.path.basename(file_paths).split('.')[0]).split('_')[-1]
        # return int(identifier)
    depth_data.sort(key=sorter)
    
    return depth_data



@jit(nopython=True)  # Compile the function with Numba
def calculate_carla_depth(depth_img, new_depth, height, width):
    # depth_img = Image.open(depth_img)
    depth_img_arr = new_depth
    # height, width, _  = depth_img_arr.shape
    # Get the dimensions of the image
    # depth_img_arr = np.zeros((depth_img.height, depth_img.width,))
    # Loop through each pixel in the depth_img
    for row in range(height):
        for col in range(width):
            pixel = depth_img[row, col]
            depth_img_arr[row, col] = 1000 * (pixel[0] + pixel[1] * 256 + pixel[2] * 256 * 256) / (
                256 * 256 * 256 - 1
            )
    return depth_img_arr


def transform_pcd(scores, depths):
    # print("scores shape", scores.shape)
    # print("scores img val", scores[:,:,])
    height, width = depths.shape

    camera_fov = args.camera_fov
    focal = width / (2.0 * np.tan(camera_fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    cx = width / 2.0  # from camera projection matrix
    cy = height / 2.0

    # voxel_size = args.voxel_size

    point_list = []
    anomaly_score_list = []

    # for i in range(height):
    #     for j in range(width):
    #         z = depths[i,j]
    #         anomaly_score = scores[i][j]
    #         # depth encoded in rgb values. For further information take a look at carla docs
    #         # depth_color[2] = B
    #         # depth_color[1] = G
    #         # depth_color[0] = R

    #         x, y = (j - cx) * z / focal, (i - cy) * z / focal
    #         coordinate = [x,y,z]

    #         if np.linalg.norm(coordinate) > 100:
    #             continue
    #         else:
    #             point_list.append(coordinate)
    #         # red, green, blue = (
    #         #     semantic_color[0],
    #         #     semantic_color[1],
    #         #     semantic_color[2],
    #         # )
    #         # point_color = [
    #         #     i / 255.0 for i in [red, green, blue]
    #         # ]  # format to o3d color values
    #         anomaly_score_list.append(anomaly_score)

    # Original crop dimensions
    original_crop = [64, 138, 896, 458]  # left, top, right, bottom

    # Original and new sizes
    original_size = (960, 600)
    new_size = (768, 512)

    # Calculate scaling factors
    scale_width = new_size[0] / original_size[0]
    scale_height = new_size[1] / original_size[1]

    # Adjust crop dimensions to the scaled size
    scaled_crop = [
        int(original_crop[0] * scale_width),  # left
        int(original_crop[1] * scale_height),  # top
        int(original_crop[2] * scale_width),  # right
        int(original_crop[3] * scale_height)   # bottom
    ]

    # Voxelization loop
    for i in range(height):
        for j in range(width):
            # Check if the pixel is within the meaningful region
            if not (scaled_crop[0] <= j < scaled_crop[2] and scaled_crop[1] <= i < scaled_crop[3]):
                continue  # Skip pixels outside the meaningful region

            z = depths[i, j]
            anomaly_score = scores[i][j]

            x, y = (j - cx) * z / focal, (i - cy) * z / focal
            coordinate = [x, y, z]

            if np.linalg.norm(coordinate) > 100:
                continue
            else:
                point_list.append(coordinate)

            anomaly_score_list.append(anomaly_score)





    depth_pcloud = o3d.geometry.PointCloud()  # create point cloud object
    depth_pcloud.points = o3d.utility.Vector3dVector(
        point_list
    )  # set pcd_np as the point cloud points
    # depth_pcloud.colors = o3d.utility.Vector3dVector(color_list)
    # point cloud needs to be rotated to fit with lidar point cloud in the future
    r_matrix = depth_pcloud.get_rotation_matrix_from_xyz(
        (-np.pi / 2, np.pi / 2, 0)
    )  # rotation matrix
    depth_pcloud.rotate(
        r_matrix, center=(0, 0, 0)
    )  # rotate depth point cloud to fit lidar point cloud
    # o3d.visualization.draw_geometries([depth_pcloud])
    depth_points = np.asarray(depth_pcloud.points)
    anomaly_score_list = np.asarray(anomaly_score_list)
    # print(np.amin(anomaly_score_list))
    # print(np.amax(anomaly_score_list))
    anomaly_score_list = np.reshape(np.array(anomaly_score_list), (len(anomaly_score_list), 1))
    return np.concatenate([depth_points, anomaly_score_list], axis=1)


def transform_pcd_o3d(depth_img_path, eval_img):
    # depth_img_pil = Image.open(depth_img_path)
    # height, width = depth_img_pil.height, depth_img_pil.width

    depth_img_array = np.load(depth_img_path)
    # depth_img_array = depth_img_array.astype(np.uint16)
    height, width = depth_img_array.shape
    print("dtype:", depth_img_array.dtype)
    # depth_img = o3d.io.read_image(depth_img_path)
    depth_img = o3d.geometry.Image(depth_img_array)


    camera_fov = args.camera_fov
    focal = width / (2.0 * np.tan(camera_fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    cx = width / 2.0  # from camera projection matrix
    cy = height / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, focal,focal, cx, cy)
    intrinsic.intrinsic_matrix = [[focal, 0, cx], [0, focal, cy], [0, 0, 1]]
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic

    # pcloud = o3d.geometry.PointCloud()
    o3d.visualization.draw_geometries([depth_img])


    # cam.extrinsic = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])
    pcloud = o3d.geometry.PointCloud().create_from_depth_image(depth_img, cam.intrinsic)
    print(pcloud)
    o3d.visualization.draw_geometries([pcloud])





def img2pcd(score_img, depth_img):
    # eval_image = np.load(score_img)
    score_arr = Image.open(score_img)
    eval_image = np.array(score_arr)

    if args.depth_preds:
        depth_img_array = np.load(depth_img)
        pcloud = transform_pcd(eval_image, depth_img_array)
        # depth_img_ = o3d.io.read_image(depth_img)
        # transform_pcd_o3d(depth_img, eval_image)

    else:
        depth_img = Image.open(depth_img)
    
        depth_img_array = calculate_carla_depth(np.array(depth_img), np.zeros((depth_img.height, depth_img.width,)),
                                            depth_img.height, depth_img.width)

        pcloud = transform_pcd(eval_image, depth_img_array)

    file_name = "voxel" + os.path.basename(str(score_img))

    return pcloud, file_name
    




def voxelize_one(pcloud, save_name, pipe=None):
    offset_x = bev_offset_forward * bev_resolution
    offset_z = 0
    voxel_points, semantics = voxel_filter(pcloud, args.voxel_resolution, WORLD_SIZE, [offset_x, 0, offset_z])
    data = np.concatenate([voxel_points, semantics], axis=1) # [:, None]
    # voxels = np.zeros(shape=cfg.voxel_size, dtype=np.uint8)
    # voxels[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]] = semantics
    # csr_voxels = sp.csr_matrix(voxels.reshape(voxels.shape[0], -1))
    np.save(f'{save_name}', data)
    # np.save(f'{save_path}/voxel_coo/voxel_coo_{name}.npy', csr_voxels)

    if pipe is not None:
        pipe.send(['x'])


def voxel_filter(pcloud, voxel_resolution, grid_size, offset):
    # pcd = np.asarray(pcloud.points)
    # sem = (np.asarray(pcloud.colors) * 255.0).astype(np.uint8)
    pcd = pcloud[:, :3]
    sem = pcloud[:, -1]
    sem = sem / 255.0
    print('sem values', sem)
    print(pcloud.shape)
    # new_sem = np.arange(len(sem))
    # for i, value in enumerate(sem): # 
    #     color_index = np.where((COLOR_PALETTE == value).all(axis = 1))
    #     # print(value)
    #     # for color in COLOR_PALETTE:
    #     #     if (value == color).all():                
    #     #         new_sem[i] = np.where((COLOR_PALETTE == value))[0][0]
    #     # print(color_index)
    #     new_sem[i] = color_index[0][0]
    # sem = new_sem
    # unique, counts = np.unique(sem, return_counts = True)
    # print(dict(zip(unique, counts)))
    grid_size = np.asarray(grid_size)
    offset = np.asarray(offset)
    offset += voxel_resolution * grid_size / 2
    pcd_b = pcd + offset
    idx = ((0 <= pcd_b) & (pcd_b < grid_size * voxel_resolution)).all(axis=1)
    pcd_b, sem_b = pcd_b[idx], sem[idx] # limit point cloud to voxel grid size

    Dx, Dy, Dz = grid_size
    # compute index for every point in a voxel
    hxyz, hmod = np.divmod(pcd_b, voxel_resolution)
    h = hxyz[:, 0] + hxyz[:, 1] * Dx + hxyz[:, 2] * Dx * Dy

    # h_n = np.nonzero(np.bincount(h.astype(np.int32)))
    h_idx = np.argsort(h)
    h, hxyz, sem_b, pcd_b, hmod = h[h_idx], hxyz[h_idx], sem_b[h_idx], pcd_b[h_idx], hmod[h_idx]
    h_n, indices = np.unique(h, return_index=True)
    n_f = h_n.shape[0]
    n_all = h.shape[0]
    voxels = np.zeros((n_f, 3), dtype=np.uint16)
    semantics = np.zeros((n_f,1), dtype=np.float64)
    # points_f = np.zeros((n_f, 3))
    # road_idx = np.where((COLOR_PALETTE == (157, 234, 50)).all(axis = 1))[0][0] # roadline 24u
    # road_idx = np.where(LABEL_CLASS == 'roadlines')[0][0]
    # voxels = []
    # semantics = []
    # points_f = []
    for i in range(n_f):
        # idx_ = (h == h_n[i])
        idx_ = np.arange(indices[i], indices[i+1]) if i < n_f - 1 else np.arange(indices[i], n_all)
        # dis = np.sum(hmod[idx_] ** 2, axis=1)

        avg_score = np.mean(sem_b[idx_])

        # semantic = sem_b[idx_][np.argmin(dis)]  # if not np.isin(sem_b[idx_], road_idx).any() else road_idx
        # semantic = np.bincount(sem_b.squeeze()[idx_]).argmax() if not np.isin(sem_b[idx_], road_idx).any() else road_idx
        voxels[i] = hxyz[idx_][0]
        semantics[i] = avg_score
        # points_f[i][2] += center[2] / 2
        # voxels.append(hxyz[idx_][0])
        # semantics.append(semantic)
        # points_f.append(pcd_b[idx_].mean(axis=0) - center)
    # debug_set, counts = np.unique(semantics, return_counts=True)
    return voxels, semantics   



def _voxelize_one(depth_file, lidar_file, cfg, save_name, pipe=None):
    pcd, sem = merge_pcd(depth_file, lidar_file, cfg.camera_position, cfg.lidar_position, cfg.fov)
    offset_x = cfg.bev_offset_forward * cfg.bev_resolution
    offset_z = cfg.offset_z * cfg.voxel_resolution
    voxel_points, semantics = voxel_filter(pcd, sem, cfg.voxel_resolution, cfg.voxel_size, [offset_x, 0, offset_z])
    data = np.concatenate([voxel_points, semantics[:, None]], axis=1)
    # voxels = np.zeros(shape=cfg.voxel_size, dtype=np.uint8)
    # voxels[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]] = semantics
    # csr_voxels = sp.csr_matrix(voxels.reshape(voxels.shape[0], -1))
    np.save(f'{save_name}', data)
    # np.save(f'{save_path}/voxel_coo/voxel_coo_{name}.npy', csr_voxels)

    if pipe is not None:
        pipe.send(['x'])

def _voxel_filter(pcd, sem, voxel_resolution, voxel_size, offset):
    voxel_size = np.asarray(voxel_size)
    offset = np.asarray(offset)
    offset += voxel_resolution * voxel_size / 2
    pcd_b = pcd + offset
    idx = ((0 <= pcd_b) & (pcd_b < voxel_size * voxel_resolution)).all(axis=1)
    pcd_b, sem_b = pcd_b[idx], sem[idx]

    Dx, Dy, Dz = voxel_size
    # compute index for every point in a voxel
    hxyz, hmod = np.divmod(pcd_b, voxel_resolution)
    h = hxyz[:, 0] + hxyz[:, 1] * Dx + hxyz[:, 2] * Dx * Dy

    # h_n = np.nonzero(np.bincount(h.astype(np.int32)))
    h_idx = np.argsort(h)
    h, hxyz, sem_b, pcd_b, hmod = h[h_idx], hxyz[h_idx], sem_b[h_idx], pcd_b[h_idx], hmod[h_idx]
    h_n, indices = np.unique(h, return_index=True)
    n_f = h_n.shape[0]
    n_all = h.shape[0]
    voxels = np.zeros((n_f, 3), dtype=np.uint16)
    semantics = np.zeros((n_f, ), dtype=np.uint8)
    # points_f = np.zeros((n_f, 3))
    road_idx = np.where(LABEL_CLASS == 'roadlines')[0][0]
    # voxels = []
    # semantics = []
    # points_f = []
    for i in range(n_f):
        # idx_ = (h == h_n[i])
        idx_ = np.arange(indices[i], indices[i+1]) if i < n_f - 1 else np.arange(indices[i], n_all)
        dis = np.sum(hmod[idx_] ** 2, axis=1)
        semantic = sem_b[idx_][np.argmin(dis)] if not np.isin(sem_b[idx_], road_idx).any() else road_idx
        # semantic = np.bincount(sem_b.squeeze()[idx_]).argmax() if not np.isin(sem_b[idx_], road_idx).any() else road_idx
        voxels[i] = hxyz[idx_][0]
        semantics[i] = semantic
        # points_f[i] = pcd_b[idx_].mean(axis=0) - center
        # points_f[i][2] += center[2] / 2
        # voxels.append(hxyz[idx_][0])
        # semantics.append(semantic)
        # points_f.append(pcd_b[idx_].mean(axis=0) - center)

    return voxels, semantics


def main():
    # depth_img_dir = os.path.join(args.dataset_path, 'DEPTH_IMG')
    scores_dir = '/disk/vanishing_data/du541/anomaly_scores/768_512_scores_anovox'
    # scores_dir = '/home/tes_unreal/Desktop/BA/RbA/anomaly_scores/swin_b_1dl/anovox'

    scores_data = sorted(os.listdir(scores_dir))

    if args.depth_preds:
        depth_data = []
        for depth_pred in sorted(os.listdir(args.dataset_path)):
            depth_data.append(os.path.join(args.dataset_path, depth_pred))
    else:
        depth_data = collect_carla_depth_img()
    

    output_path = '/disk/vanishing_data/du541//voxelpreds'
    # output_path = '/home/tes_unreal/Desktop/BA/RbA/voxelpreds'

    # os.mkdir(output_path)

    for i, score in enumerate(scores_data):
        # if i<1:
        #     break
        score = os.path.join(scores_dir, score)
        depth_img = depth_data[i]
        image_point_cloud, file_name = img2pcd(score, depth_img)
        file_path = os.path.join(output_path, file_name)
        voxelize_one(image_point_cloud, file_path)


if __name__ == "__main__":
    main()
    # for input in inputs:
    #     print(f"## Opening {input} ##")
    #     if check_file_ending(input):
    #         transform_npy(input)
    #     else:
    #         print("     --> wrong filetype")
