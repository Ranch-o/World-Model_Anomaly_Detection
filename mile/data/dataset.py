import os
from glob import glob
from PIL import Image
import csv

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader

from constants import CARLA_FPS, EGO_VEHICLE_DIMENSION, LABEL_MAP, VOXEL_LABEL, VOXEL_LABEL_CARLA
from mile.data.dataset_utils import integer_to_binary, calculate_birdview_labels, calculate_instance_mask
from mile.utils.geometry_utils import get_out_of_view_mask, calculate_geometry, lidar_to_histogram_features
from mile.utils.geometry_utils import PointCloud
from data.data_preprocessing import convert_coor_lidar


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, dataset_root=None):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.BATCHSIZE
        self.sequence_length = self.cfg.RECEPTIVE_FIELD + self.cfg.FUTURE_HORIZON

        self.dataset_root = dataset_root if dataset_root else self.cfg.DATASET.DATAROOT

        # Will be populated with self.setup()
        self.train_dataset, self.val_dataset = None, None
        self.predict_dataset = None

    def setup(self, stage=None):
        self.train_dataset = CarlaDataset(
            self.cfg, mode='train', sequence_length=self.sequence_length, dataset_root=self.dataset_root
        )
        self.val_dataset = CarlaDataset(
            self.cfg, mode='val', sequence_length=self.sequence_length, dataset_root=self.dataset_root
        )
        self.predict_dataset = CarlaDataset(
            self.cfg, mode='train', sequence_length=self.sequence_length, dataset_root=self.dataset_root
        )

        print(f'{len(self.train_dataset)} data points in {self.train_dataset.dataset_path}')
        print(f'{len(self.val_dataset)} data points in {self.val_dataset.dataset_path}')
        print(f'{len(self.predict_dataset)} data points in prediction')

        self.train_sampler = None
        self.val_sampler = None
        self.predict_sampler = range(0, len(self.predict_dataset), 200)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_sampler is None,
            num_workers=self.cfg.N_WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.N_WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=self.val_sampler,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.N_WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=self.predict_sampler,
        )



class AnovoxDataset0(Dataset):
    def __init__(
        self,
        cfg,
        dataset_root=None,
        runs_filter="*",
        img_base="RGB_IMG",
        pcd_base="PCD",
        semantic_pcd_base="SEMANTIC_PCD",
        action_base="ACTION",
        folder_exclude="Scenario_Configuration_Files",
        file_exclude="color_palette.txt",
        scenario_filter="*",
    ):
        self.cfg = cfg
        self.sequence_length = self.cfg.RECEPTIVE_FIELD + self.cfg.FUTURE_HORIZON

        self.dataset_path = os.path.join(dataset_root)
        self.intrinsics, self.extrinsics = calculate_geometry_from_config(self.cfg)

        # Iterate over all runs in the data folder
        runs = sorted(glob(os.path.join(self.dataset_path, scenario_filter)))
        runs = [run for run in runs if folder_exclude not in run and file_exclude not in run]

        self.image_paths = []
        self.pcd_paths = []
        self.semantic_pcd_paths = []
        self.action_paths = []

        for run_path in runs:
            run = os.path.basename(run_path)

            # Load image paths
            images = sorted(glob(os.path.join(self.dataset_path, run, img_base, "*.png")))
            self.image_paths.append(images)

            # Load PCD paths
            pcds = sorted(glob(os.path.join(self.dataset_path, run, pcd_base, "*.pcd")))
            self.pcd_paths.append(pcds)

            # Load Semantic PCD paths
            semantic_pcds = sorted(glob(os.path.join(self.dataset_path, run, semantic_pcd_base, "*.pcd")))
            self.semantic_pcd_paths.append(semantic_pcds)

            # Load action paths
            actions = sorted(glob(os.path.join(self.dataset_path, run, action_base, "*.csv")))
            self.action_paths.append(actions)

            # Check equal length
            if not all(len(lst) == len(images) for lst in [images, pcds, semantic_pcds, actions]):
                print("ERROR: Number of actions, images, and PCDs not equal!")

        self.data_pointers = self.get_data_pointers()

    def get_data_pointers(self):
        data_pointers = []

        for i, scenario in enumerate(self.image_paths):  # assums that the lenths of all paths is the same
            run_length = len(scenario)
            scenario_index = i

            stride = int(self.cfg.DATASET.STRIDE_SEC * CARLA_FPS)
            # Loop across all elements in the dataset, and make all elements in a sequence belong to the same run
            start_index = 0
            total_length = run_length - stride * self.sequence_length
            for i in range(start_index, total_length):
                frame_indices = range(i, i + stride * self.sequence_length, stride)
                data_pointers.append((scenario_index, list(frame_indices)))

        return data_pointers

    def __getitem__(self, i):
        batch = {}

        run_id, indices = self.data_pointers[i]
        for t in indices:
            try:
                single_element_t = self.load_single_element_time_t(run_id, t)
            except:
                print(f"{run_id}, {t} data is invalid")
                continue

            for k, v in single_element_t.items():
                batch[k] = batch.get(k, []) + [v]

        for k, v in batch.items():
            batch[k] = torch.from_numpy(np.stack(v))

        return batch

    def load_single_element_time_t(self, run_id, t):
        single_element_t = super().load_single_element_time_t(run_id, t)

        # Load PCD
        pcd_path = self.pcd_paths[run_id][t]
        pcd_data = self.load_pcd(pcd_path)
        single_element_t["pcd"] = pcd_data

        # Load Semantic PCD
        semantic_pcd_path = self.semantic_pcd_paths[run_id][t]
        semantic_pcd_data = self.load_pcd(semantic_pcd_path)
        single_element_t["semantic_pcd"] = semantic_pcd_data

        return single_element_t

    def load_pcd(self, pcd_path):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pcd_path)
        return np.asarray(pcd.points)
        pass





class AnovoxDataset(Dataset):
    def __init__(
        self,
        cfg,
        dataset_root=None,
        runs_filter="*",
        img_base="RGB_IMG",
        action_base="ACTION",
        folder_exclude="Scenario_Configuration_Files",
        file_exclude="color_palette.txt",
        scenario_filter="*",
    ):
        self.cfg = cfg
        self.sequence_length = self.cfg.RECEPTIVE_FIELD + self.cfg.FUTURE_HORIZON

        self.dataset_path = os.path.join(dataset_root)
        self.intrinsics, self.extrinsics = calculate_geometry_from_config(self.cfg)

        # Iterate over all runs in the data folder
        runs = sorted(glob(os.path.join(self.dataset_path, scenario_filter)))

        # Filter out non-scenario folder and file in AnoVox dataset
        runs = [run for run in runs if folder_exclude not in run and file_exclude not in run]

        self.image_paths = []
        self.action_paths = []
        for run_path in runs:
            run = os.path.basename(run_path)

            # Load image paths
            images = sorted(glob(os.path.join(self.dataset_path, run, img_base, "*.png")))
            self.image_paths.append(images)

            # Load action paths
            actions = sorted(glob(os.path.join(self.dataset_path, run, action_base, "*.csv")))
            self.action_paths.append(actions)

            # Check equal length
            if len(images) != len(actions):
                print("ERROR: Number of actions and images not equal!")

        self.data_pointers = self.get_data_pointers()

    def get_data_pointers(self):
        data_pointers = []

        for i, scenario in enumerate(self.image_paths):  # assums that the lenths of all paths is the same
            run_length = len(scenario)
            scenario_index = i

            stride = int(self.cfg.DATASET.STRIDE_SEC * CARLA_FPS)
            # Loop across all elements in the dataset, and make all elements in a sequence belong to the same run
            start_index = 0
            total_length = run_length - stride * self.sequence_length
            for i in range(start_index, total_length):
                frame_indices = range(i, i + stride * self.sequence_length, stride)
                data_pointers.append((scenario_index, list(frame_indices)))

        return data_pointers

    def __getitem__(self, i):
        batch = {}

        run_id, indices = self.data_pointers[i]
        for t in indices:
            try:
                single_element_t = self.load_single_element_time_t(run_id, t)
            except:
                print(f"{run_id}, {t} data is invalid")
                continue

            for k, v in single_element_t.items():
                batch[k] = batch.get(k, []) + [v]

        for k, v in batch.items():
            batch[k] = torch.from_numpy(np.stack(v))

        return batch

    def load_single_element_time_t(self, run_id, t):
        image_path = self.image_paths[run_id][t]
        action_path = self.action_paths[run_id][t]
        single_element_t = {}

        # Load image
        image = Image.open(image_path)
        image = np.asarray(image).transpose((2, 0, 1))
        single_element_t["image"] = image

        # Load action
        throttle = None
        brake = None
        steer = None

        with open(action_path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            for row in reader:
                if len(row) == 2:
                    key, value = row
                    if key.strip() == "throttle":
                        throttle = float(value.strip())
                    elif key.strip() == "brake":
                        brake = float(value.strip())
                    elif key.strip() == "steer":
                        steer = float(value.strip())

        # Check successful read
        if throttle == None or brake == None or steer == None:
            print("ERROR: Action not read actions correctly from CSV!")

        # Combine throttle and brake into one value
        throttle_brake = throttle if throttle > 0 else -brake

        single_element_t["steering"] = steer
        single_element_t["throttle_brake"] = throttle_brake

        # Geometry
        single_element_t["intrinsics"] = self.intrinsics.copy()
        single_element_t["extrinsics"] = self.extrinsics.copy()

        return single_element_t

    def __len__(self):
        return len(self.data_pointers)


    def predict_dataloader(self, batch_size=None, num_workers=1, pin_memory=True, drop_last=True):
        if batch_size is None:
            batch_size = self.cfg.BATCHSIZE  # Use the batch size from the config if not provided
    
        # Since AnovoxDataset is an instance of Dataset, we can directly use DataLoader on it
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )


# class ExtendedAnovoxDataset(AnovoxDataset):
#     def __init__(
#         self,
#         cfg,
#         dataset_root=None,
#         runs_filter="*",
#         img_base="RGB_IMG",
#         pcd_base="PCD",  # new parameter for PCD base directory
#         action_base="ACTION",
#         folder_exclude="Scenario_Configuration_Files",
#         file_exclude="color_palette.txt",
#         scenario_filter="*",
#     ):
#         super().__init__(cfg, dataset_root, runs_filter, img_base, action_base, folder_exclude, file_exclude, scenario_filter)

#         self.pcd_paths = []  # new attribute to store PCD paths
#         runs = sorted(glob(os.path.join(self.dataset_path, scenario_filter)))
#         runs = [run for run in runs if folder_exclude not in run and file_exclude not in run]

#         for run_path in runs:
#             run = os.path.basename(run_path)

#             # Load PCD paths
#             pcds = sorted(glob(os.path.join(self.dataset_path, run, pcd_base, "*.pcd")))
#             self.pcd_paths.append(pcds)

#             # Check if lengths of images, actions, and PCDs are the same
#             if len(self.image_paths[-1]) != len(self.action_paths[-1]) or len(self.image_paths[-1]) != len(pcds):
#                 print("ERROR: Number of actions, images, and PCDs not equal!")

#     def load_single_element_time_t(self, run_id, t):
#         single_element_t = super().load_single_element_time_t(run_id, t)

#         # Load PCD
#         pcd_path = self.pcd_paths[run_id][t]
#         # Here, you need a method to load your PCD data. 
#         # This is a placeholder, replace with your actual PCD loading method.
#         pcd_data = self.load_pcd(pcd_path)
#         single_element_t["pcd"] = pcd_data

#         return single_element_t

#     def load_pcd(self, pcd_path):
#         # Implement your PCD loading method here.
#         # For instance, if using Open3D:
#         # import open3d as o3d
#         # pcd = o3d.io.read_point_cloud(pcd_path)
#         # return np.asarray(pcd.points)
#         pass





class CarlaDataset(Dataset):
    def __init__(self, cfg, mode='train', sequence_length=1, dataset_root=None, towns_filter='*', runs_filter='*'):
        self.cfg = cfg
        self.mode = mode
        self.sequence_length = sequence_length

        self.dataset_path = os.path.join(dataset_root, self.cfg.DATASET.VERSION, mode)
        self.intrinsics, self.extrinsics = calculate_geometry_from_config(self.cfg)
        self.bev_out_of_view_mask = get_out_of_view_mask(self.cfg)
        self.pcd = PointCloud(
            self.cfg.POINTS.CHANNELS,
            self.cfg.POINTS.HORIZON_RESOLUTION,
            *self.cfg.POINTS.FOV,
            self.cfg.POINTS.LIDAR_POSITION,
        )

        # Iterate over all runs in the data folder

        self.data = dict()

        towns = sorted(glob(os.path.join(self.dataset_path, towns_filter)))
        for town_path in towns:
            town = os.path.basename(town_path)

            runs = sorted(glob(os.path.join(self.dataset_path, town, runs_filter)))
            for run_path in runs:
                run = os.path.basename(run_path)
                pd_dataframe_path = os.path.join(run_path, 'pd_dataframe.pkl')

                if os.path.isfile(pd_dataframe_path):
                    self.data[f'{town}/{run}'] = pd.read_pickle(pd_dataframe_path)

        self.data_pointers = self.get_data_pointers()

    def get_data_pointers(self):
        data_pointers = []

        n_filtered_run = 0
        for run, data_run in self.data.items():
            # Calculate normalised reward of the run
            run_length = len(data_run['reward'])
            cumulative_reward = data_run['reward'].sum()
            normalised_reward = cumulative_reward / run_length
            if normalised_reward < self.cfg.DATASET.FILTER_NORM_REWARD:
                n_filtered_run += 1
                continue

            stride = int(self.cfg.DATASET.STRIDE_SEC * CARLA_FPS)
            # Loop across all elements in the dataset, and make all elements in a sequence belong to the same run
            start_index = int(CARLA_FPS * self.cfg.DATASET.FILTER_BEGINNING_OF_RUN_SEC)
            total_length = len(data_run) - stride * self.sequence_length
            for i in range(start_index, total_length):
                frame_indices = range(i, i + stride * self.sequence_length, stride)
                data_pointers.append((run, list(frame_indices)))

        print(f'Filtered {n_filtered_run} runs in {self.dataset_path}')

        if self.cfg.EVAL.DATASET_REDUCTION:
            import random
            random.seed(0)
            final_size = int(len(data_pointers) / self.cfg.EVAL.DATASET_REDUCTION_FACTOR)
            data_pointers = random.sample(data_pointers, final_size)

        return data_pointers

    def __len__(self):
        return len(self.data_pointers)

    def __getitem__(self, i):
        batch = {}

        run_id, indices = self.data_pointers[i]
        for t in indices:
            try:
                single_element_t = self.load_single_element_time_t(run_id, t)
            except:
                print(f'{run_id}, {t} data is invalid')
                continue

            for k, v in single_element_t.items():
                batch[k] = batch.get(k, []) + [v]

        for k, v in batch.items():
            batch[k] = torch.from_numpy(np.stack(v))

        return batch

    def load_single_element_time_t(self, run_id, t):
        data_row = self.data[run_id].iloc[t]
        single_element_t = {}

        # Load image
        image = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['image_path'])
        )
        image = np.asarray(image).transpose((2, 0, 1))
        single_element_t['image'] = image

        # Load route map
        route_map = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['routemap_path'])
        )
        route_map = np.asarray(route_map)[None]
        # Make the grayscale image an RGB image
        _, h, w = route_map.shape
        route_map = np.broadcast_to(route_map, (3, h, w)).copy()
        single_element_t['route_map'] = route_map

        # Load bird's-eye view segmentation label
        birdview = np.asarray(Image.open(
            os.path.join(self.dataset_path, run_id, data_row['birdview_path'])
        ))
        h, w = birdview.shape
        n_classes = data_row['n_classes']
        birdview = integer_to_binary(birdview.reshape(-1), n_classes).reshape(h, w, n_classes)
        birdview = birdview.transpose((2, 0, 1))
        single_element_t['birdview'] = birdview
        birdview_label = calculate_birdview_labels(torch.from_numpy(birdview), n_classes).numpy()
        birdview_label = birdview_label[None]
        single_element_t['birdview_label'] = birdview_label

        # TODO: get person and car instance ids with json
        instance_mask = birdview[3].astype(np.bool) | birdview[4].astype(np.bool)
        instance_label, _ = scipy.ndimage.label(instance_mask[None].astype(np.int64))
        single_element_t['instance_label'] = instance_label

        # # Load lidar points clouds
        # pcd = np.load(
        #     os.path.join(self.dataset_path, run_id, data_row['points_path']),
        #     allow_pickle=True).item()  # n x 4, (x, y, z, intensity)
        # single_element_t['points_intensity'] = np.concatenate([pcd['points_xyz'], pcd['intensity'][:, None]], axis=-1)
        pcd_semantic = np.load(
            os.path.join(self.dataset_path, run_id, data_row['points_semantic_path']),
            allow_pickle=True).item()
        points = convert_coor_lidar(pcd_semantic['points_xyz'], self.cfg.POINTS.LIDAR_POSITION)

        # remap labels
        remap = np.full((max(LABEL_MAP.keys()) + 1), max(LABEL_MAP.values()), dtype=np.uint8)
        remap[list(LABEL_MAP.keys())] = list(LABEL_MAP.values())
        semantics = remap[pcd_semantic['ObjTag']]

        # mask ego-vehicle
        x, y, z = EGO_VEHICLE_DIMENSION
        ego_box = np.array([[-x / 2, -y / 2, 0], [x / 2, y / 2, z]])
        ego_idx = ((ego_box[0] < points) & (points < ego_box[1])).all(axis=1)
        semantics = semantics[~ego_idx]
        points = points[~ego_idx]
        # single_element_t['points'] = points
        # single_element_t['points_label'] = pcd_semantic['ObjTag'].astype('uint8')
        # single_element_t['points_histogram_xy'], \
        # single_element_t['points_histogram_xz'], \
        # single_element_t['points_histogram_yz'] = lidar_to_histogram_features(points, self.cfg)

        range_view_pcd_depth, range_view_pcd_xyz, range_view_pcd_sem = self.pcd.do_range_projection(points, semantics)
        if self.cfg.MODEL.LIDAR.ENABLED:
            single_element_t['range_view_pcd_xyzd'] = np.concatenate(
                [range_view_pcd_xyz, range_view_pcd_depth[..., None]], axis=-1).transpose((2, 0, 1))  # x y z d
        if self.cfg.LIDAR_SEG.ENABLED:
            single_element_t['range_view_pcd_seg'] = range_view_pcd_sem[None].astype(int)

        # Load voxels
        if self.cfg.VOXEL_SEG.ENABLED:
            voxel_data = np.load(
                os.path.join(self.dataset_path, run_id, data_row['voxel_path'])
            )
            voxel_points = voxel_data[:, :-1]
            voxel_semantics = voxel_data[:, -1]
            voxel_semantics[voxel_semantics == 255] = 0
            voxel_semantics = remap[voxel_semantics]
            voxels = np.zeros(self.cfg.VOXEL.SIZE, dtype=np.uint8)
            voxels[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]] = voxel_semantics
            single_element_t['voxel'] = voxels[None]

        # load depth_semantic image
        depth_semantic = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['depth_semantic_path'])
        )
        depth_semantic = np.asarray(depth_semantic)
        semantic_image = depth_semantic[..., -1]
        if self.cfg.LOSSES.RGB_INSTANCE:
            single_element_t['image_instance_mask'] = calculate_instance_mask(
                semantic_image[None],
                vehicle_idx=list(VOXEL_LABEL_CARLA.keys())[list(VOXEL_LABEL_CARLA.values()).index('Vehicle')],
                pedestrian_idx=list(VOXEL_LABEL_CARLA.keys())[list(VOXEL_LABEL_CARLA.values()).index('Pedestrian')],
            )

        # load semantic image
        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            single_element_t['semantic_image'] = remap[semantic_image][None].astype(int)
        # load depth
        if self.cfg.DEPTH.ENABLED:
            depth_color = depth_semantic[..., :-1].transpose((2, 0, 1)).astype(float)
            single_element_t['depth_color'] = depth_color / 255.0
            depth = (256 ** 2 * depth_color[0] + 256 * depth_color[1] + depth_color[2]) / (256 ** 3 - 1)
            depth[depth > 0.999] = -1
            single_element_t['depth'] = depth[None]

        # Load action and reward
        throttle, steering, brake = data_row['action']
        throttle_brake = throttle if throttle > 0 else -brake

        single_element_t['steering'] = np.array([steering], dtype=np.float32)
        single_element_t['throttle_brake'] = np.array([throttle_brake], dtype=np.float32)
        single_element_t['speed'] = data_row['speed']

        single_element_t['reward'] = np.array([data_row['reward']], dtype=np.float32).clip(-1.0, 1.0)
        single_element_t['value_function'] = np.array([data_row['value']], dtype=np.float32)

        # Geometry
        single_element_t['intrinsics'] = self.intrinsics.copy()
        single_element_t['extrinsics'] = self.extrinsics.copy()

        return single_element_t


def calculate_geometry_from_config(cfg):
    """ Intrinsics and extrinsics for a single camera.
    See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
    and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
    """
    # Intrinsics
    fov = cfg.IMAGE.FOV
    h, w = cfg.IMAGE.SIZE

    # Extrinsics
    forward, right, up = cfg.IMAGE.CAMERA_POSITION
    pitch, yaw, roll = cfg.IMAGE.CAMERA_ROTATION

    return calculate_geometry(fov, h, w, forward, right, up, pitch, yaw, roll)