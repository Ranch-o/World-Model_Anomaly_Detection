import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt


from mile.models.mile import Mile  # Update the import path based on your directory structure
from mile.config import get_parser, get_cfg

# from clearml import Task
# task = Task.init(project_name='bogdoll/world_model_anomaly_detection', task_name='world_model_anomaly_detection', task_type= 'testing')
import torch.nn.functional as F

import os
import socket
import time
from tqdm import tqdm

import torch
from torch.utils.tensorboard.writer import SummaryWriter
import lightning.pytorch as pl

from mile.config import get_parser, get_cfg
from mile.data.dataset import DataModule, AnovoxDataset
from mile.trainer import WorldModelTrainer

# from clearml import Task, Dataset, Model




# Data Preparation
def load_rgb_image(file_path):
    """Load and preprocess RGB image."""
    image = Image.open(file_path).convert('RGB')
    # Convert to PyTorch tensor
    image_tensor = torch.Tensor(np.array(image))
    print(image_tensor.shape)
    return image_tensor

def load_pcd(file_path):
    """Load and preprocess PCD data."""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    pcd_tensor = torch.Tensor(points)
    return pcd_tensor

# PyTorch Dataset
class AnomalyDataset(Dataset):
    def __init__(self, rgb_folder, pcd_folder):
        self.rgb_files = sorted([os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder) if f.endswith('.png')])
        self.pcd_files = sorted([os.path.join(pcd_folder, f) for f in os.listdir(pcd_folder) if f.endswith('.pcd')])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = load_rgb_image(self.rgb_files[idx])
        pcd = load_pcd(self.pcd_files[idx])
        return rgb, pcd

def visualize_rgb(image_tensor):
    plt.imshow(image_tensor.numpy().astype(np.uint8))
    plt.axis('off')
    plt.show()

def visualize_pcd(pcd_tensor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_tensor.numpy())
    o3d.visualization.draw_geometries([pcd])




def reconstruction_anomaly_data(model, anomaly_data_loader):
   # List to store reconstruction errors for each input
    mse_errors = []
    mae_errors = []
    
    # Pass input data through the model and compute reconstruction errors
    with torch.no_grad():
        for rgb, pcd in anomaly_data_loader:
            # Permute the tensor dimensions to [batch_size, channels, height, width]
            rgb = rgb.permute(0, 3, 1, 2)
            
            # Dummy tensors for additional model inputs
            dummy_speed = torch.zeros((rgb.size(0), rgb.size(1), 1), device=rgb.device)
            dummy_intrinsics = torch.zeros((rgb.size(0), rgb.size(1), 3, 3), device=rgb.device)
            dummy_extrinsics = torch.zeros((rgb.size(0), rgb.size(1), 4, 4), device=rgb.device)
            
            # Get reconstructed outputs
            output, _ = model({
                'image': rgb, 
                'pcd': pcd, 
                'speed': dummy_speed, 
                'intrinsics': dummy_intrinsics, 
                'extrinsics': dummy_extrinsics
            })
            
            # Compute MSE and MAE for RGB and PCD
            mse_rgb = F.mse_loss(output['image'], rgb)
            mse_pcd = F.mse_loss(output['lidar'], pcd)
            mae_rgb = F.l1_loss(output['image'], rgb)
            mae_pcd = F.l1_loss(output['lidar'], pcd)
            
            # Sum up the errors for a combined error per input and append to lists
            mse_errors.append(mse_rgb.item() + mse_pcd.item())
            mae_errors.append(mae_rgb.item() + mae_pcd.item())
    
    # Convert lists to numpy arrays for easier manipulation
    mse_errors = np.array(mse_errors)
    mae_errors = np.array(mae_errors)
    
    # Determine the threshold for anomalies
    # Indices of inputs identified as anomalies based on the MSE metric
    mse_threshold = mse_errors.mean() + 2 * mse_errors.std()
    # Indices of inputs identified as anomalies based on the MAE metric
    mae_threshold = mae_errors.mean() + 2 * mae_errors.std()
    
    # Identify anomalies

    mse_anomalies = np.where(mse_errors > mse_threshold)[0]
    mae_anomalies = np.where(mae_errors > mae_threshold)[0]
    
    return mse_anomalies, mae_anomalies, mse_threshold, mae_threshold, mse_errors, mae_errors


def visualize_errors(mse_errors, mae_errors, mse_threshold, mae_threshold):
    # Plotting the histogram of MSE errors
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(mse_errors, bins=50, alpha=0.7, color='blue', label='MSE Errors')
    plt.axvline(mse_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (MSE)')
    plt.title('Histogram of MSE Errors')
    plt.xlabel('MSE Error')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plotting the histogram of MAE errors
    plt.subplot(1, 2, 2)
    plt.hist(mae_errors, bins=50, alpha=0.7, color='green', label='MAE Errors')
    plt.axvline(mae_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (MAE)')
    plt.title('Histogram of MAE Errors')
    plt.xlabel('MAE Error')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_anomalous_inputs(anomaly_data_loader, anomalies_indices):
    for idx in anomalies_indices:
        rgb, pcd = anomaly_data_loader.dataset[idx]
        
        # Visualize RGB
        visualize_rgb(rgb)
        
        # Visualize PCD
        visualize_pcd(pcd)

def visualize_anomalous_regions(model, anomaly_data_loader, threshold_factor=0.5):
    """
    Visualize the anomalous regions in the RGB frames based on the reconstruction error.
    
    Args:
    - model: The trained model for reconstruction.
    - anomaly_data_loader: DataLoader for the anomaly dataset.
    - threshold_factor: Factor multiplied with max difference to create a binary mask.
    """
    
    with torch.no_grad():
        for rgb, pcd in anomaly_data_loader:
            # Permute the tensor dimensions to [batch_size, channels, height, width]
            rgb = rgb.permute(0, 3, 1, 2)
            
            # Dummy tensors for additional model inputs
            dummy_speed = torch.zeros((rgb.size(0), rgb.size(1), 1), device=rgb.device)
            dummy_intrinsics = torch.zeros((rgb.size(0), rgb.size(1), 3, 3), device=rgb.device)
            dummy_extrinsics = torch.zeros((rgb.size(0), rgb.size(1), 4, 4), device=rgb.device)
            
            # Get reconstructed outputs
            output, _ = model({
                'image': rgb, 
                'pcd': pcd, 
                'speed': dummy_speed, 
                'intrinsics': dummy_intrinsics, 
                'extrinsics': dummy_extrinsics
            })
            
            # Compute the absolute difference between original and reconstructed RGB
            difference = torch.abs(rgb - output['image'])
            
            # Convert to numpy and sum along the channel axis
            difference = difference.sum(dim=1).squeeze().numpy()
            
            # Create a binary mask based on a threshold (e.g., 50% of the max difference)
            anomaly_mask = (difference > threshold_factor * difference.max()).astype(np.uint8) * 255
            
            # Overlay the mask on the original image
            anomalous_image = rgb.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
            red_channel = anomalous_image[:, :, 0]
            red_channel[anomaly_mask == 255] = 255
            anomalous_image[:, :, 0] = red_channel
            
            # Display the anomalous image
            plt.imshow(anomalous_image)
            plt.axis('off')
            plt.show()





def prediction_anomaly_data(model, anomaly_data_loader, anomaly_threshold=0.5):
    """Prediction-based anomaly detection using the model.
    
    Args:
    - model (torch.nn.Module): The trained model for making predictions.
    - anomaly_data_loader (torch.utils.data.DataLoader): DataLoader for the anomaly dataset.
    - anomaly_threshold (float, optional): Threshold for detecting anomalies. Defaults to 0.5.
    
    Returns:
    - anomalies (list): List of indices where anomalies are detected.
    """
    
    pred_anomalies = []
    
    with torch.no_grad():
        for i, (rgb, pcd) in enumerate(anomaly_data_loader):
            # Get model predictions using the forward method
            batch_data = {'rgb': rgb, 'pcd': pcd}  # Assuming the model takes a dictionary with these keys
            predicted_output = model.forward(batch_data)
            
            # Assuming the predicted_output is a dictionary with keys 'predicted_rgb' and 'predicted_pcd'
            predicted_rgb = predicted_output.get('predicted_rgb', torch.zeros_like(rgb))
            predicted_pcd = predicted_output.get('predicted_pcd', torch.zeros_like(pcd))
            
            # Compute MSE errors
            mse_rgb = torch.mean((predicted_rgb - rgb) ** 2)
            mse_pcd = torch.mean((predicted_pcd - pcd) ** 2)
            
            # Check for anomalies and visualize
            if mse_rgb > anomaly_threshold or mse_pcd > anomaly_threshold:
                pred_anomalies.append(i)
    
    return pred_anomalies





if __name__ == "__main__":

    # Load configuration from args and potentially from a file
    args = get_parser().parse_args()
    cfg = get_cfg(args)


    # Clear any cached memory on the GPU
    # torch.cuda.empty_cache()

    # Initialize the model and load the weights
    # model = Mile(cfg, load_weights=True, checkpoint_path='/disk/vanishing_data/du541/epoch=19-step=30000.ckpt')
    

    data = DataModule(cfg) # Local Test needed 
    # data = AnovoxDataset(cfg, '/disk/vanishing_data/du541/AnoVox(Sample)/Anovox')
    data.setup() # 11.10.2023

    model = WorldModelTrainer(cfg.convert_to_dict())

    # input_model = Model(model_id='').get_local_copy() if cfg.PRETRAINED.CML_MODEL else None
    model = WorldModelTrainer(cfg.convert_to_dict(), pretrained_path=cfg.PRETRAINED.PATH)


    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    writer = SummaryWriter(log_dir=save_dir)
    
    # model = Mile(cfg)


    # with torch.no_grad():
    checkpoint = torch.load('/disk/vanishing_data/du541/epoch=19-step=30000.ckpt')
    state_dict = checkpoint['state_dict']
    # Remove 'model.' prefix from state_dict keys
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}


    model.load_state_dict(new_state_dict, strict=False)
    # model = model.cuda()

    dataloader = data.predict_dataloader()


    pbar = tqdm(total=len(dataloader),  desc='Evaluation')
    model.cuda()

    for i, data in enumerate(dataloader):
        data = {key: value.cuda() for key, value in data.items()}
        with torch.no_grad():
            output, output_imagine = model.forward(data)
            model.visualise(data, output, output_imagine, batch_idx=i, prefix='pred', writer=writer)
        pbar.update(1)

    # # Create a DataLoader for the anomaly dataset
    # anomaly_dataset = AnomalyDataset(rgb_folder="/disk/vanishing_data/du541/Scenario_7bc0952d-843d-4e79-8dda-c64aae39b9a7/RGB_IMG", pcd_folder="/disk/vanishing_data/du541/Scenario_7bc0952d-843d-4e79-8dda-c64aae39b9a7/PCD")
    # anomaly_data_loader = DataLoader(anomaly_dataset, batch_size=1, shuffle=False)


    
    # # Evaluate the model on the anomaly data
    # reconstruction_anomaly_data(model, anomaly_data_loader)

    # # Pass the data through the model, visualize, and evaluate
    # with torch.no_grad():
    #     for rgb, pcd in anomaly_data_loader:

    #         # Permute the tensor dimensions to [batch_size, channels, height, width]
    #         rgb = rgb.permute(0, 3, 1, 2)

    #         print(rgb.shape)


    #         # Create a dummy speed tensor with the same batch and sequence dimensions as the RGB data
    #         dummy_speed = torch.zeros((rgb.size(0), rgb.size(1), 1), device=rgb.device)

    #         # Create a dummy intrinsics tensor with appropriate dimensions. 
    #         # I'm assuming a (3, 3) shape for intrinsics as it's typical, but you might need to adjust.
    #         dummy_intrinsics = torch.zeros((rgb.size(0), rgb.size(1), 3, 3), device=rgb.device)

    #         # Create a dummy extrinsics tensor with appropriate dimensions. 
    #         # I'm assuming a (4, 4) shape for extrinsics as it's typical for transformation matrices.
    #         dummy_extrinsics = torch.zeros((rgb.size(0), rgb.size(1), 4, 4), device=rgb.device)


    #         print(rgb.shape)
           
    #         output, _ = model({'image': rgb, 'pcd': pcd, 'speed': dummy_speed, 'intrinsics': dummy_intrinsics, 'extrinsics': dummy_extrinsics})  # Add other required inputs as needed

    #         # Visualize reconstructed RGB
    #         visualize_rgb(output['image'][0])

    #         # Visualize reconstructed LIDAR
    #         visualize_pcd(output['lidar'][0])
            
    #         # Reconstruction based anomaly detection
    #         mse_anomalies, mae_anomalies, mse_threshold, mae_threshold, mse_errors, mae_errors = reconstruction_anomaly_data(model, anomaly_data_loader)

    #         # Plotting the histogram of MSE and MAE errors
    #         visualize_errors(mse_errors, mae_errors, mse_threshold, mae_threshold)

    #         # Visualize the anomalous RGB images and PCD data
    #         visualize_anomalous_inputs(anomaly_data_loader, mse_anomalies)

    #         # Visualize the anomalous Region
    #         visualize_anomalous_regions(model, anomaly_data_loader)