# # from urllib.request import urlopen
# from PIL import Image
# import timm

# # img = Image.open(urlopen(
# #     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
# # ))

# img = Image.open('/disk/vanishing_data/du541/carla_dataset_test/trainval/train/Town06/0019/image/image_000000018.png')

# model = timm.create_model(
#     'resnet18',
#     pretrained=True,
#     features_only=True,
# )
# model = model.eval()

# # get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

# for o in output:
#     # print shape of each feature map in output
#     # e.g.:
#     #  torch.Size([1, 64, 112, 112])
#     #  torch.Size([1, 64, 56, 56])
#     #  torch.Size([1, 128, 28, 28])
#     #  torch.Size([1, 256, 14, 14])
#     #  torch.Size([1, 512, 7, 7])

#     print(o.shape)

import torch
from torchvision import transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
import torchvision.transforms as transforms

# Open the local image file
# img = Image.open('/disk/vanishing_data/du541/carla_dataset_test/trainval/train/Town06/0019/image/image_000000018.png')
# img = Image.open('/disk/users/du541/Desktop/1.png')
img = Image.open('/disk/users/du541/Desktop/2.png')
print(img.size)

img_array = np.array(img)
height, width, channels = img_array.shape
print(f'Shape of the input image: ({height}, {width}, {channels})')


model = timm.create_model(
    'resnet18',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input dimensions expected by resnet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
])

# Preprocess the image
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# Print shape of each feature map in output
for o in output:
    print(o.shape)

# # Plot the first channel of each feature map
# for i, feature_map in enumerate(output):
#     plt.figure(figsize=(10, 10))
#     # Get the first channel of this feature map
#     channel = feature_map[0, 0].cpu().numpy()
#     plt.imshow(channel, cmap='viridis')
#     plt.title(f'Feature Map {i} - Channel 0')
#     plt.colorbar()
#     plt.show()

# # Plot feature maps
# for layer, feature_map in enumerate(output):
#     # Get the number of feature maps (channels)
#     num_feature_maps = feature_map.size(1)
    
#     # Set up the figure and axis for a grid with 'num_feature_maps' columns
#     fig, axs = plt.subplots(1, num_feature_maps, figsize=(15, 15))
    
#     for i in range(num_feature_maps):
#         # Get the ith feature map
#         fmap = feature_map[0, i].cpu().numpy()
        
#         # Plot the feature map
#         axs[i].imshow(fmap, cmap='viridis')
#         axs[i].axis('off')
    
#     plt.show()

# # Function to plot the feature maps
# def plot_feature_maps(feature_maps):
#     # Get the number of feature maps (channels)
#     num_feature_maps = feature_maps.size(1)
    
#     # Prepare the grid
#     cols = int(np.sqrt(num_feature_maps))
#     rows = int(np.ceil(num_feature_maps / cols))
    
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
#     for i in range(num_feature_maps):
#         ax = axes.flat[i]
#         ax.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
#         ax.axis('off')
    
    # plt.show()

# # Plot the feature maps from different layers
# for layer_output in output:
#     plot_feature_maps(layer_output)

def plot_feature_maps(feature_maps, layer_idx):
    # Get the number of feature maps (channels)
    num_feature_maps = feature_maps.size(1)
    
    # Prepare the grid
    cols = int(np.sqrt(num_feature_maps))
    rows = int(np.ceil(num_feature_maps / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
    for i in range(num_feature_maps):
        ax = axes.flat[i]
        ax.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
    
    # # Save the figure to disk
    # fig.savefig(f'feature_maps_layer_{layer_idx}.png')

# # Plot the feature maps from different layers
# for idx, layer_output in enumerate(output):
#     plot_feature_maps(layer_output, idx)

def load_feature_maps(folder_path):
    # Assumes feature maps are saved as Numpy arrays and loads them into a list
    feature_maps = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path)
        img_array = np.array(img)
        # feature_map = np.load(file_path, allow_pickle=True)
        feature_maps.append(torch.tensor(img_array))
    return feature_maps

def calculate_perceptual_loss(feature_maps_input, feature_maps_output):
    perceptual_loss = 0
    for fm_input, fm_output in zip(feature_maps_input, feature_maps_output):
        fm_input = fm_input.float()
        fm_output = fm_output.float()

        # print(f'fm_input: {fm_input}')
        # print(f'fm_output: {fm_output}')

        # Assumes feature maps are single-channel (grayscale). For multi-channel, you might need to adjust this.
        loss = F.mse_loss(fm_input, fm_output)
        perceptual_loss += loss.item()
    return perceptual_loss


# Load the feature maps
feature_maps_input = load_feature_maps('feature_maps_input')
feature_maps_output = load_feature_maps('feature_maps_output')
# print(f'feature_maps_input: {feature_maps_input}')
# print(f'feature_maps_output: {feature_maps_output}')

# Calculate the perceptual loss
perceptual_loss = calculate_perceptual_loss(feature_maps_input, feature_maps_output)


# Print the perceptual loss
print(f'Perceptual Loss: {perceptual_loss}')