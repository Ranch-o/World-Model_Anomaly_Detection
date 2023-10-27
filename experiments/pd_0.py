import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

input_image_path = '/disk/users/du541/Desktop/1.png'
synthesized_image_path = '/disk/users/du541/Desktop/2.png'
input_image = Image.open(input_image_path).convert('RGB')
print(input_image.size)
synthesized_image = Image.open(synthesized_image_path).convert('RGB')
print(synthesized_image.size)

original_height, original_width = input_image.size[::-1]

# Preprocess images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size if necessary
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Imagenet stats
])
input_image_tensor = transform(input_image).unsqueeze(0)
synthesized_image_tensor = transform(synthesized_image).unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_image_tensor = input_image_tensor.to(device)
synthesized_image_tensor = synthesized_image_tensor.to(device)

resnet18 = timm.create_model('resnet18', pretrained=True, features_only=True).eval()
resnet18 = resnet18.to(device) 

feature_maps_input = resnet18(input_image_tensor)
feature_maps_synthesized = resnet18(synthesized_image_tensor)


def plot_all_feature_maps(feature_maps, title):
    for layer_index, feature_map_layer in enumerate(feature_maps):
        num_feature_maps = feature_map_layer.size(1)
        print(f'Layer {layer_index}: {num_feature_maps} feature maps')


        rows = int(num_feature_maps ** 0.5)
        cols = int(num_feature_maps / rows)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

        for i, ax in enumerate(axs.flat):

            if i < num_feature_maps:
                ax.imshow(feature_map_layer[0, i].cpu().detach().numpy(), cmap='viridis')
            ax.axis('off')


        # save_folder = os.path.join(folder, title, f'Layer_{layer_index}')
        # os.makedirs(save_folder, exist_ok=True)


        # save_path = os.path.join(save_folder, f'feature_maps_layer_{layer_index}.png')
        # plt.suptitle(f'{title} - Layer {layer_index}')
        # plt.savefig(save_path)
        # plt.close(fig)  


# input_folder = 'feature_maps_original'
# synthesized_folder = 'feature_maps_synthesized'


plot_all_feature_maps(feature_maps_input, title='Feature Maps Input')
plot_all_feature_maps(feature_maps_synthesized, title='Feature Maps Synthesized')

height, width = feature_maps_input[0].shape[2:4]
print(height, width)
perceptual_difference_per_pixel = torch.zeros((height, width))


for h in range(height):
    for w in range(width):
        perceptual_difference = 0
        for fm_input, fm_synthesized in zip(feature_maps_input, feature_maps_synthesized):
            fm_input_resized = F.interpolate(fm_input, size=(height, width), mode='bilinear', align_corners=False)
            fm_synthesized_resized = F.interpolate(fm_synthesized, size=(height, width), mode='bilinear', align_corners=False)
            num_elements = fm_input_resized.size(1) 
            # num_elements = fm_input.size(1)
            pixel_difference = F.l1_loss(fm_input_resized[0, :, h, w], fm_synthesized_resized[0, :, h, w], reduction='sum') / num_elements
            # pixel_difference = F.l1_loss(fm_input[0, :, h, w], fm_synthesized[0, :, h, w], reduction='sum') / num_elements
            perceptual_difference += pixel_difference.item()
        perceptual_difference_per_pixel[h, w] = perceptual_difference


perceptual_difference_resized = F.interpolate(perceptual_difference_per_pixel.unsqueeze(0).unsqueeze(0),
                                              size=(original_height, original_width),
                                              mode='bilinear', align_corners=False).squeeze()


print(perceptual_difference_resized.shape)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))  


axs[0].imshow(input_image)
axs[0].set_title('Original Image')
axs[0].axis('off')  


axs[1].imshow(synthesized_image)
axs[1].set_title('Synthesized Image')
axs[1].axis('off')  


im = axs[2].imshow(perceptual_difference_resized.cpu().numpy(), cmap='hot')
axs[2].set_title('Perceptual Difference')
axs[2].axis('off')  


fig.colorbar(im, ax=axs[2], label='Perceptual Difference')

# # Save the figure to a file
# plt.savefig('comparison.png', bbox_inches='tight')  # This line saves the figure to a file