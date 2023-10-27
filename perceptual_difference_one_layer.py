import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
from PIL import Image

# Load images
input_image_path = '/disk/users/du541/Desktop/1.png'
synthesized_image_path = '/disk/users/du541/Desktop/2.png'
input_image = Image.open(input_image_path).convert('RGB')
synthesized_image = Image.open(synthesized_image_path).convert('RGB')

# Preprocess images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size if necessary
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Imagenet stats
])
input_image = transform(input_image).unsqueeze(0)
synthesized_image = transform(synthesized_image).unsqueeze(0)

# Load pre-trained ResNet-18
resnet18 = timm.create_model('resnet18', pretrained=True, features_only=True).eval()

# Get feature maps
feature_maps_input = resnet18(input_image)
feature_maps_synthesized = resnet18(synthesized_image)

# Select a specific layer for comparison
layer_index = 2  # Example: selecting the 3rd set of feature maps
fm_input = feature_maps_input[layer_index]
fm_synthesized = feature_maps_synthesized[layer_index]

# Get spatial dimensions of selected layer
height, width = fm_input.shape[2:4]

# Initialize tensor to store per-pixel perceptual difference
perceptual_difference_per_pixel = torch.zeros((height, width))

# Calculate perceptual difference per pixel
for h in range(height):
    for w in range(width):
        num_elements = fm_input.size(1)  # Number of channels (Mi elements)
        pixel_difference = F.l1_loss(fm_input[0, :, h, w], fm_synthesized[0, :, h, w], reduction='sum') / num_elements
        perceptual_difference_per_pixel[h, w] = pixel_difference.item()

# Normalize (assuming max possible value is known or estimated)
max_value = perceptual_difference_per_pixel.max()  # This is a simplification, max_value should be estimated
perceptual_difference_normalized = perceptual_difference_per_pixel / max_value

print(f'Perceptual Difference Per Pixel: {perceptual_difference_per_pixel}')
print(f'Normalized Perceptual Difference Per Pixel: {perceptual_difference_normalized}')

