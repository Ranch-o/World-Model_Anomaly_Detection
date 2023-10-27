import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
from PIL import Image

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# input_image = input_image.to(device)
# synthesized_image = synthesized_image.to(device)
# resnet18 = resnet18.to(device)



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

# Determine device and move tensors to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_image = input_image.to(device)
synthesized_image = synthesized_image.to(device)

# Load pre-trained ResNet-18
resnet18 = timm.create_model('resnet18', pretrained=True, features_only=True).eval()
resnet18 = resnet18.to(device)  # Move model to device

# Get feature maps
feature_maps_input = resnet18(input_image)
feature_maps_synthesized = resnet18(synthesized_image)

# Get spatial dimensions from the first set of feature maps
height, width = feature_maps_input[0].shape[2:4]

# Initialize tensor to store per-pixel perceptual difference
perceptual_difference_per_pixel = torch.zeros((height, width))

# Calculate perceptual difference per pixel
for h in range(height):
    for w in range(width):
        perceptual_difference = 0
        for fm_input, fm_synthesized in zip(feature_maps_input, feature_maps_synthesized):
            # Resize feature maps to match the spatial dimensions of the first set of feature maps
            fm_input_resized = F.interpolate(fm_input, size=(height, width), mode='bilinear', align_corners=False)
            fm_synthesized_resized = F.interpolate(fm_synthesized, size=(height, width), mode='bilinear', align_corners=False)
            num_elements = fm_input_resized.size(1)  # Number of channels (Mi elements)
            pixel_difference = F.l1_loss(fm_input_resized[0, :, h, w], fm_synthesized_resized[0, :, h, w], reduction='sum') / num_elements
            perceptual_difference += pixel_difference.item()
        perceptual_difference_per_pixel[h, w] = perceptual_difference

# Normalize (assuming max possible value is known or estimated)
max_value = perceptual_difference_per_pixel.max()  # This is a simplification, max_value should be estimated
perceptual_difference_normalized = perceptual_difference_per_pixel / max_value

print(f'Perceptual Difference Per Pixel: {perceptual_difference_per_pixel}')
print(f'Normalized Perceptual Difference Per Pixel: {perceptual_difference_normalized}')
