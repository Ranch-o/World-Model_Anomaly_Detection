import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import os

# Define paths to the gif files
input_gif_path = 'pd_visualization_results__batch_frames/original/frame_0.png'
synthesized_gif_path = 'pd_visualization_results__batch_frames/reconstructed/frame_0.png'

result_folder = 'pd_visualization_results__batch_frames'
os.makedirs(result_folder, exist_ok=True)

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet-18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet18 = timm.create_model('resnet18', pretrained=True, features_only=True).eval().to(device)

# Function to compute perceptual difference
def compute_perceptual_difference(input_frame, synthesized_frame):
    
    # Convert frames to RGB
    input_frame_rgb = input_frame.convert('RGB')
    synthesized_frame_rgb = synthesized_frame.convert('RGB')

    input_tensor = transform(input_frame_rgb).unsqueeze(0)
    print(f"Input tensor shape before normalization: {input_tensor.shape}")  # Print statement here
    input_tensor = transform(input_frame).unsqueeze(0).to(device)
    synthesized_tensor = transform(synthesized_frame).unsqueeze(0).to(device)

    feature_maps_input = resnet18(input_tensor)
    feature_maps_synthesized = resnet18(synthesized_tensor)

    height, width = feature_maps_input[0].shape[2:4]
    perceptual_difference_per_pixel = torch.zeros((height, width))

    for h in range(height):
        for w in range(width):
            perceptual_difference = 0
            for fm_input, fm_synthesized in zip(feature_maps_input, feature_maps_synthesized):
                fm_input_resized = F.interpolate(fm_input, size=(height, width), mode='bilinear', align_corners=False)
                fm_synthesized_resized = F.interpolate(fm_synthesized, size=(height, width), mode='bilinear', align_corners=False)
                num_elements = fm_input_resized.size(1)
                pixel_difference = F.l1_loss(fm_input_resized[0, :, h, w], fm_synthesized_resized[0, :, h, w], reduction='sum') / num_elements
                perceptual_difference += pixel_difference.item()
            perceptual_difference_per_pixel[h, w] = perceptual_difference

    original_dims = (input_frame.height, input_frame.width)
    perceptual_difference_resized = F.interpolate(perceptual_difference_per_pixel.unsqueeze(0).unsqueeze(0), size=original_dims, mode='bilinear', align_corners=False).squeeze()
    return perceptual_difference_resized.cpu().numpy()

# Process each frame
for i, (input_frame, synthesized_frame) in enumerate(zip(ImageSequence.Iterator(Image.open(input_gif_path)), ImageSequence.Iterator(Image.open(synthesized_gif_path)))):
    perceptual_diff = compute_perceptual_difference(input_frame, synthesized_frame)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(input_frame)
    axs[0].set_title('Original Frame')
    axs[0].axis('off')

    axs[1].imshow(synthesized_frame)
    axs[1].set_title('Synthesized Frame')
    axs[1].axis('off')

    im = axs[2].imshow(perceptual_diff, cmap='hot')
    axs[2].set_title('Perceptual Difference')
    axs[2].axis('off')
    fig.colorbar(im, ax=axs[2], label='Perceptual Difference')

    # plt.savefig(f'comparison_frame_{i}.png', bbox_inches='tight')
    # Save the visualization in the new folder
    frame_save_path = os.path.join(result_folder, f'comparison_frame_{i}.png')
    plt.savefig(frame_save_path, bbox_inches='tight')
    plt.close(fig)
