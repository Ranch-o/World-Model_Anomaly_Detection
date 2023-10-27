import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



def evaluate_reconstruction(actual_images, reconstructed_images, threshold=None):
    """
    Evaluate reconstruction-based anomaly detection.
    
    Args:
    - actual_images (tensor): A tensor containing the actual images. Shape: [num_images, height, width, channels].
    - reconstructed_images (tensor): A tensor containing the reconstructed images. Shape should match actual_images.
    - threshold (float, optional): A predefined threshold for anomaly detection. If None, it will be computed.
    
    Returns:
    - anomalies_indices (list): Indices of images identified as anomalies.
    - computed_threshold (float): The threshold used for anomaly detection.
    """
    
    # Compute the Mean Squared Error (MSE) between the actual and reconstructed images
    mse_errors = ((actual_images - reconstructed_images) ** 2).mean(axis=(1, 2, 3))
    
    # If no threshold provided, compute it as mean error + 2 standard deviations
    if threshold is None:
        threshold = mse_errors.mean() + 2 * mse_errors.std()
    
    # Identify anomalies
    anomalies_indices = np.where(mse_errors > threshold)[0]
    
    return anomalies_indices, threshold

# You can call this function with your actual and reconstructed images to get the anomalies:
# anomalies_indices, computed_threshold = evaluate_reconstruction(actual_images, reconstructed_images)


# anomalies_indices, computed_threshold = evaluate_reconstruction(actual_images, reconstructed_images)

def visualize_anomaly_in_frame0(actual_image, reconstructed_image, threshold_factor=0.88):
    """
    Visualize the anomalous regions in a single frame based on the reconstruction error.
    
    Args:
    - actual_image (tensor): A tensor containing the actual image. Shape: [height, width, channels].
    - reconstructed_image (tensor): A tensor containing the reconstructed image. Shape should match actual_image.
    - threshold_factor (float): Factor multiplied with max difference to create a binary mask.
    
    Returns:
    - Anomaly highlighted image.
    """
    
    # Compute the absolute difference between the actual and reconstructed image
    difference = np.abs(actual_image - reconstructed_image)
    
    # Sum the differences along the channel axis to get a single channel difference image
    difference = difference.sum(axis=2)

    # # Compute the Mean Squared Error (MSE) between the actual and reconstructed image
    # mse = ((actual_image - reconstructed_image) ** 2)
    
    # # Sum the MSE values along the channel axis to get a single channel difference image
    # difference = mse.sum(axis=2)

    
    # Create a binary mask based on a threshold (e.g., 50% of the max difference)
    anomaly_mask = (difference > threshold_factor * difference.max())
    
    # Create an anomaly highlighted image (using red color to highlight anomalies)
    anomaly_highlighted = actual_image.copy()
    anomaly_highlighted[anomaly_mask, 0] = 255  # Red channel
    anomaly_highlighted[anomaly_mask, 1] = anomaly_highlighted[anomaly_mask, 2] = 0  # Green & Blue channel
    
    # Display the anomaly highlighted image
    plt.imshow(anomaly_highlighted)
    plt.axis('off')
    plt.title('Anomaly Highlighted Image')
    plt.show()

    return anomaly_highlighted



def visualize_anomaly_in_frame(actual_image, reconstructed_image, lower_bound=0.6, upper_bound=0.88):
    """
    Visualize the anomalous regions in a single frame based on the reconstruction error using MSE with specific bounds.
    
    Args:
    - actual_image (tensor): A tensor containing the actual image. Shape: [height, width, channels].
    - reconstructed_image (tensor): A tensor containing the reconstructed image. Shape should match actual_image.
    - lower_bound (float): Lower bound for the MSE to create a binary mask.
    - upper_bound (float): Upper bound for the MSE to create a binary mask.
    
    Returns:
    - Anomaly highlighted image.
    """
    
    # Compute the Mean Squared Error (MSE) between the actual and reconstructed image
    mse = ((actual_image - reconstructed_image) ** 2)
    
    # Sum the MSE values along the channel axis to get a single channel difference image
    difference = mse.sum(axis=2)
    
    # Create a binary mask based on the specified MSE bounds
    anomaly_mask = (difference < upper_bound) | (difference > lower_bound)
    
    # Create an anomaly highlighted image (using red color to highlight anomalies)
    anomaly_highlighted = actual_image.copy()
    anomaly_highlighted[anomaly_mask, 0] = 255  # Red channel
    anomaly_highlighted[anomaly_mask, 1] = anomaly_highlighted[anomaly_mask, 2] = 0  # Green & Blue channel
    
    # Display the anomaly highlighted image
    plt.imshow(anomaly_highlighted)
    plt.axis('off')
    plt.title('Anomaly Highlighted Image (MSE bounds based)')
    plt.show()

    return anomaly_highlighted

# This function can be used in your script to visualize anomalies based on specific MSE bounds.





def load_image_as_array(image_path):
    """
    Load an image from the given path and return it as a numpy array.
    """
    # Open the image using PIL
    img = Image.open(image_path)
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    return img_array

# Load the actual and reconstructed images
actual_image = load_image_as_array("/fzi/ids/du541/Desktop/1.1.png")
reconstructed_image = load_image_as_array("/fzi/ids/du541/Desktop/2.1.png")

# Now you can pass these arrays to the evaluate_reconstruction_with_check function:
# anomalies_indices, computed_threshold = evaluate_reconstruction_with_check(actual_image_array, reconstructed_image_array)

anomaly_highlighted_image = visualize_anomaly_in_frame0(actual_image, reconstructed_image)