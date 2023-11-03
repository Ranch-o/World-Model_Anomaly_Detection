import numpy as np

# Load the data from .npy file
data = np.load('/disk/vanishing_data/qw825/carla_dataset_small/trainval/train/Town01/0000/points/points_000000000.npy')

# Check the shape and datatype of the loaded data
print("Shape:", data.shape)
print("Datatype:", data.dtype)

# Print the first few entries or any other inspection you'd like
print(data[:5])  # This prints the first 5 entries if it's a 1D array. Adjust accordingly for multi-dimensional arrays.
