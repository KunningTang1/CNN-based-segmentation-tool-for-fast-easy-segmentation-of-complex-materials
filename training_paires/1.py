import os
import numpy as np
import tifffile
from tifffile import imread, imwrite

def convert_mask_to_rgb(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".tif"):
            # Read the mask image
            mask = imread(os.path.join(input_dir, filename))
            
            # Check if the mask is in uint8 format
            if mask.dtype != np.uint8:
                raise TypeError(f"The image {filename} is not in uint8 format.")
            
            # Create an empty RGB image with the same dimensions as the mask
            rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
            # Assign colors to the RGB image based on the mask values
            # Red for 0, Green for 1, Blue for 2
            rgb_image[mask == 0] = [255, 0, 0]      # Red
            rgb_image[mask == 1] = [0, 255, 0]      # Green
            rgb_image[mask == 2] = [0, 0, 255]      # Blue
            
            # Save the RGB image with the same name as the mask
            imwrite(os.path.join(output_dir, filename), rgb_image)
            print(f"Saved RGB image for {filename}")

# Define the input and output directories
input_dir = './seg'
output_dir = './'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert masks to RGB and save them
convert_mask_to_rgb(input_dir, output_dir)