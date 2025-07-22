import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from inference import extract_patches, apply_background_color_rgb, save_image_correctly, extract_background_color_rgb
from models.res_attunet import AttentionResidualUNet

# Load the test image
test_image_path = '/content/313_kandal_tekvil_02_008b.jpg'
test_image_rgb = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
test_image_rgb = cv2.cvtColor(test_image_rgb, cv2.COLOR_BGR2RGB)

# Print a few pixels from the original image to verify the color channels
print(test_image_rgb[0, 0])  # Top-left corner pixel

# Extract the RGB background color
background_color_rgb = extract_background_color_rgb(test_image_rgb)

# Move to the device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the function to reconstruct an image from patches
def reconstruct_image(patches, image_shape, patch_size=64, stride=(10, 5)):
    reconstructed_image = np.zeros(image_shape)
    count_matrix = np.zeros(image_shape)

    index = 0
    for y in range(0, image_shape[0] - patch_size + 1, stride[1]):
        for x in range(0, image_shape[1] - patch_size + 1, stride[0]):
            reconstructed_image[y:y + patch_size, x:x + patch_size] += patches[index].reshape(patch_size, patch_size)
            count_matrix[y:y + patch_size, x:x + patch_size] += 1
            index += 1

    # Avoid division by zero
    count_matrix[count_matrix == 0] = 1
    reconstructed_image /= count_matrix
    return reconstructed_image

# Load the trained model
model = AttentionResidualUNet().to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/Khmer-Historical-Text-Restoration/model/64x64/attresunet_trained_64x64x25.pth'))

# Load the test image
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

# Extract patches from the test image
patch_size = 64
stride = (12, 12)
test_patches = extract_patches(test_image, patch_size, stride)

# Normalize patches and convert to a tensor
test_patches = test_patches.astype('float32') / 255.0
test_patches = torch.tensor(test_patches).unsqueeze(1).to(device)  # Add channel dimension

# Perform batch inference on patches
batch_size = 64  # Adjust based on available GPU memory
reconstructed_patches = []

with torch.no_grad():
    for i in range(0, len(test_patches), batch_size):
        batch = test_patches[i:i + batch_size]  # Create a batch of patches
        output_batch= model(batch)  # Run the batch through the model
        reconstructed_patches.append(output_batch.cpu().numpy())  # Move to CPU and convert to NumPy

# Convert reconstructed patches to a NumPy array
reconstructed_patches = np.concatenate(reconstructed_patches, axis=0)

# Reconstruct the full image
reconstructed_image = reconstruct_image(reconstructed_patches, test_image.shape, patch_size, stride)
print(f"Restored Image Min: {np.min(reconstructed_image)}, Max: {np.max(reconstructed_image)}")


# Adjust the reconstruction process for RGB
final_colored_image = apply_background_color_rgb(reconstructed_image, background_color_rgb)

# Save the final image
output_path_final = '/content/inference_313_kandal_tekvil_02_008b.jpg'

# Save the combined image in the correct format (BGR for OpenCV)
save_image_correctly(final_colored_image, output_path_final)
print(f"Final image saved at {output_path_final}")
