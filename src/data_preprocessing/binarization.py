import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_niblack, threshold_sauvola
from scipy.ndimage import minimum_filter, maximum_filter

def flat_field_correction(image, window_size=25):
    """Apply flat-field correction by dividing the image by a background estimate."""
    # Convert to grayscale for illumination correction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Estimate the illumination field using Gaussian blur (you can also try median filtering)
    illumination = cv2.GaussianBlur(gray, (window_size, window_size), 0)
    
    # Normalize the image by dividing by the illumination estimate
    corrected_image = cv2.divide(gray, illumination, scale=255.0)
    corrected_image = np.uint8(corrected_image)  # Convert to uint8 after correction
    
    return corrected_image

def otsu_binarization(image):
    """Apply Otsu's binarization method."""
    gray = image  # Use the corrected grayscale image directly
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_otsu

def niblack_binarization(image, window_size=25, k=-0.2):
    """Apply Niblack's binarization method."""
    gray = image  # Use the corrected grayscale image directly
    niblack_thresh = threshold_niblack(gray, window_size=window_size, k=k)
    binary_niblack = (gray > niblack_thresh).astype(np.uint8) * 255
    return binary_niblack

def sauvola_binarization(image, window_size=25, k=0.2):
    """Apply Sauvola's binarization method."""
    gray = image  # Use the corrected grayscale image directly
    sauvola_thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary_sauvola = (gray > sauvola_thresh).astype(np.uint8) * 255
    return binary_sauvola

def bernsen_binarization(image, window_size=25, delta=10):
    """Optimized Bernsen's binarization method using sliding window and vectorization."""
    gray = image
    binary_bernsen = np.zeros_like(gray, dtype=np.uint8)
    
    # Ensure the image is in float32 for calculations
    gray = gray.astype(np.float32)

    # Padding the image to handle borders
    pad_size = window_size
    padded_image = np.pad(gray, pad_size, mode='constant', constant_values=0)
    
    # Compute the local min and max using sliding window approach
    min_image = np.zeros_like(gray, dtype=np.float32)
    max_image = np.zeros_like(gray, dtype=np.float32)

    for y in range(pad_size, gray.shape[0] + pad_size):
        for x in range(pad_size, gray.shape[1] + pad_size):
            # Define the local window
            local_window = padded_image[y - pad_size:y + pad_size + 1, x - pad_size:x + pad_size + 1]
            
            # Compute local min and max using NumPy's min and max functions
            local_min = np.min(local_window)
            local_max = np.max(local_window)
            
            # Save min and max values for further use
            min_image[y - pad_size, x - pad_size] = local_min
            max_image[y - pad_size, x - pad_size] = local_max
    
    # Calculate local mean and apply thresholding
    local_mean = (min_image + max_image) / 2
    binary_bernsen = np.where(gray > local_mean + delta, 255, 0).astype(np.uint8)
    
    return binary_bernsen


def save_results(binary_methods, output_dir="results"):
    """Save binarized images to the specified directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, binarized_image in binary_methods.items():
        file_name = f"{method_name}.png"
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, binarized_image)
        print(f"Saved {method_name} result to {output_path}")

def display_results(original, binary_methods):
    """Display original and binarized images side by side."""
    plt.figure(figsize=(15, 10))
    plt.subplot(2, len(binary_methods) + 1, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    for i, (method_name, binarized_image) in enumerate(binary_methods.items(), start=2):
        plt.subplot(2, len(binary_methods) + 1, i)
        plt.imshow(binarized_image, cmap="gray")
        plt.title(method_name)
        plt.axis("off")
    
    plt.show()

if __name__ == "__main__":
    # Load your RGB image
    image_path = "../../data/raw/009_sr_bo_07_004.jpg" 
    image = cv2.imread(image_path)

    # Apply flat-field correction
    corrected_image = flat_field_correction(image, window_size=25)

    # Perform binarization on the corrected image
    binary_methods = {
        "Otsu's Method": otsu_binarization(corrected_image),
        "Niblack's Method": niblack_binarization(corrected_image, window_size=25, k=-0.2),
        "Sauvola's Method": sauvola_binarization(corrected_image, window_size=25, k=0.2), 
        "Bernsen's Method": bernsen_binarization(corrected_image, window_size=25, delta=10)
    }

    # Save results
    save_results(binary_methods, output_dir="binarized_results")

    # Display results
    display_results(image, binary_methods)
