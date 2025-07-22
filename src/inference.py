import numpy as np


def extract_patches(image, patch_size, stride):
    """
    Extracts overlapping patches from an image.
    """
    patches = []
    h, w = image.shape
    for y in range(0, h - patch_size + 1, stride[1]):
        for x in range(0, w - patch_size + 1, stride[0]):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    return np.array(patches)


def extract_background_color_rgb(image, mask=None):
    """
    Extract the dominant RGB background color from the image.
    If a mask is provided, only background pixels are considered.
    """
    if mask is not None:
        # Apply mask to find background pixels
        background_pixels = image[mask == 0]  # Background where mask is 0
    else:
        # Use the entire image if no mask is provided
        background_pixels = image.reshape(-1, 3)  # Flatten to [N, 3]

    # Compute the mean color of the background pixels
    background_color = np.mean(background_pixels, axis=0)
    background_color = background_color.astype(int)
    print("Extracted RGB Background Color:", background_color)
    return background_color

def apply_background_color_rgb(restored_image, background_color):
    """
    Combines restored text with the extracted background color to produce a colored RGB image.
    """
    # Get the shape of the restored image
    h, w = restored_image.shape

    print(f"Image Height: {h}")
    print(f"Image Width: {w}")

    # Create an RGB background with the extracted color
    rgb_background = np.full((h, w, 3), fill_value=background_color, dtype=np.uint8)
    print(f"Background Color (RGB): {background_color}")

    # Normalize the restored image if it's not in [0, 1]
    restored_image_normalized = np.expand_dims(restored_image, axis=-1)  # Make it (h, w, 1)
    print("Restored Image Normalized (first 5 values):", restored_image_normalized.flatten()[:5])

    # Invert the restored image to get the text mask (1 for background, 0 for text)
    text_mask = 1.0 - restored_image_normalized  # This will give us the background mask
    print("Text Mask (first 5 values):", text_mask.flatten()[:5])

    # Expand text_mask to match RGB channels
    text_mask_rgb = np.repeat(text_mask, 3, axis=-1)  # Make it (h, w, 3)
    print(f"Text Mask RGB (first 5 pixels):", text_mask_rgb[0, 0])

    # Combine the background and text
    # For text (black), we use [0, 0, 0], for the background, we use the extracted background color
    combined_image = text_mask_rgb * [0, 0, 0] + (1 - text_mask_rgb) * rgb_background
    print("Combined Image (first 5 pixels):", combined_image[0, 0])

    # Convert to uint8 for saving as an image
    combined_image_uint8 = combined_image.astype('uint8')
    print(f"Combined Image (final type): {combined_image_uint8.dtype}")
    return combined_image_uint8

def save_image_correctly(image, filename):
    """
    Save the image in the correct format for OpenCV (BGR).
    """
    # Convert RGB to BGR before saving with OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_bgr)
    print(f"Final image saved at {filename}")
