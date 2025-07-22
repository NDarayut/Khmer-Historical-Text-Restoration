import cv2
import os
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola


def flat_field_correction(image, window_size=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    illumination = cv2.GaussianBlur(gray, (window_size, window_size), 0)
    corrected_image = cv2.divide(gray, illumination, scale=255.0)
    corrected_image = np.uint8(corrected_image)
    return corrected_image


def otsu_binarization(image):
    _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_otsu


def niblack_binarization(image, window_size=25, k=-0.2):
    niblack_thresh = threshold_niblack(image, window_size=window_size, k=k)
    binary_niblack = (image > niblack_thresh).astype(np.uint8) * 255
    return binary_niblack


def sauvola_binarization(image, window_size=25, k=0.2):
    sauvola_thresh = threshold_sauvola(image, window_size=window_size, k=k)
    binary_sauvola = (image > sauvola_thresh).astype(np.uint8) * 255
    return binary_sauvola


def bernsen_binarization(image, window_size=25, delta=10):
    gray = image.astype(np.float32)
    binary_bernsen = np.zeros_like(gray, dtype=np.uint8)

    pad_size = window_size
    padded_image = np.pad(gray, pad_size, mode='constant', constant_values=0)

    min_image = np.zeros_like(gray, dtype=np.float32)
    max_image = np.zeros_like(gray, dtype=np.float32)

    for y in range(pad_size, gray.shape[0] + pad_size):
        for x in range(pad_size, gray.shape[1] + pad_size):
            local_window = padded_image[y - pad_size:y + pad_size + 1, x - pad_size:x + pad_size + 1]
            local_min = np.min(local_window)
            local_max = np.max(local_window)

            min_image[y - pad_size, x - pad_size] = local_min
            max_image[y - pad_size, x - pad_size] = local_max

    local_mean = (min_image + max_image) / 2
    binary_bernsen = np.where(gray > local_mean + delta, 255, 0).astype(np.uint8)
    return binary_bernsen


def binarize_folder(input_folder, output_folder, methods):
    os.makedirs(output_folder, exist_ok=True)
    for method_name in methods.keys():
        method_folder = os.path.join(output_folder, method_name)
        os.makedirs(method_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            file_path = os.path.join(input_folder, file_name)
            image = cv2.imread(file_path)

            corrected_image = flat_field_correction(image, window_size=25)
            for method_name, method_func in methods.items():
                binarized_image = method_func(corrected_image)
                save_path = os.path.join(output_folder, method_name, file_name)
                cv2.imwrite(save_path, binarized_image)
                print(f"Saved {method_name} result for {file_name} to {save_path}")


if __name__ == "__main__":
    input_folder = "../../data/sample"  # Folder containing input images
    output_folder = "binarized_results"  # Root folder for storing binarized images

    # Dictionary of binarization methods
    methods = {
        "Otsu's Method": otsu_binarization(img),
        #"Niblack's Method": lambda img: niblack_binarization(img, window_size=49, k=-0.25),
        "Sauvola's Method": lambda img: sauvola_binarization(img, window_size=49, k=0.25),
        #"Bernsen's Method": lambda img: bernsen_binarization(img, window_size=49, delta=10)
    }

    binarize_folder(input_folder, output_folder, methods)
