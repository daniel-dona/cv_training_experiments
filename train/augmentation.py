import cv2
import numpy as np
from scipy import ndimage
import random


def rgb_to_bw(rgb_img):
    if rgb_img.ndim != 3 or rgb_img.shape[2] not in [3, 4]:
        raise ValueError("Input must be a 3-channel (RGB) or 4-channel (RGBA) image.")

    # Define the W3C luminance coefficients for RGB to Grayscale conversion
    coefficients = np.array([0.2126, 0.7152, 0.0722])

    # Apply the coefficients to each pixel across all bands (axis=2)
    bw_img = np.dot(rgb_img[:, :, :3], coefficients)

    # Ensure the result is in uint8 format, scaling if necessary
    if bw_img.dtype != np.uint8:
        if np.ptp(bw_img) == 0:
            bw_img = np.zeros_like(bw_img, dtype=np.uint8)
        else:
            bw_img = (255 * (bw_img - np.min(bw_img)) / np.ptp(bw_img)).astype(np.uint8)

    return bw_img


def apply_histogram_equalization(rgb_img, method='global'):
    """
    Applies Histogram Equalization to an RGB image.

    Parameters:
    - rgb_img (numpy array): Input RGB image.
    - method (str, optional): Type of HE. Choices: 'global', 'clahe'. Defaults to 'global'.

    Returns:
    - he_img (numpy array): Output image after applying Histogram Equalization.
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] not in [3, 4]:
        raise ValueError("Input must be a 3-channel (RGB) or 4-channel (RGBA) image.")

    # Convert to YCrCb color space for separate luminance adjustment
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    if method == 'global':
        # Apply Global Histogram Equalization
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    elif method == 'clahe':
        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    else:
        raise ValueError("Method must be either 'global' or 'clahe'.")

    # Convert back to BGR (to maintain original color space for comparison)
    he_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return he_img


def random_data_augmentation(rgb_img):
    """
    Applies a random data augmentation transformation to the input RGB image,
    maintaining the original image dimensions.

    Parameters:
    - rgb_img (numpy array): Input RGB image.

    Returns:
    - aug_img (numpy array): Output image after applying a random data augmentation.
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] not in [3, 4]:
        raise ValueError("Input must be a 3-channel (RGB) or 4-channel (RGBA) image.")

    original_h, original_w, _ = rgb_img.shape
    aug_img = rgb_img.copy()  # Ensure the original image isn't modified

    for i in range(random.randint(0,2)):

        transformation = random.choice([
            'horizontal_flip', 'vertical_flip','color_jittering_brightness', 'color_jittering_contrast', 'color_jittering_saturation'
        ])

        if transformation == 'rotation':
            # Rotation (between -45 to 45 degrees), then crop to original dimensions
            angle = random.uniform(-45, 45)
            rotated_img = ndimage.rotate(aug_img, angle, reshape=True)
            offset_h, offset_w = (rotated_img.shape[0] - original_h) // 2, (rotated_img.shape[1] - original_w) // 2
            aug_img = rotated_img[offset_h:offset_h+original_h, offset_w:offset_w+original_w, :]

        elif transformation in ['horizontal_flip', 'vertical_flip']:
            # Flipping
            flip_code = 1 if transformation == 'horizontal_flip' else 0
            aug_img = cv2.flip(aug_img, flip_code)

        elif transformation == 'scaling':
            # Scaling (between 80% to 120%), then resize back to original dimensions
            scale_factor = random.uniform(0.8, 1.2)
            scaled_img = cv2.resize(aug_img, None, fx=scale_factor, fy=scale_factor)
            aug_img = cv2.resize(scaled_img, (original_w, original_h))

        elif transformation in ['color_jittering_brightness', 'color_jittering_contrast', 'color_jittering_saturation']:
            # Color Jittering
            if transformation == 'color_jittering_brightness':
                value = random.uniform(-50, 50)
                hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255).astype(np.uint8)
                aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            elif transformation == 'color_jittering_contrast':
                factor = random.uniform(0.5, 1.5)
                aug_img = np.clip((aug_img - [128., 128., 128.]) * factor + [128., 128., 128.], 0, 255).astype(np.uint8)
            elif transformation == 'color_jittering_saturation':
                factor = random.uniform(0.5, 1.5)
                hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255).astype(np.uint8)
                aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        else:
            pass

    return aug_img