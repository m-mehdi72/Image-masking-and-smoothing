import cv2
import numpy as np
import os
import time
from PIL import Image, ImageFilter
import glob

def process_image(image_path, mask_path, border_size, blur_radius):
    # Load the mask in grayscale and the image in color with alpha channel
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Smooth pixelated borders
    mask = blur_whole_image(mask,blur_radius)
    # Resize the mask to match the dimensions of the image
    # mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    time.sleep(10)
    # Ensure both image and mask have the same number of channels
    if img.shape[2] == 3 and mask.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Resize the mask to match the dimensions of the image
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Ensure that both image and mask have the same number of channels
    if img.shape[2] != mask.shape[2]:
        raise ValueError("Image and mask have different numbers of channels")

    # Create a 4-channel version of the mask
    mask_4d = mask.copy()

    # Apply the 4-channel mask to create the transparency
    result = cv2.bitwise_and(img, mask_4d)

    return result

def blur_whole_image(image, blur_radius):
    # Convert the image to RGBA (so it has an alpha channel)
    img_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Ensure blur_radius is an odd and positive integer
    blur_radius = max(1, blur_radius // 2) * 2 + 1

    # Apply Gaussian Blur to the entire image
    blurred_image = cv2.GaussianBlur(img_rgba, (blur_radius, blur_radius), 0)

    return blurred_image

def smooth_pixelated_borders(img, border_size, blur_radius):
    # Convert the image to RGBA (so it has an alpha channel)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Get the size of the image
    height, width, _ = img_rgba.shape

    # Create a mask to identify the pixelated borders
    mask = Image.new('L', (width, height), 0)
    mask.paste(255, (border_size, border_size, width - border_size, height - border_size))

    # Apply Gaussian Blur only to the pixelated borders
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    # Create an alpha channel from the blurred mask
    alpha_channel = np.array(mask_blurred).astype(np.uint8)

    # Apply the alpha channel to the original image
    img_smoothed = cv2.addWeighted(img_rgba, 1, np.zeros_like(img_rgba), 0, 0)
    img_smoothed[:, :, 3] = alpha_channel

    return img_smoothed

# Ensure the 'processed' directory exists
os.makedirs('processed', exist_ok=True)

# Path to the mask
mask_path = 'mask.png'

# Process all PNG images in the current directory
for image_path in glob.glob('*.PNG'):
    
    border_size = 50  # Adjust based on the size of pixelated borders
    blur_radius = 10   # Adjust based on the desired smoothness
    processed_image = process_image(image_path, mask_path, border_size, blur_radius)
    
    # Save the result in the 'processed' folder with the same name
    cv2.imwrite(os.path.join('processed', os.path.basename(image_path)), processed_image)
