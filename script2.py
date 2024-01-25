import cv2
import os
import glob
import numpy as np

def find_transition_points(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find transition points in the grayscale image
    transition_points = np.column_stack(np.where(gray_image[:-1, :] != gray_image[1:, :]))
    
    return transition_points

def apply_smoothing(image, points_to_smooth, smoothing_radius=20):
    smoothed_image = image.copy()

    for point in points_to_smooth:
        row, col = point[:2]
        min_row = max(0, row - smoothing_radius)
        max_row = min(image.shape[0], row + smoothing_radius + 1)
        min_col = max(0, col - smoothing_radius)
        max_col = min(image.shape[1], col + smoothing_radius + 1)

        neighborhood = image[min_row:max_row, min_col:max_col]
        smoothed_value = np.mean(neighborhood, axis=(0, 1))  # Take mean along channels
        smoothed_image[row, col] = smoothed_value

    return smoothed_image

def process_image(image_path, mask_path, blur_radius):
    # Load the mask in grayscale and the image in color with alpha channel
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

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

    transition_points = find_transition_points(result)

    smoothed_image = apply_smoothing(result, transition_points, smoothing_radius=blur_radius)

    return smoothed_image

# Ensure the 'processed' directory exists
os.makedirs('processed', exist_ok=True)

# Path to the mask
mask_path = 'mask.png'

# Process all PNG images in the current directory
for image_path in glob.glob('*.PNG'):
    blur_radius = 10   # Adjust based on the desired smoothness
    processed_image = process_image(image_path, mask_path, blur_radius)
    
    # Save the result in the 'processed' folder with the same name
    cv2.imwrite(os.path.join('processed', os.path.basename(image_path)), processed_image)
