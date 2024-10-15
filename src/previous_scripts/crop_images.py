import os
import cv2
import numpy as np
import argparse

def mean_adaptive_threshold(gray_images):
    """
    Applies mean adaptive thresholding to a list of grayscale images.

    Parameters:
    gray_images (list of np.ndarray): List of grayscale images.

    Returns:
    list of np.ndarray: List of binary masks obtained after thresholding.
    """
    masks = []
    for img in gray_images:
        mean_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 29, 8)
        neg_mean = cv2.bitwise_not(mean_threshold)
        masks.append(neg_mean)
    
    return masks


def post_process_masks(masks):
    """
    Post-processes binary masks by closing gaps and filling the largest contour.

    Parameters:
    masks (list of np.ndarray): List of binary masks.

    Returns:
    list of np.ndarray: List of post-processed binary masks.
    """
    filled_masks = []
    for mask in masks:

        # Closing
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Fill the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            filled = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(filled, [largest_contour], -1, 255, thickness=cv2.FILLED)
        else:
            filled = np.zeros_like(mask, dtype=np.uint8)

        filled_masks.append(filled)
    
    return filled_masks


def main(images_dir = "./data/qst1_w2/", output_masks_dir = "./data/masks/qst1_w2/", output_cropped_dir = "./data/cropped_imgs/qst1_w2/"):
    """
    Main function to process images, generate masks, and crop regions.

    Parameters:
    images_dir (str): Directory containing the input images.
    output_masks_dir (str): Directory to save the generated masks.
    output_cropped_dir (str): Directory to save the cropped images.

    Returns:
    None
    """
    # Create output directories
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_cropped_dir, exist_ok=True)

    # Read images
    gray_images = []
    rgb_images = []

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(images_dir, filename)
            img_rgb = cv2.imread(img_path)

            if img_rgb is not None:
                rgb_images.append(img_rgb)
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                gray_images.append(img_gray)
            else:
                print(f"Warning: Failed to read {img_path}")

    # Generate masks
    masks = mean_adaptive_threshold(gray_images)
    masks = post_process_masks(masks)

    # Save mask images and cropped results
    for i, mask in enumerate(masks):

        filename = f"{i:05d}.png"
        masks_save_path = os.path.join(output_masks_dir, filename)
        cropped_save_path = os.path.join(output_cropped_dir, filename)

        # Save mask
        cv2.imwrite(masks_save_path, mask)

        # Crop region (set to black if no non-zero points found in mask)
        foreground_closed = cv2.bitwise_and(rgb_images[i], rgb_images[i], mask=mask)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped_result = foreground_closed[y:y+h, x:x+w]
            cv2.imwrite(cropped_save_path, cropped_result)
        else:
            print(f"Warning: No non-zero points found in mask {filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process images, generate masks, and crop regions.')
    parser.add_argument('--images-dir', required=True, help='Directory containing the input images')
    parser.add_argument('--output-masks-dir', required=True, help='Directory to save the generated masks')
    parser.add_argument('--output-cropped-dir', required=True, help='Directory to save the cropped images')
    args = parser.parse_args()

    main(args.images_dir, args.output_masks_dir, args.output_cropped_dir)