from src.utils.histograms import change_color_space, get_pyramid_histograms
from src.utils.distance_matrix import create_distance_matrix, generate_results, generate_submission
from src.utils.images import load_images_from_directory

import numpy as np
import argparse
import cv2
import os

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

def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process input for image retrieval")

    # Files
    parser.add_argument('--queries-dir', type=str, required=True,
                        help="Name of the directory containing query images.")
    parser.add_argument('--bbdd-dir', type=str, required=True,
                        help="Name of the directory containing database images.")

    return parser.parse_args()

def main():

    # Get command-line arguments
    args = parse_args()

    # Data directories
    queries_dir = args.queries_dir
    bbdd_dir = args.bbdd_dir
    output_masks_dir = "./data/masks/qst1_w2/"
    output_cropped_dir = "./data/cropped_imgs/qst1_w2/"

    # Create masks and cropped imgages output directories
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_cropped_dir, exist_ok=True)

    # Fixed parameters
    color_space = "LAB"
    similarity_measure = "bhattacharyya"
    dimension = 1
    normalize = None
    pyramid_level = [5]
    bins = 64
    adaptive_bins = False

    # Read images
    print("Reading images with background...", end=" ")
    gray_images = []
    rgb_images = []
    for filename in os.listdir(queries_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(queries_dir, filename)
            img_rgb = cv2.imread(img_path)

            if img_rgb is not None:
                rgb_images.append(img_rgb)
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                gray_images.append(img_gray)
            else:
                print(f"Warning: Failed to read {img_path}")
    print("DONE")

    # Generate masks
    print("Generating masks and postprocessing them...", end=" ")
    masks = mean_adaptive_threshold(gray_images)
    masks = post_process_masks(masks)
    print("DONE")

    # Save mask images and cropped results
    print("Saving masks and cropped images...", end=" ")
    for i, mask in enumerate(masks):

        filename = f"{i:05d}.jpg"
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
    print("DONE")

    # Load images
    print("Loading images with background removed...", end=" ")
    test_images_raw = load_images_from_directory(output_cropped_dir)
    bbdd_images_raw = load_images_from_directory(bbdd_dir)
    print("DONE")

    # Change color space
    print("Changing color space of the images...", end=" ")
    test_images = []
    for image in test_images_raw:
        image = change_color_space(image, color_space) 
        test_images.extend(np.array([image]))
    bbdd_images = []
    for image in bbdd_images_raw:
        image = change_color_space(image, color_space) 
        bbdd_images.extend(np.array([image]))
    print("DONE")

    # Get histograms
    print("Extracting histograms...", end=" ")
    test_images_hists = []
    for image in test_images:
        image_hists = get_pyramid_histograms(image, dimension, bins, pyramid_level, adaptive_bins)
        test_images_hists.extend(np.array([image_hists]))
    bbdd_image_hists = []
    for image in bbdd_images:
        image_hists = get_pyramid_histograms(image, dimension, bins, pyramid_level, adaptive_bins)
        bbdd_image_hists.extend(np.array([image_hists]))
    print("DONE")

    # Calculate distance matrix
    print("Calculating distance matrix...", end=" ")
    distance_matrix = create_distance_matrix(test_images_hists, bbdd_image_hists, similarity_measure, normalize)
    print("DONE")

    # Generate results
    print("Generating results...", end=" ")
    results = generate_results(distance_matrix, similarity_measure)
    print("DONE")

    # Generate submission
    print("Generating submission...", end=" ")
    results_file = "result.pkl"
    generate_submission(results, k_val=10, output_path=results_file)
    print("DONE")

if __name__ == "__main__":
    main()