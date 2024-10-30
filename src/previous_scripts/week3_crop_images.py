import os
import cv2
import numpy as np
import argparse
import pickle

def compute_edges(image, threshold1=100, threshold2=200):
    """
    Computes edges in an image using the Canny edge detection algorithm.

    Parameters:
        image (ndarray): Input image, can be grayscale or RGB.
        threshold1 (int): First threshold for the hysteresis procedure.
        threshold2 (int): Second threshold for the hysteresis procedure.

    Returns:
        ndarray: Binary image with edges detected.
    """
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=3)
    
    return edges

def morphological_transformations(edges):
    """
    Applies morphological transformations to close gaps in the edges.

    Parameters:
        edges (ndarray): Binary image with edges detected.

    Returns:
        ndarray: Binary image after morphological transformations.
    """
    # Perform morphological closing to fill gaps in the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))  # Use a kernel size of your choice
    closed = cv2.dilate(edges, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  # Use a kernel size of your choice
    opened = cv2.erode(closed, kernel, iterations=1)

    return opened

def fill_with_convex_hull(edges):
    """
    Fills the detected edges with convex hulls of the largest contours.

    Parameters:
        edges (ndarray): Binary image with edges detected.

    Returns:
        tuple: A tuple containing:
            - mask (ndarray): Binary mask with filled convex hulls.
            - contours (list): List of contours found in the edges.
    """
    # Find contours of the closed image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Take the two largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Create an empty mask to draw filled convex hulls
    mask = np.zeros_like(edges)

    # Fill the convex hulls of the contours
    for contour in contours:
        if contour.size > 0:  # Ensure the contour is valid
            convex_hull = cv2.convexHull(contour)
            cv2.drawContours(mask, [convex_hull], -1, 255, thickness=cv2.FILLED)

    return mask, contours

def remove_small_segments(mask, min_area=1000):
    """
    Removes small segmented areas from a binary mask based on a minimum area threshold.
    
    Parameters:
        mask (ndarray): Binary mask where segmented areas are white (255) and background is black (0).
        min_area (int): Minimum area (in pixels) to keep. Components smaller than this will be removed.
    
    Returns:
        ndarray: Cleaned binary mask with small components removed.
    """
    # Label connected components in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Create an empty mask to store the cleaned result
    cleaned_mask = np.zeros(mask.shape, dtype=np.uint8)

    # Iterate through each component and keep only those with area above the threshold
    for i in range(1, num_labels):  # Skip label 0 as it is the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            # Retain the component in the cleaned mask
            cleaned_mask[labels == i] = 255

    return cleaned_mask

def generate_masks(imgs_list):
    """
    
    """
    masks = []
    for img in imgs_list:
        edges = compute_edges(img)
        closed = morphological_transformations(edges)
        mask, contours = fill_with_convex_hull(closed)
        mask = remove_small_segments(mask)
        masks.append(mask)
    return masks

def main(images_dir="data/denoised_1_2/", output_masks_dir="data/masks_qsd2_w3", output_cropped_dir="data/cropped_imgs_qsd2_w3"):
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
    masks = generate_masks(rgb_images)

    # List to store the number of objects (frames) per image
    frames_per_image = []

    # Save mask images and cropped results
    image_counter = 0  # Counter for unique cropped image names

    for i, mask in enumerate(masks):
        # Save the mask
        mask_filename = f"{i:05d}.png"
        masks_save_path = os.path.join(output_masks_dir, mask_filename)
        cv2.imwrite(masks_save_path, mask)

        # Detect connected components in the mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Count frames (objects of interest) in the current image
        frame_count = 0

        # Iterate over each detected component (ignoring background label 0)
        for j in range(1, num_labels):
            x, y, w, h, area = stats[j]

            # Ignore very small components (noise)
            if area > 100:  # Adjust the area threshold as needed
                # Extract and save the cropped region
                cropped_result = rgb_images[i][y:y+h, x:x+w]
                cropped_filename = f"{image_counter:05d}.jpg"
                cropped_save_path = os.path.join(output_cropped_dir, cropped_filename)
                cv2.imwrite(cropped_save_path, cropped_result)

                # Increment the counter for unique naming
                image_counter += 1
                frame_count += 1

        # Add the number of frames detected for this image to the list
        frames_per_image.append(frame_count)

    # Save the list of frame counts per image in a .pkl file
    with open("frames_per_image.pkl", "wb") as f:
        pickle.dump(frames_per_image, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process images, generate masks, and crop regions.')
    parser.add_argument('--images-dir', required=True, help='Directory containing the input images')
    parser.add_argument('--output-masks-dir', required=True, help='Directory to save the generated masks')
    parser.add_argument('--output-cropped-dir', required=True, help='Directory to save the cropped images')
    args = parser.parse_args()

    main(args.images_dir, args.output_masks_dir, args.output_cropped_dir)