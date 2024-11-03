import numpy as np
import cv2

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

def closing(edges):
    """
    Applies morphological transformations to close gaps in the edges.

    Parameters:
        edges (ndarray): Binary image with edges detected.

    Returns:
        ndarray: Binary image after morphological transformations.
    """
    # Perform morphological closing to fill gaps in the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded

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

def erosion(mask):
    """
    Applies morphological transformations to erode the mask.

    Parameters:
        mask (ndarray): Binary mask with filled convex hulls.

    Returns:
        ndarray: Binary mask after morphological transformations.
    """
    # Perform morphological erosion to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    eroded = cv2.erode(mask, kernel, iterations=1)

    return eroded

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
    Generates masks for a list of images by detecting edges, closing gaps,
    filling with convex hulls, and applying erosion.

    Parameters:
        imgs_list (list): List of images (ndarray) to process.

    Returns:
        list: List of binary masks (ndarray) for each image.
    """
    masks = []
    for img in imgs_list:
        edges = compute_edges(img)
        closed = closing(edges)
        mask, contours = fill_with_convex_hull(closed)
        mask = erosion(mask)
        mask = remove_small_segments(mask)
        masks.append(mask)
    return masks