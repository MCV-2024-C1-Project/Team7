import os
import cv2
import numpy as np
import shutil
import tqdm as tqdm
import pickle

from src.utils.images import load_images_from_directory
from src.utils.denoising import create_denoised_dataset
from src.utils.segmentation import generate_masks
from src.utils.keypoint_descriptors import *

def lowe_ratio_test (knn_matches, ratio_threshold):
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    return good_matches

def get_key_des_multi_image(images_list, method):
    """
    Identifies keypoints and calculates descriptors for each image in
    a list of loaded images using the specified method.
    
    Args:
    - images_list (list of ndarray): list of loaded images
    - method (str): method to use to extract the keypoints and descriptors
    
    Returns:
    - key_des_list (list of dictionaries): list of dictionaries, each
                    dictionary containing the keypoints and descriptors
                    for each image.
    """
    
    if method=="SIFT":
        key_des_list = get_SIFT_key_des_multi_image(images_list)
    
    elif method=="ORB":
        key_des_list = get_ORB_key_des_multi_image(images_list)
        
    elif method=="AKAZE":
        key_des_list = get_AKAZE_key_des_multi_image(images_list)
        
    elif method=="Harris-SIFT":
        key_des_list = get_key_des_multi_image(images_list, get_Harris_key, get_SIFT_descriptors)
    
    elif method=="Harris-ORB":
        key_des_list = get_key_des_multi_image(images_list, get_Harris_key, get_ORB_descriptors)

    elif method=="Harris-AKAZE":
        key_des_list = get_key_des_multi_image(images_list, get_Harris_key, get_AKAZE_descriptors)
    
    elif method=="HarrisLaplacian-SIFT":
        key_des_list = get_key_des_multi_image(images_list, get_Harris_Laplacian_keypoints, get_SIFT_descriptors)
    
    elif method=="HarrisLaplacian-ORB":
        key_des_list = get_key_des_multi_image(images_list, get_Harris_Laplacian_keypoints, get_ORB_descriptors)
    
    elif method=="HarrisLaplacian-AKAZE":
        key_des_list = get_key_des_multi_image(images_list, get_Harris_Laplacian_keypoints, get_AKAZE_descriptors)
    
    return key_des_list

def get_num_matching_descriptors(descriptors_image_1, descriptors_image_2, method, descr_method, params=[]):
    """
    Matches descriptors between two images using either Brute-Force or FLANN-based matching.

    Parameters:
        descriptors_image_1: ndarray
            Descriptors from the first image.
        descriptors_image_2: ndarray
            Descriptors from the second image.
        method: str
            Matching method to use. Options:
            - "BruteForce": Uses Brute-Force matcher.
            - "FLANN": Uses FLANN-based matcher.
        descr_method: str
            Descriptor method used for extracting features. Options:
            - "SIFT": Uses floating-point descriptors.
            - "ORB", "AKAZE": Use binary descriptors.
        params: list, optional
            Additional parameters depending on the method:
            - For "BruteForce":
                params[0]: int
                    Norm type (cv2.NORM_L2 for SIFT, cv2.NORM_HAMMING for ORB/AKAZE).
                params[1]: bool
                    Whether to use crossCheck.
            - For "FLANN":
                params[0]: dict
                    Index parameters.
                params[1]: dict
                    Search parameters.
                params[2]: int
                    Number of nearest neighbors (k).
                params[3]: float
                    Lowe's ratio for filtering matches.

    Returns:
        tuple:
            matches: list
                List of matched descriptors.
            num_matches: int
                The number of matches found.

    Notes:
        - BruteForce:
            - Uses Euclidean distance for SIFT.
            - Uses Hamming distance for ORB and AKAZE.
        - FLANN:
            - Uses KDTree for SIFT.
            - Uses LSH for ORB and AKAZE.
        - Applies Lowe's ratio test for FLANN-based matches.
    """
    if method == "BruteForce":
        if descr_method == "SIFT":
            if params:
                norm = params[0]
                crossCheck = params[1]
            else:
                norm = cv2.NORM_L2
                crossCheck = False

        elif descr_method in ["ORB","AKAZE"]:
            if params:
                norm = params[0]
                crossCheck = params[1]
            else:
                norm = cv2.NORM_HAMMING
                crossCheck = False

        matcher = cv2.BFMatcher(norm, crossCheck)
        matches = matcher.match(descriptors_image_1, descriptors_image_2)
        num_matches = len(matches)

    elif method == "FLANN":
        if descr_method == "SIFT":
            if params:
                index_params = params[0]
                search_params = params[1]
                k = params[2]
                ratio = params[3]
            else:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                k = 5
                ratio = 0.7

        elif descr_method in ["ORB","AKAZE"]:
            if params:
                index_params = params[0]
                search_params = params[1]
                k = params[2]
                ratio = params[3]
            else:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
                k = 5
                ratio = 0.7
        
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = matcher.knnMatch(descriptors_image_1, descriptors_image_2, k)
        matches = lowe_ratio_test(knn_matches, ratio)
        num_matches = len(matches)

    return matches, num_matches


def check_for_unknown_painting(num_matching_descriptors, unknown_painting_threshold):
    pass


def get_predictions(query_dir, bbdd_dir, method, matching_method, unknown_painting_threshold):
    
    # REMOVE NOISE FROM QUERY IMAGES FOR SEGMENTATION
    # ======================================================================================

    # Create a new directory for denoised images
    denoised_for_segmentation_queries_dir = 'data/denoised_for_segmentation_queries_dir'

    # Remove previous denoised images
    if os.path.exists(denoised_for_segmentation_queries_dir):
        shutil.rmtree(denoised_for_segmentation_queries_dir)

    # Execute denoising method 1
    create_denoised_dataset(
        noisy_dataset_path = query_dir,
        denoised_dataset_path = denoised_for_segmentation_queries_dir,
        method='gaussian',
        lowpass_params={'ksize': 3},
        highpass=False
    )

    # Read denoised images
    rgb_queries_denoised = []
    for filename in os.listdir(denoised_for_segmentation_queries_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(denoised_for_segmentation_queries_dir, filename)
            img_rgb = cv2.imread(img_path)
            if img_rgb is not None:
                rgb_queries_denoised.append(img_rgb)
            else:
                print(f"Warning: Failed to read {img_path}")


    # DETECT PAINTINGS IN QUERIES (SEGMENTATION)
    # ======================================================================================

    masks_queries_dir = "data/masks_queries_dir"
    cropped_queries_dir = "data/cropped_queries_dir"

    # Remove previous masks and cropped images
    if os.path.exists(masks_queries_dir):
        shutil.rmtree(masks_queries_dir)
    if os.path.exists(cropped_queries_dir):
        shutil.rmtree(cropped_queries_dir)

    # Create new directories for masks and cropped images
    os.makedirs(masks_queries_dir, exist_ok=True)
    os.makedirs(cropped_queries_dir, exist_ok=True)

    masks = generate_masks(rgb_queries_denoised)

    # Read query images
    rgb_queries = []
    for filename in os.listdir(query_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(query_dir, filename)
            img_rgb = cv2.imread(img_path)
            if img_rgb is not None:
                rgb_queries.append(img_rgb)
            else:
                print(f"Warning: Failed to read {img_path}")

    paintings_per_image = []
    image_counter = 0

    for i, mask in enumerate(tqdm.tqdm(masks, desc="Generating segmentation masks and cropping images")):
        # Save the mask
        mask_filename = f"{i:05d}.png"
        masks_save_path = os.path.join(masks_queries_dir, mask_filename)
        cv2.imwrite(masks_save_path, mask)

        # Detect connected components in the mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Count paintings (objects of interest) in the current image
        painting_count = 0

        # Collect all valid components (ignoring background label 0)
        components = []

        for j in range(1, num_labels):
            x, y, w, h, area = stats[j]

            components.append((x, y, w, h, area))  # Store component details

        # Sort components from leftmost to rightmost (higher x to lower x)
        components_sorted = sorted(components, key=lambda c: c[0])

        # Iterate over sorted components and save them
        for (x, y, w, h, area) in components_sorted:
            # Extract and save the cropped region
            cropped_result = rgb_queries[i][y:y+h, x:x+w]
            cropped_filename = f"{image_counter:05d}.jpg"
            cropped_save_path = os.path.join(cropped_queries_dir, cropped_filename)
            cv2.imwrite(cropped_save_path, cropped_result)

            # Increment the counter for unique naming
            image_counter += 1
            painting_count += 1

        # Add the number of paintings detected for this image to the list
        paintings_per_image.append(painting_count)

    # Save the list of frame counts per image in a single .pkl file
    with open("data/paintings_per_image.pkl", "wb") as f:
        pickle.dump(paintings_per_image, f)


    # REMOVE NOISE FROM QUERY IMAGES
    # =======================================================================================

    # Load list containing paintings per image
    with open('data/paintings_per_image.pkl', 'rb') as file:
        paintings_per_image = pickle.load(file)

    denoised_paintings_folder = 'data/denoised_paintings'

    # Remove previous paintings
    if os.path.exists(denoised_paintings_folder):
        shutil.rmtree(denoised_paintings_folder)

    # Create new temporary directory for denoised images
    os.makedirs(denoised_paintings_folder, exist_ok=True)

    # Using denoising method 5
    create_denoised_dataset(
        noisy_dataset_path = cropped_queries_dir,
        denoised_dataset_path = denoised_paintings_folder,
        method='wavelet',
        wavelet_params={'wavelet':'db1', 'mode':'soft', 'rescale_sigma':True},
        highpass=False
    )


    # EXTRACT LOCAL FEATURES (KEYPOINT DESCRIPTORS) FROM
    # DENOISED QUERIES AND BBDD
    # =======================================================================================

    # Load denoised query paintings and bbdd images
    query_images = load_images_from_directory(denoised_paintings_folder)
    bbdd_images = load_images_from_directory(bbdd_dir)

    # Extract keypoints and descriptors
    query_key_des_list = get_key_des_multi_image(query_images, method)
    bbdd_key_des_list = get_key_des_multi_image(bbdd_images, method)
    
    
    # GET PREDICTIONS USING MATCHING DESCRIPTORS
    # =======================================================================================
    
    # Results matrix
    results = []
    
    # For each query
    for query_image in query_key_des_list:
        
        # Get matching descriptors from each bbdd image
        num_matching_descriptors_list = []
        for bbdd_image in bbdd_key_des_list:
            
            num_matching_descriptors = get_num_matching_descriptors(query_image['descriptors'], bbdd_image['descriptors'])
            num_matching_descriptors_list.append(num_matching_descriptors)
            
        # Check if the query is an unknown painting
        unknown = check_for_unknown_painting(num_matching_descriptors_list, unknown_painting_threshold)
        
        if unknown:
            # Append unknown flag if query is unknown
            results.append([-1])
        
        else:
            # Append sorted list of predictions if painting is known
            results.append(np.argsort(num_matching_descriptors))
    
    return results


# Example usage
"""
query_dir = "./data/qsd1_w4/"
bbdd_dir = "./data/BBDD/"
get_predictions(query_dir, bbdd_dir, method="ORB", matching_method=None, unknown_painting_threshold=None)
"""