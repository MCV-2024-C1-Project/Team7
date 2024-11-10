import os
import cv2
import numpy as np
import shutil
import tqdm as tqdm
import pickle
import time
from sklearn.metrics import f1_score

from src.utils.images import load_images_from_directory
from src.utils.denoising import create_denoised_dataset
from src.utils.segmentation import generate_masks
from src.utils.keypoint_descriptors import *

def lowe_ratio_test (knn_matches, ratio_threshold):
    """
    Applies Lowe's ratio test to filter out poor matches from k-nearest neighbors (k-NN) match results.

    Args:
        knn_matches (list of tuples): A list of tuples where each tuple contains two matches (m, n).
            - `m` and `n` are typically objects with a `distance` attribute, representing the 
              distance between matched features.
        ratio_threshold (float): The threshold ratio to determine if a match is good. A smaller 
            ratio is more strict and filters out more matches.

    Returns:
        list: A list of good matches that pass Lowe's ratio test. Each match in the list is from 
        the first element of the tuple `m` in `knn_matches` that satisfies the ratio test.
    """
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) >= 2:
            m, n = match_pair[0], match_pair[1]
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches


def function_time_count(function, params):

    """
    Measures the execution time of a given function and returns both the 
    time taken and the function's result.

    Args:
        function (callable): The function to be executed.
        params (list): A tuple of parameters to pass to the function.

    Returns:
    - tuple: A tuple containing:
        - total_time (float): The time taken to execute the function, in seconds.
        - results: The output of the executed function.
    """

    start = time.time()
    results = function(*params)
    end = time.time()
    total_time = end-start

    return total_time, results


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
        key_des_list = get_key_des_wildcard_multi_image(images_list, get_Harris_key, get_SIFT_descriptors)
    
    elif method=="Harris-ORB":
        key_des_list = get_key_des_wildcard_multi_image(images_list, get_Harris_key, get_ORB_descriptors)

    elif method=="Harris-AKAZE":
        key_des_list = get_key_des_wildcard_multi_image(images_list, get_Harris_key, get_AKAZE_descriptors)
    
    elif method=="HarrisLaplacian-SIFT":
        key_des_list = get_key_des_wildcard_multi_image(images_list, get_Harris_Laplacian_keypoints, get_SIFT_descriptors)
    
    elif method=="HarrisLaplacian-ORB":
        key_des_list = get_key_des_wildcard_multi_image(images_list, get_Harris_Laplacian_keypoints, get_ORB_descriptors)
    
    elif method=="HarrisLaplacian-AKAZE":
        key_des_list = get_key_des_wildcard_multi_image(images_list, get_Harris_Laplacian_keypoints, get_AKAZE_descriptors)
    
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
                    Norm type (default: cv2.NORM_L2 for SIFT, cv2.NORM_HAMMING for ORB/AKAZE).
                params[1]: bool
                    Whether to use crossCheck (default = False).
            - For "FLANN":
                params[0]: dict
                    Index parameters.
                params[1]: dict
                    Search parameters.
                params[2]: int
                    Number of nearest neighbors (k) (default: 5).
                params[3]: float
                    Lowe's ratio for filtering matches (default: 0.7).

    Returns:
    - tuple:
        - matches: list
            List of matched descriptors.
        - num_matches: int
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
        if descr_method in ["SIFT", "Harris-SIFT", "HarrisLaplacian-SIFT"]:
            if params:
                norm = params[0]
                crossCheck = params[1]
            else:
                norm = cv2.NORM_L2
                crossCheck = False

        elif descr_method in ["ORB","AKAZE", "Harris-ORB", "Harris-AKAZE", "HarrisLaplacian-ORB", "HarrisLaplacian-AKAZE"]:
            if params:
                norm = params[0]
                crossCheck = params[1]
            else:
                norm = cv2.NORM_HAMMING
                crossCheck = False
        else:
            norm = cv2.NORM_HAMMING
            crossCheck = True

        matcher = cv2.BFMatcher(norm, crossCheck)
        matches = matcher.match(descriptors_image_1, descriptors_image_2)
        num_matches = len(matches)

    elif method == "FLANN":
        if descr_method == ["SIFT", "Harris-SIFT", "HarrisLaplacian-SIFT"]:
            if params:
                index_params = params[0]
                search_params = params[1]
                k = params[2]
                ratio = params[3]
            else:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                k = 2
                ratio = 0.7

        elif descr_method in ["ORB","AKAZE", "Harris-ORB", "Harris-AKAZE", "HarrisLaplacian-ORB", "HarrisLaplacian-AKAZE"]:
            if params:
                index_params = params[0]
                search_params = params[1]
                k = params[2]
                ratio = params[3]
            else:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
                k = 2
                ratio = 0.7
        
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = matcher.knnMatch(descriptors_image_1, descriptors_image_2, k)
        matches = lowe_ratio_test(knn_matches, ratio)
        num_matches = len(matches)

    return matches, num_matches


def check_for_unknown_painting(num_matching_descriptors_list, unknown_painting_threshold):
    """
    Determines whether an unknown painting is present by analyzing the ratio of matching descriptors.

    Args:
        num_matching_descriptors_list (list of int): A list containing the number of matching 
            descriptors for each known painting.
        unknown_painting_threshold (float): The threshold ratio used to determine if the painting 
            is unknown. If the ratio of the second highest to the highest number of matches exceeds 
            this threshold, the painting is considered unknown.

    Returns:
        bool: True if the painting is determined to be unknown, False otherwise.
    """

    matching_list_aux = sorted(num_matching_descriptors_list, reverse=True)
    max_matches = matching_list_aux[0]
    second_max_matches = matching_list_aux[1]

    ratio = second_max_matches / (max_matches+0.000001)

    return ratio


def get_predictions(query_dir, bbdd_dir, method, matching_method, matching_params=[], unknown_painting_threshold=2,
                    cache_segmented=False, cache_denoised=False):
    """
    Processes query images to identify paintings by matching their features against a database (BBDD),
    using specified image processing and matching methods.

    This function involves several steps: denoising, segmentation, feature extraction, and descriptor 
    matching. The process is optimized with optional caching for segmentation and denoising.

    Args:
        query_dir (str): Path to the directory containing query images.
        bbdd_dir (str): Path to the directory containing database (BBDD) images.
        method (str): Method used for extracting keypoints and descriptors (e.g., SIFT, ORB).
        matching_method (str): Method used for matching descriptors (e.g., brute-force, FLANN).
        matching_params (list, optional): Additional parameters for the matching method.
        unknown_painting_threshold (float, optional): Threshold for determining if a painting is unknown.
            Default is 2.
        cache_segmented (bool, optional): If True, uses cached segmented images. Default is False.
        cache_denoised (bool, optional): If True, uses cached denoised images. Default is False.

    Returns:
        list: A list of results for each query image. Each result is either:
            - A list of indices representing the best matches from the database (sorted by similarity).
            - [-1] if the painting is determined to be unknown.
    """
        
    if cache_segmented:
        print("Using cached segmented images.")
        
    if cache_denoised:
        print("Using cached denoised images.")
    
    # REMOVE NOISE FROM QUERY IMAGES FOR SEGMENTATION
    # ======================================================================================

    # Create a new directory for denoised images
    denoised_for_segmentation_queries_dir = 'data/denoised_for_segmentation_queries_dir'

    if not cache_segmented:
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

    if not cache_segmented:
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

        for i, mask in enumerate(tqdm(masks, desc="Generating segmentation masks and cropping images")):
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

    if not cache_denoised:
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
    for query_image in tqdm(query_key_des_list, desc="Matching descriptors"):
        
        # Get matching descriptors from each bbdd image
        num_matching_descriptors_list = []
        for bbdd_image in bbdd_key_des_list:
            
            # There must be at least one descriptor in the bbdd image
            if str(bbdd_image['descriptors'])!="None":
                _, num_matching_descriptors = get_num_matching_descriptors(query_image['descriptors'],
                                                                           bbdd_image['descriptors'],
                                                                           method=matching_method,
                                                                           descr_method=method,
                                                                           params=matching_params)
            else:
                num_matching_descriptors = 0
                
            num_matching_descriptors_list.append(num_matching_descriptors)
            
        # Check if the query is an unknown painting
        unknown = check_for_unknown_painting(num_matching_descriptors_list, unknown_painting_threshold)
        
        if unknown:
            # Append unknown flag if query is unknown
            results.append([-1])
        
        else:
            # Append sorted list of predictions if painting is known
            results.append(np.argsort(num_matching_descriptors_list)[::-1])
    
    return results

def generate_submission(results, paintings_per_image):
    """
    Gets the top 1 bbdd image for each query, taking into account
    whether there were one or two paintings in the image.
    
    Args:
    - results: list of lists with the ordered top results for each query
    - paintings_per_image: list with number of paintings in every query image
    
    Returns:
    - submission: a list of lists in the sumbission format
    """
    
    submission = []
    i = 0
    for num_paintings in paintings_per_image:
        if num_paintings == 1:
            submission.append([results[i][0]])
            i += 1
        elif num_paintings == 2:
            submission.append([results[i][0], results[i+1][0]])
            i += 2
            
    return submission
    
def get_unknowns_f1(submission, ground_truth):
    """
    Calculates the f1 score considering only the predictions regarding
    the unknown paintings.
    
    Args:
    - submission: results in the submission format.
    - ground_truth: groundtruth in the W4 format.
    
    Returns:
    - f1: f1 score regarding the unknown paintings predictions
    """
    
    submission_binary = [1 if sublist[0] == -1 else 0 for sublist in submission]
    groundtruth_binary = [1 if sublist[0] == -1 else 0 for sublist in ground_truth]

    f1 = f1_score(groundtruth_binary, submission_binary)
    
    return f1

# Generating results example usage
"""
query_dir = "./data/qsd1_w3/non_augmented/"
bbdd_dir = "./data/BBDD/"

results = get_predictions(query_dir, bbdd_dir,
                          method="ORB",
                          matching_method="BruteForce",
                          matching_params=[cv2.NORM_HAMMING, True],
                          unknown_painting_threshold=2,
                          cache_segmented=True,
                          cache_denoised=True)
"""

# Getting f1 example usage                       
"""
with open('./data/qsd1_w4/gt_corresps.pkl', 'rb') as f:
    ground_truth = pickle.load(f)
    
with open("data/paintings_per_image.pkl", "rb") as f:
    paintings_per_image = pickle.load(f)
    
submission = generate_submission(results, paintings_per_image)
f1 = get_unknowns_f1(submission, ground_truth)
print(f1)
"""