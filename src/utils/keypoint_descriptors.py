import cv2
import matplotlib.pyplot as plt

# Difference of Gaussians (SIFT)
# =========================================================

def get_SIFT_key_des(image):
    """
    Detects the keypoints in the given image using the SIFT detector
    and calculates the corresponding SIFT descriptors.
    
    Args:
    - image (ndarray): a loaded, untransformed image.
    
    Returns:
    - keypoints: the detected keypoints in the image (special structure
                 containing many attributes, see OpenCV doc.)
    - features (ndarray): list of the SIFT descriptors, each being 128
                          dimensions. There are as many descriptors as
                          keypoints detected.
    """
    
    # Transform to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features (keypoints and descriptors)
    keypoints, descriptors = sift.detectAndCompute(grayscale_image, None)
    
    return keypoints, descriptors

def get_SIFT_key_des_multi_image(images_list):
    """
    Given a list of loaded images, detects their keypoints using the SIFT
    detector and calculates their corresponding SIFT descriptors.
    
    Args:
    - images_list (list of ndarrays): list of loaded images.
    
    Returns:
    - key_des_list (list of dictionaries): list of dictionaries, each
                    dictionary containing the keypoints and descriptors
                    for each image.
    """
    
    # Save the keypoints and descriptors in a list
    key_des_list = []
    
    for image in images_list:
        # Get the keypoints and descriptors for each image
        keypoints, descriptors = get_SIFT_key_des(image)
        
        # Save them in a dictionary
        image_key_des = {}
        image_key_des['keypoints'] = keypoints
        image_key_des['descriptors'] = descriptors
        
        # Save the dictionary in the general list of keypoints and descriptors
        key_des_list.append(image_key_des)
        
    return key_des_list

def get_num_SIFT_matches(descriptors_image_1, descriptors_image_2):
    """
    Calculates the number of matching SIFT descriptors between two list of
    descriptors representing two different images. It uses the Brute Force
    Matcher from OpenCV with the L1 distance, as this is the recommended
    distance for SIFT. Also, crossCheck is used so that the ratio test is
    not necessary.
    
    Args:
    - descriptors_image_1: list of SIFT descriptors from image 1.
    - descriptors_image_2: list of SIFT descriptors from image 2.
    
    Returns:
    - num_matches (int): the number of matching descriptors.
    """
    
    # Create feature matcher
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Match descriptors of both images
    matches = matcher.match(descriptors_image_1, descriptors_image_2)
    
    # Calculate number of matches
    num_matches = len(matches)
    
    return matches, num_matches

def draw_SIFT_keypoints(image, keypoints, figsize=(5,5)):
    """
    Plots the image in RGB along with the SIFT keypoints found for
    the image using OpenCV drawKeypoints()
    
    Args:
    - image (ndarray): loaded image to plot (in BGR)
    - keypoints: list of keypoints for the image found using OpenCV's SIFT
    - figsize (tuple): tuple of two ints determine the figure size.
    """
    
    # Convert color space to rgb, so it can be visualized
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Call the OpenCV keypoint drawing function
    image_with_keypoints = cv2.drawKeypoints(image_rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display the result using matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(image_with_keypoints, cmap='gray')
    plt.title("SIFT Keypoints")
    plt.axis('off')
    plt.show()

def draw_SIFT_matches(image_1, image_2, keypoints_1, keypoints_2, matches, num_matches, figsize=(7,7)):
    """
    Draws two images and lines connecting their SIFT matching features.

    Args:
    - image_1 (ndarray): the first (loaded) image in BGR
    - image_2 (ndarray): the second (loaded) image in BGR
    - keypoints_1: the SIFT keypoints of the first image.
    - keypoints_2: the SIFT keypoints of the second image.
    - matches: the list of SIFT descriptors matches.
    - figsize (tuple): tuple of two ints determine the figure size.
    """
    
    # Convert color space to rgb, so it can be visualized
    image_1_rgb = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_2_rgb = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    
    # Sort matches to represent the closest ones in order
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Create image with matches using OpenCV drawMatches
    matched_img = cv2.drawMatches(image_1_rgb, keypoints_1,
                                  image_2_rgb, keypoints_2, matches[:num_matches],
                                  image_2_rgb, flags=2)
    
    
    # Plot image
    plt.figure(figsize=figsize)
    plt.imshow(matched_img, cmap='gray')
    plt.title("SIFT matches")
    plt.axis('off')
    plt.show()


# ORB (FAST + BRIEF)
# =========================================================

def get_ORB_key_des(image):
    """
    Detects the keypoints in the given image using the ORB detector
    and calculates the corresponding ORB descriptors. 
    
    Args:
    - image (ndarray): a loaded, untransformed image.
    
    Returns:
    - keypoints: the detected keypoints in the image (special structure
                 containing many attributes, see OpenCV doc.)
    - features (ndarray): list of the ORB descriptors, each being 128
                          dimensions. There are as many descriptors as
                          keypoints detected.
    """
    
    # Transform to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the SIFT detector
    orb = cv2.ORB_create()

    # Detect SIFT features (keypoints and descriptors)
    keypoints, descriptors = orb.detectAndCompute(grayscale_image, None)
    
    return keypoints, descriptors