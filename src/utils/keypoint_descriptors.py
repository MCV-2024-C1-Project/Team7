import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.feature import daisy

# =========================================================
## KEYPOINT DETECTORS
# =========================================================

# Difference of Gaussians (SIFT)

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

def get_SIFT_keypoints(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(grayscale_image, None)
    return keypoints

def get_SIFT_descriptors(image, keypoints):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(grayscale_image, keypoints)
    return descriptors


# Harris Corner Detector

def get_Harris_key(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    Detects keypoints in the given image using the Harris Corner Detector.

    Args:
    - image (ndarray): a loaded, untransformed image.
    - block_size (int): size of the neighborhood considered for corner detection.
    - ksize (int): aperture parameter of the Sobel derivative.
    - k (float): Harris detector free parameter in the equation.
    - threshold (float): threshold to identify strong corners, as a fraction of
                         the maximum corner response.

    Returns:
    - keypoints (list): list of OpenCV keypoints for compatibility with other functions.
    """
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform Harris corner detection
    harris_response = cv2.cornerHarris(grayscale_image, block_size, ksize, k)
    
    # TODO (optional, just for visualization)
    # Uncomment this code to dilate so that the corners are more clearly marked
    # harris_response = cv2.dilate(harris_response, None)
    
    # Get keypoints above the threshold
    keypoints = []
    threshold_value = threshold * harris_response.max()
    for y in range(harris_response.shape[0]):
        for x in range(harris_response.shape[1]):
            if harris_response[y, x] > threshold_value:
                keypoints.append(cv2.KeyPoint(x, y, 1))
    
    return keypoints


# Harris-Laplacian Detector

def get_Harris_Laplacian_keypoints(image):
    """
    Detects keypoints in the given image using the Harris-Laplacian detector.
    
    Args:
    - image (ndarray): Input image in BGR color format (as loaded by OpenCV).
    
    Returns:
    - keypoints (list of cv2.KeyPoint): List of keypoints detected in the image.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    harris_laplace = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
    keypoints = harris_laplace.detect(grayscale_image)

    return keypoints



# =========================================================
## LOCAL DESCRIPTORS (CAL TRIAR 3)
# =========================================================

# SIFT

def get_SIFT_descriptors(image, keypoints):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(grayscale_image, keypoints)
    return descriptors


# ORB (FAST + BRIEF)

def get_ORB_key_des(image):
    """
    Detects the keypoints in the given image using the ORB detector
    and calculates the corresponding ORB descriptors. 
    
    Args:
    - image (ndarray): a loaded, untransformed image.
    
    Returns:
    - keypoints: the detected keypoints in the image (special structure
                 containing many attributes, see OpenCV doc.)
    - features (ndarray): list of the ORB descriptors.
                        There are as many descriptors as
                        keypoints detected.
    """
    
    # Transform to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the SIFT detector
    orb = cv2.ORB_create()

    # Detect SIFT features (keypoints and descriptors)
    keypoints, descriptors = orb.detectAndCompute(grayscale_image, None)
    
    return keypoints, descriptors

def get_ORB_keypoints(image):

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()

    keypoints = orb.detect(grayscale_image, None)

    return keypoints

def get_ORB_descriptors(image, keypoints):

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()

    keypoints_ret, descriptors = orb.compute(grayscale_image, keypoints)

    return descriptors


# SURF

def get_SURF_keypoints_descriptors(image, hessianThreshold=400):
    """
    Detects keypoints and computes descriptors in the given image using the SURF (Speeded-Up Robust Features) detector.
    
    Args:
    - image (ndarray): Input image in BGR color format (as loaded by OpenCV).
    - hessianThreshold (int, optional): Threshold for the Hessian matrix used to detect keypoints.
                                         Higher values result in fewer keypoints, with only the most prominent ones detected.
                                         Default is 400.
    
    Returns:
    - keypoints (list of cv2.KeyPoint): List of detected keypoints.
    - descriptors (ndarray): Array of descriptors for each keypoint, or None if no keypoints were found.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    keypoints, descriptors = surf.detectAndCompute(grayscale_image, None)

    return keypoints, descriptors

def get_SURF_keypoints(image, hessianThreshold=400):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    keypoints = surf.detect(grayscale_image, None)
    return keypoints

def get_SURF_descriptors(image, keypoints, hessianThreshold=400):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    _, descriptors = surf.compute(grayscale_image, keypoints)
    return descriptors


# GLOH (SIFT + PCA)

def get_GLOH_descriptors(image, keypoints, n_components=128):
    """
    Computes GLOH descriptors from SIFT keypoints using PCA for dimensionality reduction.
    
    Args:
    - image (ndarray): Input image.
    - keypoints (list of KeyPoint): Detected keypoints.
    - n_components (int): Number of components for PCA.
    
    Returns:
    - gloh_descriptors (ndarray): GLOH descriptors for the keypoints.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    _, sift_descriptors = sift.compute(grayscale_image, keypoints)
    
    if sift_descriptors is not None:
        pca = PCA(n_components=n_components)
        gloh_descriptors = pca.fit_transform(sift_descriptors)
    else:
        gloh_descriptors = None
    
    return gloh_descriptors


# AKAZE 

def get_AKAZE_keypoints_descriptors(image):
    """
    Detects keypoints and computes descriptors in the given image using the AKAZE (Accelerated-KAZE) algorithm.
    
    Args:
    - image (ndarray): Input image in BGR color format (as loaded by OpenCV).
    
    Returns:
    - keypoints (list of cv2.KeyPoint): List of detected keypoints.
    - descriptors (ndarray): Array of descriptors for each keypoint, or None if no keypoints were found.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(grayscale_image, None)

    return keypoints, descriptors

def get_AKAZE_keypoints(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    keypoints = akaze.detect(grayscale_image, None)
    return keypoints

def get_AKAZE_descriptors(image, keypoints):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    _, descriptors = akaze.compute(grayscale_image, keypoints)
    return descriptors


# DAISY

def get_DAISY_descriptors(image, step=4, radius=15, rings=3, histograms=8, orientations=8):
    """
    Computes DAISY descriptors for the given image.

    Args:
    - image (ndarray): Input image in BGR color format (as loaded by OpenCV).
    - step (int): Distance between descriptor sampling points.
    - radius (int): Radius (in pixels) of the outermost ring.
    - rings (int): Number of rings.
    - histograms (int): Number of histograms per ring.
    - orientations (int): Number of orientations per histogram.

    Returns:
    - descriptors (ndarray): Array of DAISY descriptors for each keypoint.
    """
    
    # Convert image to grayscale using OpenCV
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute DAISY descriptors
    descriptors = daisy(grayscale_image, step=step, radius=radius, rings=rings, 
                        histograms=histograms, orientations=orientations, 
                        visualize=False)
    
    # Reshape descriptors for compatibility (each pixel has its own descriptor)
    descriptors = descriptors.reshape(-1, descriptors.shape[-1])
    
    return descriptors



# =========================================================
## VISUALIZATION FUNCTIONS
# =========================================================

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


def draw_Harris_keypoints(image, keypoints, figsize=(5,5)):
    """
    Draws Harris keypoints on the image for visualization.

    Args:
    - image (ndarray): loaded image to plot (in BGR)
    - keypoints: list of keypoints for the image found using Harris detector.
    - figsize (tuple): tuple of two ints determine the figure size.
    """
    
    # Convert color space to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw keypoints
    image_with_keypoints = cv2.drawKeypoints(image_rgb, keypoints, None, color=(0, 255, 0))
    
    # Display the result using matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(image_with_keypoints)
    plt.title("Harris Keypoints")
    plt.axis('off')
    plt.show()

def draw_keypoints(image, keypoints, title="Keypoints", figsize=(7,7)):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_keypoints = cv2.drawKeypoints(image_rgb, keypoints, None, color=(0, 255, 0))
    plt.figure(figsize=figsize)
    plt.imshow(image_with_keypoints)
    plt.title(title)
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


# =========================================================
## "MAIN" FUNCTIONS
# =========================================================

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

def get_key_des_multi_image(image_paths, key_function, des_function):
    """
    Given a list of image file paths, loads each image, detects its keypoints, 
    and calculates its descriptors using the specified keypoint and descriptor function.
    
    Args:
    - image_paths (list of str): List of paths to the image files.
    - key_function (function): Function to detect keypoints.
    - des_function (function): Function to compute descriptors.
    
    Returns:
    - key_des_list (list of dictionaries): List of dictionaries, each containing
                    the keypoints and descriptors for each image.
    """
    
    # List to store the keypoints and descriptors for each image
    key_des_list = []
    
    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)
        
        # Check if the image was loaded correctly
        if image is None:
            print(f"Warning: Could not load image at path: {image_path}")
            continue
        
        # Get keypoints and descriptors for the image
        keypoints = key_function(image)
        descriptors =  des_function(image)
        
        # Save them in a dictionary
        image_key_des = {
            'image_path': image_path,
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        
        # Append to the main list
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










