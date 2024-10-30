import cv2

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