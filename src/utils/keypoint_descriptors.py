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
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors