import numpy as np
import cv2
import os
import pywt
import tqdm as tqdm
from skimage.measure import shannon_entropy
from skimage.restoration import denoise_wavelet

def high_pass_filter(image, ksize=5):
    """
    Apply a high-pass filter to an image.

    Args:
        image (numpy.ndarray): The input image to be filtered.
        ksize (int, optional): The size of the kernel to be used for the 
                               Gaussian blur. Must be an odd number. 
                               Default is 5.

    Returns:
        numpy.ndarray: The high-pass filtered image.
    """
    # Apply a Gaussian blur to get low-frequency components
    low_pass = cv2.GaussianBlur(image, (ksize, ksize), 0)
    # Subtract low-frequency components from the original image
    high_pass = cv2.subtract(image, low_pass)
    return high_pass

def create_gaussian_pyramid(image, levels):
    """
    Generates a Gaussian pyramid for a given image.

    Args:
        image (numpy.ndarray): The input image for which the pyramid is to be created.
        levels (int): The number of levels in the pyramid.

    Returns:
        list: A list of images representing the Gaussian pyramid, where the first element is the original image
        and each subsequent element is a downsampled version of the previous one.
    """

    pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def create_laplacian_pyramid(gaussian_pyramid):
    """
    Generates a Laplacian pyramid from a given Gaussian pyramid.

    Args:
        gaussian_pyramid (list): A list of images representing the Gaussian pyramid, where the first element
                                 is the original image and each subsequent element is a downsampled version
                                 of the previous one.

    Returns:
        list: A list of images representing the Laplacian pyramid, where each element is the difference
              between the Gaussian-blurred image at that level and the expanded version of the next level.
    """
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid)-1, 0, -1):
        size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i-1], expanded)
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid

def apply_nlm_filter(laplacian_pyramid, h):
    """
    Apply Non-Local Means (NLM) denoising filter to each level of a Laplacian pyramid.

    Args:
        laplacian_pyramid (list of numpy.ndarray): A list of 2D arrays representing the Laplacian pyramid levels.
        h (float): Parameter regulating filter strength. Higher h value removes noise better but also removes details.

    Returns:
        list of numpy.ndarray: A list of 2D arrays representing the denoised Laplacian pyramid levels.
    """
    # Apply NLM denoising to each level of the Laplacian pyramid
    denoised_pyramid = []
    for lap in laplacian_pyramid:
        denoised_pyramid.append(cv2.fastNlMeansDenoising(lap, None, h, 7, 21))  # You can adjust the NLM parameters if needed
    return denoised_pyramid

def apply_bilateral_filter(laplacian_pyramid, d, sigma_color, sigma_space):
    """
    Apply a bilateral filter to each level of a Laplacian pyramid.

    Args:
        laplacian_pyramid (list of numpy.ndarray): A list of 2D arrays representing the Laplacian pyramid levels.
        d (int): Diameter of each pixel neighborhood used during filtering.
        sigma_color (float): Filter sigma in the color space. A larger value means that
                             farther colors within the pixel neighborhood will be mixed together.
        sigma_space (float): Filter sigma in the coordinate space. A larger value means that
                             farther pixels will influence each other as long as their colors are close enough.

    Returns:
        list of numpy.ndarray: A list of 2D arrays representing the denoised Laplacian pyramid levels.
    """
    denoised_pyramid = []
    for i, lap in enumerate(laplacian_pyramid):
        if i < len(laplacian_pyramid):
            denoised_pyramid.append(cv2.bilateralFilter(lap, d, sigma_color, sigma_space))
        else:
            denoised_pyramid.append(lap)
    return denoised_pyramid

def apply_median_filter(laplacian_pyramid, ksize):
    """
    Apply a median filter to each level of a Laplacian pyramid.

    Args:
        laplacian_pyramid (list of numpy.ndarray): A list of 2D arrays representing the Laplacian pyramid levels.
        ksize (int): Size of the kernel to be used for the median filter. Must be an odd number.

    Returns:
        list of numpy.ndarray: A list of 2D arrays representing the denoised Laplacian pyramid levels.
    """
    denoised_pyramid = []
    for i, lap in enumerate(laplacian_pyramid):
        if i < len(laplacian_pyramid):
            denoised_pyramid.append(cv2.medianBlur(lap, ksize))
        else:
            denoised_pyramid.append(lap)
    return denoised_pyramid

def apply_gaussian_filter(laplacian_pyramid, ksize):
    """
    Apply a Gaussian filter to each level of a Laplacian pyramid.

    Args:
        laplacian_pyramid (list of numpy.ndarray): A list of 2D arrays representing the Laplacian pyramid levels.
        ksize (int): Size of the kernel to be used for the Gaussian filter. Must be an odd number.

    Returns:
        list of numpy.ndarray: A list of 2D arrays representing the denoised Laplacian pyramid levels.
    """
    denoised_pyramid = []
    for i, lap in enumerate(laplacian_pyramid):
        if i < len(laplacian_pyramid):
            denoised_pyramid.append(cv2.GaussianBlur(lap, (ksize, ksize), 0))
        else:
            denoised_pyramid.append(lap)
    return denoised_pyramid

def reconstruct_image(laplacian_pyramid, gaussian_base):
    """
    Reconstructs an image from its Laplacian pyramid and a Gaussian base image.

    Args:
        laplacian_pyramid (list of numpy.ndarray): A list of images representing the Laplacian pyramid.
        gaussian_base (numpy.ndarray): The base image at the lowest resolution of the Gaussian pyramid.

    Returns:
        numpy.ndarray: The reconstructed image.
    """
    current_image = gaussian_base
    for laplacian in laplacian_pyramid:
        size = (laplacian.shape[1], laplacian.shape[0])
        current_image = cv2.pyrUp(current_image, dstsize=size)
        current_image = cv2.add(current_image, laplacian)
    return current_image

def enhance_image_with_hp(denoised_image, ksize=5):
    """
    Enhance a denoised image by adding high-frequency details using a high-pass filter.

    Args:
        denoised_image (numpy.ndarray): The input denoised image.
        ksize (int, optional): The kernel size for the high-pass filter. Default is 5.

    Returns:
        numpy.ndarray: The enhanced image with high-frequency details added.
    """
    high_pass_details = high_pass_filter(denoised_image, ksize)
    # Add high-frequency details back to the denoised image
    enhanced_image = cv2.add(denoised_image, high_pass_details)
    return enhanced_image

def laplacian_pyramid_denoising(image, lowpass_params, pyramid_levels, method):
    """
    Perform denoising on an image using Laplacian pyramid decomposition and specified lowpass filtering method.

    Args:
        image (ndarray): The input image to be denoised.
        lowpass_params (dict): Parameters for the lowpass filter method.
        pyramid_levels (int): The number of levels in the pyramid.
        method (str): The lowpass filter method to be applied.
                      Supported methods are 'gaussian', 'median', 'bilateral', and 'nlm'.

    Returns:
        ndarray: The denoised image reconstructed from the Laplacian pyramid.
    """
    gaussian_pyramid = create_gaussian_pyramid(image, pyramid_levels)
    laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)

    # Apply specified lowpass filter on the Laplacian pyramid
    if method == 'gaussian':
        denoised_pyramid = apply_gaussian_filter(laplacian_pyramid, **lowpass_params)
    elif method == 'median':
        denoised_pyramid = apply_median_filter(laplacian_pyramid, **lowpass_params)
    elif method == 'bilateral':
        denoised_pyramid = apply_bilateral_filter(laplacian_pyramid, **lowpass_params)
    elif method == 'nlm':
        denoised_pyramid = apply_nlm_filter(laplacian_pyramid, **lowpass_params)
    else:
        raise ValueError(f"Unsupported lowpass filter in Laplacian Pyramid: {method}")

    return reconstruct_image(denoised_pyramid, gaussian_pyramid[-1])

def wavelet_denoising_skimage(image, wavelet='db1', mode='soft', rescale_sigma=True):
    """
    Apply wavelet denoising using skimage's denoise_wavelet function.

    Args:
        image (ndarray): Input noisy color image.
        wavelet (str): Type of wavelet to use (e.g., 'db1' for Daubechies).
        mode (str): Thresholding mode ('soft' or 'hard').
        rescale_sigma (bool): Whether to rescale the noise's standard deviation.

    Returns:
        ndarray: Denoised image.
    """
    # Perform wavelet denoising
    denoised_image = denoise_wavelet(image, wavelet=wavelet, mode=mode, rescale_sigma=rescale_sigma)
    
    # Convert to uint8 format for visualization
    denoised_image = (denoised_image * 255).astype(np.uint8)
    
    return denoised_image

def apply_dct_denoising(image, threshold=30):
    """
    Apply DCT-based denoising to an image.
    
    This function performs denoising using the Discrete Cosine Transform (DCT). 
    It applies DCT to each color channel of the image, thresholds the DCT coefficients 
    to zero out small values (which are assumed to be noise), and then applies the 
    inverse DCT to reconstruct the denoised image.

    Args:
        image (numpy.ndarray): The input image to be denoised.
        threshold (float): The threshold value for zeroing out small DCT coefficients. 
                           Coefficients with absolute values below this threshold will be set to zero.

    Returns:
        numpy.ndarray: The denoised image.
    """
    
    # Convert image to float32 for better precision in the DCT process
    image = np.float32(image) / 255.0
    
    # Apply DCT to each color channel separately
    dct_channels = []
    for i in range(3):  # Assuming RGB
        # Apply 2D DCT (Discrete Cosine Transform)
        dct = cv2.dct(image[:, :, i])
        
        # Zero out small coefficients (thresholding)
        dct[np.abs(dct) < threshold] = 0
        
        # Apply inverse DCT to reconstruct the denoised channel
        idct = cv2.idct(dct)
        dct_channels.append(idct)
    
    # Merge the three channels back into an image
    denoised_image = cv2.merge(dct_channels)
    
    # Clip values to [0, 1] range and convert back to uint8
    denoised_image = np.clip(denoised_image * 255, 0, 255).astype(np.uint8)
    
    return denoised_image

def variance_noise_estimation(image, color_space):
    if color_space == 'grayscale':
        channel = image
    elif color_space in ['lab', 'yuv']:
        channel = image[:, :, 0]  # Use luminance channel
    else:
        channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.var(channel)

def entropy_noise_estimation(image, color_space):
    if color_space == 'grayscale':
        channel = image
    elif color_space in ['lab', 'yuv']:
        channel = image[:, :, 0]  # Use luminance channel
    else:
        channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return shannon_entropy(channel)

def laplacian_noise_estimation(image, color_space):
    if color_space == 'grayscale':
        channel = image
    elif color_space in ['lab', 'yuv']:
        channel = image[:, :, 0]
    else:
        channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(channel, cv2.CV_64F)
    return np.mean(np.abs(laplacian))

def wavelet_noise_estimation(image, color_space):
    if color_space == 'grayscale':
        channel = image
    elif color_space in ['lab', 'yuv']:
        channel = image[:, :, 0]  # Use luminance channel
    else:
        channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coeffs = pywt.wavedec2(channel, 'haar', level=2)
    high_freq_coeffs = coeffs[-1]
    return np.std(np.hstack(high_freq_coeffs))

def apply_denoising(image, method, lowpass_params=None, highpass=False, wavelet_params=None, dct_params=None, pyramid_levels=None):
    # Select denoising method
    if pyramid_levels:
        denoised_image = laplacian_pyramid_denoising(image, lowpass_params, pyramid_levels, method)

    elif method == 'wavelet' and wavelet_params:
        denoised_image = wavelet_denoising_skimage(image, **wavelet_params)

    elif method == 'dct' and dct_params:
        denoised_image = apply_dct_denoising(image, **dct_params)
    
    elif method == 'gaussian':
        denoised_image = cv2.GaussianBlur(image, (lowpass_params['ksize'], lowpass_params['ksize']), 0)

    elif method == 'median':
        denoised_image = cv2.medianBlur(image, lowpass_params['ksize'])

    elif method == 'bilateral':
        d, sigma_color, sigma_space = lowpass_params['d'], lowpass_params['sigma_color'], lowpass_params['sigma_space']
        denoised_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    elif method == 'nlm':
        h_luminance, h_color = lowpass_params['h_luminance'], lowpass_params['h_color']
        denoised_image = cv2.fastNlMeansDenoising(image, None, h_luminance, h_color, 7, 21)

    else:
        raise ValueError(f"Unsupported denoising method: {method}")

    # Apply additional high-pass filtering if requested
    if highpass:
        denoised_image = enhance_image_with_hp(denoised_image, ksize=5)

    return denoised_image

def create_denoised_dataset(noisy_dataset_path, denoised_dataset_path, method, lowpass_params=None, highpass=False, wavelet_params=None, dct_params=None, pyramid_levels=None, noise_method=None, noise_threshold=0):
    """
    Create a denoised dataset from a noisy dataset.

    Args:
    - noisy_dataset_path: Path to the directory containing noisy images.
    - denoised_dataset_path: Path to save denoised images.
    - method: The denoising method ('gaussian', 'median', 'bilateral', 'nlm', 'wavelet', 'dct', 'laplacian', 'lowpass_highpass').
    - lowpass_params: Parameters for low-pass filtering.
    - highpass: Boolean to indicate if a high-pass filter should be applied after denoising.
    - wavelet_params: Parameters for wavelet denoising.
    - dct_params: Parameters for DCT denoising.
    - pyramid_levels: Number of levels for Laplacian pyramid processing.
    - noise_method: Method for estimating noise ('variance', 'entropy', 'laplacian', 'wavelet').
    - noise_threshold: Threshold for noise estimation to decide whether to denoise the image.
    """

    # Create the denoised dataset directory if it doesn't exist
    os.makedirs(denoised_dataset_path, exist_ok=True)

    # Iterate over all images in the noisy dataset directory
    for filename in tqdm.tqdm(os.listdir(noisy_dataset_path), desc="Denoising images"):
        if filename.lower().endswith('.jpg'):  # Check for valid image file extensions
            # Construct the full path to the noisy image
            noisy_image_path = os.path.join(noisy_dataset_path, filename)

            # Read the noisy image
            noisy_image = cv2.imread(noisy_image_path)

            if not noise_method:
                denoised_image = apply_denoising(
                        noisy_image,
                        method,
                        lowpass_params=lowpass_params,
                        highpass=highpass,
                        wavelet_params=wavelet_params,
                        dct_params=dct_params,
                        pyramid_levels=pyramid_levels
                    )
                
                denoised_image_path = os.path.join(denoised_dataset_path, filename)
                cv2.imwrite(denoised_image_path, denoised_image)

            else:
                # Estimate the noise level using the specified method
                if noise_method == 'variance':
                    noise_estimate = variance_noise_estimation(noisy_image, 'grayscale')
                elif noise_method == 'entropy':
                    noise_estimate = entropy_noise_estimation(noisy_image, 'grayscale')
                elif noise_method == 'laplacian':
                    noise_estimate = laplacian_noise_estimation(noisy_image, 'grayscale')
                elif noise_method == 'wavelet':
                    noise_estimate = wavelet_noise_estimation(noisy_image, 'grayscale')
                else:
                    raise ValueError(f"Unsupported noise estimation method: {noise_method}")

                # Print the noise estimate for debugging
                #print(f"Noise estimate for {filename}: {noise_estimate}")

                # Decide whether to denoise based on the estimated noise
                if noise_estimate > noise_threshold:
                    # Apply the denoising method
                    denoised_image = apply_denoising(
                        noisy_image,
                        method,
                        lowpass_params=lowpass_params,
                        highpass=highpass,
                        wavelet_params=wavelet_params,
                        dct_params=dct_params,
                        pyramid_levels=pyramid_levels
                    )

                    # Save the denoised image
                    denoised_image_path = os.path.join(denoised_dataset_path, filename)
                    cv2.imwrite(denoised_image_path, denoised_image)
                    #print(f"Denoised image saved to: {denoised_image_path}")
                else:
                    print(f"Skipping denoising for {filename} due to low noise estimate.")