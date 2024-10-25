import numpy as np
import argparse
import cv2
import os
import pywt
import shutil
from skimage.measure import shannon_entropy
from skimage.restoration import denoise_wavelet

from src.utils.distance_matrix import generate_results, create_distance_matrix_vectors, generate_submission
from src.utils.images import load_and_preprocess_images, transform_images_color_space, load_images_from_directory
from src.utils.DCT import compute_images_block_dct, extract_dct_coefficients_zigzag

def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Input for DCT study.")

    # Files
    parser.add_argument('--queries-dir', type=str, required=True,
                        help="Name of the directory containing query images.")
    parser.add_argument('--bbdd-dir', type=str, required=True,
                        help="Name of the directory containing database images.")

    return parser.parse_args()

# HELPER FUNCTIONS
# ===========================================================
def high_pass_filter(image, ksize=5):
    """Apply a high-pass filter to the image."""
    # Apply a Gaussian blur to get low-frequency components
    low_pass = cv2.GaussianBlur(image, (ksize, ksize), 0)
    # Subtract low-frequency components from the original image
    high_pass = cv2.subtract(image, low_pass)
    return high_pass

def create_gaussian_pyramid(image, levels):
    pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def create_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid)-1, 0, -1):
        size = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i-1], expanded)
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid

def apply_nlm_filter(laplacian_pyramid, h):
    denoised_pyramid = []
    for lap in laplacian_pyramid:
        denoised_pyramid.append(cv2.fastNlMeansDenoising(lap, None, h, 7, 21))  # You can adjust the NLM parameters if needed
    return denoised_pyramid

def apply_bilateral_filter(laplacian_pyramid, d, sigma_color, sigma_space):
    denoised_pyramid = []
    for i, lap in enumerate(laplacian_pyramid):
        if i < len(laplacian_pyramid):
            denoised_pyramid.append(cv2.bilateralFilter(lap, d, sigma_color, sigma_space))
        else:
            denoised_pyramid.append(lap)
    return denoised_pyramid

def apply_median_filter(laplacian_pyramid, ksize):
    denoised_pyramid = []
    for i, lap in enumerate(laplacian_pyramid):
        if i < len(laplacian_pyramid):
            denoised_pyramid.append(cv2.medianBlur(lap, ksize))
        else:
            denoised_pyramid.append(lap)
    return denoised_pyramid

def apply_gaussian_filter(laplacian_pyramid, ksize):
    denoised_pyramid = []
    for i, lap in enumerate(laplacian_pyramid):
        if i < len(laplacian_pyramid):
            denoised_pyramid.append(cv2.GaussianBlur(lap, (ksize, ksize), 0))
        else:
            denoised_pyramid.append(lap)
    return denoised_pyramid

def reconstruct_image(laplacian_pyramid, gaussian_base):
    current_image = gaussian_base
    for laplacian in laplacian_pyramid:
        size = (laplacian.shape[1], laplacian.shape[0])
        current_image = cv2.pyrUp(current_image, dstsize=size)
        current_image = cv2.add(current_image, laplacian)
    return current_image

def enhance_image_with_hp(denoised_image, ksize=5):
    """Enhance the denoised image using high-pass filtering."""
    high_pass_details = high_pass_filter(denoised_image, ksize)
    # Add high-frequency details back to the denoised image
    enhanced_image = cv2.add(denoised_image, high_pass_details)
    return enhanced_image

def laplacian_pyramid_denoising(image, lowpass_params, pyramid_levels, method):
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

    Parameters:
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
    """Apply DCT-based denoising on an image."""
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

    Parameters:
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
    for filename in os.listdir(noisy_dataset_path):
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

# MAIN FUNCTION
# ===========================================================
def generate_submission_qst1(query_dir, bbdd_dir):
        
    # REMOVE NOISE FROM QUERY IMAGES
    # =========================================================
    
    denoised_paintings_folder = 'data/denoised_paintings'

    # Remove previous paintings
    if os.path.exists(denoised_paintings_folder):
        shutil.rmtree(denoised_paintings_folder)

    # Create new temporary directory for denoised images
    os.makedirs(denoised_paintings_folder, exist_ok=True)
    
    # Using denoising method 5
    create_denoised_dataset(
        noisy_dataset_path = query_dir,
        denoised_dataset_path = denoised_paintings_folder,
        method='wavelet',
        wavelet_params={'wavelet':'db1', 'mode':'soft', 'rescale_sigma':True},
        highpass=False
    )
    
    
    # EXTRACT TEXTURE FEATURES FROM DENOISED QUERIES AND BBDD
    # =========================================================
    
    # Using DCT with best parameters
    color = "V"
    distance_measure = "Cosine"
    block_size = 64
    num_coefs = 64

    # Load denoised query paintings and bbdd images
    query_images = load_and_preprocess_images(denoised_paintings_folder, extension=".jpg")
    bbdd_images = load_and_preprocess_images(bbdd_dir, extension=".jpg")

    # Transform color space of the images
    query_images_color = transform_images_color_space(query_images, color_space=color)
    bbdd_images_color = transform_images_color_space(bbdd_images, color_space=color)

    # Compute the DCT of the images
    query_dct_blocks = compute_images_block_dct(query_images_color, block_size)
    bbdd_dct_blocks = compute_images_block_dct(bbdd_images_color, block_size)

    # Extract first K coefficients of images DCTs
    query_feature_vectors = extract_dct_coefficients_zigzag(query_dct_blocks, num_coefs, block_size)
    bbdd_feature_vectors = extract_dct_coefficients_zigzag(bbdd_dct_blocks, num_coefs, block_size)
    
    
    # CALCULATE DISTANCES AND ORDER RESULTS
    # =========================================================
    
    # Calculate distance matrix
    distance_matrix = create_distance_matrix_vectors(query_feature_vectors, 
                                                     bbdd_feature_vectors,
                                                     distance_measure)
    # Generate sorted results
    results = generate_results(distance_matrix, distance_measure)
    
    
    # GENERATE SUBMISSION
    # =========================================================

    results_file = "result.pkl"
    generate_submission(results, k_val=10, output_path=results_file)

def main():

    # Parse command line arguments
    args = parse_args()

    # Call the main function
    generate_submission_qst1(args.queries_dir, args.bbdd_dir)

if __name__ == "__main__":
    main()