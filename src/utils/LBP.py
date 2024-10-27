from skimage.feature import local_binary_pattern
import numpy as np
from src.utils.images import split_image_into_quadrants

def compute_block_lbp(block, radius, n_points, method):
    """
    Compute the LBP histogram for a single block of the image.
    """
    lbp = local_binary_pattern(block, n_points, radius, method=method)
    
    if method=="uniform":
        # Compute LBP histogram with uniform method (bins range from 0 to n_points + 2)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    else:
        # Compute LBP histogram with default or ror method
        hist, _ = np.histogram(lbp.ravel(), bins=2**n_points, density=True)
        
    return hist

def compute_images_block_lbp(images, radius, n_points, method, n_blocks):
    """
    Given a list of images, compute the LBP for each image by dividing
    each image into blocks, computing the LBP for each block, and concatenating
    the histograms of all blocks.
    
    Parameters:
    - images: list of loaded images (grayscale)
    - radius: radius for the LBP
    - n_points: number of neighbors (points) for the LBP
    - method: method for LBP ('uniform', 'ror', etc.)
    - n_blocks: number of blocks to divide the image into (1, 4, 16, 64)
    
    Returns:
    - List of concatenated histograms for each image.
    """
    image_histograms = []
    
    for image in images:
        
        # Divide the image into blocks
        blocks = split_image_into_quadrants(image, int(np.emath.logn(4, n_blocks))+1)
        
        # Compute LBP for each block and concatenate histograms
        concatenated_histogram = []
        for block in blocks:
            block_histogram = compute_block_lbp(block, radius, n_points, method)
            concatenated_histogram.append(block_histogram)
        
        # Concatenate all block histograms into a single feature vector for the image
        concatenated_histogram = np.concatenate(concatenated_histogram)
        image_histograms.append([concatenated_histogram.astype('float32')])
    
    return image_histograms

def compute_images_block_multi_lbp(images, method, n_blocks):
    image_histograms = []
    
    for image in images:
        
        # Divide the image into blocks
        blocks = split_image_into_quadrants(image, int(np.emath.logn(4, n_blocks))+1)
        
        # Compute LBP for each block and concatenate histograms
        concatenated_histogram = []
        for block in blocks:
            
            block_histogram = compute_block_lbp(block, 1, 8, method)
            concatenated_histogram.append(block_histogram)
            
            block_histogram = compute_block_lbp(block, 2, 8, method)
            concatenated_histogram.append(block_histogram)
            
            block_histogram = compute_block_lbp(block, 3, 12, method)
            concatenated_histogram.append(block_histogram)
        
        # Concatenate all block histograms into a single feature vector for the image
        concatenated_histogram = np.concatenate(concatenated_histogram)
        image_histograms.append([concatenated_histogram.astype('float32')])
    
    return image_histograms