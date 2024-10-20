import numpy as np
import cv2

def compute_images_block_dct(images, block_size=8):
    """
    Given a list of images, computes the Discrete Cosine Transform (DCT) for each image by blocks.

    Args:
    - images (list): List of images.
    - block_size (int): The size of the blocks for which the DCT will be computed.

    Returns:
    - block_dct_images (list): A list of lists of DCTs for each image, where each image's DCT is divided into blocks.
    """

    block_dct_images = []

    for img in images:
        img_float32 = np.float32(img)  # Convert image to float32 for DCT calculation
        height, width = img.shape
        dct_blocks = []

        # Iterate over the image in block-sized steps
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Extract the block
                block = img_float32[i:i + block_size, j:j + block_size]

                # Compute the DCT for the block
                dct_block = cv2.dct(block)

                # Append the DCT of the block to the list for this image
                dct_blocks.append(dct_block)

        # Append the list of DCT blocks for this image to the final result
        block_dct_images.append(dct_blocks)

    return block_dct_images

def zigzag_indices(N):
    """
    Generates the zigzag order indices for an NxN block.
    
    Args:
    - block_size (int): The size of the DCT block (e.g., 8 for an 8x8 block).
    
    Returns:
    - zigzag (list): Indices indicating the zigzag order of the block.
    """
    indices = []
    for k in range(2 * N - 1):
        if k % 2 == 0:
            i_start = min(k, N - 1)
            i_end = max(0, k - N + 1)
            for i in range(i_start, i_end - 1, -1):
                j = k - i
                indices.append(i * N + j)
        else:
            i_start = max(0, k - N + 1)
            i_end = min(k, N - 1)
            for i in range(i_start, i_end + 1):
                j = k - i
                indices.append(i * N + j)
    return indices

def extract_dct_coefficients_zigzag(block_dct_images, num_coefs, block_size=8):
    """
    Extracts the first num_coefs coefficients in zigzag order from each block DCT for each image, and concatenates 
    these coefficients into a single vector per image.

    Args:
    - block_dct_images (list of lists): List of images, where each 
                                        image is a list of DCT blocks.
    - num_coefs (int): The number of DCT coefficients to extract from each block in zigzag order.
    - block_size (int): The size of the blocks used in DCT.

    Returns:
    - dct_vectors (list of lists): A list of vectors, each representing the concatenation 
                                   of the first K coefficients from each DCT block for each image.
    """

    # Get the zigzag indices for an 8x8 block
    zigzag = zigzag_indices(block_size)
    
    dct_vectors = []

    for blocks in block_dct_images:
        image_vector = []

        for block_dct in blocks:
            # Extract the coefficients in zigzag order
            zigzag_coeffs = zigzag_indices(block_size)
            flattened_block = block_dct.flatten()
            reordered_block_dct = flattened_block[zigzag_coeffs]
            first_coefs = reordered_block_dct[:num_coefs]

            # Append the num_coefs zigzag coefficients to the image vector
            image_vector.extend(first_coefs)

        # Append the concatenated vector for this image to the final list
        dct_vectors.append(image_vector)

    return dct_vectors