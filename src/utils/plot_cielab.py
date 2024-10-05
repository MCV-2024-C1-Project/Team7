import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

def plot_image_and_cielab_decomposition(image_path):
    # Load the image
    original_image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a NumPy array and then to CIELAB color space
    image_array = np.array(original_image) / 255.0  # Normalize pixel values to [0, 1]
    lab_image = color.rgb2lab(image_array)
    
    # Decompose the CIELAB image into its channels
    L_channel = lab_image[:, :, 0]  # L* channel
    a_channel = lab_image[:, :, 1]  # a* channel
    b_channel = lab_image[:, :, 2]  # b* channel
    
    # Plot the original image and CIELAB channels
    plt.figure(figsize=(12, 10))  # Increased height for histograms

    # Original image at the top center
    plt.subplot(3, 3, 2)  # Centered in the second column
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot CIELAB channels in a single row below
    plt.subplot(3, 3, 4)
    plt.imshow(L_channel, cmap='gray')
    plt.title('L* Channel')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(a_channel, cmap='gray')
    plt.title('a* Channel')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(b_channel, cmap='gray')
    plt.title('b* Channel')
    plt.axis('off')

    # Plot histograms for each channel
    plt.subplot(3, 3, 7)  # Histogram for L* channel
    plt.hist(L_channel.flatten(), bins=256, color='gray', alpha=0.6)
    plt.title('Histogram of L* Channel')
    plt.xlim(0, 100)  # L* channel ranges from 0 to 100
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(3, 3, 8)  # Histogram for a* channel
    plt.hist(a_channel.flatten(), bins=256, color='red', alpha=0.6)
    plt.title('Histogram of a* Channel')
    plt.xlim(-128, 127)  # a* channel ranges from -128 to 127
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(3, 3, 9)  # Histogram for b* channel
    plt.hist(b_channel.flatten(), bins=256, color='blue', alpha=0.6)
    plt.title('Histogram of b* Channel')
    plt.xlim(-128, 127)  # b* channel ranges from -128 to 127
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Example usage:
image_path = r'C:\Users\mirei\Programes\MCV\C1. Introduction to Human and Computer Vision\c1_project_team7\data\qsd1_w1\00028.jpg'  # Replace with your image path
plot_image_and_cielab_decomposition(image_path)
