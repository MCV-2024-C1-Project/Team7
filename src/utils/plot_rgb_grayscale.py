import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

def plot_image_and_channel_decomposition(image_path):
    # Load the image
    original_image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a NumPy array
    image_array = np.array(original_image) / 255.0  # Normalize pixel values to [0, 1]

    # Create the grayscale image
    grayscale_image = color.rgb2gray(image_array)  # Convert to grayscale

    # Decompose the RGB image into its channels
    R_channel = image_array[:, :, 0]  # Red channel
    G_channel = image_array[:, :, 1]  # Green channel
    B_channel = image_array[:, :, 2]  # Blue channel
    
    # Plot the original image and RGB channels with grayscale
    plt.figure(figsize=(12, 10))  # Increased height for histograms

    # Original image spanning two columns at the top center
    plt.subplot(3, 4, (2, 3))  # Centered by spanning columns 2 and 3
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot RGB channels and grayscale in a single row below
    plt.subplot(3, 4, 5)
    plt.imshow(R_channel, cmap='Reds')
    plt.title('Red Channel')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(G_channel, cmap='Greens')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(B_channel, cmap='Blues')
    plt.title('Blue Channel')
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Plot histograms for each channel
    plt.subplot(3, 4, 9)  # Histogram for Red channel
    plt.hist(R_channel.flatten(), bins=256, color='red', alpha=0.6)
    plt.title('Histogram of Red Channel')
    plt.xlim(0, 1)  # RGB channels range from 0 to 1
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(3, 4, 10)  # Histogram for Green channel
    plt.hist(G_channel.flatten(), bins=256, color='green', alpha=0.6)
    plt.title('Histogram of Green Channel')
    plt.xlim(0, 1)  # RGB channels range from 0 to 1
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(3, 4, 11)  # Histogram for Blue channel
    plt.hist(B_channel.flatten(), bins=256, color='blue', alpha=0.6)
    plt.title('Histogram of Blue Channel')
    plt.xlim(0, 1)  # RGB channels range from 0 to 1
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(3, 4, 12)  # Histogram for Grayscale channel
    plt.hist(grayscale_image.flatten(), bins=256, color='gray', alpha=0.6)
    plt.title('Histogram of Grayscale Image')
    plt.xlim(0, 1)  # Grayscale ranges from 0 to 1
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Example usage:
image_path = r'C:\Users\mirei\Programes\MCV\C1. Introduction to Human and Computer Vision\c1_project_team7\data\qsd1_w1\00028.jpg'  # Use raw string for the path
plot_image_and_channel_decomposition(image_path)
