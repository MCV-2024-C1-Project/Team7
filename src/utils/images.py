import os
import cv2
import matplotlib.pyplot as plt

def load_images_from_directory(directory):
    images = []
    # List all files in the directory
    for filename in os.listdir(directory):
        # Filter only image files based on file extensions
        if filename.endswith(('.jpg')):
            # Full path to the image file
            img_path = os.path.join(directory, filename)
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            # Check if the image was loaded successfully
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image: {img_path}")
    return images

def split_image_into_quadrants(image, level):
    """
    Recursively splits an image into quadrants based on the specified level.
    
    Args:
        image (ndarray): The image to split.
        level (int): The level of subdivision
            (e.g., level=1: no split, level=2: 4 quadrants, level=3: 16 quadrants, etc.).
        
    Returns:
        list: A list of quadrants in row-major order.
    """
    if level == 1:
        return [image]
    height, width, *rest = image.shape
    center_x, center_y = width // 2, height // 2

    # Split image into quadrants
    top_left = image[0:center_y, 0:center_x]
    top_right = image[0:center_y, center_x:width]
    bottom_left = image[center_y:height, 0:center_x]
    bottom_right = image[center_y:height, center_x:width]

    # Recursively split each quadrant
    top_left_quadrants = split_image_into_quadrants(top_left, level - 1)
    top_right_quadrants = split_image_into_quadrants(top_right, level - 1)
    bottom_left_quadrants = split_image_into_quadrants(bottom_left, level - 1)
    bottom_right_quadrants = split_image_into_quadrants(bottom_right, level - 1)

    # Calculate the number of tiles per row at this level
    tiles_per_row = 2 ** (level - 1)

    # Combine quadrants in row-major order so they can be plotted correctly
    top_row = []
    bottom_row = []
    for i in range(0, len(top_left_quadrants), tiles_per_row):
        top_row.extend(top_left_quadrants[i:i+tiles_per_row] + top_right_quadrants[i:i+tiles_per_row])
    for i in range(0, len(bottom_left_quadrants), tiles_per_row):
        bottom_row.extend(bottom_left_quadrants[i:i+tiles_per_row] + bottom_right_quadrants[i:i+tiles_per_row])

    return top_row + bottom_row

def plot_quadrants(quadrants, save=False):
    """
    Plots a list of image quadrants in a grid and optionally saves the plot.
    
    Args:
        quadrants (list): List of image quadrants to plot.
        save (bool): If True, saves the plot as 'quadrants.jpg'. Defaults to False.
    
    Returns:
        None
    """
    # Calculate the grid size
    num_quadrants = len(quadrants)
    grid_size = int(num_quadrants ** 0.5)
    
    # Create the plot
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    
    # Plot each quadrant
    for i, quadrant in enumerate(quadrants):
        ax = axs[i // grid_size, i % grid_size]
        ax.imshow(cv2.cvtColor(quadrant, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    
    if save:
        plt.savefig("quadrants.jpg")
    plt.tight_layout()
    plt.show()

def load_and_preprocess_images(folder, extension=".jpg"):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((extension)):
            image_path = os.path.join(folder, filename)
            # Read image
            image = cv2.imread(image_path)
            # Check if the image was loaded successfully
            if image is not None:
                image = preprocess_image(image, image_size=(256, 256))
                images.append(image)
            else:
                print(f"Failed to load image: {image_path}")
    return images

def preprocess_image(image, image_size=(256, 256)):
    # Resize image
    image = cv2.resize(image, image_size)
    
    return image
    
def transform_images_color_space(images, color_space="gray"):
    """
    Transforms the color space of the given images.

    Args:
    - images (list): List of loaded images (in BGR format).
    - color_space (str): The target color space.
                        "gray" - Grayscale,
                        "L"    - L channel of CIELAB,
                        "V"    - V channel of HSV.
    
    Returns:
    - transformed_images (list): List of processed images in the target color space.
    """

    transformed_images = []

    for img in images:
        if color_space == "gray":
            # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            transformed_images.append(gray_img)

        elif color_space == "L":
            # Convert the image to CIELAB and extract the L channel
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L_channel = lab_img[:, :, 0]  # Extract the L channel
            transformed_images.append(L_channel)

        elif color_space == "V":
            # Convert the image to HSV and extract the V channel
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            V_channel = hsv_img[:, :, 2]  # Extract the V channel
            transformed_images.append(V_channel)

        else:
            raise ValueError("Invalid color_space argument. Choose 'gray', 'L', or 'V'.")

    return transformed_images

def rescale_images(images, image_size):
    rescaled_images = []
    for image in images:
        rescaled_images.append(cv2.resize(image, image_size))
    return rescaled_images