import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import itertools

def create_histo_dir(base_dir, color_space, dataset):
    # Create directory for the specific color space to save histograms 

    histogram_dir = os.path.join(base_dir, dataset, color_space)
    os.makedirs(histogram_dir, exist_ok=True)
    return histogram_dir

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

def change_color_space(image, color_space):
    # Change the color space of an image 
    
    if color_space == 'RGB':
        return image
    elif color_space == 'LAB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_space == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

def get_histograms(image, dimension=1, bins=256):
    """
    Calculates histograms of an image based on specified dimensions and bins.

    Parameters
    ----------
    image : numpy.ndarray
        The input image array, which can be grayscale or color.
    dimension : int, optional
        The dimensionality of the histograms to compute (1, 2, or 3).
        This specifies the number of channels to include in each histogram.
        Default is 1.
    bins : int, optional
        The number of bins to use for the histograms along each dimension.
        Default is 256.

    Returns
    -------
    histograms : list of numpy.ndarray
        A list containing the computed histograms.
        Each histogram corresponds to a unique combination of channels
        based on the specified dimension. The different histograms
        corresponds to creating all possible combinations N choose K,
        where N is the dimensionality of the histograms, and K are the
        different dimensions of the image color space (e.g., for RGB
        and 2D histograms, it would return the 2D histograms corresponding
        to [R,G], [R,B], [G,B]).
    """
    # Determine the number of channels (dimensions) in the color space
    if len(image.shape) == 2:
        N = 1  # Grayscale image
        channels = [0]
    elif len(image.shape) == 3:
        N = image.shape[2]  # Color image
        channels = list(range(N))
    else:
        raise ValueError("Unsupported image format")

    # Validate the dimension parameter
    if dimension > N or dimension < 1:
        raise ValueError("Dimension parameter is out of valid range")

    # Generate all combinations of channels for the given dimension
    combinations = list(itertools.combinations(channels, dimension))

    histograms = []

    # Calculate histograms for each combination
    for comb in combinations:
        histSize = [bins] * dimension
        ranges = [0, 256] * dimension

        # Compute the histogram
        hist = cv2.calcHist([image], list(comb), None, histSize, ranges)
        histograms.append(hist)

    return histograms

def compare_histograms(histograms1, histograms2, distance='intersection', normalize='minmax'):
    """
    Compare two sets of histograms and compute the average distance.

    Parameters
    ----------
    histograms1 : list of numpy.ndarray
        The first set of histograms.
    histograms2 : list of numpy.ndarray
        The second set of histograms.
    distance : str, optional
        The distance metric to use.
        Default is 'intersection'.
    normalize : str, optional
        The normalization method to apply to histograms before comparison.
        Methods available:
        'minmax': cv2.NORM_MINMAX,
        'l1': cv2.NORM_L1,
        'l2': cv2.NORM_L2,
        'inf': cv2.NORM_INF

    Returns
    -------
    average_distance : float
        The average distance between the two sets of histograms.
    """
    # Validate that the two lists have the same number of histograms
    if len(histograms1) != len(histograms2):
        raise ValueError("The two sets of histograms must have the same length.")

    # Map distance strings to cv2 constants
    distance_metrics = {
        'correlation': cv2.HISTCMP_CORREL,
        'chi-square': cv2.HISTCMP_CHISQR,
        'intersection': cv2.HISTCMP_INTERSECT,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA,
        'hellinger': cv2.HISTCMP_HELLINGER,
        'kl-divergence': cv2.HISTCMP_KL_DIV
    }

    # Map normalization strings to cv2 constants
    normalization_methods = {
        'minmax': cv2.NORM_MINMAX,
        'l1': cv2.NORM_L1,
        'l2': cv2.NORM_L2,
        'inf': cv2.NORM_INF
    }

    # Check if the specified distance metric is valid
    if distance not in distance_metrics:
        raise ValueError(f"Invalid distance metric '{distance}'. Valid options are: {list(distance_metrics.keys())}")

    # Check if the specified normalization method is valid
    if normalize is not None and normalize not in normalization_methods:
        raise ValueError(f"Invalid normalization method '{normalize}'. Valid options are: {list(normalization_methods.keys())} or None")

    # Initialize total distance
    total_distance = 0

    # Iterate over each pair of histograms
    for hist1, hist2 in zip(histograms1, histograms2):
        # Flatten the histograms to 1D arrays as required by cv2.compareHist
        hist1_flat = hist1.flatten()
        hist2_flat = hist2.flatten()

        # Normalize histograms if a normalization method is specified
        if normalize is not None:
            norm_type = normalization_methods[normalize]
            hist1_flat = cv2.normalize(hist1_flat, None, alpha=0, beta=1, norm_type=norm_type)
            hist2_flat = cv2.normalize(hist2_flat, None, alpha=0, beta=1, norm_type=norm_type)

        # Compute the distance
        dist = cv2.compareHist(hist1_flat, hist2_flat, distance_metrics[distance])

        # Accumulate the distance
        total_distance += dist

    # Calculate the average distance
    average_distance = total_distance / len(histograms1)

    return average_distance


def plot_histograms(histograms, color_space, filename):
    #Visualize histograms, if more than one channel, all visible in the same plot

    # Choose colors
    if color_space in ['RGB', 'HSV', 'LAB', 'YCrCb']:
        colors = ['r', 'g', 'b']
    elif color_space == 'GRAY':
        colors = ['k'] 
    else:
        colors = ['b']  

    # Plot the histograms for each channel
    plt.figure(figsize=(10, 5))
    for i, hist in enumerate(histograms):
        plt.plot(hist, color=colors[i % len(colors)], label=f'Channel {i+1}')
    
    plt.title(f'Histograms for {filename} in {color_space} Color Space')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_histograms(histograms, filename, output_dir):
    # Save histograms in the corresponding directory, using pickle

    root, extension = os.path.splitext(filename) 
    hist_filename = root.split("_")[-1] + ".pkl" # Get image number
    hist_path = os.path.join(output_dir, hist_filename)
    with open(hist_path, 'wb') as f:
        pickle.dump(histograms, f)

# Main function
def process_images(path, base_output_dir, color_space='LAB', dataset="any", plot=False):
    """
    Generates the color histograms (for a specific color space) for all the
    images in a directory and saves them in a specific folder in pkl format.

    Parameters
    ----------
    path : str
        Directory containing the input images.
    base_output_dir : str
        Base directory where histograms will be stored.
    color_space : str, optional
        The color space to use when generating histograms. Default is 'LAB'.
    dataset : str, optional
        The dataset the images belong to. Used to create subfolders within the 
        base output directory. Default is 'any'.
    plot : bool, optional
        If True, the histograms will be plotted for each image. Default is False.
    """

    # Create the directory to later save histograms
    histogram_dir = create_histo_dir(base_output_dir, color_space, dataset)

    # Iterate through the images in the directory
    for filename in os.listdir(path):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)

            # Convert the image to the given color space
            img_converted = change_color_space(img, color_space)

            # Get histograms for the image 
            histograms = get_histograms(img_converted, color_space)

            # Save the histograms to the specific directory
            save_histograms(histograms, filename, histogram_dir)

            # Plot the histograms if requested
            if plot:
                plot_histograms(histograms, color_space, filename)

def load_histograms(hist_path):
    """
    Loads all the saved histograms at a directory
    
    Parameters
    ----------
    hist_path : str
        Relative path to the histogram directory

    Returns
    -------
    list
        A list with all the histograms in the directory
    """
    hist_list = []

    # Sort files to ensure they are in numeric order
    files = sorted(os.listdir(hist_path), key=lambda x: int(Path(x).stem))
    for file in files:
        file_path = os.path.join(hist_path, file)
        with open(file_path, 'rb') as reader:
            histogram = pickle.load(reader)
            histogram = np.concatenate(histogram) # Ensure it's 1D
            hist_list.append(histogram)
    
    return hist_list
