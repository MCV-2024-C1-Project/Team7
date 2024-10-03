import os
import cv2
import pickle
import matplotlib.pyplot as plt

def create_histo_dir(base_dir, color_space, dataset):
    # Create directory for the specific color space to save histograms 

    histogram_dir = os.path.join(base_dir, dataset, color_space)
    os.makedirs(histogram_dir, exist_ok=True)
    return histogram_dir

def change_color_space(image, color_space):
    # Change the color space of an image 
    
    if color_space == 'RGB':
        return image  #Comprovar q al llegir amb cv2 sigui rgb i q no canvii ordre de canals
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

def get_histograms(image, color_space):
    # Compute histograms for each channel

    # For when there's only one channel, as is the case of greyscale
    if len(image.shape) == 2:  
        histograms = cv2.calcHist([image], [0], None, [256], (0, 256))
    else:
        # For when there's more than one channel, we compute histogram for each one (and concat?)
        histograms = []
        channels = image.shape[2]  
        for c in range(channels):
            h = cv2.calcHist([image], [c], None, [256], (0, 256))
            histograms.append(h)
        #histograms =  FALTARIA CONCATENAR SI VOLEM Q SIGUI 1D, ARA TINC LLISTA DE LLISTES XQ HAVIA FET PLOTS

    
    return histograms


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

def process_images(path, base_output_dir, color_space='LAB', dataset="any", plot=False):
    # "Main"
    """
    Parameters:
    - path: Directory containing the input images.
    - base_output_dir: Base directory to store histograms.
    - color_space: The color space to use to get histograms.
    - dataset: The dataset the images are from. Used to create subfolders.
    - plot: Whether to plot the histograms for each image.
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

