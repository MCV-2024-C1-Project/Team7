import numpy as np
import cv2
import argparse
import os
import pandas as pd
from tqdm import tqdm

from src.utils.images import load_images_from_directory

def create_gabor_filters(kernel_size, orientations):
    gabor_filters = []
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    for angle in orientations:
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, angle, lambd, gamma, ktype=cv2.CV_32F)
        kernel /= 1.0 * kernel.sum()  # Brightness normalization
        gabor_filters.append(kernel)
    return gabor_filters

def apply_gabor_filter(img, filters):

    newimage = np.zeros_like(img)

    #always taking the most descriptive image
    for kern in filters:  
        image_filter = cv2.filter2D(img, -1, kern)  #Apply filter to image
        np.maximum(newimage, image_filter, newimage)

    return newimage

def gabor_features(filtered_img, index, num_filters):
    """
    Returns the energy, the mean and the variance of a gabor filtered image
    """
    features = {}

    features['index'] = index
    features['num_filters'] = num_filters

    energy = np.sum(filtered_img**2)
    features["energy"] = energy

    mean = np.mean(filtered_img)
    features["mean"] = mean

    variance = np.var(filtered_img)
    features["variance"] = variance

    return features

def parse_args():
    """
    Parse and return the command-line arguments
    """

    parser = argparse.ArgumentParser(description="Input for the gabor filters")

    parser.add_argument('--objective-directory', type=str, required=True,
                        help="Name of the directory of the images to apply the gabor filters.")
    parser.add_argument('--kernel-size', type=int, required=True,
                        help="Size of the kernel to apply the gabor filter")
    return parser.parse_args()
    
def main():
    args = parse_args()
    images = load_images_from_directory(args.objective_directory)
    num_orientations = [2**n for n in range(6)]
    gabor_information = []

    for num_filters in num_orientations:
        orientations = np.arange(0, np.pi, np.pi/num_filters)
        gabor_filters = create_gabor_filters(args.kernel_size, orientations)
        print("Now computing garbor with ",str(num_filters),"filters")
        for i, image in enumerate(tqdm(images)):
            filtered_img = apply_gabor_filter(image, gabor_filters)
            features = gabor_features(filtered_img, i, num_filters)
            gabor_information.append(features)
        
    df = pd.DataFrame.from_dict(gabor_information)
    save_path = os.path.join(args.objective_directory,"garbor_features.csv")
    df.to_csv(save_path)
    print(df)

if __name__ == "__main__":
    main()
