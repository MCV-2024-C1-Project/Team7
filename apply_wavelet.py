import numpy as np
import cv2
import argparse
import os
import pandas as pd
import pywt
from tqdm import tqdm

from src.utils.images import load_images_from_directory


def apply_gabor_filter(img, filters):

    newimage = np.zeros_like(img)

    #always taking the most descriptive image
    for kern in filters:  
        image_filter = cv2.filter2D(img, -1, kern)  #Apply filter to image
        np.maximum(newimage, image_filter, newimage)

    return newimage

def wavelet_features_for_df(filtered_img, direction, wavelet, index):
    """
    Returns the energy, the mean and the variance of a gabor filtered image
    """
    features = {}

    features['Image'] = index
    features['Method'] = wavelet
    features["Direction"] = direction

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
                        help="Name of the directory of the images to apply the wavelet filters.")

    return parser.parse_args()
    
def main():
    args = parse_args()
    images = load_images_from_directory(args.objective_directory)
    wavelets = ["bior1.1", "coif1", "db1", "dmey", "haar", "rbio1.1", "sym2"]
    directions = ["Horizontal", "Vertical", "Diagonal"]
    wavelet_information = []

    for wavelet in wavelets:
        print("Now computing",wavelet,"coefficients features" )
        for i, image in enumerate(tqdm(images)):
            cA, (cH, cV, cD) = pywt.dwt2(image, wavelet)
            coeff_list = [cH, cV, cD]
            for coeffs, direction in zip(coeff_list, directions):
                features = wavelet_features_for_df(coeffs, direction, wavelet, i)
                wavelet_information.append(features)
    
    df = pd.DataFrame.from_dict(wavelet_information)
    save_path = os.path.join(args.objective_directory,"wavelet_features.csv")
    df.to_csv(save_path)

if __name__ == "__main__":
    main()