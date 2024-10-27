import numpy as np
import argparse
import os
import pywt
import pickle
import pandas as pd
import cv2
import matplotlib.pyplot as plt 

from sklearn.preprocessing import normalize
from src.utils.ml_metrics import mapk
from src.utils.distance_matrix import generate_results, create_distance_matrix_vectors, create_distance_matrix
from tqdm import tqdm
from sklearn.decomposition import PCA
from src.utils.images import load_images_from_directory
from src.utils.histograms import get_histograms


def create_gabor_filters(kernel_size, orientations):
    gabor_filters = []
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    for angle in np.arange(0, np.pi, np.pi/orientations):
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

    features['Image'] = index
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
                        help="Name of the directory of the query images to apply the wavelet filters.")
    
    parser.add_argument('--bbdd-directory', type=str, required=True,
                        help="Name of the directory of the bbdd images to apply the wavelet filters.")
    
    parser.add_argument('--kernel-size', type=int, required=True,
                        help="Gabor filter kernel size.")

    return parser.parse_args()
    
def main():
    args = parse_args()
    num_orientations = [2**n for n in range(6)]
    gabor_filters = []
    args = parse_args()

    query_images = load_images_from_directory(args.objective_directory)
    query_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in query_images]
    bbdd_images = load_images_from_directory(args.bbdd_directory)
    bbdd_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in bbdd_images]
    gt_dir = os.path.join(".\data\qsd1_w3", 'gt_corresps.pkl')
    with open(gt_dir, 'rb') as reader:
        ground_truth = pickle.load(reader)

    distance_measures = ['correlation', 'chi-square', 'intersection', 'bhattacharyya', 'hellinger', 'kl-divergence']
    results_and_parameters = []

    for orientations in num_orientations:
        gabor_filters.append(create_gabor_filters(args.kernel_size, orientations))

    #for direction in directions:
        #print("Computing results for",direction,"details")
    for num_angles, gabor_filter in zip(num_orientations, gabor_filters): 
        print("Now computing results for",num_angles,"orientations") 
        query = [get_histograms(np.float32(apply_gabor_filter(img, gabor_filter))) for img in query_images]
        bbdd = [get_histograms(np.float32(apply_gabor_filter(img, gabor_filter))) for img in bbdd_images]

        for measure in distance_measures:
            print("Now computing results for",measure,"distance")
            distance_matrix = create_distance_matrix(query, bbdd, measure,normalize='l1')
            results = generate_results(distance_matrix, measure)

            mapk_val_1 = mapk(ground_truth, results, 1)
            mapk_val_5 = mapk(ground_truth, results, 5)
            
            print(f"-- MAPK@{1}: {mapk_val_1} -- MAPK@{5}: {mapk_val_5}")

            results = {
                "num_angles" : num_angles,
                "distance_measure" : measure,
                "MAP@1" : mapk_val_1,
                "MAP@5" : mapk_val_5  
            }

            results_and_parameters.append(results)
    
    df = pd.DataFrame.from_dict(results_and_parameters)
    df.to_csv("gabor_results.csv")


if __name__ == "__main__":
    main()
