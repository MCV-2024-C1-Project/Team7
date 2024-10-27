import numpy as np
import argparse
import os
import pywt
import pickle
import pandas as pd
import cv2

from sklearn.preprocessing import normalize
from src.utils.ml_metrics import mapk
from src.utils.distance_matrix import generate_results, create_distance_matrix_vectors, create_distance_matrix
from tqdm import tqdm
from sklearn.decomposition import PCA
from src.utils.images import load_images_from_directory
from src.utils.histograms import get_histograms


def wavelet_features(filtered_img):
    """
    Returns the energy, the mean and the variance of a gabor filtered image
    """
    #features = {}
    """
    features['Image'] = index
    features['Method'] = wavelet
    features["Direction"] = direction

    energy = np.sum(filtered_img**2)
    features["energy"] = energy

    mean = np.mean(filtered_img)
    features["mean"] = mean

    variance = np.var(filtered_img)
    features["variance"] = variance
    """
    
     
    pca = PCA(n_components=50)
    #processed_img = np.sqrt(np.sum(filtered_img**2, axis=-1))
    #normalized_img = normalize(processed_img, axis=1, norm='l1')
    pca_vector = pca.fit_transform(filtered_img)
    pca_vector = pca.fit_transform(pca_vector.T)
    #features["coef_vector"] = pca_vector

    return np.array(pca_vector).flatten()

def generate_wavelet(wavelet, images):
    directions = ["Horizontal", "Vertical", "Diagonal"]
    wavelet_information = {}
    for direction in directions:
        wavelet_information[direction] = []

    print("Now computing",wavelet,"coefficients features" )

    for image in tqdm(images):
        cA, (cH, cV, cD) = pywt.dwt2(image, wavelet)
        coeff_list = [cH, cV, cD]
        for coeffs, direction in zip(coeff_list, directions):
            features = wavelet_features(coeffs)
            wavelet_information[direction].append(features)
        
        #wavelet_information.append(np.array(img_features).flatten())
    
    #print(wavelet_information)

    return wavelet_information


def parse_args():
    """
    Parse and return the command-line arguments
    """

    parser = argparse.ArgumentParser(description="Input for the gabor filters")

    parser.add_argument('--objective-directory', type=str, required=True,
                        help="Name of the directory of the query images to apply the wavelet filters.")
    
    parser.add_argument('--bbdd-directory', type=str, required=True,
                        help="Name of the directory of the bbdd images to apply the wavelet filters.")

    return parser.parse_args()
    
def main():
    denoised = ["","denoised_1", "denoised_2", "denoised_3", "denoised_4", "denoised_5", "non_augmented"]
    color_scheme = ["L", "V"]
    args = parse_args()
    bbdd_vectors = {}
    for colors in color_scheme:
        if colors == "L":
            color = cv2.COLOR_RGB2LAB
        elif colors == "V":
            color = cv2.COLOR_RGB2HSV

        for i, denoise in enumerate(denoised):
            print("=== NOW USING",denoise,"DATASET WITH",colors ,"===")
            if denoise == "":
                objective_directory = args.objective_directory
            else:
                objective_directory = os.path.join(args.objective_directory,denoise)

            query_images = load_images_from_directory(objective_directory)
            if colors == 'L':
                query_images = [cv2.cvtColor(image, color)[:,:,0] for image in query_images]

            elif colors == 'V':
                query_images = [cv2.cvtColor(image, color)[:,:,2] for image in query_images]


            if i == 0:
                bbdd_images = load_images_from_directory(args.bbdd_directory)

                if colors == 'L':
                    bbdd_images = [cv2.cvtColor(image, color)[:,:,0] for image in bbdd_images]

                elif colors == 'V':
                    bbdd_images = [cv2.cvtColor(image, color)[:,:,2] for image in bbdd_images]

                gt_dir = os.path.join(".\data\qsd1_w3", 'gt_corresps.pkl')
                with open(gt_dir, 'rb') as reader:
                    ground_truth = pickle.load(reader)


            wavelets = ["bior1.1", "coif1", "db1", "dmey", "haar", "rbio1.1", "sym2"]
            directions = ["Horizontal", "Vertical", "Diagonal"]
            
            distance_measures = ["L1", "L2", "Cosine", "Pearson"]
            results_and_parameters = []

            for wavelet in wavelets:
                query_vectors = generate_wavelet(wavelet, query_images)
                if i == 0:
                    bbdd_vectors[wavelet] = generate_wavelet(wavelet, bbdd_images)

                for direction in directions:
                    print("Computing results for",direction,"details")
                    query = query_vectors[direction]
                    bbdd = bbdd_vectors[wavelet][direction]

                    for measure in distance_measures:
                        print("Now computing results for",measure,"distance")
                        distance_matrix = create_distance_matrix_vectors(query, bbdd, measure)
                        results = generate_results(distance_matrix, measure)

                        mapk_val_1 = mapk(ground_truth, results, 1)
                        mapk_val_5 = mapk(ground_truth, results, 5)
                        
                        print(f"-- MAPK@{1}: {mapk_val_1} -- MAPK@{5}: {mapk_val_5}")

                        results = {
                            "wavelet" : wavelet,
                            "direction": direction,
                            "method" : measure,
                            "MAP@1" : mapk_val_1,
                            "MAP@5" : mapk_val_5  
                        }

                        results_and_parameters.append(results)
                
            df = pd.DataFrame.from_dict(results_and_parameters)
            df.to_csv("wavelet_results_"+denoise+"_"+colors+".csv")


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    main()