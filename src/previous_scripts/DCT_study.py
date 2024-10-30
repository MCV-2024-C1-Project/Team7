import cv2
import os
import numpy as np
import pandas as pd
import pickle
import argparse

from src.utils.ml_metrics import mapk
from src.utils.distance_matrix import generate_results, create_distance_matrix_vectors
from src.utils.images import load_and_preprocess_images, transform_images_color_space
from src.utils.DCT import compute_images_block_dct, extract_dct_coefficients_zigzag

def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Input for DCT study.")

    # Files
    parser.add_argument('--queries-dir', type=str, required=True,
                        help="Name of the directory containing query images.")
    parser.add_argument('--bbdd-dir', type=str, required=True,
                        help="Name of the directory containing database images.")

    return parser.parse_args()

def dct_study(query_folder, bbdd_folder, 
              colors = ["gray", "L", "V"], 
              block_sizes = [256, 128, 64, 32, 16, 8],
              num_coefs_list = [64, 32, 16, 8, 4],
              distance_measures = ["L1", "L2", "Cosine", "Pearson"]):
    
    # Store results in a list that will later be converted to a dataframe
    full_results = []
    
    # Load and preprocess query images
    query_images = load_and_preprocess_images(query_folder, extension=".jpg")

    # Load and preprocess bbdd images
    bbdd_images = load_and_preprocess_images(bbdd_folder, extension=".jpg")
    
    for color in colors:
        
        # Load groundtruth
        gt_dir = os.path.join(query_folder, 'gt_corresps.pkl')
        with open(gt_dir, 'rb') as reader:
            ground_truth = pickle.load(reader)
        
        # Transform the color space of the query images
        query_images_color = transform_images_color_space(query_images, color_space=color)
        
        # Transform the color space of the bbdd images
        bbdd_images_color = transform_images_color_space(bbdd_images, color_space=color)
        
        for block_size in block_sizes:
            
            # Compute the DCT of the query images
            query_dct_blocks = compute_images_block_dct(query_images_color, block_size)
            
            # Compute the DCT of the bbdd images
            bbdd_dct_blocks = compute_images_block_dct(bbdd_images_color, block_size)
            
            for num_coefs in num_coefs_list:
                
                # Extract first K coefficients of query images DCTs
                query_feature_vectors = extract_dct_coefficients_zigzag(query_dct_blocks, num_coefs, block_size)
                
                # Extract first K coefficients of bbdd images DCTs
                bbdd_feature_vectors = extract_dct_coefficients_zigzag(bbdd_dct_blocks, num_coefs, block_size)
                
                for distance_measure in distance_measures:
                    
                    # Print current combination
                    print(f"{color}, {block_size} block size, {num_coefs} coefs, {distance_measure}", end=" ")
                    
                    # Calculate distance matrix
                    distance_matrix = create_distance_matrix_vectors(query_feature_vectors, 
                                                                     bbdd_feature_vectors,
                                                                     distance_measure)
                    # Generate sorted results
                    results = generate_results(distance_matrix, distance_measure)
                    
                    # Calculate metrics
                    mapk_val_1 = mapk(ground_truth, results, 1)
                    mapk_val_5 = mapk(ground_truth, results, 5)
                    
                    # Print metrics for current combination
                    print(f"-- MAPK@{1}: {mapk_val_1} -- MAPK@{5}: {mapk_val_5}")
                    
                    # Save current combination parameters and results
                    combination_results = {}
                    combination_results['Color'] = color
                    combination_results['Block size'] = block_size
                    combination_results['Num. coeffs'] = num_coefs
                    combination_results['Distance measure'] = distance_measure
                    combination_results['Feature vec. dim.'] = num_coefs*((256/block_size)**2)
                    combination_results['MAP@1'] = mapk_val_1
                    combination_results['MAP@5'] = mapk_val_5
                    full_results.append(combination_results)
                    
    # Save results to csv via df
    full_results_df = pd.DataFrame(full_results)
    full_results_df.to_csv("dct_study_results.csv")
    
    return full_results_df

def main():

    # Get command-line arguments
    args = parse_args()

    # Execute main function
    dct_study(args.queries_dir, args.bbdd_dir, 
                  colors = ["gray", "L", "V"], 
                  block_sizes = [256, 128, 64, 32, 16, 8],
                  num_coefs_list = [64, 32, 16, 8, 4],
                  distance_measures = ["L1", "L2", "Cosine", "Pearson"])

if __name__ == "__main__":
    main()
