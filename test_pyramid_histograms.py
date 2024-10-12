import os
import numpy as np
import pickle
import pandas as pd
import argparse

from src.utils.histograms import change_color_space, get_pyramid_histograms
from src.utils.distance_matrix import create_distance_matrix, generate_results
from src.utils.images import load_images_from_directory
from src.utils.ml_metrics import mapk

def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process input for image retrieval")

    # Files
    parser.add_argument('--queries-dir', type=str, required=True,
                        help="Name of the directory containing query images.")
    parser.add_argument('--bbdd-dir', type=str, required=True,
                        help="Name of the directory containing database images.")
    
    # Fixed parameters
    parser.add_argument('--color-space', type=str, required=True,
                        help="Name of the color space to use to process the images.")
    parser.add_argument('--similarity-measure', type=str, required=True,
                        help="Name of the similarity measure to compare histograms.")
    parser.add_argument('--histogram-dim', type=int, required=True,
                        help="Whether to use 1D, 2D or 3D histograms.")
    parser.add_argument('--normalize', type=str, required=True,
                        help="Which normalization technique to apply. 'None' for no normalization.")

    return parser.parse_args()

def main():
    # Get command-line arguments
    args = parse_args()

    # Data directories
    queries_dir = args.queries_dir
    bbdd_dir = args.bbdd_dir

    # Fixed parameters
    color_space = args.color_space
    similarity_measure = args.similarity_measure
    dimension=args.histogram_dim
    if args.normalize == "None":
        normalize = None
    else:
        normalize = args.normalize

    # Load grounstruth
    print("Loading groundtruth...", end=" ")
    gt_dir = os.path.join(queries_dir, 'gt_corresps.pkl')
    with open(gt_dir, 'rb') as reader:
        ground_truth = pickle.load(reader)
    print("DONE")

    # Load images
    print("Loading images...", end=" ")
    test_images_raw = load_images_from_directory(queries_dir)
    bbdd_images_raw = load_images_from_directory(bbdd_dir)
    print("DONE")

    # Change color space
    print("Changing color space of the images...", end=" ")
    test_images = []
    for image in test_images_raw:
        image = change_color_space(image, color_space) 
        test_images.extend(np.array([image]))
    bbdd_images = []
    for image in bbdd_images_raw:
        image = change_color_space(image, color_space) 
        bbdd_images.extend(np.array([image]))
    print("DONE")

    print("Calculating metrics for the different combinations...")
    # Parameters to try
    pyramid_levels_list = [[1], [2], [3], [4], [5], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
    bins_list = [(128, False), (64, False), (32, False), (16, False), (128, True)]

    # Saving the results here
    results_table_1 = []
    results_table_5 = []

    for pyramid_level in pyramid_levels_list:
        row_results_1 = []
        row_results_5 = []
        for bins in bins_list:
            adaptive_bins = bins[1]
            bins = bins[0]
            print(f"PYRAMID_LEVEL={pyramid_level}, BINS={bins}, ADAPTIVE={adaptive_bins}", end=" ")

            # Get histograms
            test_images_hists = []
            for image in test_images:
                image_hists = get_pyramid_histograms(image, dimension, bins, pyramid_level, adaptive_bins)
                test_images_hists.extend(np.array([image_hists]))
            bbdd_image_hists = []
            for image in bbdd_images:
                image_hists = get_pyramid_histograms(image, dimension, bins, pyramid_level, adaptive_bins)
                bbdd_image_hists.extend(np.array([image_hists]))

            # Calculate distance matrix
            distance_matrix = create_distance_matrix(test_images_hists, bbdd_image_hists, similarity_measure, normalize)

            # Get ordered index matrix
            results = np.array(generate_results(distance_matrix, similarity_measure)).tolist()

            # Compute the MAP@K metric
            mapk_val_1 = mapk(ground_truth, results, 1)
            mapk_val_5 = mapk(ground_truth, results, 5)
                
            print(f"-- MAPK@{1}: {mapk_val_1} -- MAPK@{5}: {mapk_val_5}")
            
            # Save result
            row_results_1.append(mapk_val_1)
            row_results_5.append(mapk_val_5)
        
        # Save results from inner loop
        results_table_1.append(row_results_1)
        results_table_5.append(row_results_5)

    # Saving results to a dataframe
    print("Saving results...", end=" ")
    bins_list_str = [str(x) for x in bins_list]
    pyramid_levels_list_str = [str(x) for x in pyramid_levels_list]

    results_df_1 = pd.DataFrame(results_table_1, columns=bins_list_str, index=pyramid_levels_list_str)
    results_df_1.index.name = 'pyramid_level'
    results_df_1.columns.name = 'bins'

    results_df_5 = pd.DataFrame(results_table_5, columns=bins_list_str, index=pyramid_levels_list_str)
    results_df_5.index.name = 'pyramid_level'
    results_df_5.columns.name = 'bins'

    # Saving results to a csv
    csv_file_1 = f'results_tests_pyramid_histograms_K{1}.csv'
    results_df_1.to_csv(csv_file_1)
    csv_file_5 = f'results_tests_pyramid_histograms_K{5}.csv'
    results_df_5.to_csv(csv_file_5)
    print("DONE")
    print("Test completed.")

if __name__ == "__main__":
    main()