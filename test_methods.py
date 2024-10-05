import os
import argparse
import numpy as np
from src.utils.histograms import load_histograms
from src.utils.distance_matrix import create_distance_matrix, generate_results
from src.utils.ml_metrics import mapk
import pickle

def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process input for image retrieval")

    parser.add_argument('--queries-hist-dir', type=str, required=True,
                        help="Name of the directory containing query histograms")
    
    parser.add_argument('--k-val', type=int, required=True,
                        help="The number of top results to retrieve (k-value)")

    parser.add_argument('--results-file', type=str, required=True,
                        help="File to save the retrieval results")

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Argument validation
    if not (0 < args.k_val <= 288):
        raise ValueError("The k-value must be a positive integer less than or equal to 288")
    
    # Retrieve groundtruth for specified queries directory
    gt_dir = os.path.join('data', args.queries_hist_dir, 'gt_corresps.pkl')
    with open(gt_dir, 'rb') as reader:
        ground_truth = pickle.load(reader)
    
    # For all similarity distances
    for method in ['Correlation', 'Chi-Square', 'Intersection', 'Bhattacharyya', 'Hellinger']:      
        # For all color spaces
        for color_space in ['GRAY', 'HSV', 'LAB', 'RGB', 'YCrCb','GRAYHSV','GRAYLAB','GRAYRGB','GRAYYCrCb']:
            # Mapping similarity measure to method index
            method_idx = ['Correlation', 'Chi-Square', 'Intersection', 'Bhattacharyya', 'Hellinger'].index(method)
            print(f'{method}{" --- "}{color_space:<12}', end=" ")
            
            # Set directories for the histograms
            bbdd_hist_dir = os.path.join('data', 'histograms', 'BBDD', color_space)
            queries_hist_dir = os.path.join('data', 'histograms', args.queries_hist_dir, color_space)

            # Load histograms
            bbdd_list = load_histograms(bbdd_hist_dir)
            query_list = load_histograms(queries_hist_dir)

            # Compute similarity matrix
            similarity_matrix = create_distance_matrix(query_list, bbdd_list, method_idx)

            # Get ordered index matrix
            results = np.array(generate_results(similarity_matrix)).tolist()

            # Compute the MAP@K metric
            mapk_val = mapk(ground_truth, results, args.k_val)

            print("MAPK: ", mapk_val)
        print("\n")


if __name__ == "__main__":
    main()