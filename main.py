import os
import argparse
from src.utils.distance_matrix import create_distance_matrix, generate_submission
from src.utils.histograms import load_histograms


def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process input for image retrieval")

    parser.add_argument('--queries-hist-dir', type=str, required=True,
                        help="Name of the directory containing query histograms")
    
    parser.add_argument('--color-space', type=str, required=True,
                        choices=['GRAY', 'HSV', 'LAB', 'RGB', 'YCrCb'],
                        help="The color space to be used. Choose from: GRAY, HSV, LAB, RGB, YCrCb")

    parser.add_argument('--similarity-measure', type=str, required=True,
                        choices=['Correlation', 'Chi-Square', 'Intersection', 'Bhattacharyya', 'Hellinger'],
                        help="The similarity measure to be used")

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

    # Mapping similarity measure to method index
    method_idx = ['Correlation', 'Chi-Square', 'Intersection', 'Bhattacharyya', 'Hellinger'].index(args.similarity_measure)

    # Set directories for the histograms
    bbdd_hist_dir = os.path.join('data', 'histograms', 'bbdd', args.color_space)
    queries_hist_dir = os.path.join('data', 'histograms', args.queries_hist_dir, args.color_space)

    # Load histograms
    bbdd_list = load_histograms(bbdd_hist_dir)
    query_list = load_histograms(queries_hist_dir)

    # Compute similarity matrix
    similarity_matrix = create_distance_matrix(query_list, bbdd_list, method_idx)

    # Generate submission results
    generate_submission(similarity_matrix, args.k_val, args.results_file)

if __name__ == "__main__":
    main()