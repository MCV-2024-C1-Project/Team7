import os
import argparse
from src.utils.distance_matrix import create_distance_matrix, generate_submission
from src.utils.histograms import load_histograms


def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process input for image retrieval")

    parser.add_argument('--queries-hist-dir', type=str,
                        help="Name of the directory containing query histograms",
                        default='qsd1_w1')
    
    parser.add_argument('--color-space', type=str,
                        choices=['GRAY', 'HSV', 'LAB', 'RGB', 'YCrCb'],
                        help="The color space to be used. Choose from: GRAY, HSV, LAB, RGB, YCrCb",
                        default='YCrCb')

    parser.add_argument('--similarity-measure', type=str,
                        choices=['Correlation', 'Chi-Square', 'Intersection', 'Bhattacharyya', 'Hellinger'],
                        help="The similarity measure to be used",
                        default='Correlation')

    parser.add_argument('--k-val', type=int,
                        help="The number of top results to retrieve (k-value)",
                        default=10)

    parser.add_argument('--results-file', type=str,
                        help="File to save the retrieval results",
                        default='result.pkl')

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