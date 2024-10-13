from src.utils.histograms import change_color_space, get_pyramid_histograms
from src.utils.distance_matrix import create_distance_matrix, generate_results, generate_submission
from src.utils.images import load_images_from_directory

import numpy as np
import argparse

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

    return parser.parse_args()

def main():

    # Get command-line arguments
    args = parse_args()

    # Data directories
    queries_dir = args.queries_dir
    bbdd_dir = args.bbdd_dir

    # Fixed parameters
    color_space = "LAB"
    similarity_measure = "bhattacharyya"
    dimension = 1
    normalize = None
    pyramid_level = [5]
    bins = 64
    adaptive_bins = False

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

    # Get histograms
    print("Extracting histograms...", end=" ")
    test_images_hists = []
    for image in test_images:
        image_hists = get_pyramid_histograms(image, dimension, bins, pyramid_level, adaptive_bins)
        test_images_hists.extend(np.array([image_hists]))
    bbdd_image_hists = []
    for image in bbdd_images:
        image_hists = get_pyramid_histograms(image, dimension, bins, pyramid_level, adaptive_bins)
        bbdd_image_hists.extend(np.array([image_hists]))
    print("DONE")

    # Calculate distance matrix
    print("Calculating distance matrix...", end=" ")
    distance_matrix = create_distance_matrix(test_images_hists, bbdd_image_hists, similarity_measure, normalize)
    print("DONE")

    # Generate results
    print("Generating results...", end=" ")
    results = generate_results(distance_matrix, similarity_measure)
    print("DONE")

    # Generate submission
    print("Generating submission...", end=" ")
    results_file = "result.pkl"
    generate_submission(results, k_val=10, output_path=results_file)
    print("DONE")

if __name__ == "__main__":
    main()