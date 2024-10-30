import os
import pickle
import pandas as pd
import argparse

from src.utils.images import load_images_from_directory, transform_images_color_space, rescale_images
from src.utils.distance_matrix import create_distance_matrix, generate_results
from src.utils.ml_metrics import mapk
from src.utils.LBP import compute_images_block_lbp, compute_images_block_multi_lbp

def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Input for the LBP study.")

    # Files
    parser.add_argument('--queries-dir', type=str, required=True,
                        help="Name of the directory containing query images.")
    parser.add_argument('--bbdd-dir', type=str, required=True,
                        help="Name of the directory containing database images.")

    return parser.parse_args()

def lbp_study(query_folder, bbdd_folder,
              colors = ["gray", "L", "V"],
              rescale_list = [True],
              radius_list = [1, 2, 3],
              points_list = [8, 10, 12],
              methods = ["default", "ror", "uniform"],
              num_blocks_list = [1, 4, 16, 64],
              distance_measures = ['intersection', 'correlation', 'bhattacharyya']):
    
    # Store results in a list that will later be converted to a dataframe
    full_results = []
    
    # Load query images
    query_images = load_images_from_directory(query_folder)

    # Load bbdd images
    bbdd_images = load_images_from_directory(bbdd_folder)
    
    # Load groundtruth
    gt_dir = os.path.join(query_folder, 'gt_corresps.pkl')
    with open(gt_dir, 'rb') as reader:
        ground_truth = pickle.load(reader)
    
    for color in colors:
        
        query_images_color = transform_images_color_space(query_images, color_space=color)
        bbdd_images_color = transform_images_color_space(bbdd_images, color_space=color)
        
        for rescale in rescale_list:
            
            if rescale:
                query_images_color = rescale_images(query_images_color, (256, 256))
                bbdd_images_color = rescale_images(bbdd_images_color, (256, 256))
        
            for radius, points in zip(radius_list, points_list):

                for method in methods:

                    for num_blocks in num_blocks_list:
                        
                        query_lbp_vectors = compute_images_block_lbp(query_images_color, radius, points, method, num_blocks)
                        bbdd_lbp_vectors = compute_images_block_lbp(bbdd_images_color, radius, points, method, num_blocks)
                        
                        for distance_measure in distance_measures:
                            
                            # Print current combination
                            print(f"{color}, rescale={rescale}, {radius}-{points}, {method}, {num_blocks}, {distance_measure}", end=" ")

                            # Calculate distance matrix
                            distance_matrix = create_distance_matrix(query_lbp_vectors, 
                                                                     bbdd_lbp_vectors,
                                                                     distance_measure,
                                                                     normalize=None)
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
                            combination_results['Rescale'] = rescale
                            combination_results['Radius'] = radius
                            combination_results['Points'] = points
                            combination_results['Method'] = method
                            combination_results['Blocks'] = num_blocks
                            combination_results['Distance'] = distance_measure
                            combination_results['MAP@1'] = mapk_val_1
                            combination_results['MAP@5'] = mapk_val_5
                            full_results.append(combination_results)
                    
    # Save results to csv via df
    full_results_df = pd.DataFrame(full_results)
    full_results_df.to_csv("lbp_study_results.csv", index=False)
    
    return full_results_df

def multiple_lbp(query_folder,
                 bbdd_folder,
                 num_blocks_list = [16, 64],
                 color = "L",
                 rescale = True,
                 radius = "Multi",
                 points = "Multi",
                 method = "default",
                 distance_measure = "bhattacharyya"):
    
    # Save results in a list, which will then be transformed into a df
    full_results = []
    
    # Load query images
    query_images = load_images_from_directory(query_folder)

    # Load bbdd images
    bbdd_images = load_images_from_directory(bbdd_folder)
    
    # Load groundtruth
    gt_dir = os.path.join(query_folder, 'gt_corresps.pkl')
    with open(gt_dir, 'rb') as reader:
        ground_truth = pickle.load(reader)
        
    query_images_color = transform_images_color_space(query_images, color_space=color)
    bbdd_images_color = transform_images_color_space(bbdd_images, color_space=color)
    
    query_images_color = rescale_images(query_images_color, (256, 256))
    bbdd_images_color = rescale_images(bbdd_images_color, (256, 256))
    
    for num_blocks in num_blocks_list:
        
        query_lbp_vectors = compute_images_block_multi_lbp(query_images_color, method, num_blocks)
        bbdd_lbp_vectors = compute_images_block_multi_lbp(bbdd_images_color, method, num_blocks)

        print(f"{num_blocks}", end=" ")

        # Calculate distance matrix
        distance_matrix = create_distance_matrix(query_lbp_vectors, 
                                                 bbdd_lbp_vectors,
                                                 distance_measure,
                                                 normalize=None)
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
        combination_results['Rescale'] = rescale
        combination_results['Radius'] = radius
        combination_results['Points'] = points
        combination_results['Method'] = method
        combination_results['Blocks'] = num_blocks
        combination_results['Distance'] = distance_measure
        combination_results['MAP@1'] = mapk_val_1
        combination_results['MAP@5'] = mapk_val_5
        full_results.append(combination_results)
    
    # Save results
    full_results_df = pd.DataFrame(full_results)
    full_results_df.to_csv("multiple_lbp_results.csv", index=False)
    
    return full_results_df

def main():

    # Get command-line arguments
    args = parse_args()

    # Execute grid function
    lbp_study(args.queries_dir,
              args.bbdd_dir,
              colors = ["gray", "L", "V"],
              rescale_list = [True],
              radius_list = [1, 2, 3],
              points_list = [8, 10, 12],
              methods = ["default", "ror", "uniform"],
              num_blocks_list = [1, 4, 16, 64],
              distance_measures = ['intersection', 'correlation', 'bhattacharyya'])
    
    # Execute multi-lbp function
    multiple_lbp(args.queries_dir,
                 args.bbdd_dir,
                 num_blocks_list = [16, 64],
                 color = "L",
                 rescale = True,
                 radius = "Multi",
                 points = "Multi",
                 method = "default",
                 distance_measure = "bhattacharyya")

if __name__ == "__main__":
    main()