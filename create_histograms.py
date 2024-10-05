from src.utils.histograms import process_images
from src.utils.histograms import save_histograms
import numpy as np
from pathlib import Path
import pickle
import os

data_dir = "./data"
histograms_dir = "./data/histograms"

datasets = ['BBDD', 'qsd1_w1', 'qst1_w1']
color_spaces = ['RGB', 'LAB', 'HSV', 'YCrCb', 'GRAY']
plot = False

def create_basic_histograms(datasets, color_spaces, data_dir, output_dir, plot):
    # Generate histograms for all methods for all datasets
    # using the basic color spaces

    for color_space in color_spaces:
        for dataset in datasets:
            image_directory = os.path.join(data_dir, dataset)
            process_images(image_directory, output_dir, color_space, dataset, plot)

def create_gray_combined_histograms(datasets, color_spaces, input_dir):
    # Generate combinations of histograms mixing gray
    # histograms with the basic color spaces

    # For each dataset
    for dataset in datasets:
        # Retrieve gray histogram files
        hist_path = os.path.join(input_dir, dataset)
        gray_path = os.path.join(hist_path,"GRAY")
        files = sorted(os.listdir(gray_path), key=lambda x: int(Path(x).stem))
        
        # For each file gray histogram file
        for file in files:
            # Retrieve grey histogram
            file_path = os.path.join(gray_path, file)
            with open(file_path, 'rb') as reader:
                gray_histogram = pickle.load(reader)
            reader.close()

            # For each color representation
            for repr in color_spaces:
                # Retrieve color histogram and combine with gray
                comb_path = os.path.join(hist_path,repr,file)
                with open(comb_path, 'rb') as reader:
                    comb_histogram = pickle.load(reader)
                    # Combination of histograms
                    comb_histogram = np.append(comb_histogram, [gray_histogram], axis=0)
                    saving_direct = gray_path+repr
                    os.makedirs(saving_direct, exist_ok=True)
                    # Save them to a new specific file for the combination
                    save_histograms(comb_histogram, file, saving_direct)
                reader.close()

create_basic_histograms(datasets, color_spaces, data_dir, histograms_dir, plot)
create_gray_combined_histograms(datasets, color_spaces[:-1], histograms_dir)