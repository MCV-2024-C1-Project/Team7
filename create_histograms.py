from utils.histograms import process_images
import os

#config (potser podriem fer config file?)
data_dir = "./data"
output_dir = "./data/histograms"
datasets = ['BBDD', 'qsd1_w1', 'qst1_w1']
color_spaces = ['RGB', 'LAB', 'HSV', 'YCrCb', 'GRAY']
plot = False

# Generate histograms for all methods for all datasets
for color_space in color_spaces:
    for dataset in datasets:
        image_directory = os.path.join(data_dir, dataset)
        process_images(image_directory, output_dir, color_space, dataset, plot)