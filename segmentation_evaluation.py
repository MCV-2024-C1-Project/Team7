import os
import cv2
import argparse
from src.utils.score_painting_retrieval import score_pixel_mask_list
from src.utils.histograms import change_color_space

def parse_args():
    """
    Parse and return the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process input for image retrieval")

    parser.add_argument('--ground-truth-directory', type=str,
                        help="Name of the directory containing the ground truth masks",
                        default=".\data\qsd2_w1")
    
    parser.add_argument('--crafted-masks-directory', type=str,
                        help="Name of the directory containing the crafted masks",
                        default=".\data\qsd2_w1_masks")

    return parser.parse_args()

args = parse_args()

crafted_dir = args.crafted_masks_directory
gt_dir = args.ground_truth_directory
color_space = "GRAY"

# List to store the images
masks = []

# Read images from the directory
for filename in os.listdir(crafted_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(crafted_dir, filename)
        img = cv2.imread(img_path)
        img = change_color_space(img, color_space)
        masks.append(img)

# List to store the gt masks
gt_masks = []

# Read images from the directory
for filename in os.listdir(gt_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(gt_dir, filename)
        img = cv2.imread(img_path)
        img = change_color_space(img, color_space)
        gt_masks.append(img)

print(score_pixel_mask_list(masks, gt_masks))