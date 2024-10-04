import os
import cv2
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from src.utils.histograms import save_histograms

input_dir = "./data/histograms"
datasets = ['BBDD', 'qsd1_w1', 'qst1_w1']

for dataset in datasets:
    hist_path = os.path.join(input_dir, dataset)
    gray_path = os.path.join(hist_path,"GRAY")
    files = sorted(os.listdir(gray_path), key=lambda x: int(Path(x).stem))

    for file in files:
        file_path = os.path.join(gray_path, file)
        with open(file_path, 'rb') as reader:
            gray_histogram = pickle.load(reader)
        reader.close()

        for repr in ['HSV', 'LAB', 'RGB', 'YCrCb']:
            comb_path = os.path.join(hist_path,repr,file)
        
            with open(comb_path, 'rb') as reader:
                comb_histogram = pickle.load(reader)
                comb_histogram = np.append(comb_histogram, [gray_histogram], axis=0)
                saving_direct = gray_path+repr
                print(saving_direct)
                os.makedirs(saving_direct, exist_ok=True)
                save_histograms(comb_histogram.tolist, file, saving_direct)
            reader.close()