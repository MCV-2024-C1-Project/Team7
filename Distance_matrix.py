import cv2
import numpy as np
import pickle
import os
from pathlib import Path

# -> is a comentary
## -> is a doubt

def load_histograms(hist_path):
    """
    Loads all the saved histograms at a directory
    
    Parameters
    ----------
    hist_path : str
        Relative path to the histogram directory

    Returns
    -------
    dict
        A dictionary with all the histograms in the directory
    """
    ## Do we want to preserve the filenames? If not with a list and not a dictionary would work

    hist_dict={}

    for file in os.listdir(hist_path):
        file_path = os.path.join(hist_path,file)
        reader = open(file_path)
        histogram = pickle.load(reader)
        reader.close()
        hist_dict[Path(file_path).stem] = histogram
    
    return hist_dict



def create_distance_matrix(query_dict, bd_dict, method):
    """
    Loads a query and computs the similarity with all the histograms of the existing images

    Parameters
    ----------
    query_dict : dict
        Dictionary of all the querys
    bd_path : dict
        Dictionary that contains the histograms of the BD
    method : int
        Specifies which similarity method to use
        1. Correlation
        2. Chi-Square
        3. Intersection
        4. Bhattacharyya
        5. Hellinger

    Returns
    -------
    str
        The name of the used query
    ndarray
        Matrix with all the similarity values
    """
    # A row for each query and a column for each element in the DB to search
    similarity_matrix = np.zeros((len(query_dict),len(bd_dict)))


    ## If we need to search each query for all elements i think there is no easy way other than doing all
    ## For comparing all the elements in a DB with themselves we can omit when ii==jj and when ii>jj -> the value will be the one in [jj][ii]

    for ii, query in enumerate(query_dict.values()):
        for jj, hist in enumerate(bd_dict.values()):
            similarity_matrix[ii][jj] = cv2.compareHist(query, hist, method)

    return similarity_matrix


