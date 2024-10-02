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
    list
        A list with all the histograms in the directory
    """
    hist_list = []

    # Sort files to ensure they are in numeric order
    files = sorted(os.listdir(hist_path), key=lambda x: int(Path(x).stem))

    for file in files:
        file_path = os.path.join(hist_path, file)
        with open(file_path, 'rb') as reader:
            histogram = pickle.load(reader)
            hist_list.append(histogram)
    
    return hist_list


def create_distance_matrix(query_list, bd_list, method):
    """
    Loads a query and computes the similarity with all the histograms of the existing images

    Parameters
    ----------
    query_list : list
        List of all the query histograms
    bd_list : list
        List that contains the histograms of the BD
    method : int
        Specifies which similarity method to use
        1. Correlation
        2. Chi-Square
        3. Intersection
        4. Bhattacharyya
        5. Hellinger

    Returns
    -------
    ndarray
        Matrix with all the similarity values
    """
    # A row for each query and a column for each element in the DB to search
    similarity_matrix = np.zeros((len(query_list), len(bd_list)))

    for ii, query in enumerate(query_list):
        for jj, hist in enumerate(bd_list):
            similarity_matrix[ii][jj] = cv2.compareHist(query, hist, method)

    return similarity_matrix


def generate_submission(similarity_matrix, k_val, output_path='result.pkl'):
    """
    Generates a submission pkl file with the top K predictions
    (sorted from most probable to least) for each query.
    The stucture is a list of lists, where each list corresponds to a query
    and its elements are the indexes of the top K predictions

    Parameters
    ----------
    similarity_matrix : ndarray
        Matrix with all the similarity values
    k_val : int
        Number of top predictions to save
    output_path : str
        Relative path to save the submission file
    """

    submission = []
    for row in similarity_matrix:
        submission.append(np.argsort(row)[::-1][:k_val])

    writer = open(output_path, 'wb')
    pickle.dump(submission, writer)
    writer.close()