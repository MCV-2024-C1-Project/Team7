import cv2
import numpy as np
import pickle
import os

def create_distance_matrix(query_list, bd_list, method):
    """
    Computes the similarity between a list of query histograms and a list 
    of database histograms.

    Parameters
    ----------
    query_list : list
        List of all the query 1D histograms
    bd_list : list
        List that contains the 1D histograms of the DB
    method : int
        Specifies which similarity method to use
        1. Correlation
        2. Chi-Square
        3. Intersection
        4. Bhattacharyya
        5. Hellinger

    Returns
    -------
    similarity_matrix: ndarray
        Matrix with all the similarity values.
    """
    # A row for each query and a column for each element in the DB to search
    similarity_matrix = np.zeros((len(query_list), len(bd_list)))

    for ii, query in enumerate(query_list):
        for jj, hist in enumerate(bd_list):
            similarity_matrix[ii][jj] = cv2.compareHist(query, hist, method)

    return similarity_matrix

def generate_results(similarity_matrix):
    """
    Generates a matrix that contains the indexes of each image sorted by score
    for each row.

    Parameters
    ----------
    similarity_matrix: ndarray
        Matrix with all the similarity values
    
    Returns
    -------
    result: list
        A list containing sorted lists of indexes by score of the similarity matrix.
    """
    result = []
    for row in similarity_matrix:
        result.append(np.argsort(row)[::-1])
    return result

def generate_submission(results, k_val, output_path='result.pkl'):
    """
    Generates a submission pkl file with the top K predictions
    (sorted from most probable to least) for each query.
    The stucture is a list of lists, where each list corresponds to a query
    and its elements are the indexes of the top K predictions

    Parameters
    ----------
    results : ndarray
        List of sorted lists of indexes by score of the similarity matrix
    k_val : int
        Number of top predictions to save
    output_path : str
        Relative path to save the submission file

    Returns
    -------
    submission: list of lists
        A list of lists where each sublist contains the the top K predicted
        indexes for a query.
    """
    #generate predictions
    submission = np.array(results)[:,:k_val].tolist()
    writer = open(output_path, 'wb')
    pickle.dump(submission, writer)
    writer.close()

    return submission