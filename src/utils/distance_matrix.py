import numpy as np
import pickle
from src.utils.histograms import compare_histograms

def create_distance_matrix(query_hists, bbdd_hists, method, normalize='minmax'):
    """
    Computes the similarity between a list of query histograms and a list 
    of database histograms.

    Parameters
    ----------
    query_hists : list
        List of all the query histograms.
    bbdd_hists : list
        List that contains the histograms of the DB.
    method : str
        Specifies which similarity method to use.
        Methods available according to OpenCV:
        'correlation': cv2.HISTCMP_CORREL,
        'chi-square': cv2.HISTCMP_CHISQR,
        'intersection': cv2.HISTCMP_INTERSECT,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA,
        'hellinger': cv2.HISTCMP_HELLINGER,
        'kl-divergence': cv2.HISTCMP_KL_DIV
    normalize : str, optional
        Specifies the normalization method to apply to histograms before comparison.
        Methods available:
        'minmax': cv2.NORM_MINMAX,
        'l1': cv2.NORM_L1,
        'l2': cv2.NORM_L2,
        'inf': cv2.NORM_INF

    Returns
    -------
    similarity_matrix: ndarray
        Matrix with all the similarity values.
    """
    # A row for each query and a column for each element in the DB to search
    similarity_matrix = np.zeros((len(query_hists), len(bbdd_hists)))

    for ii, query in enumerate(query_hists):
        for jj, bd in enumerate(bbdd_hists):
            similarity_matrix[ii][jj] = compare_histograms(query, bd, method, normalize)

    return similarity_matrix

def generate_results(similarity_matrix, distance_measure):
    """
    Generates a matrix that contains the indexes of each image sorted by score
    for each row.

    Parameters
    ----------
    similarity_matrix: ndarray
        Matrix with all the similarity values
    
    distance_measre: str
        The distance measure used to generate the similarity matrix
        Methods available according to OpenCV:
        'correlation': cv2.HISTCMP_CORREL,
        'chi-square': cv2.HISTCMP_CHISQR,
        'intersection': cv2.HISTCMP_INTERSECT,
        'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA,
        'hellinger': cv2.HISTCMP_HELLINGER,
        'kl-divergence': cv2.HISTCMP_KL_DIV
    
    Returns
    -------
    result: list
        A list containing sorted lists of indexes by score of the similarity matrix.
    """
    result = []
    if distance_measure in ["correlation", "intersection"]:
        for row in similarity_matrix:
            result.append(np.argsort(row)[::-1])
    else:
        for row in similarity_matrix:
            result.append(np.argsort(row))
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