from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr

def compare_vectors(vec1, vec2, method="L2"):
    """
    Compares two vectors using the specified method and returns the distance or similarity.
    
    Args:
    - vec1 (list or array): The first vector.
    - vec2 (list or array): The second vector.
    - method (str): The method to use for comparison. Options are:
        - "L1" for Manhattan distance (city block distance)
        - "L2" for Euclidean distance
        - "Cosine" for Cosine similarity
        - "Pearson" for Pearson correlation
    
    Returns:
    - result (float): The computed distance or similarity.
    """
    
    if method == "L1":
        # Manhattan distance (L1)
        result = cityblock(vec1, vec2)
    
    elif method == "L2":
        # Euclidean distance (L2)
        result = euclidean(vec1, vec2)
    
    elif method == "Cosine":
        # Cosine distance
        result = cosine(vec1, vec2)
    
    elif method == "Pearson":
        # Pearson correlation
        result, _ = pearsonr(vec1, vec2)
    
    else:
        raise ValueError("Unsupported method.")
    
    return result