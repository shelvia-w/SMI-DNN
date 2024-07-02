# Adapted from:
# https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py
# https://github.com/wgao9/knnie/blob/master/knnie.py
# https://github.com/wgao9/mixed_KSG/blob/master/mixed.py
# https://github.com/manuel-alvarez-chaves/unite_toolbox/blob/main/unite_toolbox/knn_estimators.py

import numpy as np
import numpy.linalg as la
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree

# ------------------------------------------------------------------------
# Mutual Information Estimators
# ------------------------------------------------------------------------

def calc_mi_dd(x, y):
    """
    Estimates the mutual information between discrete x and discrete y.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, dx_features).
    y : numpy.ndarray
        Array of shape (n_samples, dy_features).
        
    Returns
    -------
    mi : float
        Mutual information between x and y [in nats].
        
    """
    
    assert len(x) == len(y), "Arrays should have same length"
    
    mi = entropy_d(x) - centropy_d(x, y)
    return mi

def calc_ksg_mi_cc(x, y, k=3, alpha=0):
    """
    Estimates the mutual information between continuous x and continuous y using KSG.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, dx_features).
    y : numpy.ndarray
        Array of shape (n_samples, dy_features).
    k : int, optional
        The number of nearest neighbors.
    alpha : float, optional
        The threshold parameter for the PCA-based local non-uniform correction.
        
    Returns
    -------
    mi : float
        Mutual information between x and y [in nats].
    
    """
    
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    
    x = add_noise(x)
    y = add_noise(y)
    
    points = [x, y]
    points = np.hstack(points)
    # Find nearest neighbors in joint space
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    
    a = avgdigamma(x, dvec)
    b = avgdigamma(y, dvec)
    c = digamma(k)
    d = digamma(len(x))
    if alpha > 0:
        d += lnc_correction(tree, points, k, alpha)
        
    mi = -a - b + c + d
    return mi

def calc_ksg_mi_cd(x, y, k=3):
    """
    Estimates the mutual information between continuous x and discrete y using KSG.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, dx_features).
    y : numpy.ndarray
        Array of shape (n_samples, dy_features).
    k : int, optional
        The number of nearest neighbors.
        
    Returns
    -------
    mi : float
        Mutual information between x and y [in nats].
    
    """
    
    entropy_x = entropy_c(x, k)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.0
    for yval, p_y in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += p_y * entropy_c(x_given_y, k)
        else:
            if warning:
                warnings.warn(
                    "Warning, after conditioning, on y={yval} insufficient data. "
                    "Assuming maximal entropy in this case.".format(yval=yval)
                )
            entropy_x_given_y += p_y * entropy_x
            
    mi = abs(entropy_x - entropy_x_given_y)
    return mi

# ------------------------------------------------------------------------
# Entropy Estimators
# ------------------------------------------------------------------------

def entropy_d(x):
    """
    Estimates the entropy of discrete x.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d_features).
        
    Returns
    -------
    entropy : float
        Entropy of x [in nats].
        
    """
    
    unique, count = np.unique(x, return_counts=True, axis=0)
    # Convert to float as otherwise integer division results in all 0 for proba.
    proba = count.astype(float) / len(x)
    # Avoid 0 division; remove probabilities == 0.0
    # removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    
    entropy = np.sum(proba * np.log(1.0 / proba))
    return entropy

def centropy_d(x, y):
    """
    Estimates the conditional entropy of discrete x given discrete y.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d_features).
        
    Returns
    -------
    entropy : float
        (Conditional) Entropy of x given y [in nats].
        
    """
    xy = np.c_[x, y]
    
    entropy = entropy_d(xy) - entropy_d(y)
    return entropy

def entropy_c(x, k=3):
    """
    Estimates the entropy of continuous x using KSG.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, dx_features).
    k : int, optional
        The number of nearest neighbors.
    
    Returns
    -------
    entropy : float
        Entropy of x [in nats]
    
    """
    
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    
    n_samples, d_features = x.shape
    x = add_noise(x)
    
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_samples) - digamma(k) + d_features * np.log(2)
    
    entropy = const + d_features * np.log(nn).mean()
    return entropy

# ------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------

def add_noise(points, intens=1e-10):
    """
    Adds random noise to all the points.

    Parameters
    ----------
    points  : numpy.ndarray
        Array of shape (n_samples, d_features).
    intens : float, optional
        The intensity of the noise to be added [Default is 1e-10].

    Returns
    -------
    numpy.ndarray
        Array of shape (n_samples, d_features) with added random noise.
    
    """
    return points + intens * np.random.random_sample(points.shape)

def build_tree(points, metric='chebyshev'):
    """
    Constructs a spatial data structure for efficient nearest-neighbor queries.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n_samples, d_features).
    metric : str, optional
        The distance metric to use. Available options are 'euclidean',
        'manhattan', and 'chebyshev' [Default is 'chebyshev'].

    Returns
    -------
    KDTree or BallTree
        A KDTree if the number of features is less than 20, otherwise a BallTree.
                        
    """
    _, d_features = points.shape
    if d_features >= 20:
        return BallTree(points, metric=metric)
    return KDTree(points, metric=metric)

def query_neighbors(tree, points, k):
    """
    Queries the k-th nearest neighbors for a given set of points using a spatial data structure.

    Parameters
    ----------
    tree : KDTree or BallTree
        The spatial data structure (either KDTree or BallTree) used for querying.
    points : numpy.ndarray
        Array of shape (n_samples, d_features).
    k : int
        The number of nearest neighbors to query.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_samples,) containing the distances to the k-th nearest neighbor
        for each input point.
             
    """
    return tree.query(points, k=k + 1)[0][:, k]

def count_neighbors(tree, points, r):
    """
    Counts the number of neighbors within a given radius for each point in the
    input array using a spatial data structure.
    
    Parameters
    ----------
    tree : KDTree or BallTree
        The spatial data structure (either KDTree or BallTree) used for querying.
    points : numpy.ndarray
        Array of shape (n_samples, d_features).
    r : float
        The radius within which to count the neighbors.
    
    Returns
    -------
    numpy.ndarray
        Array of shape (n_samples,) containing the count of neighbors within the radius r 
        for each input point.
    """
    return tree.query_radius(points, r, count_only=True)

def avgdigamma(points, dvec):
    """
    Computes the average digamma function value for the number of neighbors within a
    specified radius for each point.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n_samples, d_features)
    dvec: numpy.ndarray
        Array of shape (n_samples,), representing the radius within which to count
        the neighbors for each point.
    
    Returns
    -------
    float
        The average value of the digamma function of all the points.
    """
    
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

def lnc_correction(tree, points, k, alpha):
    """
    Estimates the PCA-based local non-uniform correction (LNC) term for the KSG estimator.
    
    Parameters
    ----------
    tree : KDTree or BallTree
        The spatial data structure (either KDTree or BallTree) used for querying.
    points : numpy.ndarray
        Array of shape (n_samples, d_features)
    k : int
        The number of nearest neighbors to query.
    alpha : float, optional
        The threshold parameter for the PCA-based local non-uniform correction.

    Returns
    -------
    float
        The estimated LNC term.
    
    """
    
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space
        knn = tree.query(point[None, :], k=k + 1, return_distance=False)[0]
        knn_points = points[knn]
        # Substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e
