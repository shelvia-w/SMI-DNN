# Adapted from:
# https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py
# https://github.com/wgao9/knnie/blob/master/knnie.py
# https://github.com/wgao9/mixed_KSG/blob/master/mixed.py
# https://github.com/manuel-alvarez-chaves/unite_toolbox/blob/main/unite_toolbox/knn_estimators.py

import numpy as np
import numpy.linalg as la
import warnings
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree


def _as_2d_array(values, name, dtype=None):
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return values


def _validate_pair(x, y, k=None, continuous_x=True, continuous_y=True):
    x = _as_2d_array(x, "x", dtype=float if continuous_x else None)
    y = _as_2d_array(y, "y", dtype=float if continuous_y else None)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples")
    if k is not None:
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if k > len(x) - 1:
            raise ValueError("k must be smaller than the number of samples")
    if continuous_x and not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if continuous_y and not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values")
    return x, y

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
    
    x, y = _validate_pair(x, y, continuous_x=False, continuous_y=False)
    
    mi = entropy_d(x) - centropy_d(x, y)
    return mi

def calc_ksg_mi_cc(x, y, k=3, alpha=0, noise=1e-10, random_state=None):
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
    
    x, y = _validate_pair(x, y, k=k)
    rng = np.random.default_rng(random_state)
    
    x = add_noise(x, intens=noise, rng=rng)
    y = add_noise(y, intens=noise, rng=rng)
    
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

def calc_ksg_mi_cd(
    x,
    y,
    k=3,
    warn_on_small_class=True,
    small_class_action="raise",
    clip_negative=False,
    noise=1e-10,
    random_state=None,
    return_details=False,
    warning=None,
):
    """
    Estimates mutual information between continuous x and discrete y by entropy
    decomposition, I(X;Y) = H(X) - H(X|Y).
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, dx_features).
    y : numpy.ndarray
        Array of shape (n_samples, dy_features).
    k : int, optional
        The number of nearest neighbors.
    warn_on_small_class : bool, optional
        Whether to warn when a label class has too few samples for the chosen k.
    small_class_action : {'raise', 'global', 'skip'}, optional
        How to handle classes with count <= k. 'raise' is statistically safest.
        'global' uses H(X) for that class for backward compatibility. 'skip'
        excludes the class from H(X|Y), which changes the target distribution.
    clip_negative : bool, optional
        If True, clip negative finite-sample estimates to zero. The default keeps
        negative estimates visible as diagnostics.
    noise : float, optional
        Standard deviation of centered jitter added before continuous entropy
        estimation. Set to 0 to disable jitter.
    random_state : int or np.random.Generator, optional
        Seed or generator for jitter reproducibility.
    return_details : bool, optional
        If True, return a dictionary with estimate diagnostics.
        
    Returns
    -------
    mi : float
        Mutual information between x and y [in nats].
    
    """
    
    if warning is not None:
        warn_on_small_class = warning

    if small_class_action not in {"raise", "global", "skip"}:
        raise ValueError("small_class_action must be one of {'raise', 'global', 'skip'}")

    x, y = _validate_pair(x, y, k=k, continuous_x=True, continuous_y=False)
    rng = np.random.default_rng(random_state)

    entropy_x = entropy_c(x, k, noise=noise, rng=rng)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.0
    skipped_mass = 0.0
    small_classes = []
    for yval, p_y in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += p_y * entropy_c(x_given_y, k, noise=noise, rng=rng)
        else:
            small_classes.append((yval.copy(), len(x_given_y), float(p_y)))
            if warn_on_small_class:
                warnings.warn(
                    "After conditioning on y={yval}, only {n} samples remain; "
                    "k={k} requires at least k+1 samples.".format(
                        yval=yval, n=len(x_given_y), k=k
                    ),
                    RuntimeWarning,
                )
            if small_class_action == "raise":
                raise ValueError(
                    "At least one discrete class has <= k samples. Lower k, merge "
                    "rare classes, or set small_class_action explicitly."
                )
            if small_class_action == "global":
                entropy_x_given_y += p_y * entropy_x
            elif small_class_action == "skip":
                skipped_mass += p_y
            
    mi = entropy_x - entropy_x_given_y
    if clip_negative:
        mi = max(0.0, mi)

    if return_details:
        return {
            "mi": float(mi),
            "entropy_x": float(entropy_x),
            "entropy_x_given_y": float(entropy_x_given_y),
            "small_classes": small_classes,
            "skipped_probability_mass": float(skipped_mass),
            "k": k,
        }
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

def entropy_c(x, k=3, noise=1e-10, rng=None):
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
    
    x = _as_2d_array(x, "x", dtype=float)
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if k > len(x) - 1:
        raise ValueError("k must be smaller than the number of samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    
    n_samples, d_features = x.shape
    x = add_noise(x, intens=noise, rng=rng)
    
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_samples) - digamma(k) + d_features * np.log(2)
    
    entropy = const + d_features * np.log(nn).mean()
    return entropy

# ------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------

def add_noise(points, intens=1e-10, rng=None):
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
    points = np.asarray(points, dtype=float)
    if intens is None or intens == 0:
        return points
    if intens < 0:
        raise ValueError("intens must be non-negative")
    rng = np.random.default_rng(rng)
    return points + intens * rng.normal(size=points.shape)

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
