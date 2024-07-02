import numpy as np
from estimators.knn_estimators import calc_ksg_mi_cd
from estimators.neural_estimators import calc_neural_mi

def sample_from_sphere(d):
    """
    Generates a random sample from the surface of a unit sphere in d-dimensional space.
    
    Parameters
    ----------
    d : int
        The dimensionality of the space.
    
    Returns
    -------
    np.ndarray
        A d-dimensional unit vector sampled uniformly from the surface of the unit sphere.
        
    """
    
    vec = np.random.randn(d, 1)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def compute_smi(x, y, proj_x=True, proj_y=False, n_projs=1000, method='ksg_cd'):
    """
    Computes the Sliced Mutual Information (SMI) between x and y.
    
    Parameters
    ----------
    x : np.ndarray
        An array of shape (n_samples, dx_features).
    y : np.ndarray
        An array of shape (n_samples, dy_features).
    proj_x : bool, optional
        Whether to project x [Default is True].
    proj_y : bool, optional
        Whether to project y [Default is False].
    n_projs : int, optional
        The number of random projections to use for estimating the sliced mutual information [Default is 1000].
    method : str, optional
        The method to use for mutual information estimation. Available options are 'ksg_cd', 'ksg_cc', 'neural' [Default is 'ksg_cd'].
    
    Returns
    -------
    SMI : float
        The estimated SMI between x and y.
        
    """

    mi_list = []
    for i in range(n_projs):
        theta = sample_from_sphere(x.shape[1])
        phi = sample_from_sphere(y.shape[1])
        thetaX = np.dot(x, theta) if proj_x else x
        phiY = np.dot(y, phi) if proj_y else y
        if method == 'ksg_cd':
            mi_list.append(calc_ksg_mi_cd(thetaX,phiY))
        elif method == 'ksg_cc':
            mi_list.append(calc_ksg_mi_cc(thetaX,phiY))
        elif method == 'neural':
            thetaX_tensor = tf.convert_to_tensor(thetaX)
            phiY_tensor = tf.convert_to_tensor(phiY)
            dataset = tf.data.Dataset.from_tensor_slices((thetaX_tensor, phiY_tensor)).batch(512)
            mi_list.append(calc_neural_mi(dataset, n_epochs, critic='separable', train_obj='js_fgan', eval_type='smile', print_mi=False))
    smi = np.mean(mi_list)
    return smi