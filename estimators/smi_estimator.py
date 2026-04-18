from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
from estimators.knn_estimators import calc_ksg_mi_cc, calc_ksg_mi_cd


def _as_2d_array(values, name, dtype=float):
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if dtype is not None and not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def sample_from_sphere(d, rng=None):
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
    
    if not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive integer")
    rng = np.random.default_rng(rng)
    vec = rng.normal(size=(d, 1))
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def _projection_mi(args):
    (
        x,
        y,
        theta,
        phi,
        proj_x,
        proj_y,
        method,
        n_epochs,
        batch_size,
        estimator_kwargs,
    ) = args

    theta_x = np.dot(x, theta) if proj_x else x
    phi_y = np.dot(y, phi) if proj_y else y

    if method == 'ksg_cd':
        return calc_ksg_mi_cd(theta_x, phi_y, **estimator_kwargs)
    if method == 'ksg_cc':
        return calc_ksg_mi_cc(theta_x, phi_y, **estimator_kwargs)
    if method == 'neural':
        import tensorflow as tf
        from estimators.neural_estimators import calc_neural_mi

        theta_x_tensor = tf.convert_to_tensor(theta_x)
        phi_y_tensor = tf.convert_to_tensor(phi_y)
        dataset = tf.data.Dataset.from_tensor_slices((theta_x_tensor, phi_y_tensor)).batch(batch_size)
        return calc_neural_mi(
            dataset,
            n_epochs,
            **estimator_kwargs,
        )
    raise ValueError("method must be one of {'ksg_cd', 'ksg_cc', 'neural'}")


def compute_smi(
    x,
    y,
    proj_x=True,
    proj_y=False,
    n_projs=1000,
    method='ksg_cd',
    n_epochs=100,
    random_state=None,
    estimator_kwargs=None,
    return_details=False,
    n_jobs=1,
    batch_size=512,
    return_projections=False,
):
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
    n_epochs : int, optional
        Number of critic-training epochs used when method='neural' [Default is 100].
    random_state : int or np.random.Generator, optional
        Seed or generator for reproducible random projections and estimator jitter.
    estimator_kwargs : dict, optional
        Keyword arguments forwarded to the selected MI estimator.
    return_details : bool, optional
        If True, return a dictionary with per-projection estimates and uncertainty.
    n_jobs : int, optional
        Number of worker threads used to evaluate projection estimates. Use -1 for
        all available CPU cores [Default is 1].
    batch_size : int, optional
        Batch size for the neural estimator [Default is 512].
    return_projections : bool, optional
        If True and return_details=True, include sampled projection vectors.
    
    Returns
    -------
    SMI : float
        The estimated SMI between x and y.
        
    """

    if method not in {'ksg_cd', 'ksg_cc', 'neural'}:
        raise ValueError("method must be one of {'ksg_cd', 'ksg_cc', 'neural'}")
    if not isinstance(n_projs, int) or n_projs < 1:
        raise ValueError("n_projs must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be a positive integer or -1")

    y_dtype = None if method == 'ksg_cd' and not proj_y else float
    x = _as_2d_array(x, "x", dtype=float)
    y = _as_2d_array(y, "y", dtype=y_dtype)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples")

    if method == 'ksg_cd' and proj_y:
        raise ValueError("method='ksg_cd' expects discrete y; use proj_y=False or method='ksg_cc'")
    if method == 'neural' and n_projs > 1:
        warnings.warn(
            "method='neural' trains a fresh critic for each projection. This is "
            "usually very expensive; consider a small n_projs or a dedicated neural SMI routine.",
            RuntimeWarning,
        )

    rng = np.random.default_rng(random_state)
    estimator_kwargs = dict(estimator_kwargs or {})
    if estimator_kwargs.get('return_details'):
        raise ValueError("compute_smi requires scalar per-projection estimates; use return_details on compute_smi instead")
    if method == 'neural':
        estimator_kwargs.setdefault('print_mi', False)
    if method in {'ksg_cd', 'ksg_cc'} and 'random_state' not in estimator_kwargs:
        estimator_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n_projs)
    else:
        estimator_seeds = [estimator_kwargs.get('random_state')] * n_projs

    thetas = [
        sample_from_sphere(x.shape[1], rng) if proj_x else None
        for _ in range(n_projs)
    ]
    phis = [
        sample_from_sphere(y.shape[1], rng) if proj_y else None
        for _ in range(n_projs)
    ]

    tasks = []
    for i in range(n_projs):
        kwargs = dict(estimator_kwargs)
        if method in {'ksg_cd', 'ksg_cc'} and 'random_state' not in kwargs:
            kwargs['random_state'] = int(estimator_seeds[i])
        tasks.append((
            x,
            y,
            thetas[i],
            phis[i],
            proj_x,
            proj_y,
            method,
            n_epochs,
            batch_size,
            kwargs,
        ))

    if n_jobs == -1:
        import os

        n_workers = os.cpu_count() or 1
    else:
        n_workers = n_jobs
    if method == 'neural' and n_workers != 1:
        warnings.warn(
            "Parallel neural SMI can create multiple TensorFlow models at once; "
            "forcing n_jobs=1 for stability.",
            RuntimeWarning,
        )
        n_workers = 1

    if n_workers == 1 or n_projs == 1:
        mi_values = [_projection_mi(task) for task in tasks]
    else:
        n_workers = min(n_workers, n_projs)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            mi_values = list(executor.map(_projection_mi, tasks))

    mi_values = np.asarray(mi_values, dtype=float)
    smi = float(np.mean(mi_values))

    if not return_details:
        return smi

    result = {
        "smi": smi,
        "projection_mi": mi_values,
        "stderr": float(np.std(mi_values, ddof=1) / np.sqrt(n_projs)) if n_projs > 1 else np.nan,
        "std": float(np.std(mi_values, ddof=1)) if n_projs > 1 else np.nan,
        "n_projs": n_projs,
        "method": method,
        "n_jobs": n_workers,
    }
    if return_projections:
        result["theta"] = thetas
        result["phi"] = phis
    return result
