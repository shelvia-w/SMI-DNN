from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from estimators.knn_estimators import calc_ksg_mi_cc, calc_ksg_mi_cd
from estimators.neural_estimators import calc_neural_mi
from estimators.smi_estimator import compute_smi


@dataclass(frozen=True)
class EstimateResult:
    estimate: float
    stderr: float | None
    details: dict[str, Any]


def _labels_as_column(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels).reshape(-1, 1)


def estimate_mi(
    x: np.ndarray,
    y: np.ndarray,
    estimator: str,
    seed: int,
    k: int = 3,
    n_projs: int = 1000,
    n_jobs: int = 1,
    neural_epochs: int = 50,
    batch_size: int = 512,
) -> EstimateResult:
    x = np.asarray(x, dtype="float32")
    y = _labels_as_column(y)

    if estimator == "ksg_cd":
        value = calc_ksg_mi_cd(x, y, k=k, random_state=seed)
        return EstimateResult(float(value), None, {"k": k})
    if estimator == "ksg_cc":
        value = calc_ksg_mi_cc(x, y.astype("float32"), k=k, random_state=seed)
        return EstimateResult(float(value), None, {"k": k})
    if estimator == "neural":
        ds = tf.data.Dataset.from_tensor_slices((x, y.astype("float32"))).batch(batch_size)
        value = calc_neural_mi(ds, neural_epochs, print_mi=False)
        return EstimateResult(float(value), None, {"neural_epochs": neural_epochs})
    if estimator == "smi_ksg_cd":
        result = compute_smi(
            x,
            y,
            method="ksg_cd",
            n_projs=n_projs,
            n_jobs=n_jobs,
            random_state=seed,
            estimator_kwargs={"k": k},
            return_details=True,
        )
        return EstimateResult(result["smi"], result["stderr"], result)
    if estimator == "smi_ksg_cc":
        result = compute_smi(
            x,
            y.astype("float32"),
            proj_x=True,
            proj_y=True,
            method="ksg_cc",
            n_projs=n_projs,
            n_jobs=n_jobs,
            random_state=seed,
            estimator_kwargs={"k": k},
            return_details=True,
        )
        return EstimateResult(result["smi"], result["stderr"], result)
    if estimator == "smi_neural":
        result = compute_smi(
            x,
            y.astype("float32"),
            proj_x=True,
            proj_y=True,
            method="neural",
            n_projs=n_projs,
            random_state=seed,
            n_epochs=neural_epochs,
            batch_size=batch_size,
            estimator_kwargs={"print_mi": False},
            return_details=True,
        )
        return EstimateResult(result["smi"], result["stderr"], result)
    raise ValueError("Unsupported estimator")
