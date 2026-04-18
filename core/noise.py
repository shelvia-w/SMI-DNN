from __future__ import annotations

import numpy as np


def apply_symmetric_label_noise(
    labels: np.ndarray,
    num_classes: int,
    noise_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError("noise_ratio must be in [0, 1]")
    labels = np.asarray(labels, dtype="int64")
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    mask = rng.random(len(labels)) < noise_ratio
    for idx in np.flatnonzero(mask):
        current = labels[idx]
        choices = np.delete(np.arange(num_classes), current)
        noisy[idx] = rng.choice(choices)
    return noisy, mask
