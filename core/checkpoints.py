from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from .utils import ensure_dir


def checkpoint_path(run_dir: str | Path, epoch: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"epoch_{epoch:03d}.keras"


def save_checkpoint(model: tf.keras.Model, run_dir: str | Path, epoch: int) -> Path:
    path = checkpoint_path(run_dir, epoch)
    ensure_dir(path.parent)
    model.save(path, include_optimizer=True)
    return path


def load_checkpoint(path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path)
