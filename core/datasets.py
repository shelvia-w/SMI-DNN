from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    input_shape: tuple[int, ...]
    num_classes: int


def _normalize_images(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32") / 255.0
    if x.ndim == 3:
        x = x[..., None]
    return x


def load_dataset(name: str, test_fraction: float = 1.0) -> DatasetBundle:
    name = name.lower()
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        num_classes = 10
    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        num_classes = 10
    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        num_classes = 10
    elif name == "cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset")

    x_train = _normalize_images(x_train)
    x_test = _normalize_images(x_test)
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    if not 0.0 < test_fraction <= 1.0:
        raise ValueError("test_fraction must be in (0, 1]")
    if test_fraction < 1.0:
        n_test = max(1, int(len(x_test) * test_fraction))
        x_test = x_test[:n_test]
        y_test = y_test[:n_test]

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_shape=tuple(x_train.shape[1:]),
        num_classes=num_classes,
    )


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
