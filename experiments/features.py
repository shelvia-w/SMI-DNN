from __future__ import annotations

import numpy as np
import tensorflow as tf


def validate_layers(model: tf.keras.Model, layer_names: list[str]) -> list[str]:
    available = {layer.name for layer in model.layers}
    missing = [name for name in layer_names if name not in available]
    if missing:
        raise ValueError(f"Unknown layer names: {missing}. Available layers: {sorted(available)}")
    return layer_names


def make_feature_model(model: tf.keras.Model, layer_names: list[str]) -> tf.keras.Model:
    validate_layers(model, layer_names)
    outputs = [model.get_layer(name).output for name in layer_names]
    return tf.keras.Model(model.input, outputs)


def flatten_activations(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values.reshape(values.shape[0], -1)


def extract_activations(
    model: tf.keras.Model,
    x: np.ndarray,
    layer_names: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    feature_model = make_feature_model(model, layer_names)
    outputs = feature_model.predict(x, batch_size=batch_size, verbose=0)
    if len(layer_names) == 1:
        outputs = [outputs]
    return {
        name: flatten_activations(values)
        for name, values in zip(layer_names, outputs)
    }


def select_feature_layers(
    model: tf.keras.Model,
    requested_layers: list[str],
    include_all_layers: bool,
) -> list[str]:
    from .models import default_analysis_layers, penultimate_layer_name

    if requested_layers:
        return validate_layers(model, requested_layers)
    if include_all_layers:
        return default_analysis_layers(model)
    return [penultimate_layer_name(model)]
