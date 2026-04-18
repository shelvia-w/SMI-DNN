from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input


def build_mlp5(
    input_shape: tuple[int, ...],
    num_classes: int,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
) -> tf.keras.Model:
    del use_batch_norm
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = layers.Flatten(name="flatten")(inputs)
    for idx in range(1, 5):
        x = layers.Dense(1024, activation="relu", name=f"fc{idx}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout{idx}")(x)
    outputs = layers.Dense(num_classes, activation=None, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="mlp5")


def build_cnn5(
    input_shape: tuple[int, ...],
    num_classes: int,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = True,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = inputs
    for idx, stride in enumerate([2, 1, 2, 1], start=1):
        x = layers.Conv2D(512, 3, strides=stride, padding="same", activation=None, name=f"conv{idx}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn{idx}")(x)
        x = layers.ReLU(name=f"relu{idx}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout{idx}")(x)
    x = layers.Conv2D(num_classes, 1, strides=1, padding="same", activation=None, name="conv_logits")(x)
    outputs = layers.GlobalAveragePooling2D(name="gap")(x)
    return tf.keras.Model(inputs, outputs, name="cnn5")


def _imagenet_input_adapter(inputs, preprocess, name_prefix: str):
    x = inputs
    if inputs.shape[-1] == 1:
        x = layers.Lambda(lambda t: tf.image.grayscale_to_rgb(t), name=f"{name_prefix}_gray_to_rgb")(x)
    # Small datasets are kept at 32x32 for practicality; this is the smallest
    # ImageNet-backbone input accepted by Keras for these applications.
    x = layers.Resizing(32, 32, name=f"{name_prefix}_resize_32")(x)
    x = layers.Lambda(lambda t: preprocess(t * 255.0), name=f"{name_prefix}_preprocess")(x)
    return x


def build_vgg16(
    input_shape: tuple[int, ...],
    num_classes: int,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
) -> tf.keras.Model:
    del use_batch_norm
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = _imagenet_input_adapter(inputs, vgg_preprocess_input, "vgg")
    base = VGG16(include_top=False, weights="imagenet", input_shape=(32, 32, 3), pooling="avg")
    base.trainable = True
    x = base(x)
    x = layers.Dense(4096, activation="relu", name="fc1")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(4096, activation="relu", name="fc2")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation=None, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="vgg16")


def build_resnet50(
    input_shape: tuple[int, ...],
    num_classes: int,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
) -> tf.keras.Model:
    del use_batch_norm
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = _imagenet_input_adapter(inputs, resnet_preprocess_input, "resnet")
    base = ResNet50(include_top=False, weights="imagenet", input_shape=(32, 32, 3), pooling="avg")
    base.trainable = True
    x = base(x)
    x = layers.Dense(4096, activation="relu", name="fc1")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(4096, activation="relu", name="fc2")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation=None, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="resnet50")


def build_model(
    name: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = True,
) -> tf.keras.Model:
    if name == "mlp5":
        return build_mlp5(input_shape, num_classes, dropout_rate, use_batch_norm)
    if name in {"cnn5", "cnn5_bn", "cnn6"}:
        return build_cnn5(input_shape, num_classes, dropout_rate, use_batch_norm)
    if name == "vgg16":
        return build_vgg16(input_shape, num_classes, dropout_rate, use_batch_norm)
    if name == "resnet50":
        return build_resnet50(input_shape, num_classes, dropout_rate, use_batch_norm)
    raise ValueError("Unsupported model")


def default_analysis_layers(model: tf.keras.Model) -> list[str]:
    if model.name == "mlp5":
        return ["fc1", "fc2", "fc3", "fc4"]
    if model.name == "cnn5":
        return ["relu1", "relu2", "relu3", "relu4"]
    if model.name in {"vgg16", "resnet50"}:
        return ["fc1", "fc2"]

    candidates = []
    for layer in model.layers:
        if isinstance(layer, (layers.Dense, layers.Conv2D, layers.GlobalAveragePooling2D, layers.ReLU)):
            if layer.name not in {"logits", "conv_logits", "gap"}:
                candidates.append(layer.name)
    return candidates


def penultimate_layer_name(model: tf.keras.Model) -> str:
    layers_to_use = default_analysis_layers(model)
    if not layers_to_use:
        raise ValueError(f"No default hidden layers are defined for model {model.name!r}")
    return layers_to_use[-1]
