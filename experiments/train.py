from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from .datasets import make_tf_dataset


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float

    @property
    def generalization_gap(self) -> float:
        return self.train_accuracy - self.test_accuracy


@dataclass
class AccuracyPlateauScheduler:
    best_accuracy: float = -1.0
    wait: int = 0


@dataclass
class EarlyStopState:
    best_accuracy: float = -1.0
    best_epoch: int = 0
    wait: int = 0


def compile_model(model: tf.keras.Model, learning_rate: float, momentum: float = 0.9) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def update_learning_rate_on_plateau(
    model: tf.keras.Model,
    train_accuracy: float,
    scheduler: AccuracyPlateauScheduler,
    patience: int = 10,
    factor: float = 0.9,
) -> float:
    if train_accuracy > scheduler.best_accuracy:
        scheduler.best_accuracy = train_accuracy
        scheduler.wait = 0
    else:
        scheduler.wait += 1
    if scheduler.wait >= patience:
        current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        new_lr = current_lr * factor
        try:
            model.optimizer.learning_rate.assign(new_lr)
        except AttributeError:
            tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        scheduler.wait = 0
        return new_lr
    return float(tf.keras.backend.get_value(model.optimizer.learning_rate))


def should_stop_on_accuracy_plateau(
    train_accuracy: float,
    epoch: int,
    state: EarlyStopState,
    patience: int = 20,
    min_delta: float = 0.0,
) -> bool:
    if train_accuracy > state.best_accuracy + min_delta:
        state.best_accuracy = train_accuracy
        state.best_epoch = epoch
        state.wait = 0
        return False
    state.wait += 1
    return state.wait >= patience


def evaluate_model(
    model: tf.keras.Model,
    x,
    y,
    batch_size: int,
) -> tuple[float, float]:
    loss, accuracy = model.evaluate(x, y, batch_size=batch_size, verbose=0)
    return float(loss), float(accuracy)


def train_one_epoch(
    model: tf.keras.Model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size: int,
    epoch: int,
    seed: int,
) -> EpochMetrics:
    train_ds = make_tf_dataset(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed + epoch)
    model.fit(train_ds, epochs=1, verbose=0)
    train_loss, train_acc = evaluate_model(model, x_train, y_train, batch_size)
    test_loss, test_acc = evaluate_model(model, x_test, y_test, batch_size)
    return EpochMetrics(
        epoch=epoch,
        train_loss=train_loss,
        train_accuracy=train_acc,
        test_loss=test_loss,
        test_accuracy=test_acc,
    )
