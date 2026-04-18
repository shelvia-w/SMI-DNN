from __future__ import annotations

import argparse
import csv
import logging

from .checkpoints import save_checkpoint
from .config import add_common_args, config_from_args, experiment_dir
from .datasets import load_dataset
from .features import extract_activations
from .models import build_model, default_analysis_layers
from .noise import apply_symmetric_label_noise
from .estimators import estimate_mi
from .train import AccuracyPlateauScheduler, compile_model, train_one_epoch, update_learning_rate_on_plateau
from .utils import configure_logging, ensure_dir, sample_indices, set_global_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run clean/noisy-label MI/SMI benchmark.")
    add_common_args(parser, default_setup="memorization")
    parser.add_argument("--noise-ratio", type=float, default=0.4)
    parser.add_argument("--conditions", default="clean,noisy", help="Comma-separated: clean,noisy")
    return parser.parse_args()


def run():
    configure_logging()
    args = parse_args()
    config = config_from_args(args)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    root = experiment_dir(config, "label_noise")
    ensure_dir(root)
    write_json(root / "config.json", config.to_dict() | {"conditions": conditions})

    dataset = load_dataset(config.dataset, test_fraction=config.test_fraction)
    rows = []

    for seed in config.seeds:
        for condition in conditions:
            set_global_seed(seed)
            y_train = dataset.y_train
            corruption_rate = 0.0
            if condition == "noisy":
                y_train, mask = apply_symmetric_label_noise(
                    dataset.y_train,
                    dataset.num_classes,
                    config.noise_ratio,
                    seed,
                )
                corruption_rate = float(mask.mean())
            elif condition != "clean":
                raise ValueError("conditions must contain only clean and/or noisy")

            model = compile_model(
                build_model(config.model, dataset.input_shape, dataset.num_classes),
                config.learning_rate,
                config.momentum,
            )
            lr_scheduler = AccuracyPlateauScheduler()
            layer_names = list(config.layers) or default_analysis_layers(model)
            idx = sample_indices(len(dataset.x_train), config.max_mi_samples, seed)
            run_dir = root / f"seed_{seed}" / condition
            ensure_dir(run_dir)

            for epoch in range(1, config.epochs + 1):
                metrics = train_one_epoch(
                    model,
                    dataset.x_train,
                    y_train,
                    dataset.x_test,
                    dataset.y_test,
                    config.batch_size,
                    epoch,
                    seed,
                )
                current_lr = update_learning_rate_on_plateau(
                    model,
                    metrics.train_accuracy,
                    lr_scheduler,
                    patience=config.lr_patience,
                    factor=config.lr_decay_factor,
                )
                if config.save_checkpoints:
                    save_checkpoint(model, run_dir, epoch)

                activations = extract_activations(model, dataset.x_train[idx], layer_names, config.batch_size)
                for layer_name, reps in activations.items():
                    estimate = estimate_mi(
                        reps,
                        y_train[idx],
                        config.estimator,
                        seed=seed + epoch,
                        k=config.k,
                        n_projs=config.n_projs,
                        n_jobs=config.n_jobs,
                        batch_size=config.batch_size,
                    )
                    rows.append({
                        "experiment": "label_noise",
                        "dataset": config.dataset,
                        "model": config.model,
                        "estimator": config.estimator,
                        "seed": seed,
                        "condition": condition,
                        "noise_ratio": config.noise_ratio if condition == "noisy" else 0.0,
                        "observed_corruption_rate": corruption_rate,
                        "epoch": epoch,
                        "layer": layer_name,
                        "mi": estimate.estimate,
                        "stderr": estimate.stderr,
                        "train_loss": metrics.train_loss,
                        "train_accuracy": metrics.train_accuracy,
                        "test_loss": metrics.test_loss,
                        "test_accuracy": metrics.test_accuracy,
                        "learning_rate": current_lr,
                    })
                logging.info("seed=%s condition=%s epoch=%s done", seed, condition, epoch)

    results_path = root / "results.csv"
    if not rows:
        raise RuntimeError("No results were produced")
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("wrote %s", results_path)


if __name__ == "__main__":
    run()
