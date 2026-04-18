from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass

from .checkpoints import save_checkpoint
from .config import (
    LABEL_NOISE_RATIOS,
    add_common_args,
    config_from_args,
    default_dropout_rates,
    experiment_dir,
    parse_bool,
    parse_csv_floats,
    parse_csv_strings,
)
from .datasets import load_dataset
from .estimators import estimate_mi
from .features import extract_activations, select_feature_layers
from .models import build_model
from .noise import apply_symmetric_label_noise
from .train import (
    AccuracyPlateauScheduler,
    EarlyStopState,
    compile_model,
    should_stop_on_accuracy_plateau,
    train_one_epoch,
    update_learning_rate_on_plateau,
)
from .utils import configure_logging, ensure_dir, sample_indices, set_global_seed, write_json


@dataclass(frozen=True)
class Trial:
    variation: str
    dropout_rate: float
    noise_ratio: float
    use_batch_norm: bool


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SMI/generalization-gap experiment.")
    add_common_args(parser, default_setup="generalization")
    parser.add_argument(
        "--sweep",
        default="all",
        choices=["all", "dropout", "label_noise", "batch_norm", "single"],
        help="Which generalization variation to run.",
    )
    parser.add_argument("--dropout-rates", default="", help="Comma-separated rates. Empty uses paper-target defaults.")
    parser.add_argument("--noise-ratios", default="", help="Comma-separated ratios. Empty uses paper-target defaults.")
    parser.add_argument(
        "--batch-norm-values",
        default="true,false",
        help="Comma-separated booleans used for cnn5 BN sweeps.",
    )
    parser.add_argument("--include-all-layers", action="store_true", help="Compute SMI for all named hidden layers.")
    parser.add_argument("--noise-ratio", type=float, default=0.0, help="Used only with --sweep single.")
    return parser.parse_args()


def _batch_norm_values(args, model: str) -> tuple[bool, ...]:
    if model not in {"cnn5", "cnn5_bn", "cnn6"}:
        return (False,)
    return tuple(parse_bool(value) for value in parse_csv_strings(args.batch_norm_values)) or (True, False)


def build_trials(args, config) -> list[Trial]:
    dropout_rates = parse_csv_floats(args.dropout_rates) or default_dropout_rates(config.model)
    noise_ratios = parse_csv_floats(args.noise_ratios) or LABEL_NOISE_RATIOS
    bn_values = _batch_norm_values(args, config.model)
    trials: list[Trial] = []

    if args.sweep in {"all", "dropout"}:
        for dropout_rate in dropout_rates:
            for use_bn in bn_values:
                trials.append(Trial("dropout", dropout_rate, 0.0, use_bn))

    if args.sweep in {"all", "label_noise"}:
        # Smallest reasonable assumption: label-noise sweeps isolate label corruption
        # by using dropout=0, while still comparing both CNN BN settings.
        for noise_ratio in noise_ratios:
            for use_bn in bn_values:
                trials.append(Trial("label_noise", 0.0, noise_ratio, use_bn))

    if args.sweep in {"all", "batch_norm"} and config.model in {"cnn5", "cnn5_bn", "cnn6"}:
        for use_bn in bn_values:
            trials.append(Trial("batch_norm", 0.0, 0.0, use_bn))

    if args.sweep == "single":
        trials.append(Trial("single", config.dropout_rate, config.noise_ratio, config.use_batch_norm))

    seen = set()
    unique_trials = []
    for trial in trials:
        key = (trial.variation, trial.dropout_rate, trial.noise_ratio, trial.use_batch_norm)
        if key not in seen:
            seen.add(key)
            unique_trials.append(trial)
    return unique_trials


def _trial_id(trial: Trial, seed: int) -> str:
    bn = "bn1" if trial.use_batch_norm else "bn0"
    return f"{trial.variation}_drop{trial.dropout_rate:g}_noise{trial.noise_ratio:g}_{bn}_seed{seed}"


def run_trial(config, dataset, trial: Trial, seed: int, include_all_layers: bool) -> list[dict]:
    set_global_seed(seed)
    y_train = dataset.y_train
    observed_corruption_rate = 0.0
    if trial.noise_ratio > 0:
        y_train, mask = apply_symmetric_label_noise(dataset.y_train, dataset.num_classes, trial.noise_ratio, seed)
        observed_corruption_rate = float(mask.mean())

    model = compile_model(
        build_model(
            config.model,
            dataset.input_shape,
            dataset.num_classes,
            dropout_rate=trial.dropout_rate,
            use_batch_norm=trial.use_batch_norm,
        ),
        config.learning_rate,
        config.momentum,
    )
    layer_names = select_feature_layers(model, list(config.layers), include_all_layers)
    idx = sample_indices(len(dataset.x_train), config.max_mi_samples, seed)
    lr_scheduler = AccuracyPlateauScheduler()
    stop_state = EarlyStopState()
    rows = []
    final_metrics = None
    final_lr = config.learning_rate

    run_dir = experiment_dir(config, "generalization") / _trial_id(trial, seed)
    ensure_dir(run_dir)

    for epoch in range(1, config.epochs + 1):
        final_metrics = train_one_epoch(
            model,
            dataset.x_train,
            y_train,
            dataset.x_test,
            dataset.y_test,
            config.batch_size,
            epoch,
            seed,
        )
        final_lr = update_learning_rate_on_plateau(
            model,
            final_metrics.train_accuracy,
            lr_scheduler,
            patience=config.lr_patience,
            factor=config.lr_decay_factor,
        )
        if config.save_checkpoints:
            save_checkpoint(model, run_dir, epoch)
        if should_stop_on_accuracy_plateau(
            final_metrics.train_accuracy,
            epoch,
            stop_state,
            patience=config.early_stop_patience,
            min_delta=config.min_accuracy_delta,
        ):
            logging.info(
                "early stop trial=%s seed=%s epoch=%s best_train_acc=%.4f",
                trial,
                seed,
                epoch,
                stop_state.best_accuracy,
            )
            break

    if final_metrics is None:
        raise RuntimeError("Training produced no metrics")

    activations = extract_activations(model, dataset.x_train[idx], layer_names, config.batch_size)
    for layer_name, reps in activations.items():
        estimate = estimate_mi(
            reps,
            y_train[idx],
            config.estimator,
            seed=seed + final_metrics.epoch,
            k=config.k,
            n_projs=config.n_projs,
            n_jobs=config.n_jobs,
            batch_size=config.batch_size,
        )
        rows.append({
            "experiment": "generalization",
            "variation": trial.variation,
            "dataset": config.dataset,
            "model": model.name,
            "estimator": config.estimator,
            "seed": seed,
            "dropout_rate": trial.dropout_rate,
            "noise_ratio": trial.noise_ratio,
            "use_batch_norm": trial.use_batch_norm,
            "observed_corruption_rate": observed_corruption_rate,
            "epoch": final_metrics.epoch,
            "best_train_accuracy": stop_state.best_accuracy,
            "best_train_epoch": stop_state.best_epoch,
            "layer": layer_name,
            "mi": estimate.estimate,
            "smi": estimate.estimate,
            "stderr": estimate.stderr,
            "train_loss": final_metrics.train_loss,
            "train_accuracy": final_metrics.train_accuracy,
            "test_loss": final_metrics.test_loss,
            "test_accuracy": final_metrics.test_accuracy,
            "generalization_gap": final_metrics.generalization_gap,
            "learning_rate": final_lr,
            "smi_samples": len(idx),
            "smi_projections": config.n_projs,
            "smi_label_source": "training_labels_used",
        })
    return rows


def run():
    configure_logging()
    args = parse_args()
    config = config_from_args(args)
    trials = build_trials(args, config)
    root = experiment_dir(config, "generalization")
    ensure_dir(root)
    write_json(root / "config.json", config.to_dict() | {
        "sweep": args.sweep,
        "include_all_layers": args.include_all_layers,
        "trials": [trial.__dict__ for trial in trials],
    })

    dataset = load_dataset(config.dataset, test_fraction=config.test_fraction)
    rows = []
    for seed in config.seeds:
        for trial in trials:
            logging.info("starting seed=%s trial=%s", seed, trial)
            rows.extend(run_trial(config, dataset, trial, seed, args.include_all_layers))

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
