from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate MI/SMI experiment CSV files.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more results.csv files.")
    parser.add_argument("--output", required=True, help="Output aggregate CSV path.")
    parser.add_argument("--correlations-output", default="", help="Optional output CSV for SMI/gap correlations.")
    return parser.parse_args()


def _ensure_generalization_gap(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "generalization_gap" not in data.columns and {"train_accuracy", "test_accuracy"} <= set(data.columns):
        data["generalization_gap"] = data["train_accuracy"] - data["test_accuracy"]
    if "smi" not in data.columns and "mi" in data.columns:
        data["smi"] = data["mi"]
    return data


def aggregate(paths: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    data = _ensure_generalization_gap(pd.concat(frames, ignore_index=True))
    group_cols = [
        col for col in [
            "experiment",
            "variation",
            "dataset",
            "model",
            "estimator",
            "condition",
            "dropout_rate",
            "noise_ratio",
            "use_batch_norm",
            "epoch",
            "layer",
        ]
        if col in data.columns
    ]
    value_cols = [
        col
        for col in [
            "mi",
            "smi",
            "stderr",
            "train_accuracy",
            "test_accuracy",
            "generalization_gap",
            "learning_rate",
            "observed_corruption_rate",
        ]
        if col in data.columns
    ]
    summary = data.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary


def correlations(paths: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    data = _ensure_generalization_gap(pd.concat(frames, ignore_index=True))
    if not {"smi", "generalization_gap", "layer"} <= set(data.columns):
        return pd.DataFrame()

    group_cols = [
        col for col in ["experiment", "variation", "dataset", "model", "estimator", "layer"]
        if col in data.columns
    ]
    rows = []
    for key, group in data.groupby(group_cols, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        valid = group[["smi", "generalization_gap"]].dropna()
        row["n"] = len(valid)
        if len(valid) >= 2 and valid["smi"].nunique() > 1 and valid["generalization_gap"].nunique() > 1:
            row["pearson"] = valid["smi"].corr(valid["generalization_gap"], method="pearson")
            row["spearman"] = valid["smi"].corr(valid["generalization_gap"], method="spearman")
        else:
            row["pearson"] = float("nan")
            row["spearman"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    args = parse_args()
    output = Path(args.output)
    ensure_dir(output.parent)
    aggregate(args.input).to_csv(output, index=False)
    if args.correlations_output:
        corr_output = Path(args.correlations_output)
        ensure_dir(corr_output.parent)
        correlations(args.input).to_csv(corr_output, index=False)


if __name__ == "__main__":
    run()
