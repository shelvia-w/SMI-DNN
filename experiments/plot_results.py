from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MI/SMI benchmark results.")
    parser.add_argument("--input", required=True, help="Raw or aggregate CSV file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", default="all", choices=["all", "scatter", "layers", "effects", "lines"])
    parser.add_argument("--x", default="epoch", choices=["epoch", "layer"])
    parser.add_argument("--hue", default="variation", choices=["condition", "layer", "estimator", "variation", "model", "dataset"])
    return parser.parse_args()


def _col(data: pd.DataFrame, name: str) -> str:
    return f"{name}_mean" if f"{name}_mean" in data.columns else name


def _ensure_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "smi" not in data.columns and "mi" in data.columns:
        data["smi"] = data["mi"]
    if "generalization_gap" not in data.columns and {"train_accuracy", "test_accuracy"} <= set(data.columns):
        data["generalization_gap"] = data["train_accuracy"] - data["test_accuracy"]
    return data


def plot_lines(data: pd.DataFrame, output_dir: Path, x: str, hue: str) -> None:
    ensure_dir(output_dir)
    y_col = _col(data, "smi")
    err_col = "smi_std" if "smi_std" in data.columns else None
    for layer, layer_df in data.groupby("layer", dropna=False):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for key, group in layer_df.groupby(hue, dropna=False):
            group = group.sort_values(x)
            xs = group[x].astype(str) if x == "layer" else group[x]
            ax.plot(xs, group[y_col], marker="o", label=str(key))
            if err_col is not None and x != "layer":
                ax.fill_between(xs, group[y_col] - group[err_col], group[y_col] + group[err_col], alpha=0.2)
        ax.set_title(f"SMI estimates: {layer}")
        ax.set_xlabel(x)
        ax.set_ylabel("SMI")
        ax.legend(frameon=False)
        fig.tight_layout()
        safe_layer = str(layer).replace("/", "_")
        fig.savefig(output_dir / f"{safe_layer}_{x}_by_{hue}.png", dpi=200)
        plt.close(fig)


def plot_smi_gap_scatter(data: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    smi_col = _col(data, "smi")
    gap_col = _col(data, "generalization_gap")
    for keys, group in data.groupby([col for col in ["dataset", "model", "layer"] if col in data.columns], dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        title = " / ".join(str(value) for value in key_values)
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        hue_col = "variation" if "variation" in group.columns else None
        if hue_col:
            for hue, hue_group in group.groupby(hue_col, dropna=False):
                ax.scatter(hue_group[smi_col], hue_group[gap_col], label=str(hue), alpha=0.8)
            ax.legend(frameon=False)
        else:
            ax.scatter(group[smi_col], group[gap_col], alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("SMI")
        ax.set_ylabel("train accuracy - test accuracy")
        fig.tight_layout()
        safe = "_".join(str(value).replace("/", "_") for value in key_values)
        fig.savefig(output_dir / f"smi_vs_gap_{safe}.png", dpi=200)
        plt.close(fig)


def plot_layer_comparison(data: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    smi_col = _col(data, "smi")
    group_cols = [col for col in ["dataset", "model", "variation"] if col in data.columns]
    for keys, group in data.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        grouped = group.groupby("layer", dropna=False)[smi_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.bar(grouped["layer"].astype(str), grouped[smi_col])
        ax.set_title(" / ".join(str(value) for value in key_values))
        ax.set_xlabel("layer")
        ax.set_ylabel("mean SMI")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        safe = "_".join(str(value).replace("/", "_") for value in key_values)
        fig.savefig(output_dir / f"layers_{safe}.png", dpi=200)
        plt.close(fig)


def plot_effects(data: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    gap_col = _col(data, "generalization_gap")
    smi_col = _col(data, "smi")
    for effect in ["dropout_rate", "noise_ratio", "use_batch_norm"]:
        if effect not in data.columns:
            continue
        for layer, group in data.groupby("layer", dropna=False):
            grouped = group.groupby(effect, dropna=False)[[smi_col, gap_col]].mean().reset_index()
            fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
            xs = grouped[effect].astype(str) if effect == "use_batch_norm" else grouped[effect]
            ax1.plot(xs, grouped[smi_col], marker="o", color="tab:blue", label="SMI")
            ax1.set_xlabel(effect)
            ax1.set_ylabel("SMI", color="tab:blue")
            ax2 = ax1.twinx()
            ax2.plot(xs, grouped[gap_col], marker="s", color="tab:red", label="gap")
            ax2.set_ylabel("generalization gap", color="tab:red")
            ax1.set_title(f"{effect}: {layer}")
            fig.tight_layout()
            safe_layer = str(layer).replace("/", "_")
            fig.savefig(output_dir / f"effect_{effect}_{safe_layer}.png", dpi=200)
            plt.close(fig)


def run():
    args = parse_args()
    data = _ensure_columns(pd.read_csv(args.input))
    output_dir = Path(args.output_dir)
    if args.kind in {"all", "scatter"}:
        plot_smi_gap_scatter(data, output_dir)
    if args.kind in {"all", "layers"} and "layer" in data.columns:
        plot_layer_comparison(data, output_dir)
    if args.kind in {"all", "effects"}:
        plot_effects(data, output_dir)
    if args.kind == "lines":
        plot_lines(data, output_dir, args.x, args.hue)


if __name__ == "__main__":
    run()
