from __future__ import annotations

import argparse

from . import aggregate_results, plot_results, run_generalization, run_label_noise, run_layer_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Unified entry point for benchmark experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generalization")
    subparsers.add_parser("label-noise")
    subparsers.add_parser("layer-analysis")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("plot")
    args, remainder = parser.parse_known_args()
    return args, remainder


def run():
    args, remainder = parse_args()
    import sys

    sys.argv = [sys.argv[0], *remainder]
    if args.command == "generalization":
        run_generalization.run()
    elif args.command == "label-noise":
        run_label_noise.run()
    elif args.command == "layer-analysis":
        run_layer_analysis.run()
    elif args.command == "aggregate":
        aggregate_results.run()
    elif args.command == "plot":
        plot_results.run()
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    run()
