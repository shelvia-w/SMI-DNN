# Sliced Mutual Information for Deep Neural Network Analysis

This repository studies deep neural networks through **Sliced Mutual Information (SMI)**, a projection-based mutual-information measure for analyzing learned representations, label information, and training dynamics.

This is the official code repository for the paper **"Using Sliced Mutual Information to Study Memorization and Generalization in Deep Neural Networks"**.

The current codebase focuses on reproducible Python experiments for:

- **Layer-wise representation analysis:** tracking how MI/SMI changes across hidden layers and training epochs.
- **Generalization diagnostics:** comparing SMI with train/test behavior under dropout, label noise, and batch-normalization sweeps.
- **Memorization under label noise:** measuring how clean and corrupted labels shape learned representations.

## Research Questions

This project is organized around three practical questions:

1. How does MI/SMI between representations and labels evolve across depth and training time?
2. Can SMI separate useful label structure from memorized noise?
3. How do regularization choices such as dropout and batch normalization affect information captured in hidden layers?

## Repository Structure

```text
SMI-DNN/
|-- estimators/
|   |-- smi_estimator.py        # Sliced MI over random projections
|   |-- knn_estimators.py       # KSG continuous/discrete and continuous/continuous MI estimators
|   `-- neural_estimators.py    # Neural variational MI estimators
|-- experiments/
|   |-- main.py                 # Unified CLI entry point
|   |-- run_generalization.py   # Dropout, label-noise, batch-norm, and single-run sweeps
|   |-- run_label_noise.py      # Clean vs noisy-label benchmark runner
|   |-- run_layer_analysis.py   # Per-epoch layer-wise MI/SMI analysis
|   |-- aggregate_results.py    # Summary statistics and optional SMI/gap correlations
|   |-- plot_results.py         # Plot generation from raw or aggregated CSV outputs
|   |-- config.py               # CLI defaults, choices, and validation
|   |-- datasets.py             # TensorFlow/Keras dataset loading
|   |-- models.py               # MLP, CNN, VGG16, and ResNet50 builders
|   `-- train.py                # Training loop utilities
|-- notebooks/
|   |-- SMI_Layers_Training.ipynb
|   `-- SMI_Label_Noise.ipynb
|-- requirements.txt
`-- README.md
```

## Setup

Create an environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run commands from the repository root so Python can import both `estimators` and `experiments`.

To use the notebooks:

```bash
jupyter notebook notebooks
```

The experiment scripts use TensorFlow/Keras datasets. Dataset files, and ImageNet weights for `vgg16` or `resnet50`, may be downloaded by TensorFlow the first time those options are used.

## Method Overview

SMI estimates mutual information by projecting high-dimensional variables onto random low-dimensional directions. For variables `X` and `Y`, the estimator repeatedly samples projection vectors, estimates MI on each projected pair, and averages the projection-level estimates.

The implementation supports:

- Continuous-discrete KSG estimation with `method="ksg_cd"`
- Continuous-continuous KSG estimation with `method="ksg_cc"`
- Neural variational estimation with `method="neural"`
- Parallel projection evaluation through `n_jobs`
- Reproducible projection sampling through `random_state`
- Optional per-projection diagnostics through `return_details=True`

Basic usage:

```python
from estimators.smi_estimator import compute_smi

smi = compute_smi(
    x=representations,
    y=labels,
    proj_x=True,
    proj_y=False,
    n_projs=1000,
    method="ksg_cd",
    random_state=42,
    n_jobs=-1,
)
```

Request details when you need projection-level estimates and Monte Carlo uncertainty:

```python
result = compute_smi(
    representations,
    labels,
    method="ksg_cd",
    random_state=42,
    n_jobs=-1,
    return_details=True,
)

print(result["smi"], result["stderr"])
```

## Experiments

The Python pipeline is the preferred path for reproducible runs. The notebooks are useful for interactive launch, inspection, and lightweight analysis.

You can run each module directly:

```bash
python -m experiments.run_layer_analysis --help
python -m experiments.run_label_noise --help
python -m experiments.run_generalization --help
```

Or use the unified entry point:

```bash
python -m experiments.main layer-analysis --help
python -m experiments.main label-noise --help
python -m experiments.main generalization --help
python -m experiments.main aggregate --help
python -m experiments.main plot --help
```

### Layer-Wise Analysis

Run a layer-wise benchmark over selected training epochs:

```bash
python -m experiments.run_layer_analysis \
  --setup generalization \
  --dataset mnist \
  --model mlp5 \
  --estimator smi_ksg_cd \
  --epochs 50 \
  --analysis-epochs 1,5,10,25,50 \
  --seeds 0,1,2 \
  --max-mi-samples 10000 \
  --n-projs 500 \
  --n-jobs -1
```

If `--analysis-epochs` is omitted, MI/SMI is computed at every epoch. If `--layers` is omitted, the default hidden layers for the selected model are used.

### Label-Noise Benchmark

Run clean and noisy-label conditions:

```bash
python -m experiments.run_label_noise \
  --setup memorization \
  --dataset fashion_mnist \
  --model cnn5_bn \
  --estimator smi_ksg_cd \
  --epochs 50 \
  --seeds 0,1,2 \
  --conditions clean,noisy \
  --noise-ratio 0.4 \
  --max-mi-samples 10000 \
  --n-projs 500 \
  --n-jobs -1
```

Noisy-label runs compute MI/SMI against the labels used for training, so the `noisy` condition uses the corrupted training labels.

### Generalization Sweeps

Run the broader generalization experiment, which can sweep dropout rates, label-noise ratios, and batch-normalization settings:

```bash
python -m experiments.run_generalization \
  --setup generalization \
  --dataset cifar10 \
  --model cnn5_bn \
  --estimator smi_ksg_cd \
  --sweep all \
  --epochs 50 \
  --seeds 0,1,2 \
  --max-mi-samples 10000 \
  --n-projs 500 \
  --n-jobs -1
```

Useful `--sweep` values:

- `all`: run dropout, label-noise, and batch-normalization variations.
- `dropout`: sweep dropout rates only.
- `label_noise`: sweep label-noise ratios only.
- `batch_norm`: compare batch-normalization settings for CNN-style models.
- `single`: run one configuration using `--dropout-rate`, `--noise-ratio`, and `--use-batch-norm`.

### Aggregation And Plotting

Aggregate one or more raw `results.csv` files:

```bash
python -m experiments.aggregate_results \
  --input outputs/label_noise/fashion_mnist/cnn5/smi_ksg_cd/results.csv \
  --output outputs/label_noise_summary.csv \
  --correlations-output outputs/label_noise_correlations.csv
```

Generate plots from a raw or aggregated CSV:

```bash
python -m experiments.plot_results \
  --input outputs/label_noise_summary.csv \
  --output-dir outputs/figures \
  --kind all \
  --x epoch \
  --hue condition
```

## Supported Options

Datasets:

- Generalization: `mnist`, `fashion_mnist`, `cifar10`, `cifar100`
- Memorization: `mnist`, `fashion_mnist`

Models:

- Generalization: `mlp5`, `cnn5`, `cnn5_bn`, `cnn6`, `vgg16`, `resnet50`
- Memorization: `mlp5`, `cnn5`, `cnn5_bn`, `cnn6`

Estimators:

- Direct MI: `ksg_cd`, `ksg_cc`, `neural`
- Sliced MI: `smi_ksg_cd`, `smi_ksg_cc`, `smi_neural`

Default hidden layers:

- `mlp5`: `fc1`, `fc2`, `fc3`, `fc4`
- `cnn5`, `cnn5_bn`, `cnn6`: `relu1`, `relu2`, `relu3`, `relu4`
- `vgg16`, `resnet50`: `fc1`, `fc2`

## Defaults And Outputs

Important CLI defaults:

- Optimizer: SGD with learning rate `0.01` and momentum `0.9`
- Epochs: `200`
- Batch size: `32` for MLP/CNN models, `256` for `vgg16` and `resnet50`
- Seeds: `0,1,2`
- SMI projections: `1000`
- MI sample cap: `10000`
- KNN neighbors: `k=3`
- Output root: `outputs/`

Outputs are written under:

```text
outputs/<experiment>/<dataset>/<canonical_model>/<estimator>/
```

For example, `cnn5_bn` and `cnn6` outputs are stored under canonical model name `cnn5`.

Each run writes:

- `config.json`: resolved experiment configuration
- `results.csv`: row-level MI/SMI estimates, model metrics, layers, seeds, and conditions
- Optional checkpoints when `--save-checkpoints` is enabled

## Reproducibility Notes

- Use `--seeds` for explicit seed control.
- Use `--analysis-epochs` to reduce expensive MI/SMI evaluation frequency.
- Use `--layers` to restrict analysis to specific named layers.
- Symmetric label corruption is seed-controlled.
- Random SMI projections and KNN jitter are controlled by the estimator seed.
- SMI estimates depend on sampled projection directions; increase `--n-projs` for lower variance at higher computational cost.
- KSG estimators report information in nats.
- Neural SMI trains a fresh critic per projection and is much more expensive than KSG-based SMI.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/SMI_Layers_Training.ipynb` | Interactive layer-wise SMI workflow. |
| `notebooks/SMI_Label_Noise.ipynb` | Interactive clean vs noisy-label workflow. |

## References

This repository builds on standard mutual-information estimators and neural MI objectives, including:

- Kraskov, Stoegbauer, and Grassberger, "Estimating mutual information", 2004.
- Belghazi et al., "Mutual Information Neural Estimation", 2018.
- van den Oord, Li, and Vinyals, "Representation Learning with Contrastive Predictive Coding", 2018.
- Poole et al., "On Variational Bounds of Mutual Information", 2019.
- Song and Ermon, "Understanding the Limitations of Variational Mutual Information Estimators", 2020.

## Citation

If this repository supports your work, please cite the paper:

```bibtex
@article{smi_dnn_memorization_generalization,
  title = {Using Sliced Mutual Information to Study Memorization and Generalization in Deep Neural Networks},
  author = {TODO},
  journal = {TODO},
  year = {TODO}
}
```

Please also cite any estimator papers relevant to the parts you use.
