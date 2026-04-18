from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    setup: str = "generalization"
    dataset: str = "mnist"
    model: str = "mlp5"
    estimator: str = "smi_ksg_cd"
    output_dir: str = "outputs"
    batch_size: int = 0
    epochs: int = 200
    learning_rate: float = 1e-2
    momentum: float = 0.9
    lr_decay_factor: float = 0.9
    lr_patience: int = 10
    early_stop_patience: int = 20
    min_accuracy_delta: float = 0.0
    seeds: tuple[int, ...] = (0, 1, 2)
    noise_ratio: float = 0.0
    dropout_rate: float = 0.0
    use_batch_norm: bool = True
    layers: tuple[str, ...] = ()
    analysis_epochs: tuple[int, ...] = ()
    max_mi_samples: int = 10000
    n_projs: int = 1000
    n_jobs: int = 1
    k: int = 3
    test_fraction: float = 1.0
    save_checkpoints: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        data["layers"] = list(self.layers)
        data["analysis_epochs"] = list(self.analysis_epochs)
        return data


SETUPS = {"generalization", "memorization"}
DATASETS = {"mnist", "fashion_mnist", "cifar10", "cifar100"}
MODELS = {"mlp5", "cnn5", "cnn5_bn", "cnn6", "vgg16", "resnet50"}
ESTIMATORS = {"ksg_cd", "ksg_cc", "neural", "smi_ksg_cd", "smi_ksg_cc", "smi_neural"}
GENERALIZATION_DATASETS = {"mnist", "fashion_mnist", "cifar10", "cifar100"}
GENERALIZATION_MODELS = {"mlp5", "cnn5", "cnn5_bn", "cnn6", "vgg16", "resnet50"}
MEMORIZATION_DATASETS = {"mnist", "fashion_mnist"}
MEMORIZATION_MODELS = {"mlp5", "cnn5", "cnn5_bn", "cnn6"}

MLP_CNN_DROPOUT_RATES = (0.1, 0.2, 0.3, 0.4, 0.5)
TRANSFER_DROPOUT_RATES = (0.1, 0.2, 0.3, 0.4)
LABEL_NOISE_RATIOS = (0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def parse_csv_ints(value: str | None) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_floats(value: str | None) -> tuple[float, ...]:
    if value is None or value == "":
        return ()
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str | None) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def default_batch_size(model: str) -> int:
    if model in {"vgg16", "resnet50"}:
        return 256
    return 32


def default_dropout_rates(model: str) -> tuple[float, ...]:
    if model in {"vgg16", "resnet50"}:
        return TRANSFER_DROPOUT_RATES
    return MLP_CNN_DROPOUT_RATES


def canonical_model_name(model: str) -> str:
    if model == "cnn6":
        return "cnn5"
    if model == "cnn5_bn":
        return "cnn5"
    return model


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    if config.setup not in SETUPS:
        raise ValueError(f"setup must be one of {sorted(SETUPS)}")
    if config.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}")
    if config.model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    if config.estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}")
    if config.setup == "generalization":
        if config.dataset not in GENERALIZATION_DATASETS:
            raise ValueError(f"generalization datasets must be one of {sorted(GENERALIZATION_DATASETS)}")
        if config.model not in GENERALIZATION_MODELS:
            raise ValueError(f"generalization models must be one of {sorted(GENERALIZATION_MODELS)}")
    if config.setup == "memorization":
        if config.dataset not in MEMORIZATION_DATASETS:
            raise ValueError(f"memorization datasets must be one of {sorted(MEMORIZATION_DATASETS)}")
        if config.model not in MEMORIZATION_MODELS:
            raise ValueError(f"memorization models must be one of {sorted(MEMORIZATION_MODELS)}")
    if config.batch_size < 0:
        raise ValueError("batch_size must be positive, or 0 to use the architecture default")
    if config.batch_size == 0:
        config.batch_size = default_batch_size(config.model)
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be positive")
    if not 0.0 <= config.noise_ratio <= 1.0:
        raise ValueError("noise_ratio must be in [0, 1]")
    if not 0.0 <= config.dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")
    if config.max_mi_samples < 2:
        raise ValueError("max_mi_samples must be at least 2")
    if config.n_projs < 1:
        raise ValueError("n_projs must be positive")
    if config.k < 1:
        raise ValueError("k must be positive")
    return config


def add_common_args(parser, default_setup="generalization"):
    parser.add_argument("--setup", default=default_setup, choices=sorted(SETUPS))
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASETS))
    parser.add_argument("--model", default="mlp5", choices=sorted(MODELS))
    parser.add_argument("--estimator", default="smi_ksg_cd", choices=sorted(ESTIMATORS))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses 32 for MLP/CNN and 256 for VGG/ResNet.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs before the accuracy-plateau stop.")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-decay-factor", type=float, default=0.9)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--min-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated integer seeds.")
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--use-batch-norm", default="true", help="Boolean; meaningful for cnn5.")
    parser.add_argument("--layers", default="", help="Comma-separated layer names. Empty means default analysis layers.")
    parser.add_argument("--analysis-epochs", default="", help="Comma-separated epochs to analyze. Empty means all epochs.")
    parser.add_argument("--max-mi-samples", type=int, default=10000)
    parser.add_argument("--n-projs", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser


def config_from_args(args) -> ExperimentConfig:
    config = ExperimentConfig(
        setup=args.setup,
        dataset=args.dataset,
        model=args.model,
        estimator=args.estimator,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        lr_decay_factor=args.lr_decay_factor,
        lr_patience=args.lr_patience,
        early_stop_patience=args.early_stop_patience,
        min_accuracy_delta=args.min_accuracy_delta,
        seeds=parse_csv_ints(args.seeds),
        dropout_rate=args.dropout_rate,
        use_batch_norm=parse_bool(args.use_batch_norm),
        layers=parse_csv_strings(args.layers),
        analysis_epochs=parse_csv_ints(args.analysis_epochs),
        max_mi_samples=args.max_mi_samples,
        n_projs=args.n_projs,
        n_jobs=args.n_jobs,
        k=args.k,
        test_fraction=args.test_fraction,
        save_checkpoints=args.save_checkpoints,
    )
    if hasattr(args, "noise_ratio"):
        config.noise_ratio = args.noise_ratio
    return validate_config(config)


def experiment_dir(config: ExperimentConfig, name: str) -> Path:
    return Path(config.output_dir) / name / config.dataset / canonical_model_name(config.model) / config.estimator
