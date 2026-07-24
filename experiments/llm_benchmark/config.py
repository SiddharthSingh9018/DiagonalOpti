import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    data_dir: str = "data"
    train_tokens: int = 200_000
    val_tokens: int = 20_000
    synthetic_noise: float = 0.08


@dataclass
class TrainConfig:
    batch_size: int = 16
    base_steps: int = 200
    duration_multipliers: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    eval_interval: int = 25
    checkpoint_interval: int = 200
    grad_clip: float = 1.0
    target_losses: list[float] = field(default_factory=lambda: [3.0, 2.8, 2.5])
    wall_clock_checkpoints_sec: list[float] = field(default_factory=lambda: [30.0, 60.0, 120.0])


@dataclass
class OptimizerConfig:
    names: list[str] = field(default_factory=lambda: ["AdamW", "Lion", "Sophia", "Muon", "DiagonalOptimiser"])
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {
            "AdamW": 3e-4,
            "Lion": 1e-4,
            "Sophia": 3e-4,
            "Muon": 2e-4,
            "DiagonalOptimiser": 3e-4,
        }
    )
    weight_decay: float = 0.01


@dataclass
class ExperimentConfig:
    experiment_name: str = "tinygpt_optimizer_benchmark"
    output_dir: str = "results"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    device: str = "auto"
    compile_model: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizers: OptimizerConfig = field(default_factory=OptimizerConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _dataclass_to_dict(getattr(obj, key)) for key in obj.__dataclass_fields__}
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _dataclass_to_dict(value) for key, value in obj.items()}
    return obj


def load_config(path: str | Path) -> ExperimentConfig:
    default = _dataclass_to_dict(ExperimentConfig())
    with Path(path).open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    merged = _merge_dict(default, user_config)
    return ExperimentConfig(
        experiment_name=merged["experiment_name"],
        output_dir=merged["output_dir"],
        seeds=merged["seeds"],
        device=merged["device"],
        compile_model=merged.get("compile_model", False),
        model=ModelConfig(**merged["model"]),
        data=DataConfig(**merged["data"]),
        train=TrainConfig(**merged["train"]),
        optimizers=OptimizerConfig(**merged["optimizers"]),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(_dataclass_to_dict(config), f, indent=2)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return _dataclass_to_dict(config)
