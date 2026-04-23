"""Centralized project configuration.

Edit values in this file only. Scripts should import these configs
instead of redefining parameters locally.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    data_dir: Path = Path("EuroSAT_RGB")
    checkpoints_dir: Path = Path("checkpoints")
    outputs_dir: Path = Path("outputs")


@dataclass
class DataPrepConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    image_size: int = 64
    flatten: bool = True
    normalize: bool = True
    file_prefix: str = "eurosat_split"
    show_progress: bool = True
    progress_every: int = 500


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 2e-5
    input_dim: int = 64 * 64 * 3
    hidden_dim1: int = 256
    hidden_dim2: int = 256
    output_dim: int = 10
    activation: str = "relu"
    lr_decay_gamma: float = 0.95


@dataclass
class SearchConfig:
    mode: str = "grid"  # "random" or "grid"
    num_trials: int = 25
    seed: int = 42

    # If > 0, override TRAIN.epochs during search; else use TRAIN.epochs
    epochs_per_trial: int = 30

    # Random-search space
    lr_log10_min: float = -4.0
    lr_log10_max: float = -1.5
    wd_log10_min: float = -6.0
    wd_log10_max: float = -3.0
    hidden_dim1_candidates: tuple[int, ...] = (64, 128, 256, 384, 512)
    hidden_dim2_candidates: tuple[int, ...] = (32, 64, 96, 128, 192, 256)

    # Grid-search space requested by user
    grid_learning_rates: tuple[float, ...] = (0.01, 0.009, 0.008)
    grid_hidden_dim_pairs: tuple[tuple[int, int], ...] = ((256, 256), (512, 64))
    grid_weight_decays: tuple[float, ...] = (5e-6, 2e-5, 2e-4)


@dataclass
class TestConfig:
    checkpoint_name: str = "best_model.npz"


PATHS = Paths()
DATA_PREP = DataPrepConfig()
TRAIN = TrainConfig()
SEARCH = SearchConfig()
TEST = TestConfig()
