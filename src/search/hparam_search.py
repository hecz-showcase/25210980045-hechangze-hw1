"""Hyperparameter search: tune LR, hidden dims, and weight decay with random/grid search."""

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import SEARCH, TRAIN
from src.models.mlp import MLPConfig, ThreeLayerMLP
from src.training.optimizer import SGD, SGDConfig
from src.training.trainer import Trainer


@dataclass
class SearchResult:
    trial_id: int
    config: Dict[str, float]
    best_val_acc: float
    best_val_loss: float


def _sample_hparams(rng: np.random.Generator) -> Dict[str, float]:
    learning_rate = 10 ** rng.uniform(SEARCH.lr_log10_min, SEARCH.lr_log10_max)
    weight_decay = 10 ** rng.uniform(SEARCH.wd_log10_min, SEARCH.wd_log10_max)

    if not SEARCH.hidden_dim1_candidates:
        raise ValueError("SEARCH.hidden_dim1_candidates must not be empty")
    if not SEARCH.hidden_dim2_candidates:
        raise ValueError("SEARCH.hidden_dim2_candidates must not be empty")

    hidden_dim1 = int(rng.choice(np.asarray(SEARCH.hidden_dim1_candidates, dtype=np.int64)))
    valid_h2 = [h for h in SEARCH.hidden_dim2_candidates if h <= hidden_dim1]
    if not valid_h2:
        raise ValueError("No valid hidden_dim2 candidate <= hidden_dim1.")
    hidden_dim2 = int(rng.choice(np.asarray(valid_h2, dtype=np.int64)))

    return {
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "hidden_dim1": float(hidden_dim1),
        "hidden_dim2": float(hidden_dim2),
    }


def _grid_hparams() -> List[Dict[str, float]]:
    combos: List[Dict[str, float]] = []
    for lr in SEARCH.grid_learning_rates:
        for h1, h2 in SEARCH.grid_hidden_dim_pairs:
            for wd in SEARCH.grid_weight_decays:
                combos.append(
                    {
                        "learning_rate": float(lr),
                        "weight_decay": float(wd),
                        "hidden_dim1": float(h1),
                        "hidden_dim2": float(h2),
                    }
                )
    return combos


def _run_single_trial(
    trial_id: int,
    sampled: Dict[str, float],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    checkpoints_dir: Path,
    seed: int,
    epochs: int,
    ckpt_prefix: str,
    verbose: bool,
) -> Tuple[SearchResult, Path]:
    input_dim = int(x_train.shape[1])
    output_dim = int(np.max(y_train) + 1)

    model = ThreeLayerMLP(
        MLPConfig(
            input_dim=input_dim,
            hidden_dim1=int(sampled["hidden_dim1"]),
            hidden_dim2=int(sampled["hidden_dim2"]),
            output_dim=output_dim,
            activation=TRAIN.activation,
        ),
        seed=seed + trial_id,
    )

    optimizer = SGD(
        SGDConfig(
            learning_rate=float(sampled["learning_rate"]),
            lr_decay_gamma=TRAIN.lr_decay_gamma,
        )
    )

    trial_ckpt_name = f"{ckpt_prefix}_trial_{trial_id:03d}.npz"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=TRAIN.batch_size,
        weight_decay=float(sampled["weight_decay"]),
        checkpoint_dir=checkpoints_dir,
        checkpoint_name=trial_ckpt_name,
        seed=seed + trial_id,
        verbose=verbose,
    )

    history = trainer.fit()
    best_val_acc = float(np.max(history["val_acc"]))
    best_val_loss = float(np.min(history["val_loss"]))

    result = SearchResult(
        trial_id=trial_id,
        config=sampled,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
    )
    return result, checkpoints_dir / trial_ckpt_name


def _finalize_best_checkpoint(trial_ckpt_paths: List[Path], best_idx: int, checkpoints_dir: Path, final_name: str) -> Path:
    final_best_ckpt = checkpoints_dir / final_name
    best_trial_ckpt = trial_ckpt_paths[best_idx]
    if best_trial_ckpt.exists():
        shutil.copy2(best_trial_ckpt, final_best_ckpt)

    for path in trial_ckpt_paths:
        if path.exists():
            path.unlink(missing_ok=True)

    return final_best_ckpt


def run_random_search(
    num_trials: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    checkpoints_dir: Path,
    seed: int | None = None,
) -> List[SearchResult]:
    if num_trials <= 0:
        raise ValueError("num_trials must be > 0")

    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    trial_seed = SEARCH.seed if seed is None else seed
    epochs = SEARCH.epochs_per_trial if SEARCH.epochs_per_trial > 0 else TRAIN.epochs
    rng = np.random.default_rng(trial_seed)

    results: List[SearchResult] = []
    trial_ckpt_paths: List[Path] = []
    best_idx = -1
    best_acc = -np.inf

    for trial_id in range(1, num_trials + 1):
        sampled = _sample_hparams(rng)
        result, trial_ckpt = _run_single_trial(
            trial_id=trial_id,
            sampled=sampled,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            checkpoints_dir=checkpoints_dir,
            seed=trial_seed,
            epochs=epochs,
            ckpt_prefix="random_search",
            verbose=True,
        )
        results.append(result)
        trial_ckpt_paths.append(trial_ckpt)

        if result.best_val_acc > best_acc:
            best_acc = result.best_val_acc
            best_idx = trial_id - 1

        print(
            f"[Random Trial {trial_id:03d}/{num_trials}] "
            f"lr={sampled['learning_rate']:.6g}, "
            f"h1={int(sampled['hidden_dim1'])}, h2={int(sampled['hidden_dim2'])}, "
            f"wd={sampled['weight_decay']:.6g} -> best_val_acc={result.best_val_acc:.4f}"
        )

    if best_idx >= 0:
        final_path = _finalize_best_checkpoint(
            trial_ckpt_paths=trial_ckpt_paths,
            best_idx=best_idx,
            checkpoints_dir=checkpoints_dir,
            final_name="best_model_random.npz",
        )
        print(f"[Random Search] Best checkpoint saved to: {final_path}")

    return results


def run_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    checkpoints_dir: Path,
    seed: int | None = None,
) -> List[SearchResult]:
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    trial_seed = SEARCH.seed if seed is None else seed
    epochs = SEARCH.epochs_per_trial if SEARCH.epochs_per_trial > 0 else TRAIN.epochs
    combos = _grid_hparams()

    if not combos:
        raise ValueError("Grid search space is empty. Check SearchConfig grid fields.")

    results: List[SearchResult] = []
    trial_ckpt_paths: List[Path] = []
    best_idx = -1
    best_acc = -np.inf

    for trial_id, sampled in enumerate(combos, start=1):
        result, trial_ckpt = _run_single_trial(
            trial_id=trial_id,
            sampled=sampled,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            checkpoints_dir=checkpoints_dir,
            seed=trial_seed,
            epochs=epochs,
            ckpt_prefix="grid_search",
            verbose=True,
        )
        results.append(result)
        trial_ckpt_paths.append(trial_ckpt)

        if result.best_val_acc > best_acc:
            best_acc = result.best_val_acc
            best_idx = trial_id - 1

        print(
            f"[Grid Trial {trial_id:03d}/{len(combos)}] "
            f"lr={sampled['learning_rate']:.6g}, "
            f"h1={int(sampled['hidden_dim1'])}, h2={int(sampled['hidden_dim2'])}, "
            f"wd={sampled['weight_decay']:.6g} -> best_val_acc={result.best_val_acc:.4f}"
        )

    if best_idx >= 0:
        final_path = _finalize_best_checkpoint(
            trial_ckpt_paths=trial_ckpt_paths,
            best_idx=best_idx,
            checkpoints_dir=checkpoints_dir,
            final_name="best_model_grid.npz",
        )
        print(f"[Grid Search] Best checkpoint saved to: {final_path}")

    return results


def save_search_results(results: List[SearchResult], output_dir: Path, mode: str) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda r: r.best_val_acc, reverse=True)
    prefix = f"{mode.lower()}_search_results"

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"

    json_payload = [
        {
            "trial_id": r.trial_id,
            "learning_rate": r.config["learning_rate"],
            "hidden_dim1": int(r.config["hidden_dim1"]),
            "hidden_dim2": int(r.config["hidden_dim2"]),
            "weight_decay": r.config["weight_decay"],
            "best_val_acc": r.best_val_acc,
            "best_val_loss": r.best_val_loss,
        }
        for r in sorted_results
    ]
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial_id",
                "learning_rate",
                "hidden_dim1",
                "hidden_dim2",
                "weight_decay",
                "best_val_acc",
                "best_val_loss",
            ]
        )
        for r in sorted_results:
            writer.writerow(
                [
                    r.trial_id,
                    r.config["learning_rate"],
                    int(r.config["hidden_dim1"]),
                    int(r.config["hidden_dim2"]),
                    r.config["weight_decay"],
                    r.best_val_acc,
                    r.best_val_loss,
                ]
            )

    return json_path, csv_path
