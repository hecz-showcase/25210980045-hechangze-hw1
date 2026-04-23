"""Search entrypoint: run random/grid search and record hyperparameter performance."""

import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PREP, PATHS, SEARCH, TRAIN
from src.search.hparam_search import run_grid_search, run_random_search, save_search_results


def load_preprocessed_splits() -> Dict[str, np.ndarray]:
    split_path = PATHS.outputs_dir / f"{DATA_PREP.file_prefix}.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}. Run scripts/prepare_data.py first.")

    data = np.load(split_path)
    required = ("x_train", "y_train", "x_val", "y_val")
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key in split file: {key}")

    return {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "x_val": data["x_val"],
        "y_val": data["y_val"],
    }


def main() -> None:
    split = load_preprocessed_splits()
    mode = SEARCH.mode.lower().strip()
    epochs_per_trial = SEARCH.epochs_per_trial if SEARCH.epochs_per_trial > 0 else TRAIN.epochs

    print(
        f"[Search] mode={mode}, epochs_per_trial={epochs_per_trial}, "
        f"batch_size={TRAIN.batch_size}, seed={SEARCH.seed}"
    )

    if mode == "grid":
        total = len(SEARCH.grid_learning_rates) * len(SEARCH.grid_hidden_dim_pairs) * len(SEARCH.grid_weight_decays)
        print(f"[Search] Grid trials={total}")
        results = run_grid_search(
            x_train=split["x_train"],
            y_train=split["y_train"],
            x_val=split["x_val"],
            y_val=split["y_val"],
            checkpoints_dir=PATHS.checkpoints_dir,
            seed=SEARCH.seed,
        )
    elif mode == "random":
        print(f"[Search] Random trials={SEARCH.num_trials}")
        results = run_random_search(
            num_trials=SEARCH.num_trials,
            x_train=split["x_train"],
            y_train=split["y_train"],
            x_val=split["x_val"],
            y_val=split["y_val"],
            checkpoints_dir=PATHS.checkpoints_dir,
            seed=SEARCH.seed,
        )
    else:
        raise ValueError("SEARCH.mode must be 'grid' or 'random'")

    json_path, csv_path = save_search_results(results, PATHS.outputs_dir / "logs", mode=mode)

    best = max(results, key=lambda r: r.best_val_acc)
    print("[Search] Finished.")
    print(
        f"[Search] Best trial={best.trial_id}, best_val_acc={best.best_val_acc:.4f}, "
        f"lr={best.config['learning_rate']:.6g}, "
        f"h1={int(best.config['hidden_dim1'])}, h2={int(best.config['hidden_dim2'])}, "
        f"wd={best.config['weight_decay']:.6g}"
    )
    print(f"[Search] Results JSON: {json_path}")
    print(f"[Search] Results CSV:  {csv_path}")


if __name__ == "__main__":
    main()
