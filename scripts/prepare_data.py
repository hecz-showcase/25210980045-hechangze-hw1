"""Data preparation entrypoint: load EuroSAT_RGB, split train/val/test, and save to disk."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PREP, PATHS
from src.data.dataset import prepare_and_save_splits


def main() -> None:
    split, npz_path, meta_path = prepare_and_save_splits(
        data_dir=PATHS.data_dir,
        output_dir=PATHS.outputs_dir,
        train_ratio=DATA_PREP.train_ratio,
        val_ratio=DATA_PREP.val_ratio,
        seed=DATA_PREP.seed,
        image_size=(DATA_PREP.image_size, DATA_PREP.image_size),
        flatten=DATA_PREP.flatten,
        normalize=DATA_PREP.normalize,
        file_prefix=DATA_PREP.file_prefix,
        show_progress=DATA_PREP.show_progress,
        progress_every=DATA_PREP.progress_every,
    )

    print("Data preparation completed.")
    print(f"train: {split.x_train.shape[0]} samples")
    print(f"val:   {split.x_val.shape[0]} samples")
    print(f"test:  {split.x_test.shape[0]} samples")
    print(f"classes: {len(split.class_to_idx)}")
    print(f"saved arrays: {npz_path}")
    print(f"saved meta:   {meta_path}")


if __name__ == "__main__":
    main()
