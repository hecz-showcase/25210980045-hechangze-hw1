"""Data loader and preprocessing for EuroSAT_RGB with train/val/test split."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_to_idx: Dict[str, int]


def _list_classes(data_dir: Path) -> Tuple[Path, ...]:
    classes = tuple(sorted(p for p in data_dir.iterdir() if p.is_dir()))
    if not classes:
        raise ValueError(f"No class subdirectories found under: {data_dir}")
    return classes


def _list_images_for_class(class_dir: Path) -> List[Path]:
    image_paths = sorted(class_dir.glob("*.jpg"))
    image_paths += sorted(class_dir.glob("*.jpeg"))
    image_paths += sorted(class_dir.glob("*.png"))
    return image_paths


def load_eurosat_rgb(
    data_dir: Path,
    image_size: Tuple[int, int] = (64, 64),
    flatten: bool = True,
    normalize: bool = True,
    show_progress: bool = False,
    progress_every: int = 500,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Read EuroSAT RGB images and return features, labels, and class mapping."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    class_dirs = _list_classes(data_dir)
    class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}

    class_to_images: Dict[str, List[Path]] = {}
    total_images = 0
    for class_dir in class_dirs:
        paths = _list_images_for_class(class_dir)
        class_to_images[class_dir.name] = paths
        total_images += len(paths)

    if total_images == 0:
        raise ValueError(f"No images found under: {data_dir}")

    x_list = []
    y_list = []
    loaded = 0
    started_at = time.time()

    if show_progress:
        print(f"[Data] Start loading from: {data_dir}")
        print(f"[Data] Classes: {len(class_dirs)}, images: {total_images}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_id = class_to_idx[class_name]
        image_paths = class_to_images[class_name]

        if not image_paths:
            continue

        if show_progress:
            print(f"[Data] Loading class '{class_name}' ({len(image_paths)} images)...")

        for image_path in image_paths:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if img.size != image_size:
                    img = img.resize(image_size, Image.BILINEAR)

                arr = np.asarray(img, dtype=np.float32)
                if normalize:
                    arr /= 255.0
                if flatten:
                    arr = arr.reshape(-1)

                x_list.append(arr)
                y_list.append(class_id)

            loaded += 1
            if show_progress and (loaded % max(1, progress_every) == 0 or loaded == total_images):
                pct = (loaded / total_images) * 100.0
                elapsed = time.time() - started_at
                print(f"[Data] Loaded {loaded}/{total_images} ({pct:.2f}%), elapsed {elapsed:.1f}s")

    x = np.stack(x_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    if show_progress:
        elapsed = time.time() - started_at
        print(f"[Data] Done. Shape x={x.shape}, y={y.shape}, elapsed {elapsed:.1f}s")

    return x, y, class_to_idx


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> SplitData:
    """Create stratified train/val/test split.

    Test ratio is computed as: 1 - train_ratio - val_ratio.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = np.random.default_rng(seed)
    train_idx_parts = []
    val_idx_parts = []
    test_idx_parts = []

    classes = np.unique(y)
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = cls_idx.size

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))

        if n >= 3 and (n_train + n_val) >= n:
            n_val = max(0, n_val - 1)

        train_part = cls_idx[:n_train]
        val_part = cls_idx[n_train : n_train + n_val]
        test_part = cls_idx[n_train + n_val :]

        train_idx_parts.append(train_part)
        val_idx_parts.append(val_part)
        test_idx_parts.append(test_part)

    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    test_idx = np.concatenate(test_idx_parts)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    class_to_idx = {str(int(cls)): int(cls) for cls in classes}
    return SplitData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
        class_to_idx=class_to_idx,
    )


def save_split_data(
    split_data: SplitData,
    output_dir: Path,
    file_prefix: str = "eurosat_split",
) -> Tuple[Path, Path]:
    """Save split arrays to .npz and metadata to .json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{file_prefix}.npz"
    meta_path = output_dir / f"{file_prefix}_meta.json"

    np.savez_compressed(
        npz_path,
        x_train=split_data.x_train,
        y_train=split_data.y_train,
        x_val=split_data.x_val,
        y_val=split_data.y_val,
        x_test=split_data.x_test,
        y_test=split_data.y_test,
    )

    metadata = {
        "class_to_idx": split_data.class_to_idx,
        "num_train": int(split_data.x_train.shape[0]),
        "num_val": int(split_data.x_val.shape[0]),
        "num_test": int(split_data.x_test.shape[0]),
        "feature_dim": int(split_data.x_train.shape[1]) if split_data.x_train.ndim == 2 else None,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return npz_path, meta_path


def prepare_and_save_splits(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    image_size: Tuple[int, int] = (64, 64),
    flatten: bool = True,
    normalize: bool = True,
    file_prefix: str = "eurosat_split",
    show_progress: bool = False,
    progress_every: int = 500,
) -> Tuple[SplitData, Path, Path]:
    """End-to-end utility: load data, split dataset, and save to disk."""
    x, y, class_to_idx = load_eurosat_rgb(
        data_dir=data_dir,
        image_size=image_size,
        flatten=flatten,
        normalize=normalize,
        show_progress=show_progress,
        progress_every=progress_every,
    )

    split = split_dataset(
        x=x,
        y=y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    split.class_to_idx = class_to_idx

    npz_path, meta_path = save_split_data(split, output_dir=output_dir, file_prefix=file_prefix)
    return split, npz_path, meta_path


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield mini-batches for SGD training."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    indices = np.arange(x.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
