"""Tester module: load best checkpoint and evaluate on test set."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.eval.metrics import accuracy, confusion_matrix
from src.models.mlp import MLPConfig, ThreeLayerMLP


class Tester:
    """Tester for inference and metrics on test data."""

    def __init__(self, checkpoint_path: Path, activation: str = "relu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.activation = activation
        self.model: Optional[ThreeLayerMLP] = None

    def _build_model_from_checkpoint(self, ckpt: np.lib.npyio.NpzFile) -> ThreeLayerMLP:
        required = ("W1", "b1", "W2", "b2", "W3", "b3")
        for key in required:
            if key not in ckpt:
                raise KeyError(f"Checkpoint missing parameter: {key}")

        w1 = ckpt["W1"]
        w2 = ckpt["W2"]
        w3 = ckpt["W3"]

        cfg = MLPConfig(
            input_dim=int(w1.shape[0]),
            hidden_dim1=int(w1.shape[1]),
            hidden_dim2=int(w2.shape[1]),
            output_dim=int(w3.shape[1]),
            activation=self.activation,
        )
        model = ThreeLayerMLP(cfg, seed=0)

        # Overwrite randomly initialized params with trained checkpoint params.
        for key in required:
            model.params[key] = ckpt[key]

        return model

    def load_checkpoint(self) -> None:
        """Load best checkpoint and reconstruct model from saved weights."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        ckpt = np.load(self.checkpoint_path)
        self.model = self._build_model_from_checkpoint(ckpt)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Run test evaluation and return accuracy and confusion matrix."""
        if self.model is None:
            self.load_checkpoint()
        assert self.model is not None

        y_pred = self.model.predict(x_test)
        num_classes = int(max(np.max(y_test), np.max(y_pred)) + 1)

        acc = accuracy(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, num_classes=num_classes)

        return {
            "accuracy": np.array(acc, dtype=np.float32),
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }


def load_test_split(split_npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load x_test and y_test from preprocessed split file."""
    split_npz_path = Path(split_npz_path)
    if not split_npz_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_npz_path}")

    data = np.load(split_npz_path)
    if "x_test" not in data or "y_test" not in data:
        raise KeyError("Split file must contain x_test and y_test")

    return data["x_test"], data["y_test"]


def load_class_names(meta_json_path: Path) -> List[str]:
    """Load class names from split metadata JSON (sorted by class index)."""
    meta_json_path = Path(meta_json_path)
    if not meta_json_path.exists():
        return []

    import json

    meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
    class_to_idx = meta.get("class_to_idx", {})
    if not class_to_idx:
        return []

    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]
