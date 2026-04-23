"""Trainer module: epoch loop, validation, logging, and best-checkpoint saving."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.dataset import iterate_minibatches
from src.models.losses import l2_regularization, softmax_cross_entropy
from src.models.mlp import ThreeLayerMLP
from src.training.optimizer import SGD
from src.eval.metrics import accuracy


class Trainer:
    """End-to-end trainer for ThreeLayerMLP.

    This class handles:
    - mini-batch SGD training
    - cross-entropy + optional L2 regularization
    - validation loss/accuracy tracking
    - best-checkpoint saving by validation accuracy
    - learning-rate decay per epoch
    """

    def __init__(
        self,
        model: ThreeLayerMLP,
        optimizer: SGD,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        weight_decay: float,
        checkpoint_dir: Path,
        checkpoint_name: str = "best_model.npz",
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of samples")
        if x_val.shape[0] != y_val.shape[0]:
            raise ValueError("x_val and y_val must have the same number of samples")
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.model = model
        self.optimizer = optimizer

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.verbose = verbose

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / checkpoint_name

        self.best_val_acc: float = -np.inf
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

    def _compute_l2(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute L2 penalty and merge its gradients into current grads."""
        if self.weight_decay <= 0.0:
            return 0.0, grads

        l2_penalty, l2_grads = l2_regularization(params, self.weight_decay)
        merged = {k: v.copy() for k, v in grads.items()}
        for key, g in l2_grads.items():
            if key in merged:
                merged[key] = merged[key] + g
        return float(l2_penalty), merged

    def _train_one_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """Run one epoch over training set and update model parameters."""
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        batch_seed = self.seed + epoch_idx
        for x_batch, y_batch in iterate_minibatches(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            shuffle=True,
            seed=batch_seed,
        ):
            logits = self.model.forward(x_batch)
            ce_loss, grad_logits = softmax_cross_entropy(logits, y_batch)
            self.model.backward(grad_logits)

            l2_penalty, merged_grads = self._compute_l2(self.model.params, self.model.grads)
            batch_loss = ce_loss + l2_penalty

            self.optimizer.step(self.model.params, merged_grads)

            pred = np.argmax(logits, axis=1)
            total_loss += float(batch_loss) * x_batch.shape[0]
            total_correct += int(np.sum(pred == y_batch))
            total_seen += x_batch.shape[0]

        avg_loss = total_loss / total_seen
        avg_acc = total_correct / total_seen
        return float(avg_loss), float(avg_acc)

    def _evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate loss and accuracy on a full dataset split."""
        logits = self.model.forward(x)
        ce_loss, _ = softmax_cross_entropy(logits, y)
        l2_penalty, _ = l2_regularization(self.model.params, self.weight_decay) if self.weight_decay > 0 else (0.0, {})
        loss = float(ce_loss + l2_penalty)

        pred = np.argmax(logits, axis=1)
        acc = accuracy(y, pred)
        return loss, acc

    def fit(self) -> Dict[str, List[float]]:
        """Run full training loop and return history dict."""
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch_idx=epoch)
            val_loss, val_acc = self._evaluate(self.x_val, self.y_val)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(float(self.optimizer.cfg.learning_rate))

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_best_checkpoint(self.best_checkpoint_path)

            if self.verbose:
                print(
                    f"[Epoch {epoch:03d}/{self.epochs}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                    f"lr={self.optimizer.cfg.learning_rate:.6f}"
                )

            self.optimizer.decay_lr()

        if self.verbose:
            print(f"Best val_acc={self.best_val_acc:.4f}, checkpoint={self.best_checkpoint_path}")

        return self.history

    def save_best_checkpoint(self, path: Optional[Path] = None) -> None:
        """Save current model params to .npz checkpoint."""
        save_path = Path(path) if path is not None else self.best_checkpoint_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_path,
            W1=self.model.params["W1"],
            b1=self.model.params["b1"],
            W2=self.model.params["W2"],
            b2=self.model.params["b2"],
            W3=self.model.params["W3"],
            b3=self.model.params["b3"],
            best_val_acc=np.array([self.best_val_acc], dtype=np.float32),
        )

    def load_checkpoint(self, path: Optional[Path] = None) -> None:
        """Load model params from .npz checkpoint into current model."""
        ckpt_path = Path(path) if path is not None else self.best_checkpoint_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = np.load(ckpt_path)
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            self.model.params[key] = ckpt[key]

        if "best_val_acc" in ckpt:
            self.best_val_acc = float(np.asarray(ckpt["best_val_acc"]).reshape(-1)[0])
