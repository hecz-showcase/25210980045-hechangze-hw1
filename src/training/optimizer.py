"""Optimizer module: SGD updates and learning-rate decay strategy."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SGDConfig:
    learning_rate: float
    lr_decay_gamma: float 

class SGD:
    """
    Vanilla SGD optimizer.
    """
    def __init__(self, cfg: SGDConfig) -> None:
        self.cfg = cfg

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """Run one SGD parameter update using provided gradients."""
        for key, param in params.items():
            if key not in grads:
                raise KeyError(f"Missing gradient for parameter: {key}")
            param -= self.cfg.learning_rate * grads[key]

    def decay_lr(self) -> None:
        """Apply multiplicative learning-rate decay."""
        self.cfg.learning_rate *= self.cfg.lr_decay_gamma
