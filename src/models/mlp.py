"""Three-layer MLP: parameter init, forward pass, backward pass, and cache."""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from src.models.activations import (
    relu,
    relu_backward,
    sigmoid,
    sigmoid_backward,
    tanh,
    tanh_backward,
)


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dim1: int
    hidden_dim2: int
    output_dim: int
    activation: str = "relu"


class ThreeLayerMLP:
    """Three-layer MLP with two configurable hidden layers and activation."""

    def __init__(self, cfg: MLPConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed

        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Dict[str, np.ndarray] = {}

        self._act_forward, self._act_backward = self._resolve_activation(cfg.activation)
        self._init_params()

    def _resolve_activation(
        self, activation: str
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        act = activation.lower()
        if act == "relu":
            return relu, relu_backward
        if act == "sigmoid":
            return sigmoid, sigmoid_backward
        if act == "tanh":
            return tanh, tanh_backward
        raise ValueError("Unsupported activation. Use one of: relu, sigmoid, tanh")

    def _init_params(self) -> None:
        rng = np.random.default_rng(self.seed)

        if self.cfg.activation.lower() == "relu":
            # He initialization is a good default for ReLU.
            w1_scale = np.sqrt(2.0 / self.cfg.input_dim)
            w2_scale = np.sqrt(2.0 / self.cfg.hidden_dim1)
            w3_scale = np.sqrt(2.0 / self.cfg.hidden_dim2)
        else:
            # Xavier initialization is usually safer for sigmoid/tanh.
            w1_scale = np.sqrt(1.0 / self.cfg.input_dim)
            w2_scale = np.sqrt(1.0 / self.cfg.hidden_dim1)
            w3_scale = np.sqrt(1.0 / self.cfg.hidden_dim2)

        self.params["W1"] = rng.standard_normal((self.cfg.input_dim, self.cfg.hidden_dim1)) * w1_scale
        self.params["b1"] = np.zeros(self.cfg.hidden_dim1, dtype=np.float32)

        self.params["W2"] = rng.standard_normal((self.cfg.hidden_dim1, self.cfg.hidden_dim2)) * w2_scale
        self.params["b2"] = np.zeros(self.cfg.hidden_dim2, dtype=np.float32)

        self.params["W3"] = rng.standard_normal((self.cfg.hidden_dim2, self.cfg.output_dim)) * w3_scale
        self.params["b3"] = np.zeros(self.cfg.output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass and store intermediate values for backward pass.

        Args:
            x: Input tensor of shape [batch_size, input_dim].

        Returns:
            logits: Output tensor of shape [batch_size, output_dim].
        """
        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = self._act_forward(z1)

        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = self._act_forward(z2)

        logits = a2 @ self.params["W3"] + self.params["b3"]

        self.cache["x"] = x
        self.cache["z1"] = z1
        self.cache["a1"] = a1
        self.cache["z2"] = z2
        self.cache["a2"] = a2
        self.cache["logits"] = logits

        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        """Backpropagate gradients from logits to all trainable parameters.

        Args:
            grad_logits: Gradient of loss w.r.t. logits, shape [batch_size, output_dim].
        """
        if not self.cache:
            raise RuntimeError("forward() must be called before backward().")

        x = self.cache["x"]
        z1 = self.cache["z1"]
        a1 = self.cache["a1"]
        z2 = self.cache["z2"]
        a2 = self.cache["a2"]

        dW3 = a2.T @ grad_logits
        db3 = np.sum(grad_logits, axis=0)

        da2 = grad_logits @ self.params["W3"].T
        dz2 = self._act_backward(da2, z2)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.params["W2"].T
        dz1 = self._act_backward(da1, z1)

        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.grads["W1"] = dW1
        self.grads["b1"] = db1
        self.grads["W2"] = dW2
        self.grads["b2"] = db2
        self.grads["W3"] = dW3
        self.grads["b3"] = db3

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict integer class ids for input batch."""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
