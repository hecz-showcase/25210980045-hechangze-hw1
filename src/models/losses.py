"""Loss functions: cross-entropy and L2 regularization with gradients."""

from typing import Dict, Tuple

import numpy as np


def softmax_cross_entropy(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return cross-entropy loss and softmax gradient for given logits and true labels."""
    # Compute softmax probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy loss
    loss = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-15))

    # Compute gradient
    grad = probs.copy()
    grad[np.arange(len(y_true)), y_true] -= 1
    grad /= len(y_true)

    return loss, grad


def l2_regularization(params: Dict[str, np.ndarray], weight_decay: float) -> Tuple[float, Dict[str, np.ndarray]]:
    """Return L2 penalty and gradients for each weight matrix."""
    penalty = 0.0
    gradients = {key: np.zeros_like(param) for key, param in params.items()}

    for key, param in params.items():
        if key.startswith('W'):  # Only apply L2 to weight matrices, not biases
            penalty += 0.5 * weight_decay * np.sum(param ** 2)
            gradients[key] = weight_decay * param

    return penalty, gradients
