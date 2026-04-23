"""Activation functions: forward and backward for ReLU/Sigmoid/Tanh."""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    '''relu(x) = max(0, x)'''
    return np.maximum(0, x)


def relu_backward(grad_out: np.ndarray, x: np.ndarray) -> np.ndarray:
    '''relu_backward(grad_out, x) = grad_out * (x > 0)'''
    return grad_out * (x > 0)
    

def sigmoid(x: np.ndarray) -> np.ndarray:
    '''sigmoid(x) = 1 / (1 + exp(-x))'''
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(grad_out: np.ndarray, x: np.ndarray) -> np.ndarray:
    '''sigmoid_backward(grad_out, x) = grad_out * sigmoid(x) * (1 - sigmoid(x))'''
    s = sigmoid(x)
    return grad_out * s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    '''tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))'''
    return np.tanh(x)


def tanh_backward(grad_out: np.ndarray, x: np.ndarray) -> np.ndarray:
    '''tanh_backward(grad_out, x) = grad_out * (1 - tanh(x) ** 2)'''
    return grad_out * (1 - np.tanh(x) ** 2)
