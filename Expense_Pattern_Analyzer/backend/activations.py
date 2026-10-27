"""Activation functions and their derivatives (NumPy only)."""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid: squashes any number into (0, 1)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_derivative(a):
    """Derivative expressed using the sigmoid OUTPUT a = sigmoid(z)."""
    return a * (1.0 - a)


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    """Derivative w.r.t. the pre-activation z."""
    return (z > 0).astype(float)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    """Derivative w.r.t. the pre-activation z."""
    return 1.0 - np.tanh(z) ** 2


# name -> (function, derivative that takes the pre-activation z)
ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
}
