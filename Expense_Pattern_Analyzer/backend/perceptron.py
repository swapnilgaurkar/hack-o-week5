"""A single artificial neuron (perceptron) for teaching."""
import numpy as np


class Perceptron:
    """output = activation(weights . inputs + bias)."""

    def __init__(self, weights, bias=0.0, activation="step"):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        if activation not in ("step", "sigmoid"):
            raise ValueError("activation must be 'step' or 'sigmoid'")
        self.activation = activation

    def weighted_sum(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        if inputs.shape != self.weights.shape:
            raise ValueError("inputs and weights must have the same length")
        return float(np.dot(self.weights, inputs) + self.bias)

    def predict(self, inputs):
        z = self.weighted_sum(inputs)
        if self.activation == "step":
            return 1.0 if z >= 0 else 0.0
        return float(1.0 / (1.0 + np.exp(-z)))


def demo(inputs=(0.8, 0.3), weights=(0.9, -0.5), bias=-0.2):
    """Return the numbers behind one perceptron decision (used by the Model Lab)."""
    p = Perceptron(weights, bias, "step")
    z = p.weighted_sum(inputs)
    return {"inputs": list(inputs), "weights": list(weights), "bias": bias,
            "weighted_sum": z, "output": p.predict(inputs)}
