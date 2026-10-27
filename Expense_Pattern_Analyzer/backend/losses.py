"""Loss functions. Predictions are clipped so log(0) never happens."""
import numpy as np

EPS = 1e-12


def binary_cross_entropy(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.clip(np.asarray(y_prob, dtype=float).reshape(-1), EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


LOSSES = {"bce": binary_cross_entropy, "mse": mean_squared_error}
