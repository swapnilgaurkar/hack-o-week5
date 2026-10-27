"""Feature scaling. Statistics are learned from TRAINING data only."""
import numpy as np

from .synthetic_data import FEATURES


class Standardizer:
    def __init__(self, mean=None, std=None, features=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)
        self.features = list(features or FEATURES)

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=float)
        self.mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        self.std = np.where(std < 1e-12, 1.0, std)   # avoid divide-by-zero
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("Standardizer is not fitted")
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "features": self.features}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"], d["features"])
