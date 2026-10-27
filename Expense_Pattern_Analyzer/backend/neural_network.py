"""Fully connected network (inputs -> hidden -> 1 sigmoid output) in pure NumPy."""
import numpy as np

from .activations import ACTIVATIONS, sigmoid
from .losses import LOSSES


class NeuralNetwork:
    def __init__(self, n_inputs=5, n_hidden=8, activation="relu", seed=42):
        if activation not in ACTIVATIONS:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.n_inputs, self.n_hidden = int(n_inputs), int(n_hidden)
        self.activation = activation
        rng = np.random.default_rng(seed)
        # He init for ReLU, Xavier init for Tanh; biases start at zero.
        s1 = np.sqrt((2.0 if activation == "relu" else 1.0) / self.n_inputs)
        s2 = np.sqrt(1.0 / self.n_hidden)
        self.params = {
            "W1": rng.normal(0, s1, (self.n_inputs, self.n_hidden)),
            "b1": np.zeros((1, self.n_hidden)),
            "W2": rng.normal(0, s2, (self.n_hidden, 1)),
            "b2": np.zeros((1, 1)),
        }

    # ---- forward propagation -------------------------------------------
    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_inputs:
            raise ValueError(f"X must have shape (n, {self.n_inputs})")
        act = ACTIVATIONS[self.activation][0]
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = act(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = sigmoid(z2)
        return a2, {"X": X, "z1": z1, "a1": a1, "a2": a2}

    def predict_proba(self, X):
        return self.forward(X)[0]

    # ---- backpropagation -----------------------------------------------
    def backward(self, y, cache, loss="bce"):
        m = cache["X"].shape[0]
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if y.shape[0] != m:
            raise ValueError("y and X must have the same number of rows")
        a2 = cache["a2"]
        if loss == "bce":
            dz2 = (a2 - y) / m                      # BCE + sigmoid simplifies to (p - y)
        elif loss == "mse":
            dz2 = 2.0 * (a2 - y) * a2 * (1 - a2) / m
        else:
            raise ValueError("loss must be 'bce' or 'mse'")
        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * ACTIVATIONS[self.activation][1](cache["z1"])
        grads = {"W1": cache["X"].T @ dz1, "b1": dz1.sum(axis=0, keepdims=True),
                 "W2": dW2, "b2": db2}
        for k, g in grads.items():
            if g.shape != self.params[k].shape:
                raise ValueError(f"gradient shape mismatch for {k}")
            if not np.all(np.isfinite(g)):
                raise FloatingPointError(f"non-finite gradient for {k}")
        return grads

    # ---- training ------------------------------------------------------
    def fit(self, X, y, optimizer, epochs=100, batch_size=64, loss="bce",
            X_val=None, y_val=None, seed=0):
        """Mini-batch training. Returns {'train_loss': [...], 'val_loss': [...]}."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        loss_fn = LOSSES[loss]
        rng = np.random.default_rng(seed)
        hist = {"train_loss": [], "val_loss": []}
        n = X.shape[0]
        for _ in range(int(epochs)):
            idx = rng.permutation(n)                # shuffle every epoch
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                _, cache = self.forward(X[b])
                optimizer.step(self.params, self.backward(y[b], cache, loss))
            hist["train_loss"].append(loss_fn(y, self.predict_proba(X)))
            if X_val is not None:
                hist["val_loss"].append(loss_fn(y_val, self.predict_proba(X_val)))
        if not all(np.all(np.isfinite(p)) for p in self.params.values()):
            raise FloatingPointError("training diverged (try a smaller learning rate)")
        return hist

    def get_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params):
        for k in self.params:
            arr = np.asarray(params[k], dtype=float)
            if arr.shape != self.params[k].shape:
                raise ValueError(f"saved parameter {k} has wrong shape")
            self.params[k] = arr.copy()
