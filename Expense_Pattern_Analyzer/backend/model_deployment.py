"""Training pipeline, model saving/loading (Joblib) and the in-memory model service.

SECURITY: joblib uses pickle internally. Only the app's OWN local model file is ever
loaded; there is no upload endpoint and the path is fixed by the server.
"""
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .evaluation import classification_metrics, split_data
from .neural_network import NeuralNetwork
from .optimizers import make_optimizer
from .preprocessing import Standardizer
from .synthetic_data import FEATURES, generate_dataset

FORMAT_VERSION = 1
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "budget_model.joblib"


def risk_label(p):
    return "low" if p < 0.33 else ("moderate" if p < 0.66 else "high")


def train_model(optimizer="adam", activation="relu", learning_rate=0.01, epochs=100,
                n_hidden=8, n_samples=3000, seed=42):
    """Train on synthetic data. Returns (net, scaler, metadata)."""
    X, y = generate_dataset(n_samples, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_data(X, y, seed=seed)
    scaler = Standardizer().fit(Xtr)                      # fit on TRAIN only
    Xtr_s, Xva_s, Xte_s = (scaler.transform(a) for a in (Xtr, Xva, Xte))
    net = NeuralNetwork(len(FEATURES), n_hidden, activation, seed)
    opt = make_optimizer(optimizer, learning_rate)
    before = net.get_params()
    hist = net.fit(Xtr_s, ytr, opt, epochs, 64, "bce", Xva_s, yva, seed)
    changed = any(not np.allclose(before[k], net.params[k]) for k in before)
    meta = {
        "optimizer": opt.name, "activation": activation, "learning_rate": float(learning_rate),
        "epochs": int(epochs), "n_hidden": int(n_hidden), "seed": int(seed),
        "loss": "binary_cross_entropy", "data": "synthetic",
        "split": {"train": len(Xtr), "validation": len(Xva), "test": len(Xte)},
        "loss_history": hist["train_loss"], "val_loss_history": hist["val_loss"],
        "initial_loss": hist["train_loss"][0], "final_loss": hist["train_loss"][-1],
        "metrics": classification_metrics(yte, net.predict_proba(Xte_s)),
        "metrics_dataset": "held-out synthetic test split",
        "weights_updated": bool(changed),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return net, scaler, meta


def save_model(path, net, scaler, meta):
    """Save a versioned dict of arrays + config (not the live object) atomically."""
    payload = {
        "format_version": FORMAT_VERSION,
        "architecture": {"n_inputs": net.n_inputs, "n_hidden": net.n_hidden, "n_outputs": 1,
                         "output_activation": "sigmoid"},
        "hidden_activation": net.activation,
        "params": net.get_params(),
        "feature_order": list(scaler.features),
        "scaler": scaler.to_dict(),
        "metadata": meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return str(path)


def load_model(path):
    """Load the app's own model file. Returns (net, scaler, meta)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("No saved model found. Train and save a model first.")
    d = joblib.load(path)
    if not isinstance(d, dict) or d.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported or corrupt model file")
    if d["feature_order"] != FEATURES:
        raise ValueError("Saved model feature order does not match the application")
    a = d["architecture"]
    net = NeuralNetwork(a["n_inputs"], a["n_hidden"], d["hidden_activation"])
    net.set_params(d["params"])
    return net, Standardizer.from_dict(d["scaler"]), d["metadata"]


class ModelService:
    """Holds the current model in memory (thread-safe)."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.net = self.scaler = self.meta = None
        self.saved = False
        self.lock = threading.Lock()

    def train(self, **cfg):
        net, scaler, meta = train_model(**cfg)
        with self.lock:
            self.net, self.scaler, self.meta, self.saved = net, scaler, meta, False
        return meta

    def save(self):
        with self.lock:
            if self.net is None:
                raise RuntimeError("No trained model to save. Train a model first.")
            save_model(self.model_path, self.net, self.scaler, self.meta)
            self.saved = True

    def load(self):
        net, scaler, meta = load_model(self.model_path)
        with self.lock:
            self.net, self.scaler, self.meta, self.saved = net, scaler, meta, True
        return meta

    def predict(self, features):
        with self.lock:
            if self.net is None:
                raise RuntimeError("No model available. Train or load a model first.")
            x = np.array([[features[f] for f in FEATURES]], dtype=float)
            return float(self.net.predict_proba(self.scaler.transform(x))[0, 0])

    def status(self):
        with self.lock:
            return {"trained": self.net is not None, "saved_in_memory_matches_disk": self.saved,
                    "model_file_exists": self.model_path.is_file(),
                    "metadata": self.meta}
