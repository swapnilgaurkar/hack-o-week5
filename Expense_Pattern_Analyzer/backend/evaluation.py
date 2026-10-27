"""Train/validation/test split and classification metrics."""
import numpy as np


def split_data(X, y, val_frac=0.15, test_frac=0.15, seed=0):
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    pred = (np.asarray(y_prob).reshape(-1) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}, "n_samples": int(len(y_true))}
