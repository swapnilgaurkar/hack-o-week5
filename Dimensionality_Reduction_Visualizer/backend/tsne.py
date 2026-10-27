import numpy as np
from sklearn.manifold import TSNE

from preprocessing import standardized_numeric_data, get_class_values


def safe_perplexity(requested, n_samples):
    try:
        value = float(requested)
    except (TypeError, ValueError):
        value = 30.0

    # sklearn requires perplexity < number of samples.
    # Keep a useful range for an educational visualization.
    upper = max(1.0, n_samples - 1.0)
    value = min(max(value, 2.0), upper)

    # For very tiny datasets, use the largest valid value.
    if n_samples <= 3:
        value = max(1.0, min(value, n_samples - 1.0))

    return value


def run_tsne(df, class_column=None, perplexity=30):
    X, feature_names = standardized_numeric_data(df)
    n_samples = X.shape[0]

    if n_samples < 3:
        raise ValueError("t-SNE needs at least 3 data rows.")

    if X.shape[1] < 1:
        raise ValueError("At least one numerical feature is required for t-SNE.")

    actual_perplexity = safe_perplexity(perplexity, n_samples)

    # Keep the method stable and reproducible for a classroom demo.
    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
        max_iter=1000
    )

    reduced = tsne.fit_transform(X)

    return {
        "x": reduced[:, 0].round(6).tolist(),
        "y": reduced[:, 1].round(6).tolist(),
        "feature_names": feature_names,
        "perplexity": round(float(actual_perplexity), 2),
        "class_values": get_class_values(df, class_column),
        "x_label": "Dimension 1",
        "y_label": "Dimension 2"
    }
