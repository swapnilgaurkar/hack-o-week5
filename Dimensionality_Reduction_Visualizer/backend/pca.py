import numpy as np
from sklearn.decomposition import PCA

from preprocessing import standardized_numeric_data, get_class_values


def run_pca(df, class_column=None):
    X, feature_names = standardized_numeric_data(df)

    if X.shape[0] < 2:
        raise ValueError("PCA needs at least 2 data rows.")

    # PCA needs at least two dimensions for a 2D visualization.
    if X.shape[1] < 2:
        raise ValueError(
            "PCA needs at least 2 numerical features to create a 2D visualization."
        )

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    return {
        "x": reduced[:, 0].round(6).tolist(),
        "y": reduced[:, 1].round(6).tolist(),
        "feature_names": feature_names,
        "explained_variance": [round(float(v) * 100, 2) for v in explained],
        "cumulative_variance": [round(float(v) * 100, 2) for v in cumulative],
        "class_values": get_class_values(df, class_column),
        "x_label": "PC1",
        "y_label": "PC2"
    }
