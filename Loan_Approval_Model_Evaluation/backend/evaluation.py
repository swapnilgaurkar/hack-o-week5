from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluate_model(model, X_test, y_test, predictions):
    probabilities = model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="accuracy")

    cm = confusion_matrix(y_test, predictions)

    return {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "confusion_matrix": cm.tolist(),
        "cross_validation_mean": round(float(cv_scores.mean()), 4),
        "cross_validation_std": round(float(cv_scores.std()), 4)
    }
