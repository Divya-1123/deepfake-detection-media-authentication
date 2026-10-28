"""Evaluation metrics named in the synopsis: accuracy, precision, recall, F1, confusion matrix (+ ROC-AUC, specificity)."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def compute_metrics(y_true, fake_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(fake_prob)
    y_pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])           # rows = actual [real, fake], cols = predicted
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    out = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,   # share of REAL images correctly called real
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, prob)) if len(set(y_true.tolist())) == 2 else None
    return out


def best_threshold(y_true, fake_prob) -> float:
    """Threshold maximising balanced accuracy (TPR + TNR - 1), robust to real/fake class imbalance."""
    y_true, prob = np.asarray(y_true), np.asarray(fake_prob)
    best_t, best_j = 0.5, -1.0
    for t in np.unique(np.round(prob, 4)):
        pred = prob >= t
        tpr = (pred & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
        tnr = (~pred & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
        if tpr + tnr - 1 > best_j:
            best_j, best_t = tpr + tnr - 1, float(t)
    return best_t
