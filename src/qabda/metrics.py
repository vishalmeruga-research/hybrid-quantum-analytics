from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return out
