"""
metrics.py

Utility functions for evaluating **classification models**.

Functions:
    - classify_metrics: Compute common classification metrics (accuracy, precision, recall, F1, ROC-AUC)
      at a specified probability threshold.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def classify_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float = 0.5) -> dict:
    """
    Compute classification metrics at a given probability threshold.

    Args:
        y_true (np.ndarray): Ground truth binary labels (0/1).
        probs (np.ndarray): Predicted probabilities (values between 0 and 1).
        thr (float, optional): Classification threshold (default: 0.5).

    Returns:
        dict: Dictionary containing:
            - threshold: float, probability threshold used
            - accuracy: float, overall accuracy
            - precision: float, proportion of predicted positives that are correct
            - recall: float, proportion of true positives correctly identified
            - f1: float, harmonic mean of precision and recall
            - roc_auc: float, area under ROC curve (threshold-independent)
    """
    # Convert probabilities to binary predictions using threshold
    preds = (probs >= thr).astype(int)

    # Compute metrics
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs))
    }
