"""
plots.py

Visualization utilities for evaluating classification models.

Functions:
    - plot_pr: Plot Precision–Recall curve.
    - plot_roc: Plot ROC curve.
    - plot_confusion: Plot confusion matrix heatmap.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix


def plot_pr(y_true, probs, title: str = "Precision–Recall Curve"):
    """
    Plot a Precision–Recall curve.

    Args:
        y_true (array-like): Ground truth binary labels (0/1).
        probs (array-like): Predicted probabilities.
        title (str, optional): Plot title (default: "Precision–Recall Curve").
    """
    p, r, _ = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(r, p, linewidth=2, color="navy")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_roc(y_true, probs, title: str = "ROC Curve"):
    """
    Plot a Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (array-like): Ground truth binary labels (0/1).
        probs (array-like): Predicted probabilities.
        title (str, optional): Plot title (default: "ROC Curve").
    """
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, linewidth=2, color="darkorange")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_confusion(y_true, preds, title: str = "Confusion Matrix"):
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true (array-like): Ground truth binary labels (0/1).
        preds (array-like): Predicted class labels (0/1).
        title (str, optional): Plot title (default: "Confusion Matrix").
    """
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                annot_kws={"size": 12})
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
