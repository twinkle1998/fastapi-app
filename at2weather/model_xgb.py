"""
model_xgb.py

XGBoost classifier training utilities for predicting
"Will it rain in the next 7 days?" (binary classification).

Functions:
    - train_xgb_classifier: Train XGBoost classifier with GridSearchCV (time-series aware).
    - pick_thresholds: Select probability thresholds based on F1 score and recall floor.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier


def train_xgb_classifier(X_train, y_train, random_state: int = 42):
    """
    Train an XGBoost classifier with hyperparameter tuning using GridSearchCV.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training feature matrix.
        y_train (pd.Series or np.ndarray): Binary training labels (0/1).
        random_state (int, optional): Random seed (default: 42).

    Returns:
        tuple:
            - best_estimator_ (XGBClassifier): Trained best model.
            - best_params_ (dict): Best hyperparameters from grid search.
    """
    # Handle class imbalance (scale positive weight)
    neg, pos = np.bincount(y_train)
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

    # Base model
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",       # AUC-PR is robust to class imbalance
        tree_method="hist",        # Efficient tree construction
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    # Hyperparameter search space
    param_grid = {
        "n_estimators": [400, 800],
        "learning_rate": [0.01, 0.03],
        "max_depth": [4, 6],
        "min_child_weight": [1, 2],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [1, 5],
        "reg_alpha": [0, 0.5]
    }

    # TimeSeriesSplit preserves chronological order for validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Grid search (optimize for F1 score)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=tscv,
        n_jobs=-1,
        verbose=2
    )

    # Fit and return best model
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def pick_thresholds(y_true, probs, recall_floor: float = 0.90):
    """
    Determine probability thresholds for classification.

    Args:
        y_true (np.ndarray): Ground truth labels.
        probs (np.ndarray): Predicted probabilities.
        recall_floor (float, optional): Minimum recall required for threshold A (default: 0.90).

    Returns:
        tuple:
            - bestA (dict): Threshold with recall >= recall_floor and best F1.
            - bestB (dict): Threshold that maximizes F1 score overall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)

    # Compute F1 scores for all thresholds
    f1s = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6)

    # Threshold B: maximize F1 score
    b_idx = int(np.argmax(f1s))
    bestB = {
        "thr": float(thresholds[b_idx]),
        "f1": float(f1s[b_idx]),
        "prec": float(precisions[b_idx]),
        "rec": float(recalls[b_idx])
    }

    # Threshold A: satisfy recall floor, then maximize F1
    bestA = {"thr": 0.5, "f1": -1, "prec": 0, "rec": 0}
    for i, thr in enumerate(thresholds):
        if recalls[i] >= recall_floor:
            if f1s[i] > bestA["f1"]:
                bestA = {
                    "thr": float(thr),
                    "f1": float(f1s[i]),
                    "prec": float(precisions[i]),
                    "rec": float(recalls[i])
                }

    return bestA, bestB
