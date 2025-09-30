"""
metrics_reg.py

Utility functions for evaluating **regression models**.

Functions:
    - regression_metrics: Compute standard regression error and performance metrics.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute common regression evaluation metrics.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: Dictionary containing:
            - MAE: Mean Absolute Error (average absolute difference).
            - RMSE: Root Mean Squared Error (penalizes large errors).
            - MedAE: Median Absolute Error (robust to outliers).
            - R2: Coefficient of determination (explained variance).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MedAE": float(medae),
        "R2": float(r2),
    }
