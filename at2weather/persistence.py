"""
persistence.py

Utility functions for saving trained models and artifacts to disk.

Functions:
    - ensure_dir: Create directory if it does not exist.
    - save_model: Save trained model (via joblib).
    - save_json: Save dictionary/metadata as JSON.
"""

import json
from pathlib import Path
import joblib


def ensure_dir(path: str):
    """
    Ensure a directory exists. Create it if missing.

    Args:
        path (str): Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model, path: str):
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained model object (e.g., XGBoost, CatBoost, sklearn).
        path (str): File path for saving the model (e.g., "models/xgb_model.pkl").
    """
    ensure_dir(Path(path).parent.as_posix())
    joblib.dump(model, path)


def save_json(obj: dict, path: str):
    """
    Save a Python dictionary as a JSON file.

    Args:
        obj (dict): Dictionary to save (e.g., best parameters, metrics).
        path (str): File path for saving JSON (e.g., "results/metrics.json").
    """
    ensure_dir(Path(path).parent.as_posix())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
