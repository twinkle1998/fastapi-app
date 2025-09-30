"""
model_catboost_reg.py

CatBoost regressor training utilities for predicting
"Total precipitation in the next 3 days" (regression task).

Functions:
    - train_catboost_regressor: Train CatBoost regressor with hyperparameter tuning (Optuna-based).
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
import optuna


def train_catboost_regressor(X_train, y_train, n_trials: int = 30, random_state: int = 42):
    """
    Train a CatBoost regressor with hyperparameter tuning using Optuna.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Target values.
        n_trials (int, optional): Number of Optuna trials (default: 30).
        random_state (int, optional): Random seed (default: 42).

    Returns:
        tuple:
            - best_model (CatBoostRegressor): Trained best CatBoost model.
            - best_params (dict): Best hyperparameters found by Optuna.
    """

    # Prepare dataset for CatBoost
    train_pool = Pool(X_train, y_train)

    # Define Optuna objective
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "loss_function": "RMSE",
            "random_seed": random_state,
            "verbose": 0,
        }

        model = CatBoostRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=3)

        mae_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
            preds = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, preds))

        return np.mean(mae_scores)

    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({"loss_function": "RMSE", "random_seed": random_state, "verbose": 0})

    # Train final model on full training set
    best_model = CatBoostRegressor(**best_params)
    best_model.fit(X_train, y_train, verbose=100)

    return best_model, best_params
