"""
split.py

Time-based splitting utilities for training, validation, testing, 
and deployment sets.

Functions:
    - time_based_split: Perform chronological splits of weather dataset 
      into train, validation, test, and deploy sets.
"""

import pandas as pd
from typing import List, Tuple
from datetime import date


def time_based_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_years: Tuple[int, int] = (2015, 2021),
    val_years: Tuple[int, int] = (2022, 2023),
    test_year: int = 2024,
    deploy_year: int = 2025,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series]:
    """
    Perform chronological train/val/test/deploy split based on year.

    Args:
        df (pd.DataFrame): Input dataset containing a datetime column `time`.
        feature_cols (List[str]): Columns to be used as features.
        target_col (str): Name of the target column.
        train_years (Tuple[int, int], optional): Start and end year for training set.
        val_years (Tuple[int, int], optional): Start and end year for validation set.
        test_year (int, optional): Year for hold-out test set.
        deploy_year (int, optional): Year for deployment (latest data, up to today).
        verbose (bool, optional): If True, print sizes and date ranges for each split.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, X_deploy, y_deploy)
    """
    df = df.copy()
    df["year"] = df["time"].dt.year

    # Separate features and target
    X = df[feature_cols]
    y = df[target_col]

    # Masks for each split
    train_mask = df["year"].between(train_years[0], train_years[1])
    val_mask = df["year"].between(val_years[0], val_years[1])
    test_mask = df["year"] == test_year

    # Deployment set → only include rows from deploy_year up to today
    today = pd.to_datetime(date.today())
    deploy_mask = (df["year"] == deploy_year) & (df["time"] <= today)

    # Build splits
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    X_deploy, y_deploy = X[deploy_mask], y[deploy_mask]

    if verbose:
        print("Dataset Splits")
        print(f"Train:   {X_train.shape}, {y_train.shape} | {y_train.index.min()} → {y_train.index.max()}")
        print(f"Val:     {X_val.shape}, {y_val.shape} | {y_val.index.min()} → {y_val.index.max()}")
        print(f"Test:    {X_test.shape}, {y_test.shape} | {y_test.index.min()} → {y_test.index.max()}")
        print(f"Deploy:  {X_deploy.shape}, {y_deploy.shape} | {y_deploy.index.min()} → {y_deploy.index.max()}")

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        X_deploy, y_deploy
    )
