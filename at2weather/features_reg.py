"""
features_reg.py

Feature engineering utilities for the **regression task**:
Predicting the total precipitation over the next 3 days in Sydney.

Two versions are provided:
    - build_tree_features_for_regression_train: drops NaNs (best for training)
    - build_tree_features_for_regression_infer: keeps NaNs (best for deployment)
"""

import numpy as np
import pandas as pd


def add_target_precip_3d(df: pd.DataFrame, target_col="precip_3d") -> pd.DataFrame:
    """Create target: total precipitation over the next 3 days."""
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out[target_col] = (
        out["precipitation_sum"].shift(-1).fillna(0)
        + out["precipitation_sum"].shift(-2).fillna(0)
        + out["precipitation_sum"].shift(-3).fillna(0)
    )
    return out


def _base_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features but do not drop NaNs."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    # Drop irrelevant
    df = df.drop(columns=["year", "sunrise", "sunset", "snowfall_sum"], errors="ignore")

    # --- Rainfall lags & rolling ---
    for k in [1, 3, 7]:
        df[f"rain_lag{k}"] = df["rain_sum"].shift(k)
    df["rain_rolling7"] = df["rain_sum"].rolling(7).mean()

    # --- Temperature lags & rolling ---
    df["temp_lag1"] = df["temperature_2m_mean"].shift(1)
    df["temp_lag7"] = df["temperature_2m_mean"].shift(7)
    df["temp_rolling7"] = df["temperature_2m_mean"].rolling(7).mean()

    # --- Radiation rolling ---
    df["rad_rolling7"] = df["shortwave_radiation_sum"].rolling(7).mean()

    # --- Seasonality ---
    doy = df["time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    return df


def build_tree_features_for_regression_train(df: pd.DataFrame, target_col="precip_3d") -> pd.DataFrame:
    """Training mode: build features and drop NaNs (clean dataset)."""
    df = _base_regression_features(df)
    return df.dropna().reset_index(drop=True)


def build_tree_features_for_regression_infer(df: pd.DataFrame, target_col="precip_3d") -> pd.DataFrame:
    """Inference mode: build features but keep NaNs (tree models can handle them)."""
    df = _base_regression_features(df)
    return df.reset_index(drop=True)