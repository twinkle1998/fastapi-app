"""
features.py

Feature engineering utilities for the **classification task**:
Predicting whether it will rain in the next 7 days in Sydney.

Two versions are provided:
    - build_tree_features_for_classification_train: drops NaNs (best for training)
    - build_tree_features_for_classification_infer: keeps NaNs (best for deployment)
"""

import numpy as np
import pandas as pd


def add_target_will_rain_in_7d(df: pd.DataFrame, target_col="will_rain_in_7d") -> pd.DataFrame:
    """Create binary target: will it rain in the next 7 days?"""
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out[target_col] = out["rain_sum"].shift(-7).fillna(0)
    out[target_col] = (out[target_col] > 0).astype(int)
    return out


def _base_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features but do not drop NaNs."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    # Drop redundant or irrelevant columns
    df = df.drop(columns=["year", "sunrise", "sunset", "snowfall_sum"], errors="ignore")

    # --- Wind encoding ---
    if "winddirection_10m_dominant" in df.columns:
        df["wind_dir_sin"] = np.sin(np.radians(df["winddirection_10m_dominant"]))
        df["wind_dir_cos"] = np.cos(np.radians(df["winddirection_10m_dominant"]))
        df = df.drop(columns=["winddirection_10m_dominant"], errors="ignore")

    # --- Rainfall lags & rolling ---
    for k in [1, 2, 3, 7, 14]:
        df[f"rain_lag{k}"] = df["rain_sum"].shift(k)
    df["rain_roll3"] = df["rain_sum"].rolling(3).mean()
    df["rain_roll7"] = df["rain_sum"].rolling(7).mean()
    df["rain_roll14"] = df["rain_sum"].rolling(14).mean()
    df["rain_volatility7"] = df["rain_sum"].rolling(7).std()

    # --- Dry spell (days since last rain) ---
    last_rain_grp = (df["rain_sum"] > 0).astype(int).replace(0, np.nan).ffill().fillna(0).cumsum()
    df["days_since_rain"] = df.groupby(last_rain_grp).cumcount()

    # --- Temperature lags & rolling ---
    df["temp_mean_lag1"] = df["temperature_2m_mean"].shift(1)
    df["temp_mean_lag3"] = df["temperature_2m_mean"].shift(3)
    df["temp_mean_lag7"] = df["temperature_2m_mean"].shift(7)
    df["temp_roll7"] = df["temperature_2m_mean"].rolling(7).mean()
    df["temp_drop_1d"] = df["temperature_2m_mean"].shift(1) - df["temperature_2m_mean"]

    # --- Radiation rolling ---
    df["rad_roll3"] = df["shortwave_radiation_sum"].rolling(3).mean()
    df["rad_roll7"] = df["shortwave_radiation_sum"].rolling(7).mean()

    # --- Sunshine rolling ---
    if "sunshine_duration" in df.columns:
        df["sunshine_roll7"] = df["sunshine_duration"].rolling(7).mean()
        df["sunshine_roll14"] = df["sunshine_duration"].rolling(14).mean()

    # --- Wind lags ---
    df["windspeed_10m_max_lag1"] = df["windspeed_10m_max"].shift(1)
    df["windgusts_10m_max_lag1"] = df["windgusts_10m_max"].shift(1)

    # --- Humidity proxy ---
    df["humidity_proxy"] = df["precipitation_hours"] / (df["daylight_duration"] + 1e-6)

    # --- Storm flag ---
    df["storm_flag"] = ((df["rain_lag1"] > 10) & (df["windgusts_10m_max_lag1"] > 50)).astype(int)

    # --- Seasonality (day of year cycle) ---
    doy = df["time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    return df


def build_tree_features_for_classification_train(df: pd.DataFrame, target_col="will_rain_in_7d") -> pd.DataFrame:
    """Training mode: build features and drop NaNs (clean dataset)."""
    df = _base_classification_features(df)
    return df.dropna().reset_index(drop=True)


def build_tree_features_for_classification_infer(df: pd.DataFrame, target_col="will_rain_in_7d") -> pd.DataFrame:
    """Inference mode: build features but keep NaNs (tree models can handle them)."""
    df = _base_classification_features(df)
    return df.reset_index(drop=True)