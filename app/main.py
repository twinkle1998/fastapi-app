from fastapi import FastAPI, Query
from datetime import datetime, timedelta
import pandas as pd
import joblib

# Import our custom modules
from at2weather.data_fetcher import fetch_and_save_sydney_data
from at2weather.features import build_tree_features_for_classification_infer
from at2weather.features_reg import build_tree_features_for_regression_infer

# ------------------------------
# Load trained models
# ------------------------------
RAIN_MODEL_PATH = "models/xgb_classifier.pkl"
PRECIP_MODEL_PATH = "models/catboost_regressor.pkl"

model_rain = joblib.load(RAIN_MODEL_PATH)
model_precip = joblib.load(PRECIP_MODEL_PATH)

# Feature lists (must match training)
RAIN_FEATURE_COLS = ['apparent_temperature_mean', 'daylight_duration', 'shortwave_radiation_sum', 'wind_dir_sin', 'wind_dir_cos', 'rain_lag1', 'rain_lag2', 'rain_lag3', 'rain_lag7', 'rain_lag14', 'rain_roll3', 'rain_roll7', 'rain_roll14', 'rain_volatility7', 'temp_mean_lag1', 'temp_mean_lag3', 'temp_mean_lag7', 'temp_roll7', 'temp_drop_1d', 'rad_roll3', 'rad_roll7', 'sunshine_roll7', 'sunshine_roll14', 'windspeed_10m_max_lag1', 'humidity_proxy', 'storm_flag', 'doy_sin']


PRECIP_FEATURE_COLS = [
    "daylight_duration", "temperature_2m_min",
    "precipitation_hours", "precipitation_sum",
    "temperature_2m_mean", "temperature_2m_max",
    "sunshine_duration"
]

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(
    title="Sydney Weather Prediction API",
    description="""
    **Project Objectives:**  
    - Predict whether it will rain in 7 days (classification).  
    - Predict 3-day cumulative precipitation (regression).  

    **Endpoints:**  
    - `/` : Project overview.  
    - `/health/` : Health check.  
    - `/predict/rain/` : Predict if it will rain in 7 days.  
    - `/predict/precipitation/fall/` : Predict precipitation in next 3 days.  
    """,
    version="1.0.0",
)

# ------------------------------
# Root & health
# ------------------------------
@app.get("/")
def root():
    return {
        "message": "Sydney Weather Prediction API",
        "endpoints": {
            "health": "/health/",
            "predict_rain": "/predict/rain/?date=YYYY-MM-DD",
            "predict_precipitation": "/predict/precipitation/fall/?date=YYYY-MM-DD",
        },
        "github_repo": "YOUR_GITHUB_LINK_HERE"
    }


@app.get("/health/")
def health():
    return {"status": "200", "message": "Weather Prediction API is healthy!"}


# ------------------------------
# Rain prediction (7-day)
# ------------------------------
@app.get("/predict/rain/")
def predict_rain(date: str = Query(..., description="Date in format YYYY-MM-DD")):
    try:
        input_date = pd.to_datetime(date)

        # Fetch data including enough history
        start_year = input_date.year - 1  # need past lags
        df = fetch_and_save_sydney_data(start_year)
        df = pd.concat([df, fetch_and_save_sydney_data(input_date.year)], ignore_index=True)

        # Build features (infer mode: keeps NaNs)
        df_feat = build_tree_features_for_classification_infer(df)
        df_feat = df_feat.sort_values("time").reset_index(drop=True)

        # Target date + 7 days later
        pred_date = input_date + timedelta(days=7)

        row = df_feat[df_feat["time"] == input_date]
        if row.empty:
            return {"error": f"No features available for {date}"}

        X_input = row[RAIN_FEATURE_COLS]
        pred_prob = model_rain.predict_proba(X_input)[0, 1]
        will_rain = bool(pred_prob >= 0.5)

        return {
            "input_date": date,
            "prediction": {
                "date": pred_date.strftime("%Y-%m-%d"),
                "will_rain": will_rain,
            }
        }

    except Exception as e:
        return {"error": str(e)}


# ------------------------------
# Precipitation prediction (3-day sum)
# ------------------------------
@app.get("/predict/precipitation/fall/")
def predict_precipitation(date: str = Query(..., description="Date in format YYYY-MM-DD")):
    try:
        input_date = pd.to_datetime(date)

        # Fetch data including enough history
        start_year = input_date.year - 1
        df = fetch_and_save_sydney_data(start_year)
        df = pd.concat([df, fetch_and_save_sydney_data(input_date.year)], ignore_index=True)

        # Build features (infer mode)
        df_feat = build_tree_features_for_regression_infer(df)
        df_feat = df_feat.sort_values("time").reset_index(drop=True)

        row = df_feat[df_feat["time"] == input_date]
        if row.empty:
            return {"error": f"No features available for {date}"}

        X_input = row[PRECIP_FEATURE_COLS]
        y_pred = model_precip.predict(X_input)[0]

        return {
            "input_date": date,
            "prediction": {
                "start_date": (input_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                "end_date": (input_date + timedelta(days=3)).strftime("%Y-%m-%d"),
                "precipitation_fall": round(float(y_pred), 2)
            }
        }

    except Exception as e:
        return {"error": str(e)}