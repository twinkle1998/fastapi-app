"""
Package initializer for machine learning project.

This file exposes the core modules of the package so they can be easily imported 
at the package level. By defining `__all__`, we explicitly control which 
submodules are available when someone runs:

    from studentpkg import *

Each listed module corresponds to a specific stage in the ML pipeline 
(data fetching, feature engineering, model training, evaluation, plotting).
"""

__all__ = [
    # Data pipeline
    "data_fetcher",   # Responsible for fetching raw/processed weather data
    "features",       # Feature engineering for classification models
    "features_reg",   # Feature engineering for regression models
    "split",          # Train-test split utilities

    # Modeling (XGBoost baseline models)
    "model_xgb",      # XGBoost models for classification tasks
    "model_xgb_reg",  # XGBoost models for regression tasks

    # Evaluation metrics
    "metrics",        # Metrics utilities for classification
    "metrics_reg",    # Metrics utilities for regression

    # Persistence & Visualization
    "persistence",    # Save/load trained models and artifacts
    "plots",          # Visualization utilities (performance, feature importance, etc.)

    "model_catboost_reg",   

]


