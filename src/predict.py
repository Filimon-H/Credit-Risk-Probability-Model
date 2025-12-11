"""Model inference utilities for credit risk prediction.

This module handles loading the trained model and making predictions.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

# Singleton for loaded model
_model: Optional[Any] = None
_model_metadata: Optional[Dict[str, Any]] = None


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def get_model_path() -> Path:
    """Return the path to the saved model."""
    return get_project_root() / "models" / "best_model.pkl"


def get_metadata_path() -> Path:
    """Return the path to the model metadata."""
    return get_project_root() / "models" / "model_metadata.json"


def load_model() -> Any:
    """Load the trained model (singleton pattern).

    Returns
    -------
    Any
        The trained sklearn pipeline.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    global _model

    if _model is not None:
        return _model

    model_path = get_model_path()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run scripts/save_best_model.py to create it."
        )

    _model = joblib.load(model_path)
    return _model


def load_model_metadata() -> Dict[str, Any]:
    """Load model metadata (name, metrics, etc.).

    Returns
    -------
    Dict[str, Any]
        Model metadata dictionary.
    """
    global _model_metadata

    if _model_metadata is not None:
        return _model_metadata

    metadata_path = get_metadata_path()

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            _model_metadata = json.load(f)
    else:
        _model_metadata = {"model_name": "unknown", "metrics": {}}

    return _model_metadata


def is_model_loaded() -> bool:
    """Check if the model is currently loaded."""
    return _model is not None


def get_feature_columns() -> list:
    """Return the list of feature columns in the expected order.

    This must match the order used during training.
    """
    return [
        "txn_count",
        "total_amount",
        "avg_amount",
        "std_amount",
        "min_amount",
        "max_amount",
        "total_value",
        "avg_value",
        "avg_txn_hour",
        "std_txn_hour",
        "weekend_txn_ratio",
        "net_amount",
        "n_credits",
        "n_debits",
        "productcategory_financial_services_ratio",
        "productcategory_airtime_ratio",
        "productcategory_utility_bill_ratio",
        "channelid_channelid_3_ratio",
        "channelid_channelid_2_ratio",
        "channelid_channelid_5_ratio",
        "providerid_providerid_4_ratio",
        "providerid_providerid_6_ratio",
        "providerid_providerid_5_ratio",
        "recency_days",
        "frequency",
        "monetary",
    ]


def predict_single(features: Dict[str, Any]) -> Dict[str, Any]:
    """Make a prediction for a single customer.

    Parameters
    ----------
    features : Dict[str, Any]
        Dictionary of customer features (must contain all required fields).

    Returns
    -------
    Dict[str, Any]
        Prediction result with keys:
        - is_high_risk: int (0 or 1)
        - probability_high_risk: float
        - model_name: str
        - model_version: str
    """
    model = load_model()
    metadata = load_model_metadata()

    # Build DataFrame with features in correct order
    feature_cols = get_feature_columns()
    row_data = {col: features[col] for col in feature_cols}
    X = pd.DataFrame([row_data])

    # Predict
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0, 1]

    return {
        "is_high_risk": int(y_pred),
        "probability_high_risk": float(y_proba),
        "model_name": metadata.get("model_name", "unknown"),
        "model_version": "1.0.0",
    }
