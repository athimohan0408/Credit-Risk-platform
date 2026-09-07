"""
src/ml/predictor.py
-------------------
Lightweight inference wrapper. Loads the trained LightGBM model and encoders
from models/, and exposes a single `predict()` function for the Streamlit UI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR   = _PROJECT_ROOT / "models"

# Fallback used only when metrics.json predates the calibrated training run.
DEFAULT_THRESHOLD = 0.15


def _calibrate(p: float, metrics: dict) -> float:
    """
    Convert a raw model score into a real probability of default.

    The model is trained with `scale_pos_weight`, which fits the odds of an
    artificially balanced population — a typical applicant scores around 0.54.
    Dividing the odds back down by that weight restores the true prior. See
    `src.ml.train.calibrate` for the derivation.
    """
    w = float(metrics.get("calibration", {}).get("scale_pos_weight",
              metrics.get("scale_pos_weight", 1.0)))
    if w <= 1.0:
        return p
    return p / (p + (1.0 - p) * w)


def _load_artifacts():
    """Load model, feature list, encoders and metrics (cached on first call)."""
    model        = joblib.load(_MODELS_DIR / "lgbm_model.pkl")
    feature_cols = joblib.load(_MODELS_DIR / "feature_cols.pkl")
    cat_encoders = joblib.load(_MODELS_DIR / "cat_encoders.pkl")
    with open(_MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return model, feature_cols, cat_encoders, metrics


# Module-level cache
_cache: dict = {}


def get_artifacts():
    # Keyed on "model" rather than on the cache being empty, so that populating
    # any other cache entry first does not suppress this load.
    if "model" not in _cache:
        _cache["model"], _cache["feature_cols"], _cache["cat_encoders"], _cache["metrics"] = _load_artifacts()
    return _cache["model"], _cache["feature_cols"], _cache["cat_encoders"], _cache["metrics"]


def get_train_medians():
    """
    Median of every model feature on the training set, or None if the artifact
    predates this file (in which case `preprocess_test` warns and falls back).
    """
    if "train_medians" not in _cache:
        path = _MODELS_DIR / "train_medians.pkl"
        _cache["train_medians"] = joblib.load(path) if path.exists() else None
    return _cache["train_medians"]


def predict(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict default probability for a single applicant.

    Parameters
    ----------
    applicant_data : dict mapping column names to raw values (as they would
                     appear in application_train.csv, e.g. DAYS_BIRTH=-10000,
                     EXT_SOURCE_2=0.55, NAME_CONTRACT_TYPE='Cash loans').

    Returns
    -------
    dict with keys:
        default_probability : float  calibrated probability of default (0–1)
        raw_model_score     : float  uncalibrated model output, for reference
        risk_label          : str    'LOW' / 'MEDIUM' / 'HIGH'
        risk_score          : int    0–100, scaled from the calibrated probability
        baseline_rate       : float  portfolio default rate, for comparison
        risk_multiple       : float  how many times the baseline this applicant is
        threshold           : float  the model's decision threshold
        decision            : str    'APPROVE' / 'REVIEW' at that threshold
    """
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))
    from src.data.preprocessor import preprocess_test

    model, feature_cols, cat_encoders, metrics = get_artifacts()

    # Wrap in DataFrame (single row)
    df = pd.DataFrame([applicant_data])

    # Use preprocess_test to apply same transformations as training
    X = preprocess_test(df, cat_encoders, feature_cols, get_train_medians())
    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])

    raw = float(model.predict_proba(X)[0, 1])
    prob = _calibrate(raw, metrics)

    baseline = float(metrics.get("default_rate", 0.0807))
    threshold = float(
        metrics.get("operating_point", {}).get("threshold", DEFAULT_THRESHOLD)
    )

    # Bands are anchored to the portfolio baseline rather than to arbitrary
    # constants, so "MEDIUM" always means "riskier than an average applicant".
    if prob < baseline:
        label = "LOW"
    elif prob < 2.5 * baseline:
        label = "MEDIUM"
    else:
        label = "HIGH"

    return {
        "default_probability": round(prob, 4),
        "raw_model_score":     round(raw, 4),
        "risk_label":          label,
        "risk_score":          int(round(prob * 100)),
        "baseline_rate":       round(baseline, 4),
        "risk_multiple":       round(prob / baseline, 2) if baseline else None,
        "threshold":           threshold,
        "decision":            "REVIEW" if prob >= threshold else "APPROVE",
    }


def get_metrics() -> dict:
    """Return the metrics dict from metrics.json."""
    _, _, _, metrics = get_artifacts()
    return metrics


def get_feature_importance() -> pd.Series:
    """Return feature importances as a sorted Series."""
    model, feature_cols, _, _ = get_artifacts()
    return pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
