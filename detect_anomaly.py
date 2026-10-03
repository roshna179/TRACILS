"""
Score a single landing-summary record against the trained model.

This is the function to eventually call from the live TRACILS
pipeline: once a landing event finishes and you have its aggregated
stats (duration, loc/gs deviation and trend, altitude change, gate
samples -- the same shape ils_checker.py / landing_tracker.py already
compute per approach), hand that dict to score_landing() and get back
whether it looked statistically unusual compared to the training data.

Run directly for a worked example:
    python detect_anomaly.py
"""

import os
import json

import joblib
import pandas as pd

from features import build_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

_cache = {}


def _load_artifacts():
    if not _cache:
        _cache["model"] = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
        _cache["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
            _cache["columns"] = json.load(f)
    return _cache["model"], _cache["scaler"], _cache["columns"]


def score_landing(record: dict) -> dict:
    """
    record: a dict with the same fields as one row of
        TRACILS_landing_data.csv (at minimum, everything in
        features.NUMERIC_COLUMNS; runway/coverage are optional).

    Returns {"anomaly_score": float, "is_anomaly": bool, "verdict": str}.
    Lower anomaly_score = more anomalous; is_anomaly follows the same
    "auto" threshold the model was trained with.
    """
    model, scaler, columns = _load_artifacts()
    df = pd.DataFrame([record])
    X = build_features(df, enforce_columns=columns)
    X_scaled = scaler.transform(X)

    score = float(model.decision_function(X_scaled)[0])
    is_anomaly = bool(model.predict(X_scaled)[0] == -1)

    return {
        "anomaly_score": round(score, 4),
        "is_anomaly": is_anomaly,
        "verdict": "ABNORMAL" if is_anomaly else "NORMAL",
    }


if __name__ == "__main__":
    print("Example 1: a typical landing from the training data")
    normal_example = {
        "duration": 538.1,
        "runway": "RWY 32",
        "coverage": "left_coverage",
        "loc_deviation": 0.25,
        "loc_abs_deviation": 0.29,
        "gs_deviation": 2.17,
        "gs_abs_deviation": 1.34,
        "loc_trend": -0.1,
        "gs_trend": 0.81,
        "altitude_change": -308.5,
        "usable_gate_samples": 5,
        "gate_min": -160.6,
        "gate_max": 486.2,
    }
    print(normal_example)
    print(score_landing(normal_example))

    print("\nExample 2: an implausible landing (large localizer excursion)")
    abnormal_example = dict(normal_example)
    abnormal_example["loc_deviation"] = 3.8
    abnormal_example["loc_abs_deviation"] = 3.8
    print(abnormal_example)
    print(score_landing(abnormal_example))
