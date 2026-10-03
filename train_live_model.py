"""
Train the LIVE variant of the TRACILS anomaly model.

Why a second model: landing_logger.py's gate_min / gate_max /
usable_gate_samples are an unvalidated placeholder (see its docstring),
so feeding them to the original model inflated false alarms on real
normal landings from 10% to 26% in testing. This variant trains on the
same 100 rows but WITHOUT those three columns, so live scoring only
uses features the live system computes reliably. Same algorithm, same
contamination, same random_state.

Also saves training-score statistics so the UI can show "more typical
than X% of training landings" (percentile of the decision score).

Run from inside src/:   python train_live_model.py
"""
import os, json
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import features as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP = ["usable_gate_samples", "gate_min", "gate_max"]

F.RAW_NUMERIC_COLUMNS = [c for c in F.RAW_NUMERIC_COLUMNS if c not in DROP]
F.NUMERIC_COLUMNS = F.RAW_NUMERIC_COLUMNS + F.ENGINEERED_COLUMNS

df = pd.read_csv(os.path.join(BASE, "data", "TRACILS_landing_data.csv"))
X = F.build_features(df)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
model = IsolationForest(n_estimators=500, contamination=0.1, random_state=42).fit(Xs)
scores = np.sort(model.decision_function(Xs))

md = os.path.join(BASE, "models")
joblib.dump(model, os.path.join(md, "live_isolation_forest.joblib"))
joblib.dump(scaler, os.path.join(md, "live_scaler.joblib"))
meta = {
    "columns": list(X.columns),
    "n_train": int(len(df)),
    "n_estimators": 500,
    "contamination": 0.1,
    "train_scores_sorted": [round(float(s), 5) for s in scores],
    "feature_mean": {c: float(v) for c, v in zip(X.columns, scaler.mean_)},
    "feature_std": {c: float(v) for c, v in zip(X.columns, scaler.scale_)},
}
with open(os.path.join(md, "live_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("Saved live model:", list(X.columns))
