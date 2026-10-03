"""
ml_live.py -- live Isolation Forest scoring for TRACILS.

Which baseline is the model trained on?  (checked in this order)
  1. FIELD   a model trained on >= MIN_FIELD_ROWS REAL landings logged by
             this system (data/TRACILS_landing_data_live.csv). It is
             rebuilt automatically every RETRAIN_EVERY new landings.
  2. SIM     a starter model trained on simulated normal approaches
             (sim_baseline.py) until enough real landings exist.
The UI always says which one is active.

Why not the original 100-row CSV: its numbers do not match what the live
pipeline measures (see sim_baseline.py), which made every normal landing
look abnormal.

Models are (re)built on first run instead of shipped as binary files, so
they always match the scikit-learn version installed on your machine.

Honest limits (also shown in the UI):
  - Scores stay PROVISIONAL until most of the approach has been seen.
  - The SIM baseline rests on invented assumptions. Real traffic that
    differs from them gets flagged; if too many landings are flagged the
    ML card shows a baseline-mismatch warning.
"""
import os, json, bisect, time, threading
from collections import deque

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import landing_logger

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "ml_model")
LIVE_CSV = landing_logger.LIVE_LOG_PATH

MIN_FIELD_ROWS = 30
RETRAIN_EVERY = 10
MIN_SAMPLES = 5
CONTAMINATION = 0.1

BASE_COLS = ["duration", "loc_deviation", "loc_abs_deviation", "gs_deviation",
             "gs_abs_deviation", "loc_trend", "gs_trend", "altitude_change"]
COLUMNS = BASE_COLS + ["loc_rate", "gs_rate"]

FRIENDLY = {
    "duration": "Approach duration", "loc_deviation": "Final localizer offset",
    "loc_abs_deviation": "Avg localizer offset", "gs_deviation": "Final glideslope offset",
    "gs_abs_deviation": "Avg glideslope offset", "loc_trend": "Localizer drift",
    "gs_trend": "Glideslope drift", "altitude_change": "Altitude lost",
    "loc_rate": "Localizer offset rate", "gs_rate": "Glideslope offset rate",
}

_lock = threading.RLock()
_art = {}
_recent = deque(maxlen=20)      # True = final verdict was ABNORMAL


# ------------------------------------------------------------------ training
def _frame(rows):
    df = pd.DataFrame(rows)[BASE_COLS].astype(float)
    dur = df["duration"].replace(0, np.nan)
    df["loc_rate"] = (df["loc_abs_deviation"] / dur).fillna(0.0)
    df["gs_rate"] = (df["gs_abs_deviation"] / dur).fillna(0.0)
    return df[COLUMNS]


def _fit_and_save(df, prefix, label, n_train):
    os.makedirs(MODEL_DIR, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df)
    model = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42).fit(Xs)
    scores = np.sort(model.decision_function(Xs))
    joblib.dump(model, os.path.join(MODEL_DIR, f"{prefix}_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{prefix}_scaler.joblib"))
    meta = {
        "tier": prefix, "label": label, "n_train": int(n_train),
        "sklearn": sklearn.__version__, "trained_at": time.time(),
        "median_duration": float(df["duration"].median()),
        "train_scores_sorted": [round(float(s), 5) for s in scores],
        "feature_mean": {c: float(v) for c, v in zip(COLUMNS, scaler.mean_)},
        "feature_std": {c: float(v) if v > 0 else 1.0 for c, v in zip(COLUMNS, scaler.scale_)},
    }
    with open(os.path.join(MODEL_DIR, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f)
    _art.clear()
    return meta


def train_sim():
    import sim_baseline
    df = _frame(sim_baseline.build_feature_dicts())
    return _fit_and_save(df, "sim", "simulated approaches", len(df))


def live_row_count():
    try:
        d = pd.read_csv(LIVE_CSV)
    except Exception:
        return 0, None
    d = d[(d["duration"] >= 150)]
    return len(d), d


def retrain_from_live(force=False):
    """Train the FIELD model from real logged landings. True if (re)trained."""
    n, d = live_row_count()
    if n < MIN_FIELD_ROWS:
        return False
    meta = _read_meta("field")
    if meta and not force and n < meta["n_train"] + RETRAIN_EVERY and meta.get("sklearn") == sklearn.__version__:
        return False
    d = d.dropna(subset=BASE_COLS)
    if len(d) < MIN_FIELD_ROWS:
        return False
    with _lock:
        _fit_and_save(_frame(d.to_dict("records")), "field", "live landings", len(d))
    print(f"  🤖 ML baseline rebuilt from {len(d)} real logged landings")
    return True


# ------------------------------------------------------------------- loading
def _read_meta(prefix):
    p = os.path.join(MODEL_DIR, f"{prefix}_meta.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _ensure():
    """Pick the best usable model, (re)building anything stale or missing."""
    with _lock:
        if _art:
            return
        retrain_from_live()
        for prefix in ("field", "sim"):
            meta = _read_meta(prefix)
            if prefix == "field" and meta and meta.get("sklearn") != sklearn.__version__:
                if not retrain_from_live(force=True):
                    meta = None
            if prefix == "sim" and (not meta or meta.get("sklearn") != sklearn.__version__):
                meta = train_sim()
            if meta and os.path.exists(os.path.join(MODEL_DIR, f"{prefix}_model.joblib")):
                _art.update(model=joblib.load(os.path.join(MODEL_DIR, f"{prefix}_model.joblib")),
                            scaler=joblib.load(os.path.join(MODEL_DIR, f"{prefix}_scaler.joblib")),
                            meta=meta)
                return


def _min_duration(meta):
    return max(150.0, 0.8 * meta.get("median_duration", 400.0))


# ------------------------------------------------------------------- public
def model_info():
    try:
        _ensure()
        meta = _art["meta"]
    except Exception as e:
        return {"available": False, "error": str(e)}
    n_real, _ = live_row_count()
    warn = len(_recent) >= 6 and (sum(_recent) / len(_recent)) > 0.4
    return {
        "available": True, "name": "Isolation Forest",
        "kind": "Unsupervised anomaly detector",
        "tier": meta["tier"],
        "trained_on": f"{meta['n_train']} {meta['label']}",
        "n_train": meta["n_train"],
        "live_rows": n_real, "live_needed": MIN_FIELD_ROWS,
        "baseline_warning": bool(warn),
        "recent_flag_rate": round(sum(_recent) / len(_recent), 2) if _recent else None,
    }


def score_features(f):
    _ensure()
    model, scaler, meta = _art["model"], _art["scaler"], _art["meta"]
    row = _frame([f]).iloc[0]
    x = np.array([[float(row[c]) for c in COLUMNS]])
    xs = scaler.transform(x)
    score = float(model.decision_function(xs)[0])
    is_anom = bool(model.predict(xs)[0] == -1)
    train = meta["train_scores_sorted"]
    pct = round(100.0 * bisect.bisect_left(train, score) / len(train))

    z = xs[0]
    reasons = []
    for i in np.argsort(-np.abs(z))[:3]:
        if abs(z[i]) < 1.0:
            continue
        c = COLUMNS[i]
        reasons.append({"feature": FRIENDLY.get(c, c), "value": round(float(x[0][i]), 3),
                        "typical": round(meta["feature_mean"][c], 3),
                        "direction": "high" if z[i] > 0 else "low",
                        "sigma": round(float(abs(z[i])), 1)})
    return {"verdict": "ABNORMAL" if is_anom else "NORMAL", "score": round(score, 4),
            "percentile": pct, "reasons": reasons, "baseline": meta["tier"]}


def score_final(f):
    """Whole-approach verdict used when a landing is logged."""
    _ensure()
    meta = _art["meta"]
    if f["n_samples"] < MIN_SAMPLES or f["duration"] < _min_duration(meta):
        return {"verdict": "NOT_SCORED", "baseline": meta["tier"], "reasons": [],
                "note": f"only {f['n_samples']} fixes over {int(f['duration'])} s "
                        f"(need {MIN_SAMPLES}+ fixes)"}
    return score_features(f)


def live_assessment(callsign, runway):
    """Per aircraft, each poll. Never raises."""
    try:
        _ensure()
        meta = _art["meta"]
        f = landing_logger.peek_features(callsign)
        n = f["n_samples"] if f else 0
        need = _min_duration(meta)
        if not f or n < MIN_SAMPLES or f["duration"] < need:
            return {"verdict": "COLLECTING", "samples": n, "needed": MIN_SAMPLES,
                    "baseline": meta["tier"],
                    "progress": round(min(1.0, min(n / MIN_SAMPLES, (f["duration"] if f else 0) / need)), 2)}
        out = score_features(f)
        out["samples"] = n
        out["provisional"] = True
        return out
    except Exception as e:
        return {"verdict": "UNAVAILABLE", "note": str(e)}


def after_landing_logged(verdict):
    """Called by landing_logger after each logged landing."""
    _recent.append(verdict == "ABNORMAL")
    try:
        retrain_from_live()
    except Exception as e:
        print(f"  (ML retrain skipped: {e})")


if __name__ == "__main__":
    print(model_info())
