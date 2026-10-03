"""
Check the trained Isolation Forest against:
  - the real data it was trained on (all labeled "normal" by the
    existing rule-based checker) -- ideally almost none of these get
    flagged (those that do are worth a look, but aren't necessarily
    wrong -- see train_isolation_forest.py's note on this)
  - synthetic abnormal data (never seen during training) -- ideally
    most/all of these DO get flagged, since they were deliberately
    pushed outside the range of anything in the real data

IMPORTANT: this evaluates against MULTIPLE independent synthetic draws
(12 by default), not just the one saved to
data/synthetic_abnormal_landings.csv. Scoring against a single draw
was tried first and turned out to be misleading -- one particular
random draw happened to read an 80% catch rate, while the honest
12-seed average for that same model was only 55.6%. A single small
synthetic sample is noisy enough that it can make a model look
noticeably better (or worse) than it consistently is; averaging over
several independent draws is what caught that.

This is still a sanity check on a generated test set, not a rigorous
benchmark against real faults -- treat the numbers as "does this look
like it's working, consistently", not as a validated real-world
detection rate.

Run from inside src/ (after train_isolation_forest.py; this script
generates its own synthetic draws, so generate_synthetic_abnormal.py
is optional -- it's useful for having one saved, inspectable copy):
    python evaluate_model.py
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import build_features
from generate_synthetic_abnormal import make_synthetic

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "TRACILS_landing_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_EVAL_SEEDS = 12


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        columns = json.load(f)
    return model, scaler, columns


def score(df, model, scaler, columns):
    X = build_features(df, enforce_columns=columns)
    X_scaled = scaler.transform(X)
    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)
    return scores, preds


def main():
    if not os.path.exists(os.path.join(MODEL_DIR, "isolation_forest.joblib")):
        raise SystemExit(
            "models/isolation_forest.joblib not found -- run "
            "train_isolation_forest.py first."
        )

    model, scaler, columns = load_artifacts()
    real = pd.read_csv(DATA_PATH)
    real_scores, real_preds = score(real, model, scaler, columns)
    real_flag_rate = (real_preds == -1).mean()

    synth_catch_rates = []
    last_synth_scores = None
    for seed in range(N_EVAL_SEEDS):
        synth = make_synthetic(n=20, seed=seed)
        synth_scores, synth_preds = score(synth, model, scaler, columns)
        synth_catch_rates.append((synth_preds == -1).mean())
        last_synth_scores = synth_scores  # keep one draw's scores for the plot

    mean_catch = np.mean(synth_catch_rates) * 100
    std_catch = np.std(synth_catch_rates) * 100

    print("=" * 60)
    print(f"  EVALUATION  ({N_EVAL_SEEDS} independent synthetic draws, 20 rows each)")
    print("  (synthetic abnormal data is generated for this check only --")
    print("   it is not real flight telemetry)")
    print("=" * 60)
    print(f"  Real 'normal' landings flagged anomalous : {real_flag_rate*100:.1f}%  "
          f"({int(real_flag_rate*len(real))}/{len(real)})")
    print(f"      -> lower is better (these are labeled normal)")
    print(f"  Synthetic abnormal landings caught       : {mean_catch:.1f}% "
          f"(+/- {std_catch:.1f} across {N_EVAL_SEEDS} draws)")
    print(f"      -> higher is better; the +/- matters -- a single draw can")
    print(f"         mislead, which is why this averages over {N_EVAL_SEEDS} of them")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(real_scores, bins=20, alpha=0.7, label="real (labeled normal)")
    plt.hist(last_synth_scores, bins=20, alpha=0.7, label="synthetic abnormal (one draw, for illustration)")
    plt.axvline(0, color="black", linestyle="--", linewidth=1, label="decision boundary (score=0)")
    plt.xlabel("Isolation Forest anomaly score (lower = more anomalous)")
    plt.ylabel("count")
    plt.title(f"Score separation (headline catch rate: {mean_catch:.0f}% avg over {N_EVAL_SEEDS} draws)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "score_separation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved plot: outputs/score_separation.png")
    print("(the plot shows one representative draw; the printed catch rate above")
    print(f" is the {N_EVAL_SEEDS}-draw average and is the number to trust)")


if __name__ == "__main__":
    main()
