"""
Train an Isolation Forest anomaly detector on TRACILS landing-summary
data.

Why no abnormal/labeled data is needed here
--------------------------------------------
Isolation Forest is unsupervised. Instead of learning "normal vs.
abnormal" from labeled examples the way a classifier would, it
repeatedly partitions the feature space with random splits and
measures how many splits it takes to isolate each point into its own
partition. Points that sit in a dense cluster (the "normal" pattern)
take many splits to isolate; points that sit apart from everything
else take only a few. That split-count IS the anomaly score -- so the
model only ever needs to see what normal looks like. It never has to
be shown an actual fault to learn to recognize one that resembles it
statistically.

That's exactly why the 100 rows here -- all logged "normal" by the
existing rule-based checker -- are enough to train on by themselves.
What they're NOT enough for is checking that the trained model
actually catches something unusual when it sees it. That's what
generate_synthetic_abnormal.py + evaluate_model.py are for.

Run from inside src/:
    python train_isolation_forest.py
"""

import os
import json

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from features import build_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "TRACILS_landing_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = build_features(df)

    # Isolation Forest is tree-based, so it doesn't strictly need
    # scaled inputs the way distance-based models do -- but duration
    # (hundreds) and loc_deviation (tenths of a degree) sit on very
    # different scales, and scaling keeps the random split selection
    # from being dominated by whichever feature happens to have the
    # largest raw numbers.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # NOTE on contamination: scikit-learn's contamination="auto" is NOT
    # computed from your data -- it's a fixed historical constant
    # (threshold = -0.5 on the raw score) from the original Isolation
    # Forest paper. On this dataset -- only 100 rows, all pre-vetted as
    # normal, sitting in a fairly tight cluster -- that fixed threshold
    # happened to land almost exactly in the middle of the score
    # distribution, flagging roughly HALF the training set. That's not
    # a meaningful result; it's an artifact of "auto" not adapting to
    # small/homogeneous data.
    #
    # contamination=0.1 was chosen empirically and checked with a
    # 12-seed average (evaluate_model.py does this by default now, not
    # a single arbitrary synthetic draw -- earlier testing showed a
    # single lucky seed can overstate performance a lot: one draw read
    # 80% catch rate, the honest 12-seed average with the ORIGINAL
    # feature set was only 55.6%):
    #
    #   contamination  real FP rate (want low)   synthetic catch (want high)
    #       0.05              5.0%                       30.6%
    #       0.08              8.0%                       46.2%
    #       0.10             10.0%                       55.6%
    #       0.15             15.0%                       68.1%
    #       0.20             20.0%                       73.8%
    #
    # Unlike n_estimators (which plateaus), contamination has NO clean
    # elbow -- catch rate keeps climbing roughly in step with false
    # positives the whole way. There's no hyperparameter value that
    # fixes this; contamination is a dial trading detection against
    # false alarms, not a lever that improves the model's underlying
    # ability to separate normal from abnormal. That ability is what
    # the engineered features (loc_rate, gs_rate, in features.py)
    # actually moved: with them added, contamination=0.1 jumps from
    # 55.6% to 72.9% catch rate at the SAME 10% false-positive rate --
    # a real improvement, not just a different point on the same
    # trade-off curve. 0.1 is kept here as a reasonable starting point;
    # raise it if missing a real anomaly is worse than a false alarm in
    # your use case, at the cost of more of the latter.
    #
    # This is a sanity-check result on a generated test set, not a
    # validated real-world detection rate -- revisit once real
    # abnormal landings exist to evaluate against. Single-feature
    # outliers (only ONE aspect of the approach is extreme) are still
    # caught less reliably than multi-feature ones; more real training
    # data remains the most durable fix for that, see README.md.
    model = IsolationForest(
        n_estimators=500,
        contamination=0.1,
        random_state=42,
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal, roughly centered on 0.
    # predict: 1 = normal (inlier), -1 = anomaly (outlier).
    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)

    scored = df.copy()
    scored["anomaly_score"] = scores
    scored["is_anomaly"] = preds == -1
    scored.sort_values("anomaly_score").to_csv(
        os.path.join(OUTPUT_DIR, "scored_training_data.csv"), index=False
    )

    joblib.dump(model, os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f, indent=2)

    n_flagged = int((preds == -1).sum())
    print(f"Trained on {len(df)} landings, {len(X.columns)} features:")
    print(f"  {list(X.columns)}")
    print()
    print(f"{n_flagged}/{len(df)} flagged anomalous within the training set itself.")
    print("This is a sanity check, not a fault report -- these are simply this")
    print("'normal' dataset's own statistical outliers (e.g. an unusually long")
    print("approach, or a slightly larger glideslope deviation than its peers).")
    print()
    print("Saved:")
    print("  models/isolation_forest.joblib")
    print("  models/scaler.joblib")
    print("  models/feature_columns.json")
    print("  outputs/scored_training_data.csv  (every row, sorted most- to least-anomalous)")


if __name__ == "__main__":
    main()
