"""
Generate a small set of clearly-synthetic ABNORMAL landings.

These are NOT used to train the model (see train_isolation_forest.py
for why Isolation Forest doesn't need that). They exist purely so
evaluate_model.py has something with a known label to check the
trained model against -- i.e. "if we show it something that should
obviously look wrong, does its anomaly score actually say so?"

Each synthetic row starts from a real normal row and pushes ONE aspect
of it well outside the range seen anywhere in the real data (localizer
or glideslope deviation several times larger than any real example,
an approach that finishes in a fraction of the normal time, etc.),
so a reasonable detector should flag it.

Run from inside src/:
    python generate_synthetic_abnormal.py
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "TRACILS_landing_data.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "synthetic_abnormal_landings.csv")

N_SYNTHETIC = 20
SEED = 42


def make_synthetic(n=N_SYNTHETIC, seed=SEED):
    rng = np.random.default_rng(seed)
    real = pd.read_csv(DATA_PATH)
    base = real.sample(n, replace=True, random_state=seed).reset_index(drop=True)

    kinds = [
        "loc_excursion", "gs_excursion", "both_excursion",
        "erratic_trend", "short_duration", "few_samples",
    ]

    rows = []
    for i, row in base.iterrows():
        r = row.copy()
        kind = rng.choice(kinds)

        if kind == "loc_excursion":
            # Real data tops out around 0.42 deg -- push well past it.
            r["loc_deviation"] = float(rng.choice([-1, 1]) * rng.uniform(2.0, 4.5))
            r["loc_abs_deviation"] = abs(r["loc_deviation"])

        elif kind == "gs_excursion":
            # Real data tops out around 2.65 deg.
            r["gs_deviation"] = float(rng.choice([-1, 1]) * rng.uniform(4.0, 8.0))
            r["gs_abs_deviation"] = abs(r["gs_deviation"])

        elif kind == "both_excursion":
            r["loc_deviation"] = float(rng.choice([-1, 1]) * rng.uniform(2.0, 4.0))
            r["loc_abs_deviation"] = abs(r["loc_deviation"])
            r["gs_deviation"] = float(rng.choice([-1, 1]) * rng.uniform(3.0, 6.0))
            r["gs_abs_deviation"] = abs(r["gs_deviation"])

        elif kind == "erratic_trend":
            # A deviation that's swinging hard over the approach,
            # rather than settling toward the centerline/glidepath.
            r["loc_trend"] = float(rng.choice([-1, 1]) * rng.uniform(3.0, 6.0))
            r["gs_trend"] = float(rng.choice([-1, 1]) * rng.uniform(3.0, 6.0))

        elif kind == "short_duration":
            # Real approaches here run 470-650s; a few tens of seconds
            # isn't a real tracked approach.
            r["duration"] = float(rng.uniform(30, 90))

        elif kind == "few_samples":
            # Too little data near the gate to trust the reading at all.
            r["usable_gate_samples"] = int(rng.integers(0, 2))

        r["aircraft"] = f"SYN{i:03d}"
        r["icao24"] = "synthetic"
        r["status"] = "synthetic_abnormal"
        r["remarks"] = f"synthetic: {kind}"
        rows.append(r)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    synthetic = make_synthetic()
    synthetic.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(synthetic)} synthetic abnormal rows to:")
    print(f"  {OUT_PATH}")
    print()
    print("Reminder: these are generated for evaluation only. They are never")
    print("read by train_isolation_forest.py, and should never be presented")
    print("as real flight data.")
