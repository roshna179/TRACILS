"""
Single source of truth for turning a raw TRACILS landing-summary row
into the numeric feature matrix the model trains on and scores.

train_isolation_forest.py, evaluate_model.py, and detect_anomaly.py all
import build_features() from here instead of each doing their own
column selection -- the same lesson from the TRACILS map-plotting bugs
applies here: two independent copies of "what counts as a feature"
would eventually disagree, and a model scoring live data with slightly
different features than it was trained on is a silent, hard-to-notice
bug.
"""

import pandas as pd

# Columns straight from TRACILS_landing_data.csv that are genuinely
# informative about HOW the approach/landing went.
RAW_NUMERIC_COLUMNS = [
    "duration",              # how long the approach was tracked, seconds
    "loc_deviation",         # signed localizer deviation, degrees
    "loc_abs_deviation",     # |localizer deviation|
    "gs_deviation",          # signed glideslope deviation, degrees
    "gs_abs_deviation",      # |glideslope deviation|
    "loc_trend",             # localizer deviation drift over the approach
    "gs_trend",              # glideslope deviation drift over the approach
    "altitude_change",       # ft lost over the tracked window
    "usable_gate_samples",   # how many clean position fixes near the gate
    "gate_min",              # min lateral/vertical gate reading
    "gate_max",              # max lateral/vertical gate reading
]

# Engineered on top of the raw columns above. Kept because they were
# tested, not assumed: averaged over a 12-seed synthetic evaluation
# (see evaluate_model.py), adding loc_rate/gs_rate raised the
# synthetic-anomaly catch rate from 55.6% to 72.9% at the SAME 10%
# false-positive rate on real normal data.
#
# Why they help: dividing deviation by duration packs two signals into
# one number -- it's large both for a genuinely large deviation AND
# for a deviation that happened over a suspiciously short approach
# (duration sits in the denominator) -- which is exactly the kind of
# single-dimension-looking anomaly Isolation Forest otherwise
# under-catches on a small dataset (see train_isolation_forest.py's
# notes on that limitation).
#
# Two other candidates were tried and deliberately left OUT because
# testing didn't support them: "gate_range" (gate_max - gate_min) made
# things WORSE (55.6% -> 48.8%) -- another weak dimension diluting the
# model's random splits, same failure mode as the constant one-hot
# columns below. "combined_severity" (loc_abs + gs_abs) tested neutral
# on top of loc_rate/gs_rate, so it was left out too, to keep the
# feature set as small as it can be while still working.
ENGINEERED_COLUMNS = ["loc_rate", "gs_rate"]

NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS

# Categorical columns worth one-hot encoding IF they vary. In the
# current dataset every row is "RWY 32" / "left_coverage", so these
# contribute nothing yet -- but a future dataset with both runways, or
# both coverage directions, should be able to use this signal without
# anyone having to remember to add it back in.
CATEGORICAL_COLUMNS = ["runway", "coverage"]

# Columns that are identifiers or labels, not features, and are
# intentionally never fed to the model:
#   aircraft, icao24   -- identify WHICH flight, not HOW it flew
#   start_time, end_time -- absolute clock time; duration already
#                            captures the only useful signal in these
#   status, remarks    -- this is the existing rule-based verdict,
#                          which is exactly what we're trying to add
#                          an independent, statistical check alongside
#   unused_1           -- entirely empty in this dataset


def _add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Guard against duration == 0 (shouldn't happen on real data, but
    # a malformed or synthetic row could hit it) -- avoid inf/NaN
    # rather than let them silently corrupt scaling downstream.
    safe_duration = df["duration"].replace(0, pd.NA)
    df["loc_rate"] = (df["loc_abs_deviation"] / safe_duration).fillna(0.0)
    df["gs_rate"] = (df["gs_abs_deviation"] / safe_duration).fillna(0.0)
    return df


def build_features(df: pd.DataFrame, enforce_columns=None) -> pd.DataFrame:
    """
    df: a dataframe with at least RAW_NUMERIC_COLUMNS (CATEGORICAL_COLUMNS
        are used if present; ENGINEERED_COLUMNS are computed here, not
        expected to already exist in df).
    enforce_columns: the exact, ordered column list produced at
        training time (models/feature_columns.json). When given, the
        output is reindexed to match it exactly -- filling any missing
        one-hot category with 0 and dropping anything unexpected. This
        is what makes it safe to score a single new landing record
        (which can't possibly produce the full spread of one-hot
        categories on its own) consistently with how the model was
        trained.
    """
    missing_raw = [c for c in RAW_NUMERIC_COLUMNS if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing expected numeric columns: {missing_raw}")

    df = _add_engineered(df)
    numeric = df[NUMERIC_COLUMNS].copy()

    cat_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if cat_present:
        dummies = pd.get_dummies(df[cat_present].astype(str), prefix=cat_present)
    else:
        dummies = pd.DataFrame(index=df.index)

    features = pd.concat([numeric, dummies], axis=1)

    if enforce_columns is not None:
        # Inference time: match training's exact, already-pruned column
        # list -- don't re-derive anything here.
        features = features.reindex(columns=enforce_columns, fill_value=0)
    else:
        # Training time only: drop any column that's constant across
        # every row (e.g. a one-hot category that never varies, like
        # "runway_RWY 32" when every row in this dataset IS RWY 32).
        # A constant column can never help isolate any point -- every
        # random split drawn on it is a no-op -- so leaving it in just
        # wastes a share of the model's random splits on a dimension
        # that does nothing, diluting the splits spent on dimensions
        # that actually carry signal. This mattered in practice here:
        # removing two such columns measurably improved how reliably
        # the model caught single-dimension outliers.
        constant_cols = [c for c in features.columns if features[c].nunique(dropna=False) <= 1]
        if constant_cols:
            features = features.drop(columns=constant_cols)

    return features
