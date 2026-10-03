# TRACILS Landing Anomaly Detector (Isolation Forest)

Flags statistically unusual landings from TRACILS's per-landing ILS
summary data (localizer/glideslope deviation, trend, altitude change,
duration, gate samples), using an **Isolation Forest** — an
unsupervised anomaly-detection model.

## Do we need abnormal (faulty) landing data to train this?

**No — and that's the point of Isolation Forest.**

Most ML models you'd reach for (logistic regression, a random forest
*classifier*, a neural net) are *supervised*: they need examples of
both classes — normal and abnormal — labeled, to learn the boundary
between them. Your dataset only has normal landings, so a supervised
model couldn't be trained on it at all.

Isolation Forest is different: it's **unsupervised**. It works by
randomly slicing up the feature space over and over, and measuring how
many slices it takes to isolate each point on its own:

- A landing that sits in the dense cluster of "typical" landings takes
  **many** random slices to separate from its neighbors.
- A landing that's off on its own somewhere — an unusually large
  localizer deviation, a strangely short approach, whatever — takes
  only a **few** slices to isolate, because there's nothing else
  nearby to get in the way.

That slice-count *is* the anomaly score. The model never needs to be
shown an actual fault — it only needs enough normal examples to know
what "the dense cluster" looks like, so it can recognize when
something doesn't belong to it. That's exactly what your 100-row
`TRACILS_landing_data.csv` provides.

### So what's the synthetic abnormal data for, then?

Not training — **checking the trained model actually works**.
`generate_synthetic_abnormal.py` makes a small set of landings with
one aspect pushed well outside anything seen in the real data (a
localizer deviation several times larger than the real max, an
approach that "finishes" in 40 seconds instead of ~500, etc.), clearly
labeled `synthetic_abnormal`, and `evaluate_model.py` checks whether
the trained model's anomaly score actually calls these out. This is a
sanity check on a small, artificial set — not a validated real-world
detection rate — but it's the best check possible without any real
faulty landings on hand. If/when TRACILS logs a real abnormal landing
(an actual LOC/GS alert, say), drop it into `data/` and treat it the
same way: score it, see if the model agrees it's unusual.

## Folder structure

```
tracils_ml/
├── data/
│   ├── TRACILS_landing_data.csv          # your real data (100 normal landings)
│   └── synthetic_abnormal_landings.csv   # generated — see above — not real flights
├── src/
│   ├── features.py                       # single source of truth for feature selection
│   ├── train_isolation_forest.py         # trains + saves the model
│   ├── generate_synthetic_abnormal.py    # makes the synthetic eval set
│   ├── evaluate_model.py                 # checks the model against real + synthetic data
│   └── detect_anomaly.py                 # scores one new landing (for live use later)
├── models/                               # saved model + scaler + feature list (after training)
├── outputs/                              # scored CSV + evaluation plot (after running)
├── requirements.txt
└── README.md
```

## Setup

```bash
cd tracils_ml
pip install -r requirements.txt
```

## Run it

All scripts are run from inside `src/`:

```bash
cd src

# 1. Train on the real normal data
python train_isolation_forest.py

# 2. Generate the synthetic abnormal set (for evaluation only)
python generate_synthetic_abnormal.py

# 3. Check the trained model against both
python evaluate_model.py

# 4. Score a single new landing (worked example included)
python detect_anomaly.py
```

Outputs land in `outputs/`:
- `scored_training_data.csv` — every training row with its anomaly
  score, sorted most- to least-anomalous (worth a skim: whatever's at
  the top is this dataset's own statistical outliers)
- `score_separation.png` — histogram showing how well the model
  separates real normal landings from the synthetic abnormal ones

## Features used

See `src/features.py` for the exact list and reasoning. In short:
localizer/glideslope deviation (signed and absolute), their trend over
the approach, altitude change, approach duration, gate-sample stats,
plus two engineered features — `loc_rate`/`gs_rate` (deviation divided
by duration) — added because testing showed they measurably improve
detection (see "A caution on interpreting results" below). Identifiers
(`aircraft`, `icao24`), absolute timestamps, and the existing
rule-based `status`/`remarks` columns are deliberately **not** fed to
the model — the point of this model is to be an independent
statistical check, not to just re-read the existing verdict.

`runway` and `coverage` are one-hot encoded if they vary — in the
current dataset every row is `RWY 32` / `left_coverage`, so those
columns are constant and `features.py` automatically drops them before
training (a constant column can never help isolate an outlier, and
leaving it in just dilutes the model's random splits across a
dimension that does nothing — this measurably hurt detection in
testing before it was dropped). The code is ready for a dataset that
includes more than one runway or coverage direction without needing
changes.

## Wiring this into the live TRACILS system (next step, not done here)

`detect_anomaly.score_landing(record)` is written to be called from
the live pipeline: once `landing_tracker.py` / `ils_checker.py` finish
tracking an approach and have its aggregated stats in the same shape
as a CSV row, pass that dict in and get back a statistical
normal/abnormal verdict to show alongside the existing rule-based one.
This repo doesn't wire that up automatically, since it depends on
exactly how you want to assemble per-landing aggregates in the live
server — happy to do that next if useful.

## A caution on interpreting results

Measured properly this time: averaged over **12 independent synthetic
draws** (20 rows each), not a single arbitrary sample. That distinction
matters — an earlier, single-draw evaluation of this same model read
an 80% catch rate; the honest 12-draw average turned out to be lower.
Always trust `evaluate_model.py`'s printed average (it regenerates its
own draws every run) over any single number quoted here.

| | |
|---|---|
| Real "normal" landings flagged anomalous | 10/100 (10%) — false positives, best effort |
| Synthetic abnormal landings caught | 70.8% average, ±11.0% across 12 draws |

Three things genuinely worth knowing, not just caveats to skim past:

- **A single-run evaluation can mislead, and did.** Before this project
  started averaging over multiple synthetic draws, one lucky draw
  showed 80% caught; the robust 12-draw average for that exact same
  model was 55.6%. Small synthetic test sets (20 rows) are noisy by
  nature — this is why `evaluate_model.py` generates 12 independent
  sets and averages them instead of scoring just one.
- **Two engineered features measurably helped, two didn't — all four
  were tested, not assumed.** Adding `loc_rate`/`gs_rate` (deviation
  magnitude divided by approach duration — see `features.py`) raised
  the 12-draw-average catch rate from 55.6% to 70.8% at the *same* 10%
  false-positive rate. A `gate_range` feature was tried and made things
  *worse* (another dimension diluting the model's splits, same failure
  mode as the constant columns described above). A `combined_severity`
  feature tested neutral on top of `loc_rate`/`gs_rate`. Both losers
  were left out.
- **Single-feature outliers are still caught less reliably than
  multi-feature ones.** A landing with two things wrong at once (e.g.
  both localizer AND glideslope off) is far easier for the model to
  isolate than one with only a single feature pushed to an extreme.
  With only ~13 features and 100 training rows, Isolation Forest's
  random splits don't always land on the one feature that matters
  before a single-dimension outlier "blends in" along all its other,
  perfectly ordinary dimensions. This is a known characteristic of the
  algorithm on small datasets, not something `contamination` tuning or
  further feature engineering alone fully fixes (see the table in
  `train_isolation_forest.py`'s comments for what else was tried).
  **More real training rows is the most durable fix** — detection
  should keep improving as more real landings accumulate.
- With only 100 training rows (and all logged "normal" to begin with),
  this model's sense of "normal" is only as broad as what's in that
  100-row sample. If real operations include approaches this sample
  didn't happen to cover (different wind conditions, a different time
  of day, etc.), the model may flag those as anomalous simply because
  it's never seen them — not because anything was actually wrong.
- `contamination=0.1` was chosen empirically against the synthetic set,
  not tuned against any real confirmed fault (none exist in this
  dataset yet). Treat it as a starting point to revisit once real
  abnormal landings are available to check against.
