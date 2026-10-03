"""
sim_baseline.py -- a STARTER definition of "normal approach" in the exact
format TRACILS's live pipeline produces.

Why this exists
---------------
The 100-row TRACILS_landing_data.csv the model was first trained on does
not match what the live pipeline measures. Its gs_deviation averages
+2.17 deg (live, on a textbook 3 deg path, gives ~0), it only covers about
630 ft of descent over ~550 s (live tracking starts 15 NM out and covers
several thousand feet), and its trend scales differ. Against that data a
perfectly normal live approach sits 5 to 13 standard deviations away, so
the model called every landing abnormal.

Until enough REAL live landings have been logged to train on
(ml_live.MIN_FIELD_ROWS), the model is trained on simulated normal
approaches produced here. They go through landing_logger.features_from_
samples, the same function the live system uses, so the feature
definitions match exactly. The simulation's assumptions (speeds, glide
angle spread, intercept altitudes, lateral wander) are reasonable but
INVENTED; the UI says "simulated baseline" for as long as this is in use,
and ml_live replaces it automatically with a model trained on real logged
landings once enough exist.
"""
import math
import random

import landing_logger
import ils_checker

# distance from the threshold to the localizer antenna, in NM
ANT_NM = (ils_checker.haversine_nm(*ils_checker.RWY14, *ils_checker.RWY32) * 1852
          + ils_checker.LOC_ANTENNA_BEYOND_M) / 1852.0

N_SIM = 300
SEED = 7


def simulate_approach(rng, loc_offset_nm=None, extra_alt_ft=0.0, lateral_decay=True):
    """One approach as a list of per-poll samples (landing_logger format)."""
    poll = rng.uniform(40, 52)
    speed_kt = rng.uniform(115, 160)
    decel = rng.uniform(0.0, 0.25)          # fraction of speed shed by the end
    glide = rng.gauss(3.0, 0.25)
    intercept_alt = rng.uniform(2000, 3500)
    alt_bias = rng.gauss(0, 120)
    d = rng.uniform(7.0, 12.0)          # first fix once established on final
    d_end = rng.uniform(0.5, 3.0)
    y = max(-0.6, min(0.6, rng.gauss(0, 0.3))) if loc_offset_nm is None else loc_offset_nm
    t = 0.0
    samples = []
    start = d
    while d > d_end:
        path_alt = d * 6076.12 * math.tan(math.radians(glide))
        alt = min(path_alt, intercept_alt) + alt_bias + rng.gauss(0, 60) + extra_alt_ft
        if lateral_decay:
            y *= math.exp(-0.18)             # drifts toward the centerline
        yy = y + rng.gauss(0, 0.006)
        loc = math.degrees(math.atan2(yy, d + ANT_NM))
        gs_angle = math.degrees(math.atan2(max(alt, 0), d * 6076.12))
        samples.append({
            "time": t, "icao24": "sim", "runway": "RWY 32",
            "loc_deviation": round(loc, 2), "gs_deviation": round(gs_angle - 3.0, 2),
            "loc_alert": False, "gs_alert": False,
            "altitude_ft": alt, "threshold_distance_nm": d,
        })
        frac = (start - d) / max(start, 1e-6)
        v = speed_kt * (1 - decel * frac)
        d -= v * poll / 3600.0
        t += poll + rng.uniform(-2, 2)
    return samples


def build_feature_dicts(n=N_SIM, seed=SEED):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = simulate_approach(rng)
        if len(s) >= 5:
            out.append(landing_logger.features_from_samples(s))
    return out


def abnormal_feature_dicts(seed=99):
    """Evaluation only: approaches with a clear fault, for sanity checks."""
    rng = random.Random(seed)
    cases = {
        "wide localizer": dict(loc_offset_nm=rng.choice([-1, 1]) * 1.2, lateral_decay=False),
        "high on glidepath": dict(extra_alt_ft=1800),
        "low on glidepath": dict(extra_alt_ft=-1500),
    }
    out = []
    for name, kw in cases.items():
        for _ in range(20):
            s = simulate_approach(rng, **kw)
            if len(s) >= 5:
                out.append((name, landing_logger.features_from_samples(s)))
    return out
