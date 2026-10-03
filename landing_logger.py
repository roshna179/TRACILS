"""
landing_logger.py
==================
Accumulates per-aircraft ILS samples during a tracked approach and,
when map_plot.py finalizes a landing (see _finalize_landed), turns
that history into one row matching TRACILS_landing_data.csv's schema
and appends it to data/TRACILS_landing_data_live.csv -- a SEPARATE,
growing file. The original curated 100-row dataset the ML model in
tracils_ml/ was trained and evaluated against is never touched by
this module.

############################################################
# READ THIS BEFORE TRUSTING gate_min / gate_max / usable_gate_samples
############################################################
These three fields' exact original definition is NOT known. Nothing
in the TRACILS codebase as built computes a "gate" value --
TRACILS_landing_data.csv was supplied as existing data, and its
generation logic was never seen. Looking at the real numbers: across
all 100 rows, gate_min is ALWAYS negative (-467.1 to -156.9) and
gate_max is ALWAYS positive (356.8 to 609.8). That rules out the
obvious guesses -- lateral offset from centerline would be small and
roughly centered on zero, not this shape; altitude doesn't fit either.

What's computed below is a BEST-EFFORT PLACEHOLDER, not a confirmed
match: lateral offset from the extended runway centerline, in FEET,
over the final approach "gate" (inside LANDING_GATE_NM of the runway
threshold):
    offset_ft = distance_to_threshold_ft * tan(radians(loc_deviation))
gate_min/gate_max are this value's min/max across the gate-window
samples; usable_gate_samples is how many samples fell inside that
window. This placeholder is roughly SYMMETRIC around zero -- it does
NOT reproduce the real data's always-negative-min/always-positive-max
pattern, which is a concrete, visible sign that it is not the same
definition the original data used.

Every row this module logs is tagged in "remarks" so these rows are
easy to find and exclude. DO NOT merge
data/TRACILS_landing_data_live.csv into the training set used by
tracils_ml/ until this is confirmed or corrected -- ideally by finding
(or asking for) whatever script originally produced
TRACILS_landing_data.csv, so this module can be fixed to match it
exactly.
############################################################

Hook points (already wired into map_plot.py / server.py in this
checkout):
  - record_sample(ac, ils_result) is called once per poll cycle, per
    aircraft, from server.py's poll loop. It's a safe no-op for
    aircraft with no runway/deviation data yet.
  - finalize(callsign, coverage_reason) is called from
    map_plot._finalize_landed(), at the exact moment a landing is
    logged on the map -- same trigger, same data, so the live log and
    the map can never disagree about when a landing happened.
"""

import os
import csv
import math
import time

from ils_checker import nearest_runway, heading_difference, HEADING_TOLERANCE

# "Established on final" window (see record_sample)
ESTABLISHED_MAX_NM = 12
ESTABLISHED_MAX_ALT_FT = 4500
LOC_CAPTURE_DEG = 8

# Last ML verdict computed by finalize(), incl. reasons (read by map_plot)
last_verdict = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_LOG_PATH = os.path.join(BASE_DIR, "data", "TRACILS_landing_data_live.csv")

LANDING_GATE_NM = 2.0   # matches ils_checker.LANDING_DISTANCE

CSV_FIELDS = [
    "aircraft", "icao24", "runway", "start_time", "end_time", "duration",
    "coverage", "loc_deviation", "loc_abs_deviation", "gs_deviation",
    "gs_abs_deviation", "loc_trend", "gs_trend", "altitude_change",
    "unused_1", "usable_gate_samples", "gate_min", "gate_max", "status",
    "remarks", "ml_verdict", "ml_score", "ml_percentile", "ml_baseline",
]

_history = {}   # callsign -> list of sample dicts, oldest first


def record_sample(ac, ils_result):
    """
    Call once per poll cycle for each aircraft, passing ils_checker's
    check_aircraft() result for it. No-ops if there's no runway/
    deviation data yet (aircraft too far out, or missing heading).
    """
    if not ils_result.get("runway") or ils_result.get("loc_deviation") is None:
        return

    # Only samples where the aircraft is ESTABLISHED ON FINAL: close enough,
    # low enough, pointing along the runway, and inside the localizer capture
    # envelope. Without this, base legs, vectoring turns, departures and
    # overflights inside the 15 NM circle were logged as if they were part of
    # the approach, inflating the deviation features and flagging normal
    # landings as abnormal.
    heading = ac.get("heading")
    alt = ac.get("altitude_ft")
    rw = nearest_runway(ac)
    if (heading is None or alt is None or ac.get("on_ground")
            or rw["distance"] > ESTABLISHED_MAX_NM or alt > ESTABLISHED_MAX_ALT_FT
            or heading_difference(heading, rw["heading"]) > HEADING_TOLERANCE
            or abs(ils_result["loc_deviation"]) > LOC_CAPTURE_DEG):
        return

    callsign = ac.get("callsign")
    if not callsign:
        return

    runway_info = nearest_runway(ac)

    _history.setdefault(callsign, []).append({
        # demo flights carry their own (accelerated) clock in "_t"
        "time": ac.get("_t", time.time()),
        "icao24": ac.get("icao24"),
        "runway": ils_result["runway"],
        "loc_deviation": ils_result["loc_deviation"],
        "gs_deviation": ils_result["gs_deviation"],
        "loc_alert": ils_result.get("loc_alert", False),
        "gs_alert": ils_result.get("gs_alert", False),
        "altitude_ft": ac.get("altitude_ft"),
        "threshold_distance_nm": runway_info["distance"],
    })


def _linear_trend(xs, ys):
    """Slope of ys vs xs via simple least squares; 0.0 if <2 points."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def features_from_samples(samples):
    """
    Single source of truth for turning a list of per-poll samples into
    the per-approach summary features (duration, deviations, trends,
    altitude change). Used by finalize() for the logged CSV row AND by
    ml_live.py for live scoring, so the two can never disagree.
    """
    samples = sorted(samples, key=lambda s: s["time"])
    start_time = samples[0]["time"]
    end_time = samples[-1]["time"]
    duration = round(end_time - start_time, 1)

    loc_values = [s["loc_deviation"] for s in samples]
    gs_values = [s["gs_deviation"] for s in samples]
    rel_times = [s["time"] - start_time for s in samples]

    loc_abs_deviation = round(sum(abs(v) for v in loc_values) / len(loc_values), 2)
    gs_abs_deviation = round(sum(abs(v) for v in gs_values) / len(gs_values), 2)
    # degrees per 100 seconds (see finalize() notes on scale)
    loc_trend = round(_linear_trend(rel_times, loc_values) * 100, 3)
    gs_trend = round(_linear_trend(rel_times, gs_values) * 100, 3)

    altitude_values = [s["altitude_ft"] for s in samples if s["altitude_ft"] is not None]
    altitude_change = round(altitude_values[-1] - altitude_values[0], 1) if len(altitude_values) >= 2 else 0.0

    return {
        "start_time": start_time, "end_time": end_time, "duration": duration,
        "loc_deviation": loc_values[-1], "loc_abs_deviation": loc_abs_deviation,
        "gs_deviation": gs_values[-1], "gs_abs_deviation": gs_abs_deviation,
        "loc_trend": loc_trend, "gs_trend": gs_trend,
        "altitude_change": altitude_change,
        "n_samples": len(samples),
    }


def discard(callsign):
    """Forget an in-progress approach without logging it (demo clean-up)."""
    _history.pop(callsign, None)


def peek_features(callsign):
    """Features for an approach still in progress (does NOT consume history)."""
    samples = _history.get(callsign, [])
    if len(samples) < 2:
        return None
    return features_from_samples(samples)


def finalize(callsign, coverage_reason, persist=True):
    """
    Call exactly when a landing is finalized (map_plot._finalize_landed).

    coverage_reason: "left_coverage" or "stagnant_position" -- which
    mechanism triggered the finalization (see map_plot.update_landed_events).

    Returns the logged row dict, or None if there wasn't enough history
    to log anything meaningful (fewer than 2 samples -- can't compute a
    trend or a real duration from one point).
    """
    samples = _history.pop(callsign, [])
    if len(samples) < 2:
        return None

    samples.sort(key=lambda s: s["time"])
    f = features_from_samples(samples)
    start_time, end_time, duration = f["start_time"], f["end_time"], f["duration"]
    loc_deviation, gs_deviation = f["loc_deviation"], f["gs_deviation"]
    loc_abs_deviation, gs_abs_deviation = f["loc_abs_deviation"], f["gs_abs_deviation"]
    loc_trend, gs_trend = f["loc_trend"], f["gs_trend"]
    altitude_change = f["altitude_change"]

    runway_name = samples[-1]["runway"]
    any_alert = any(s["loc_alert"] or s["gs_alert"] for s in samples)

    # --- gate_min / gate_max / usable_gate_samples: SEE MODULE DOCSTRING.
    # Best-effort placeholder, not validated against the original
    # data's generation logic.
    gate_samples = [s for s in samples if s["threshold_distance_nm"] <= LANDING_GATE_NM]
    used_fallback = False
    if not gate_samples:
        # Coverage ended before the aircraft got inside the gate
        # window (e.g. ADS-B lost at 5nm, still descending) -- fall
        # back to the full approach rather than logging an empty gate.
        gate_samples = samples
        used_fallback = True

    gate_offsets_ft = []
    for s in gate_samples:
        dist_ft = s["threshold_distance_nm"] * 6076.12
        gate_offsets_ft.append(dist_ft * math.tan(math.radians(s["loc_deviation"])))

    usable_gate_samples = len(gate_samples)
    gate_min = round(min(gate_offsets_ft), 1)
    gate_max = round(max(gate_offsets_ft), 1)

    remarks = (
        "auto-logged from live TRACILS session; gate_min/gate_max/"
        "usable_gate_samples are an UNVALIDATED placeholder (see "
        "landing_logger.py docstring) -- do not merge into training "
        "data without confirming this matches the original definition"
    )
    if used_fallback:
        remarks += "; no samples inside the gate window, used full approach as fallback"

    row = {
        "aircraft": callsign,
        "icao24": samples[-1].get("icao24") or "",
        "runway": runway_name,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "duration": duration,
        "coverage": coverage_reason,
        "loc_deviation": loc_deviation,
        "loc_abs_deviation": loc_abs_deviation,
        "gs_deviation": gs_deviation,
        "gs_abs_deviation": gs_abs_deviation,
        "loc_trend": loc_trend,
        "gs_trend": gs_trend,
        "altitude_change": altitude_change,
        "unused_1": "",
        "usable_gate_samples": usable_gate_samples,
        "gate_min": gate_min,
        "gate_max": gate_max,
        "status": "flagged_by_rule_based_checker" if any_alert else "normal",
        "remarks": remarks,
    }

    # Independent ML verdict (Isolation Forest) alongside the rule-based one.
    try:
        import ml_live
        verdict = ml_live.score_final(f)
        last_verdict.clear()
        last_verdict.update(verdict)
        row["ml_verdict"] = verdict["verdict"]
        row["ml_score"] = verdict.get("score", "")
        row["ml_percentile"] = verdict.get("percentile", "")
        row["ml_baseline"] = verdict.get("baseline", "")
        print(f"  ML {callsign}: {verdict['verdict']} | fixes={f['n_samples']} dur={f['duration']}s "
              f"loc_abs={f['loc_abs_deviation']} gs_abs={f['gs_abs_deviation']} "
              f"loc_trend={f['loc_trend']} gs_trend={f['gs_trend']} alt_change={f['altitude_change']}")
        for r in verdict.get("reasons", []):
            print(f"     why: {r['feature']} {r['value']} ({r['sigma']} sigma {r['direction']}; typical {r['typical']})")
    except Exception as e:   # never let ML break logging
        row["ml_verdict"], row["ml_score"], row["ml_percentile"], row["ml_baseline"] = "UNAVAILABLE", "", "", ""
        print(f"  (ML scoring skipped: {e})")

    if persist:
        _append_row(row)
        try:
            import ml_live
            ml_live.after_landing_logged(row.get("ml_verdict"))
        except Exception:
            pass
    else:
        # Simulated (demo) landing: scored and shown, but NEVER written to the
        # real landing log and never allowed to influence ML retraining.
        print(f"  (demo landing {callsign}: not logged, not used for training)")
    return row


def _append_row(row):
    os.makedirs(os.path.dirname(LIVE_LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LIVE_LOG_PATH)
    with open(LIVE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  📝 Logged landing: {row['aircraft']} on {row['runway']} "
          f"-> {LIVE_LOG_PATH}")
    print(f"     (gate_min/gate_max are an unvalidated placeholder -- see landing_logger.py)")
