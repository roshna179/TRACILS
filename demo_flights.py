"""
demo_flights.py -- inject SIMULATED flights into the live TRACILS picture.

Purpose: prove the system works without waiting for a real abnormal approach.
A demo flight is just another aircraft dict (same fields OpenSky gives us), so
it travels through exactly the same pipeline as real traffic: ILS maths ->
landing logger -> Isolation Forest -> ATC sequence/spacing -> map. Nothing on
the detection side is faked or special-cased.

Differences from real traffic (all deliberate, all visible in the UI):
  * Every demo aircraft is tagged synthetic=True and shown as SIMULATED.
  * It runs on an accelerated clock (VSTEP virtual seconds per real TICK_S) so
    a full 11 NM approach takes about a minute instead of five.
  * Its landing is scored and displayed but is NEVER written to the real
    landing log and never used to retrain the ML baseline.
"""
import math
import random
import threading
import time

import ils_checker as ilc

TICK_S = 2.0          # real seconds between demo updates
VSTEP = 12.0          # virtual seconds advanced per tick (6x speed-up)
TOUCHDOWN_NM = 0.25   # demo flight is removed when it gets this close

SCENARIOS = {
    "normal": {
        "label": "Normal approach (control)",
        "blurb": "Stable on the localizer and a 3 degree glidepath. Should stay green.",
    },
    "drift": {
        "label": "Drifting off the centerline",
        "blurb": "Starts centred, then drifts sideways. Localizer deviation grows until the card turns red.",
    },
    "high": {
        "label": "Too high on glidepath",
        "blurb": "Lined up but about 1800 ft above the path. Shows amber, then the glidepath warning.",
    },
    "spacing": {
        "label": "Two aircraft, spacing conflict",
        "blurb": "A faster follower closes on a slower leader. Gap shrinks below the minimum.",
    },
}

_lock = threading.RLock()
_flights = []
_counter = [100]


def _next_callsign():
    _counter[0] += 1
    return f"DEMO{_counter[0]}"


def _position(runway, d_nm, y_nm):
    """lat/lon of a point d_nm out along the extended centerline, y_nm to the pilot's left."""
    info = ilc.RUNWAY_INFO[runway]
    out_brg = (info["heading"] + 180) % 360
    lat, lon = ilc.destination_point(info["threshold"][0], info["threshold"][1], out_brg, d_nm * 1852)
    if abs(y_nm) > 1e-9:
        lat, lon = ilc.destination_point(lat, lon, (out_brg + 90) % 360 if y_nm > 0 else (out_brg - 90) % 360,
                                         abs(y_nm) * 1852)
    return lat, lon


def _make_flight(runway, scenario, d0, speed_kt, decel, **kw):
    rng = random.Random()
    return {
        "callsign": _next_callsign(), "icao24": f"demo{_counter[0]:02x}",
        "runway": runway, "scenario": scenario,
        "d0": d0, "d": d0, "y": 0.0, "speed_kt": speed_kt, "decel": decel,
        "glide": 3.0, "intercept_alt": 3000.0, "extra_alt": kw.get("extra_alt", 0.0),
        "drift_rate": kw.get("drift_rate", 0.0), "drift_cap": kw.get("drift_cap", 0.0),
        "side": kw.get("side", 1),
        "speed_gain": kw.get("speed_gain", 0.0),
        "vt": 0.0, "t_start": time.time(), "rng": rng, "heading_off": 0.0,
    }


def inject(scenario, runway="RWY 32"):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if runway not in ilc.RUNWAY_INFO:
        raise ValueError(f"unknown runway {runway!r}")
    rng = random.Random()
    made = []
    with _lock:
        if scenario == "normal":
            made.append(_make_flight(runway, scenario, 9.0, rng.uniform(130, 145), 0.12))
        elif scenario == "drift":
            made.append(_make_flight(runway, scenario, 9.0, 140, 0.10,
                                     drift_rate=0.10, drift_cap=0.55, side=rng.choice([-1, 1])))
        elif scenario == "high":
            made.append(_make_flight(runway, scenario, 9.0, 140, 0.10, extra_alt=1800))
        elif scenario == "spacing":
            # leader is nearer the runway and slower; the follower starts 2.6 NM behind and is faster
            made.append(_make_flight(runway, scenario, 8.5, 130, 0.05))
            made.append(_make_flight(runway, scenario, 11.1, 152, 0.05))
        _flights.extend(made)
    return [_describe(f) for f in made]


def _aircraft(f):
    d, y = f["d"], f["y"]
    lat, lon = _position(f["runway"], d, f["side"] * y)   # y is a magnitude; side picks left/right
    rwy_hdg = ilc.RUNWAY_INFO[f["runway"]]["heading"]
    path_alt = ilc.FIELD_ELEV_FT + d * 6076.12 * math.tan(math.radians(f["glide"]))
    alt = min(path_alt, f["intercept_alt"]) + f["extra_alt"] + f["rng"].gauss(0, 35)
    frac = (f["d0"] - d) / max(f["d0"], 1e-6)
    kt = (f["speed_kt"] + f["speed_gain"] * frac) * (1 - f["decel"] * frac)
    return {
        "callsign": f["callsign"], "icao24": f["icao24"],
        "lat": lat, "lon": lon,
        "altitude_ft": round(max(alt, ilc.FIELD_ELEV_FT)),
        "heading": round((rwy_hdg + f["heading_off"]) % 360, 1),
        "speed_ms": kt / 1.94384,
        "distance_nm": round(ilc.haversine_nm(ilc.TRV_LAT, ilc.TRV_LON, lat, lon), 1),
        "on_ground": False,
        "on_approach": True,
        "synthetic": True,
        "_t": f["t_start"] + f["vt"],      # accelerated clock for landing_logger
        "_kt": kt,
    }


def _describe(f):
    ac = _aircraft(f)
    return {"callsign": f["callsign"], "runway": f["runway"], "scenario": f["scenario"],
            "lat": ac["lat"], "lon": ac["lon"]}


def current_aircraft():
    """Aircraft dicts for every live demo flight (does not advance time)."""
    with _lock:
        return [_aircraft(f) for f in _flights]


def step():
    """Advance every demo flight by VSTEP virtual seconds; drop any that touched down."""
    with _lock:
        alive = []
        for f in _flights:
            ac = _aircraft(f)
            dd = ac["_kt"] * VSTEP / 3600.0
            f["d"] -= dd
            f["vt"] += VSTEP
            if f["drift_rate"]:
                new_y = min(f["drift_cap"], f["y"] + f["drift_rate"] * dd)
                moving = new_y > f["y"] + 1e-9
                f["y"] = new_y
                # drifting sideways: nose points slightly toward the side it is drifting to
                f["heading_off"] = (-f["side"]) * math.degrees(math.atan(f["drift_rate"])) if moving else 0.0
            if f["d"] > TOUCHDOWN_NM:
                alive.append(f)
        _flights[:] = alive


def clear():
    with _lock:
        _flights.clear()


def active():
    with _lock:
        return [{"callsign": f["callsign"], "scenario": f["scenario"], "runway": f["runway"],
                 "label": SCENARIOS[f["scenario"]]["label"], "distance_nm": round(f["d"], 1)}
                for f in _flights]


def is_active():
    with _lock:
        return bool(_flights)
