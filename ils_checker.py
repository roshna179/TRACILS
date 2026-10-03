"""
ILS / landing-detection logic for TRACILS.
=============================================================
This module is the SINGLE SOURCE OF TRUTH for:
  - runway geometry (thresholds, headings)
  - "which runway is this aircraft closest to" (nearest_runway)
  - "is this aircraft landing" (detect_landing)
  - localizer / glideslope deviation math (check_aircraft)

Previously map_plot.py had its own copy of nearest_runway/detect_landing
using two runways (14/32, +-25 deg heading tolerance) while this file had
a second, DIFFERENT copy (is_on_approach) hard-coded to a single runway
heading of 032. Because the two disagreed, an aircraft correctly flagged
"LANDING" by map_plot (via RWY14, heading ~140) would fail this file's
is_on_approach() check (which only accepted headings near 032), so
check_aircraft() returned loc_deviation/gs_angle/gs_deviation = None.
That is why the map could show "LANDING - ILS OK" while the deviation
numbers never actually calculated -- there was no real math running,
just a status string from a different function that silently disagreed
with the ILS math's own idea of what "on approach" meant.

Fix: there is now exactly one runway/landing model, defined here, and
map_plot.py imports it instead of keeping its own copy.

--------------------------------------------------------------------
A note on what "ILS deviation" means here vs. in a real aircraft
--------------------------------------------------------------------
A real ILS receiver never sees a runway threshold or GPS position at
all. It listens to two overlapping VHF/UHF beams, one modulated at
90 Hz and one at 150 Hz. On the runway centerline (and on the glide
path) both tones are received at equal depth of modulation; the
receiver reports this as "0 dots". Drift left/right (or above/below
the glide path) and one tone's modulation depth grows relative to the
other -- that difference (DDM, "Difference in Depth of Modulation")
IS the deviation signal, shown to the pilot as needle deflection in
"dots" (full-scale is conventionally +/-2.5 deg for the localizer and
+/-0.7 deg for the glideslope, i.e. 2 dots each side).

We have no radio front end here -- only ADS-B position reports -- so
we can't measure DDM directly. What we do instead is the geometrically
equivalent thing: compute the aircraft's actual angular position
relative to the runway centerline and glide path from its lat/lon/alt,
which is exactly the angle a real localizer/glideslope antenna array
would be encoding into DDM at that same physical position. We then
express that angle on the same "dots" scale a cockpit CDI/ILS needle
uses, so the numbers mean the same thing a pilot would see, even
though we derived them from geometry instead of modulation depth.
"""

import math
import time

# ------------------------------------------------------------------
# TRV airport reference + runway geometry
# ------------------------------------------------------------------

TRV_LAT = 8.4821
TRV_LON = 76.9199

RWY14 = (8.4905583, 76.9117283)   # threshold used when landing to the SE
RWY32 = (8.4721776, 76.9296490)   # threshold used when landing to the NW

# Runway headings are DERIVED from the threshold coordinates above, in TRUE
# bearing. They used to be hard-coded to 140 / 320, which are the MAGNETIC
# names of the runway. Mixing a magnetic heading with true-bearing math
# rotated the centerline about 4 deg away from the real runway axis
# (true axis here is ~136 / 316), so every aircraft flying a perfect
# approach showed ~4 deg of localizer error and was flagged.
# `haversine_nm` / `initial_bearing` are defined further down, so the same
# formula is repeated inline here.
import math as _m

def _true_bearing(p1, p2):
    la1, la2 = _m.radians(p1[0]), _m.radians(p2[0])
    dl = _m.radians(p2[1] - p1[1])
    y = _m.sin(dl) * _m.cos(la2)
    x = _m.cos(la1) * _m.sin(la2) - _m.sin(la1) * _m.cos(la2) * _m.cos(dl)
    return (_m.degrees(_m.atan2(y, x)) + 360) % 360

RWY14_HEADING = round(_true_bearing(RWY14, RWY32), 2)   # landing toward the SE
RWY32_HEADING = round(_true_bearing(RWY32, RWY14), 2)   # landing toward the NW

RUNWAY_INFO = {
    "RWY 14": {"threshold": RWY14, "heading": RWY14_HEADING},
    "RWY 32": {"threshold": RWY32, "heading": RWY32_HEADING},
}

# Detection parameters (previously duplicated/divergent between files)
MAX_APPROACH_DISTANCE = 15       # NM from threshold to be considered "approaching"
MAX_LANDING_ALTITUDE  = 5000     # ft
HEADING_TOLERANCE     = 25       # degrees, aircraft heading vs runway heading
LOC_ANTENNA_BEYOND_M  = 300      # localizer antenna distance past the far runway end
FIELD_ELEV_FT         = 13       # VOTV elevation (4 m), ft above sea level
LANDING_DISTANCE      = 2        # NM -- inside this, always call it "landing"

# Real-world full-scale CDI deflection references (used to express our
# geometric deviation on the same "dots" scale a pilot's ILS needle uses)
LOC_FULL_SCALE_DEG = 2.5   # +/-2.5 deg = 2 dots, localizer
GS_FULL_SCALE_DEG  = 0.7   # +/-0.7 deg = 2 dots, glideslope

# Alert thresholds (fraction-of-a-dot sensitivity for the on-map alert color)
LOC_LIMIT_DEG = 0.5
GS_LIMIT_DEG  = 0.5

GLIDE_SLOPE = 3.0   # nominal glidepath angle, degrees


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065  # nm
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def initial_bearing(lat1, lon1, lat2, lon2):
    """Bearing (deg, 0-360) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def heading_difference(h1, h2):
    return abs((h1 - h2 + 180) % 360 - 180)


def signed_angle_diff(a, b):
    """a - b, normalized to [-180, 180]."""
    d = (a - b + 180) % 360 - 180
    return d


def destination_point(lat, lon, bearing_deg, distance_m):
    """Point reached from (lat,lon) travelling distance_m along bearing_deg."""
    R = 6371000
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R)
        + math.cos(lat1) * math.sin(distance_m / R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lat2), math.degrees(lon2)


TOUCHDOWN_OFFSET_M = 300  # standard touchdown aim point, past the threshold


def get_touchdown_point(runway_name):
    info = RUNWAY_INFO.get(runway_name)
    if not info:
        return None
    threshold_lat, threshold_lon = info["threshold"]
    return destination_point(threshold_lat, threshold_lon, info["heading"], TOUCHDOWN_OFFSET_M)


def nearest_runway(ac):
    lat, lon = ac["lat"], ac["lon"]
    d14 = haversine_nm(lat, lon, RWY14[0], RWY14[1])
    d32 = haversine_nm(lat, lon, RWY32[0], RWY32[1])
    if d14 < d32:
        return {"runway": "RWY 14", "threshold": RWY14, "heading": RWY14_HEADING, "distance": d14}
    return {"runway": "RWY 32", "threshold": RWY32, "heading": RWY32_HEADING, "distance": d32}


# ------------------------------------------------------------------
# Landing detection (stateful: needs altitude history per callsign)
# ------------------------------------------------------------------

_alt_state = {}   # callsign -> {"last": latest altitude seen, "prev": altitude before that}


def is_descending(ac):
    """
    Idempotent: the answer only changes when the aircraft's altitude changes,
    so calling this several times in one cycle (check_aircraft, the map status,
    landed-event tracking...) always gives the same result. The earlier version
    overwrote its memory on every call, so only the FIRST call each cycle could
    ever say "descending".
    """
    callsign = ac.get("callsign")
    altitude = ac.get("altitude_ft")
    if altitude is None or not callsign:
        return False
    st = _alt_state.get(callsign)
    if st is None:
        _alt_state[callsign] = {"last": altitude, "prev": altitude}
        return False
    if altitude != st["last"]:
        st["prev"], st["last"] = st["last"], altitude
    return st["last"] < st["prev"] - 50   # ignore ADS-B altitude jitter


def detect_landing(ac):
    """
    True/False + the runway dict this aircraft is being judged against.
    A single, non-duplicated definition -- both RWY14 and RWY32 are
    considered, so an aircraft on a real, valid approach is never
    missed just because it isn't lined up on one specific runway.
    """
    heading = ac.get("heading")
    altitude = ac.get("altitude_ft")
    if heading is None or altitude is None:
        return False, None

    runway = nearest_runway(ac)

    if runway["distance"] > MAX_APPROACH_DISTANCE:
        return False, runway
    if altitude > MAX_LANDING_ALTITUDE:
        return False, runway

    if heading_difference(heading, runway["heading"]) > HEADING_TOLERANCE:
        return False, runway

    descending = is_descending(ac)

    if runway["distance"] <= LANDING_DISTANCE:
        return True, runway
    if descending:
        return True, runway
    return False, runway


# ------------------------------------------------------------------
# ILS deviation math (runs for ANY aircraft with an identified nearby
# runway, not only ones already flagged "landing" -- a real localizer/
# glideslope signal is capturable for miles before touchdown)
# ------------------------------------------------------------------

def check_loc(ac, runway):
    """
    True localizer deviation: is the aircraft's *position* left/right
    of the extended runway centerline, as seen from the runway
    threshold? (Not just "is its nose pointed the right way" -- an
    aircraft can be perfectly lined up on heading while still being
    laterally offset from the centerline, and vice versa.)
    """
    # A real localizer antenna sits beyond the FAR end of the runway, not at the
    # threshold. Measuring the angle from the threshold makes a tiny sideways
    # offset look huge near the runway (4 deg at 1.5 NM is only ~200 m), so the
    # angle is measured from the antenna position instead: far-end threshold
    # plus LOC_ANTENNA_BEYOND_M further along the runway.
    far = RWY14 if runway["runway"] == "RWY 32" else RWY32
    ant_lat, ant_lon = destination_point(far[0], far[1], runway["heading"], LOC_ANTENNA_BEYOND_M)
    expected_bearing = (runway["heading"] + 180) % 360   # centerline, extended outward
    actual_bearing = initial_bearing(ant_lat, ant_lon, ac["lat"], ac["lon"])
    deviation_deg = signed_angle_diff(actual_bearing, expected_bearing)
    dots = deviation_deg / (LOC_FULL_SCALE_DEG / 2)
    return round(deviation_deg, 2), round(dots, 2)


def check_glide_slope(ac, runway):
    threshold_lat, threshold_lon = runway["threshold"]
    dist_nm = haversine_nm(threshold_lat, threshold_lon, ac["lat"], ac["lon"])
    # height ABOVE THE THRESHOLD, not above sea level
    altitude_ft = max(0.0, (ac.get("altitude_ft") or 0) - FIELD_ELEV_FT)
    if dist_nm <= 0.01:
        return 0.0, -GLIDE_SLOPE, round(-GLIDE_SLOPE / (GS_FULL_SCALE_DEG / 2), 2)
    dist_ft = dist_nm * 6076.12
    angle = math.degrees(math.atan(altitude_ft / dist_ft))
    deviation = angle - GLIDE_SLOPE
    dots = deviation / (GS_FULL_SCALE_DEG / 2)
    return round(angle, 2), round(deviation, 2), round(dots, 2)


def check_aircraft(aircraft):
    """
    Run ILS checks on a single aircraft, using the SAME runway/landing
    determination the map uses (detect_landing/nearest_runway), so the
    status text and the deviation numbers can never disagree again.
    """
    landing, runway = detect_landing(aircraft)

    result = {
        "callsign":      aircraft.get("callsign", "N/A"),
        "distance_nm":   aircraft.get("distance_nm"),
        "altitude_ft":   aircraft.get("altitude_ft"),
        "runway":        runway["runway"] if runway else None,
        "on_approach":   landing,
        "loc_deviation": None,
        "loc_dots":      None,
        "gs_angle":      None,
        "gs_deviation":  None,
        "gs_dots":       None,
        "loc_alert":     False,
        "gs_alert":      False,
        "status":        "CRUISING",
    }

    # Compute real ILS-style numbers whenever a runway is close enough to
    # be meaningful (matches map_plot's "APPROACHING" radius), not only
    # once the stricter "LANDING" flag is true -- a real ILS needle comes
    # alive miles before that.
    if runway and runway["distance"] <= MAX_APPROACH_DISTANCE:
        loc_dev, loc_dots = check_loc(aircraft, runway)
        gs_angle, gs_dev, gs_dots = check_glide_slope(aircraft, runway)

        result["loc_deviation"] = loc_dev
        result["loc_dots"] = loc_dots
        result["gs_angle"] = gs_angle
        result["gs_deviation"] = gs_dev
        result["gs_dots"] = gs_dots
        result["loc_alert"] = abs(loc_dev) > LOC_LIMIT_DEG
        result["gs_alert"] = abs(gs_dev) > GS_LIMIT_DEG

        if landing:
            result["status"] = "ALERT" if (result["loc_alert"] or result["gs_alert"]) else "OK"
        else:
            result["status"] = "APPROACHING"

    return result


def run_ils_checks(aircraft_list):
    """Console/log version, used by main.py and this file's __main__."""
    near = []
    cruising = []
    for a in aircraft_list:
        _, rwy = detect_landing(a)
        if rwy and rwy["distance"] <= MAX_APPROACH_DISTANCE:
            near.append(a)
        else:
            cruising.append(a)

    print(f"\n{'='*55}")
    print(f"  AIRCRAFT SUMMARY")
    print(f"{'='*55}")
    print(f"  Total within 250nm : {len(aircraft_list)}")
    print(f"  Cruising           : {len(cruising)}  (no ILS check)")
    print(f"  Near TRV runways   : {len(near)}  (ILS check active)")
    print(f"{'='*55}")

    results = []
    for ac in near:
        r = check_aircraft(ac)
        results.append(r)
        print(f"\n  Aircraft : {r['callsign']}  ({r['runway']})")
        print(f"  Distance : {r['distance_nm']} nm   Altitude: {r['altitude_ft']} ft")
        print(f"  LOC dev  : {r['loc_deviation']} deg ({r['loc_dots']} dots)  "
              f"{'** ALERT **' if r['loc_alert'] else 'OK'}")
        print(f"  GS angle : {r['gs_angle']} deg (ideal 3.0), dev {r['gs_deviation']} deg "
              f"({r['gs_dots']} dots)  {'** ALERT **' if r['gs_alert'] else 'OK'}")
        print(f"  Status   : {r['status']}")

    if not near:
        print("\n  No aircraft currently near a TRV runway approach.\n")

    return results


if __name__ == "__main__":
    from adsb_receiver import get_aircraft
    print("Fetching aircraft...")
    planes = get_aircraft()
    run_ils_checks(planes)
