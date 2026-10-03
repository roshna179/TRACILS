import time
import folium
import webbrowser
import os
import math
import json

from adsb_receiver import get_aircraft
import landing_logger
import ml_live
import atc_metrics
import version
from ils_checker import (
    check_aircraft,
    detect_landing,
    nearest_runway,
    get_touchdown_point,
    RUNWAY_INFO,
    TRV_LAT,
    TRV_LON,
    RWY14,
    RWY32,
)

# ============================================================
# NOTE ON THIS FILE'S PREVIOUS BUG (round 1 of fixes)
# ============================================================
# map_plot.py used to keep its OWN copy of nearest_runway/detect_landing
# (dual-runway, 15nm/5000ft/25deg) while ils_checker.py had a SECOND,
# incompatible copy (single-runway, 10nm/3000ft/30deg, heading 032 only).
# The two disagreed, so an aircraft this file correctly called "LANDING"
# would fail ils_checker's own approach test -- meaning check_aircraft()
# handed back loc_deviation / gs_angle / gs_deviation = None even while
# the map showed "LANDING - ILS OK". There is now exactly one copy of
# this logic, in ils_checker.py, imported above.
# ============================================================

# ============================================================
# LANDING EVENT TRACKING (persists across draw_map calls)
# ============================================================

last_known_aircraft = {}      # callsign -> last position + stagnation tracking
landed_events = []            # persisted landing markers
LANDED_DISPLAY_SECONDS = 900  # keep landed markers up for 15 minutes

# --------------------------------------------------------------
# "Standing still" / "back and forth at landing" fixes (round 2)
# --------------------------------------------------------------
# Previously, an aircraft was only marked "landed" once it vanished
# entirely from the OpenSky feed. Near the ground, ADS-B coverage can
# keep reporting the SAME stale position for several poll cycles before
# finally dropping the aircraft (or OpenSky's own state-vector snapshot
# just repeats the last position it has). The result: the marker sat
# frozen in one spot, still labeled "LANDING - ILS OK", for a long
# stretch before anything changed.
#
# Fix: track how long an aircraft's position has stopped moving while
# it's flagged as landing. If it hasn't moved more than
# STAGNANT_DISTANCE_M for STAGNANT_CYCLES consecutive polls, treat it
# as landed immediately -- we don't need to wait for it to disappear.
STAGNANT_DISTANCE_M = 120   # ~1 city block; real ground movement exceeds this fast
STAGNANT_CYCLES = 2         # consecutive polls with no meaningful movement

# That stagnation fix introduced a NEW problem: OpenSky often keeps
# reporting the SAME aircraft as "still landing" for a while after it
# has already been logged as landed (it hasn't technically vanished).
# That put two markers on the map for one aircraft at once -- the live
# "LANDING" icon (still being redrawn each poll wherever OpenSky reports
# it) and the static purple "landed" ghost -- which is what looked like
# the plane "going back and forth": really it was two markers for the
# same aircraft, overlapping and swapping z-order each redraw.
#
# Fix: once an aircraft is logged as landed, suppress it from the live
# aircraft layer for a cooldown window so only the landed marker shows.
RECENTLY_LANDED_COOLDOWN_S = 120
recently_landed = {}   # callsign -> time.time() it was logged as landed


def is_recently_landed(callsign):
    ts = recently_landed.get(callsign)
    if ts is None:
        return False
    if time.time() - ts > RECENTLY_LANDED_COOLDOWN_S:
        del recently_landed[callsign]
        return False
    return True


def _meters_between(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _finalize_landed(callsign, info, coverage_reason="left_coverage"):
    touchdown = get_touchdown_point(info["runway"])
    ev = {
        "callsign": callsign,
        "last_lat": info["lat"],       # last real ADS-B ping (still airborne)
        "last_lon": info["lon"],
        "lat": touchdown[0] if touchdown else info["lat"],   # where we draw the icon
        "lon": touchdown[1] if touchdown else info["lon"],
        "runway": info["runway"],
        "altitude": info["altitude"],
        "time": time.time(),
        "ml": None,
        "synthetic": bool(info.get("synthetic")),
    }
    landed_events.insert(0, ev)
    recently_landed[callsign] = time.time()

    # Turn this aircraft's accumulated ILS samples (recorded each poll
    # cycle via landing_logger.record_sample, called from server.py)
    # into one TRACILS_landing_data.csv-shaped row, appended to a
    # separate, growing live log. coverage_reason records WHY this
    # finalized (vanished from the feed vs. stagnant position), the
    # same distinction update_landed_events already makes.
    try:
        row = landing_logger.finalize(callsign, coverage_reason, persist=not ev["synthetic"])
        if row and row.get("ml_verdict"):
            ev["ml"] = {
                "verdict": row["ml_verdict"],
                "score": row["ml_score"],
                "percentile": row["ml_percentile"],
                "rule_flagged": row["status"] != "normal",
                "baseline": row.get("ml_baseline", ""),
                "reasons": landing_logger.last_verdict.get("reasons", []),
                "note": landing_logger.last_verdict.get("note", ""),
            }
    except Exception as e:
        print(f"  ⚠️  landing_logger failed to log {callsign}: {e}")


def update_landed_events(aircraft_list):
    """
    Compare this cycle's aircraft to aircraft we were tracking as
    'landing' last cycle. An aircraft becomes a "landed" marker when
    EITHER:
      (a) it vanishes from the feed entirely (original behaviour), or
      (b) its position stops changing for STAGNANT_CYCLES polls in a
          row while it's still flagged "landing" (fixes "standing
          still for a long time").
    Once logged as landed, it's added to recently_landed so the live
    layer (see build_live_payload/draw_map) skips drawing it for a
    cooldown window -- fixing the "going back and forth" double-marker
    bug that the stagnation fix introduced.
    """
    global last_known_aircraft

    current_callsigns = set()

    for ac in aircraft_list:
        callsign = ac.get("callsign")
        if not callsign:
            continue
        current_callsigns.add(callsign)

        landing, runway = detect_landing(ac)
        if landing and runway:
            prev = last_known_aircraft.get(callsign)
            if ac.get("_repeat") and prev is not None:
                # Same OpenSky fix re-sent between real polls (demo ticks rebuild the
                # display faster than OpenSky updates). It is not new evidence that the
                # aircraft stopped moving, so leave its stagnation counter alone.
                continue
            if prev is not None:
                moved_m = _meters_between(prev["lat"], prev["lon"], ac["lat"], ac["lon"])
                stagnant_count = prev.get("stagnant_count", 0) + 1 if moved_m < STAGNANT_DISTANCE_M else 0
            else:
                stagnant_count = 0

            last_known_aircraft[callsign] = {
                "lat": ac["lat"],
                "lon": ac["lon"],
                "runway": runway["runway"],
                "altitude": ac.get("altitude_ft"),
                "stagnant_count": stagnant_count,
                "synthetic": bool(ac.get("synthetic")),
            }

            if stagnant_count >= STAGNANT_CYCLES:
                # Position has stopped changing -- don't wait for the
                # feed to drop it, call it landed now.
                info = last_known_aircraft.pop(callsign)
                _finalize_landed(callsign, info, coverage_reason="stagnant_position")
                current_callsigns.discard(callsign)  # let it re-enter fresh if it somehow reappears moving

    vanished = set(last_known_aircraft.keys()) - current_callsigns
    for callsign in vanished:
        info = last_known_aircraft.pop(callsign)
        _finalize_landed(callsign, info, coverage_reason="left_coverage")


def purge_synthetic():
    """Remove every trace of demo flights (live tracking, landed markers, history)."""
    for cs in [c for c, i in last_known_aircraft.items() if i.get("synthetic")]:
        last_known_aircraft.pop(cs, None)
        landing_logger.discard(cs)
    for ev in [e for e in landed_events if e.get("synthetic")]:
        landed_events.remove(ev)
        recently_landed.pop(ev["callsign"], None)


def get_active_landed_events():
    now = time.time()
    fresh = [e for e in landed_events if now - e["time"] <= LANDED_DISPLAY_SECONDS]
    landed_events[:] = fresh
    return fresh


# ============================================================
# DRAW MAP
# ============================================================

def draw_map(aircraft_list):

    # --------------------------------------------------------
    # CREATE MAP
    # --------------------------------------------------------

    m = folium.Map(
        location=[TRV_LAT, TRV_LON],
        zoom_start=7
    )


    # ========================================================
    # AIRPORT MARKER
    # ========================================================

    folium.Marker(

        location=[
            TRV_LAT,
            TRV_LON
        ],

        popup="""
        <b>TRV</b><br>
        Trivandrum International Airport
        """,

        tooltip="TRV Airport",

        icon=folium.Icon(
            color="red",
            icon="plane",
            prefix="fa"
        )

    ).add_to(m)


    # ========================================================
    # RUNWAY
    # ========================================================

    folium.PolyLine(

        locations=[
            RWY14,
            RWY32
        ],

        color="black",
        weight=12,
        opacity=0.9,

        tooltip="TRV Runway 14/32"

    ).add_to(m)


    # ========================================================
    # RUNWAY CENTERLINE
    # ========================================================

    folium.PolyLine(

        locations=[
            RWY14,
            RWY32
        ],

        color="white",
        weight=2,

        dash_array="10,10"

    ).add_to(m)


    # ========================================================
    # RWY 14
    # ========================================================

    folium.Marker(

        location=RWY14,

        tooltip="RWY 14",

        popup="RWY 14 Threshold",

        icon=folium.DivIcon(

            html="""
            <div style="
                font-size:13px;
                font-weight:bold;
                color:black;
                background:white;
                padding:4px;
                border:1px solid black;
            ">
            RWY 14
            </div>
            """

        )

    ).add_to(m)


    # ========================================================
    # RWY 32
    # ========================================================

    folium.Marker(

        location=RWY32,

        tooltip="RWY 32",

        popup="RWY 32 Threshold",

        icon=folium.DivIcon(

            html="""
            <div style="
                font-size:13px;
                font-weight:bold;
                color:black;
                background:white;
                padding:4px;
                border:1px solid black;
            ">
            RWY 32
            </div>
            """

        )

    ).add_to(m)


    # ========================================================
    # 250 NM MONITORING ZONE
    # ========================================================

    folium.Circle(

        location=[
            TRV_LAT,
            TRV_LON
        ],

        radius=250 * 1852,

        color="blue",

        fill=False,

        weight=1,

        tooltip="250 NM Monitoring Zone"

    ).add_to(m)


    # ========================================================
    # 10 NM APPROACH ZONE
    # ========================================================

    folium.Circle(

        location=[
            TRV_LAT,
            TRV_LON
        ],

        radius=10 * 1852,

        color="orange",

        fill=True,

        fill_opacity=0.05,

        weight=1,

        tooltip="10 NM Approach Zone"

    ).add_to(m)


    # ========================================================
    # AIRCRAFT
    # ========================================================

    landing_count = 0
    update_landed_events(aircraft_list)
    active_landings = get_active_landed_events()

    for ac in aircraft_list:

        callsign_check = ac.get("callsign", "N/A")

        # Skip drawing a live marker for an aircraft just logged as
        # landed -- see is_recently_landed() docstring above for why.
        if is_recently_landed(callsign_check):
            continue

        try:

            ils = check_aircraft(ac)

        except Exception:

            ils = {

                "loc_alert": False,
                "gs_alert": False,
                "loc_deviation": 0,
                "loc_dots": 0,
                "gs_angle": 0,
                "gs_deviation": 0,
                "gs_dots": 0,

            }

        # ----------------------------------------------------
        # DETECT LANDING
        # ----------------------------------------------------

        landing, runway = detect_landing(ac)

        alert = (
            ils["loc_alert"]
            or
            ils["gs_alert"]
        )


        # ----------------------------------------------------
        # DETERMINE STATUS
        # ----------------------------------------------------

        if landing:

            landing_count += 1

            if alert:

                color = "red"
                status = "LANDING - ILS ALERT"

            else:

                color = "green"
                status = "LANDING - ILS OK"

        elif (
            runway
            and runway["distance"] <= 15
        ):

            color = "orange"
            status = "APPROACHING TRV"

        else:

            color = "blue"
            status = "CRUISING"


        # ----------------------------------------------------
        # POPUP
        # ----------------------------------------------------

        runway_name = (
            runway["runway"]
            if runway
            else "N/A"
        )

        runway_distance = (
            round(runway["distance"], 2)
            if runway
            else "N/A"
        )


        popup_text = f"""

        <div style="width:250px">

        <h4>TRACILS Aircraft</h4>

        <b>Callsign:</b>
        {ac.get('callsign', 'N/A')}
        <br>

        <b>Status:</b>
        {status}
        <br><br>

        <b>Altitude:</b>
        {ac.get('altitude_ft', 'N/A')} ft
        <br>

        <b>Heading:</b>
        {ac.get('heading', 'N/A')}°
        <br>

        <b>Distance from TRV:</b>
        {ac.get('distance_nm', 'N/A')} NM
        <br>

        <b>Runway:</b>
        {runway_name}
        <br>

        <b>Distance from threshold:</b>
        {runway_distance} NM

        <hr>

        <b>Localizer deviation:</b>
        {ils['loc_deviation']}° ({ils.get('loc_dots')} dots)
        <br>

        <b>Glide slope:</b>
        {ils['gs_angle']}°
        <br>

        <b>GS deviation:</b>
        {ils['gs_deviation']}° ({ils.get('gs_dots')} dots)

        <hr>

        <b>Status:</b>
        {status}

        </div>

        """


        # ----------------------------------------------------
        # AIRCRAFT MARKER
        # ----------------------------------------------------

        folium.Marker(

            location=[
                ac["lat"],
                ac["lon"]
            ],

            popup=folium.Popup(
                popup_text,
                max_width=300
            ),

            tooltip=(
                f"{ac.get('callsign', 'UNKNOWN')} | "
                f"{status}"
            ),

            icon=folium.Icon(

                color=color,

                icon="plane",

                prefix="fa"

            )

        ).add_to(m)


        # ====================================================
        # DRAW APPROACH LINE
        # ====================================================

        if landing and runway:

            folium.PolyLine(

                locations=[

                    [
                        ac["lat"],
                        ac["lon"]
                    ],

                    runway["threshold"]

                ],

                color="green"
                if not alert
                else "red",

                weight=3,

                dash_array="8,8",

                opacity=0.8,

                tooltip=(
                    f"{ac.get('callsign', 'Aircraft')} "
                    f"approach to "
                    f"{runway['runway']}"
                )

            ).add_to(m)


    # ========================================================
    # LANDED AIRCRAFT MARKERS
    # (runs once, after all current aircraft are drawn)
    # ========================================================

    for ev in active_landings:

        landed_time = time.strftime(
            "%H:%M:%S",
            time.localtime(ev["time"])
        )

        runway_heading = RUNWAY_INFO.get(ev["runway"], {}).get("heading", 0)
        # fa-plane's default artwork points ~45° off north, so offset the
        # rotation to roughly align the nose with the runway heading
        icon_rotation = runway_heading - 45

        popup_text = f"""
        <div style="width:220px">
        <b>{ev['callsign']}</b><br>
        <b>LANDED</b> on {ev['runway']} at {landed_time}<br>
        Last altitude before touchdown: {ev['altitude']} ft
        </div>
        """

        # Ghost line: last real ADS-B ping -> the runway touchdown point,
        # showing the final glide that ADS-B coverage missed
        folium.PolyLine(
            locations=[
                [ev["last_lat"], ev["last_lon"]],
                [ev["lat"], ev["lon"]],
            ],
            color="purple",
            weight=2,
            opacity=0.6,
            dash_array="4,8",
            tooltip=f"{ev['callsign']} final glide to touchdown"
        ).add_to(m)

        # Aircraft icon sitting on the runway itself, at the touchdown point
        folium.Marker(
            location=[ev["lat"], ev["lon"]],
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{ev['callsign']} | LANDED {landed_time}",
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    transform: rotate({icon_rotation}deg);
                    font-size: 22px;
                    color: purple;
                    text-shadow: 0 0 3px white, 0 0 3px white;
                ">
                    <i class="fa fa-plane"></i>
                </div>
                """
            )
        ).add_to(m)


    # ========================================================
    # AUTO-ZOOM TO FIT ALL AIRCRAFT
    # (prevents traffic from being off-screen at load time)
    # ========================================================

    all_points = [[TRV_LAT, TRV_LON]] + [
        [ac["lat"], ac["lon"]] for ac in aircraft_list
    ]

    if len(all_points) > 1:
        m.fit_bounds(all_points)


    # ========================================================
    # LEGEND
    # ========================================================

    legend = f"""

    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 1000;
        background: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid grey;
        font-size: 13px;
    ">

    <b>TRACILS</b><br><br>

    🔴 Landing + ILS Alert<br>

    🟢 Landing + ILS OK<br>

    🟠 Approaching TRV<br>

    🔵 Cruising<br>

    🟣 Landed (last {LANDED_DISPLAY_SECONDS // 60} min)<br>

    ⚫ Runway 14/32<br>

    <hr>

    <b>Landing Aircraft:</b>
    {landing_count}

    </div>

    """

    m.get_root().html.add_child(
        folium.Element(legend)
    )


    # ========================================================
    # AUTO REFRESH
    # ========================================================

    output = "tracils_map.html"

    m.save(output)


    with open(
        output,
        "r",
        encoding="utf-8"
    ) as f:

        html = f.read()


    html = html.replace(

        "<head>",

        """
        <head>

        <meta
        http-equiv="refresh"
        content="10">

        """

    )


    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)


    print(
        f"Map updated: "
        f"{len(aircraft_list)} aircraft"
    )

    print(
        f"Landing aircraft: "
        f"{landing_count}"
    )


    # ========================================================
    # OPEN BROWSER FIRST TIME
    # ========================================================

    if not os.path.exists(
        "_map_opened.flag"
    ):

        webbrowser.open(

            "file://"
            + os.path.abspath(output)

        )

        open(
            "_map_opened.flag",
            "w"
        ).close()


# ============================================================
# MAIN
# ============================================================

# ============================================================
# LIVE-UPDATE MODE (no page reload, no lost zoom/pan)
# ============================================================
#
# The functions below replace the old "regenerate the whole HTML file
# and meta-refresh the browser every N seconds" approach. Instead:
#
#   build_static_map()   -> called ONCE, when the server starts.
#                            Draws everything that never changes
#                            (tiles, airport marker, runway, zones,
#                            legend shell) and injects a small JS
#                            polling loop into the page.
#
#   build_live_payload() -> called every refresh cycle by server.py.
#                            Returns a plain JSON-serializable dict
#                            (no folium/HTML) describing the current
#                            aircraft + landed markers. The browser's
#                            JS fetches this from /api/data and swaps
#                            only the aircraft/landed marker layers,
#                            leaving the map's current zoom/pan alone.
# ============================================================

def _status_for(ac, ils):
    """Work out landing/approach/cruise status + marker color for one aircraft."""
    landing, runway = detect_landing(ac)
    alert = bool(ils["loc_alert"] or ils["gs_alert"])

    if landing:
        color = "#FF5C5C" if alert else "#3ACB7E"
        status = "LANDING - ILS ALERT" if alert else "LANDING - ILS OK"
    elif runway and runway["distance"] <= 15:
        color = "#F2A73B"
        status = "APPROACHING TRV"
    else:
        color = "#5B9BD9"
        status = "CRUISING"

    return landing, runway, alert, color, status


def build_live_payload(aircraft_list):
    """
    Package this cycle's aircraft + landing data as a plain dict
    (JSON-serializable — no folium objects) for the live-updating
    frontend. This is what server.py hands back from /api/data.
    """
    update_landed_events(aircraft_list)
    active_landings = get_active_landed_events()

    aircraft_out = []
    entries = []
    landing_count = 0

    for ac in aircraft_list:
        callsign = ac.get("callsign", "N/A")

        # Don't send a live-layer entry for an aircraft we just logged
        # as landed -- OpenSky often keeps reporting it briefly after
        # touchdown, and sending both was the "going back and forth"
        # bug (two markers for one aircraft, drawn on top of each other
        # each redraw). The landed ghost marker covers it instead.
        if is_recently_landed(callsign):
            continue

        try:
            ils = check_aircraft(ac)
        except Exception:
            ils = {
                "loc_alert": False,
                "gs_alert": False,
                "loc_deviation": 0,
                "loc_dots": 0,
                "gs_angle": 0,
                "gs_deviation": 0,
                "gs_dots": 0,
            }

        landing, runway, alert, color, status = _status_for(ac, ils)
        if landing:
            landing_count += 1

        ml = None
        if runway and runway["distance"] <= 15:
            ml = ml_live.live_assessment(callsign, runway["runway"])

        entries.append({"ac": ac, "ils": ils, "runway": runway, "ml": ml})

        aircraft_out.append({
            "synthetic":           bool(ac.get("synthetic")),
            "ml":                  ml,
            "callsign":            callsign,
            "lat":                 ac["lat"],
            "lon":                 ac["lon"],
            "altitude_ft":         ac.get("altitude_ft"),
            "heading":             ac.get("heading"),
            "speed_ms":            ac.get("speed_ms"),
            "distance_nm":         ac.get("distance_nm"),
            "status":              status,
            "color":               color,
            "runway":              runway["runway"] if runway else None,
            "runway_distance":     round(runway["distance"], 2) if runway else None,
            # only draw the dashed approach line while actually landing
            "runway_threshold":    list(runway["threshold"]) if (landing and runway) else None,
            "approach_line_color": ("#FF5C5C" if alert else "#3ACB7E") if (landing and runway) else None,
            "loc_deviation":       ils["loc_deviation"],
            "loc_dots":            ils.get("loc_dots"),
            "gs_angle":            ils["gs_angle"],
            "gs_deviation":        ils["gs_deviation"],
            "gs_dots":             ils.get("gs_dots"),
        })

    # Controller-facing layer: ETA, sequence, spacing, ONE merged status.
    try:
        atc_per, atc_seq, atc_att = atc_metrics.build_atc(entries)
    except Exception as e:
        print(f"  (ATC metrics skipped: {e})")
        atc_per, atc_seq, atc_att = {}, [], []
    for item in aircraft_out:
        a = atc_per.get(item["callsign"])
        item["atc"] = a
        if a:
            item["color"] = a["color"]
            if item.get("approach_line_color"):
                item["approach_line_color"] = a["color"]

    landed_out = []
    for ev in active_landings:
        landed_time = time.strftime("%H:%M:%S", time.localtime(ev["time"]))
        runway_heading = RUNWAY_INFO.get(ev["runway"], {}).get("heading", 0)

        landed_out.append({
            "callsign":      ev["callsign"],
            "lat":           ev["lat"],
            "lon":           ev["lon"],
            "last_lat":      ev["last_lat"],
            "last_lon":      ev["last_lon"],
            "runway":        ev["runway"],
            "altitude":      ev["altitude"],
            "landed_time":   landed_time,
            "synthetic":     bool(ev.get("synthetic")),
            "ml":            ev.get("ml"),
            # the new aircraft glyph points due north by default, so the
            # rotation is just the runway heading itself -- no offset needed
            "icon_rotation": runway_heading,
        })

    return {
        "timestamp":     time.time(),
        "aircraft":      aircraft_out,
        "landed":        landed_out,
        "landing_count": landing_count,
        "total":         len(aircraft_list),
        "ml_model":      ml_live.model_info(),
        "build":         version.BUILD,
        "sequence":      atc_seq,
        "attention":     atc_att,
        "atc_config":    {"min_spacing_nm": atc_metrics.MIN_SPACING_NM},
    }


def build_static_map(output="tracils_map.html", poll_interval_seconds=45):
    """
    Build the parts of the map that NEVER change after page load:
    base tiles, airport marker, runway, centerline, zones, and the
    legend shell. Injects a small JS block that polls /api/data on
    an interval and updates only the aircraft/landed marker layers
    in place, so the user's zoom/pan is never disturbed and the page
    itself never reloads.

    poll_interval_seconds should match server.py's REFRESH_SECONDS,
    which is itself derived from your OpenSky credit budget (see
    adsb_receiver.py) rather than a guessed number.

    Call this ONCE, at server startup — not every refresh cycle.
    """
    REFRESH_SECONDS_JS = int(poll_interval_seconds * 1000)

    m = folium.Map(
        location=[TRV_LAT, TRV_LON], zoom_start=9,
    )

    # A quiet radar-beacon mark instead of a generic map pin — pulses
    # gently, like the rotating beacon on a real airfield.
    folium.Marker(
        location=[TRV_LAT, TRV_LON],
        popup="<b>TRV</b><br>Trivandrum International Airport",
        tooltip="TRV Airport",
        icon=folium.DivIcon(
            html="""
                <div class="tracils-beacon">
                    <div class="beacon-ring"></div>
                    <div class="beacon-core"></div>
                </div>
            """,
            icon_size=(22, 22), icon_anchor=(11, 11),
        ),
    ).add_to(m)

    # Runway body: a mid slate-grey so it reads clearly against the
    # near-black basemap (plain black, as before, would vanish into it).
    folium.PolyLine(
        locations=[RWY14, RWY32], color="#4B5A67", weight=12, opacity=0.95,
        tooltip="TRV Runway 14/32",
    ).add_to(m)

    folium.PolyLine(
        locations=[RWY14, RWY32], color="#E7ECF0", weight=2, dash_array="10,10",
    ).add_to(m)

    for loc, label in ((RWY14, "RWY 14"), (RWY32, "RWY 32")):
        folium.Marker(
            location=loc,
            tooltip=label,
            popup=f"{label} Threshold",
            icon=folium.DivIcon(html=f"""
                <div class="tracils-rwy-tag">{label}</div>
            """),
        ).add_to(m)

    folium.Circle(
        location=[TRV_LAT, TRV_LON], radius=250 * 1852, color="#3E6E93",
        fill=False, weight=1, tooltip="250 NM Monitoring Zone",
    ).add_to(m)

    folium.Circle(
        location=[TRV_LAT, TRV_LON], radius=10 * 1852, color="#F2A73B",
        fill=True, fill_opacity=0.06, weight=1, tooltip="10 NM Approach Zone",
    ).add_to(m)

    # ------------------------------------------------------------
    # Design tokens + chrome: brand mark (top-left), live status
    # pill (top-right), and the main HUD readout panel (bottom-left).
    # ------------------------------------------------------------
    chrome = f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body {{ background: #0A0F14; }}
        /* Free OpenStreetMap tiles, re-tinted into a dark scope with a
           CSS filter -- no API key needed, unlike CartoDB's dark tiles.
           Only the base tile pane is affected; markers, lines and the
           HUD chrome live in separate Leaflet panes untouched by this. */
        .leaflet-tile-pane {{
            filter: invert(1) hue-rotate(180deg) brightness(0.95) contrast(0.9) saturate(0.6);
        }}

        :root {{
            --panel: rgba(15, 20, 26, 0.86);
            --border: #263340;
            --text: #DCE6ED;
            --text-muted: #7C8B98;
            --accent-cruise: #5B9BD9;
            --accent-approach: #F2A73B;
            --accent-ok: #3ACB7E;
            --accent-alert: #FF5C5C;
            --accent-landed: #B98CE0;
            --font-sans: 'IBM Plex Sans', -apple-system, sans-serif;
            --font-mono: 'IBM Plex Mono', ui-monospace, monospace;
        }}

        .tracils-beacon {{ position: relative; width: 22px; height: 22px; }}
        .tracils-beacon .beacon-core {{
            position: absolute; top: 7px; left: 7px; width: 8px; height: 8px;
            border-radius: 50%; background: var(--accent-approach);
            box-shadow: 0 0 6px var(--accent-approach);
        }}
        .tracils-beacon .beacon-ring {{
            position: absolute; top: 0; left: 0; width: 22px; height: 22px;
            border-radius: 50%; border: 2px solid var(--accent-approach);
            opacity: 0.6; animation: beacon-pulse 2.2s ease-out infinite;
        }}
        @keyframes beacon-pulse {{
            0%   {{ transform: scale(0.3); opacity: 0.8; }}
            100% {{ transform: scale(1.7); opacity: 0; }}
        }}

        .tracils-rwy-tag {{
            font-family: var(--font-mono); font-size: 11px; font-weight: 600;
            color: var(--text); background: rgba(15, 20, 26, 0.92);
            padding: 3px 7px; border: 1px solid #3A4652; border-radius: 4px;
            letter-spacing: 0.03em; white-space: nowrap;
        }}

        #tracils-brand {{
            position: fixed; top: 18px; left: 70px; z-index: 1000;
            font-family: var(--font-sans); pointer-events: none;
            text-shadow: 0 1px 6px rgba(0,0,0,0.7);
        }}
        #tracils-brand .brand-name {{
            font-size: 17px; font-weight: 700; color: var(--text);
            letter-spacing: 0.02em;
        }}
        #tracils-brand .brand-sub {{
            font-size: 11.5px; color: var(--text-muted); margin-top: 2px;
        }}

        #tracils-status-pill {{
            position: fixed; top: 16px; right: 18px; z-index: 1000;
            display: flex; align-items: center; gap: 9px;
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 20px; padding: 8px 14px; backdrop-filter: blur(8px);
            font-family: var(--font-sans); font-size: 12.5px; color: var(--text);
        }}
        #tracils-status-pill .live-dot {{
            width: 8px; height: 8px; border-radius: 50%; background: var(--accent-ok);
            animation: live-pulse 2s infinite;
        }}
        #tracils-status-pill .live-dot.stale {{
            background: var(--accent-alert); animation: none;
        }}
        #tracils-status-pill .live-clock {{
            font-family: var(--font-mono); color: var(--text-muted);
            border-left: 1px solid var(--border); padding-left: 9px; margin-left: 2px;
        }}
        @keyframes live-pulse {{
            0%   {{ box-shadow: 0 0 0 0 rgba(58,203,126,0.55); }}
            70%  {{ box-shadow: 0 0 0 7px rgba(58,203,126,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(58,203,126,0); }}
        }}

        #tracils-atc {{
            position: fixed; top: 64px; right: 18px; z-index: 1000; width: 272px;
            max-height: calc(100vh - 100px); overflow-y: auto;
            background: var(--panel); border: 1px solid var(--border);
            border-radius: 14px; padding: 14px 14px 10px; backdrop-filter: blur(10px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.45);
            font-family: var(--font-sans); color: var(--text);
        }}
        #tracils-atc .atc-title {{ font-size: 10.5px; letter-spacing: .09em; color: var(--text-muted); margin-bottom: 8px; }}
        #tracils-atc .atc-empty {{ font-size: 12px; color: var(--text-muted); padding: 2px 0 6px; }}
        #tracils-atc .atc-row {{ display: grid; grid-template-columns: 16px 1fr auto auto 10px; gap: 8px; align-items: center; font-size: 12.5px; padding: 3px 0; }}
        #tracils-atc .atc-row .n {{ color: var(--text-muted); font-family: var(--font-mono); }}
        #tracils-atc .atc-row .cs {{ font-weight: 600; }}
        #tracils-atc .atc-row .cs small {{ color: var(--text-muted); font-weight: 400; margin-left: 5px; }}
        #tracils-atc .atc-row .eta {{ font-family: var(--font-mono); color: #B8C6D1; }}
        #tracils-atc .atc-row .gap {{ font-family: var(--font-mono); color: var(--text-muted); font-size: 11px; min-width: 46px; text-align: right; }}
        #tracils-atc .atc-row .st {{ width: 9px; height: 9px; border-radius: 50%; }}
        #tracils-atc .atc-card {{ border: 1px solid var(--border); border-radius: 10px; padding: 9px 11px; margin-top: 8px; background: rgba(10,15,20,0.6); }}
        #tracils-atc .atc-card .hd {{ display: flex; justify-content: space-between; align-items: center; font-size: 10.5px; letter-spacing: .08em; margin-bottom: 5px; }}
        #tracils-atc .atc-card .ml {{ background: #2a2140; color: #B98CE0; padding: 1px 6px; border-radius: 4px; font-size: 10.5px; letter-spacing: 0; }}
        #tracils-atc .atc-card .hl {{ font-size: 13px; font-weight: 600; }}
        #tracils-atc .atc-card .dt {{ font-size: 11.5px; color: #9FB0BD; margin-top: 3px; }}
        #tracils-atc .cdi {{ position: relative; height: 14px; margin: 8px 0 2px; }}
        #tracils-atc .cdi .tr {{ position: absolute; left: 0; right: 0; top: 6px; height: 1px; background: var(--border); }}
        #tracils-atc .cdi .dot {{ position: absolute; top: 3px; width: 7px; height: 7px; border-radius: 50%; border: 1px solid #4B5A67; }}
        #tracils-atc .cdi .mid {{ position: absolute; left: 50%; top: 0; width: 1px; height: 14px; background: #7C8B98; }}
        #tracils-atc .cdi .mk {{ position: absolute; top: 1px; width: 12px; height: 12px; border-radius: 2px; margin-left: -6px; }}
        #tracils-atc .cdi-cap {{ font-size: 11px; color: #9FB0BD; }}
        #tracils-atc .act {{ font-size: 12px; margin-top: 7px; padding-top: 7px; border-top: 1px solid var(--border); }}
        #tracils-atc .act span {{ color: var(--text-muted); }}
        #tracils-atc .atc-foot {{ font-size: 10.5px; color: var(--text-muted); margin-top: 8px; }}
        .leaflet-tooltip.tracils-label {{
            background: rgba(15,20,26,0.92); color: #DCE6ED; border: 1px solid #263340;
            border-left-width: 3px; border-radius: 4px; padding: 3px 7px;
            font-family: 'IBM Plex Sans', sans-serif; font-size: 11.5px; line-height: 1.35;
            box-shadow: none;
        }}
        .leaflet-tooltip.tracils-label:before {{ display: none; }}
        .tracils-label small {{ color: #7C8B98; font-size: 10.5px; }}
        .simtag {{ background: #1b3550; color: #7FC4FF; font-size: 9px; font-weight: 600;
                  padding: 0 4px; border-radius: 3px; margin-left: 4px; letter-spacing: .07em; }}

        #tracils-panel {{
            position: fixed; bottom: 24px; left: 24px; z-index: 1000;
            width: 232px; background: var(--panel); border: 1px solid var(--border);
            border-radius: 14px; padding: 18px 18px 14px; backdrop-filter: blur(10px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.45);
            font-family: var(--font-sans); color: var(--text);
        }}
        #tracils-panel .hero-value {{
            font-family: var(--font-mono); font-size: 40px; font-weight: 600;
            line-height: 1; color: var(--text); transition: color 0.3s;
        }}
        #tracils-panel .hero-value.active {{ color: var(--accent-approach); }}
        #tracils-panel .hero-label {{ font-size: 12px; color: var(--text-muted); margin-top: 5px; }}
        #tracils-panel .panel-row {{
            display: flex; justify-content: space-between; align-items: baseline;
            font-size: 12.5px; margin-top: 10px;
        }}
        #tracils-panel .row-label {{ color: var(--text-muted); }}
        #tracils-panel .row-value {{ font-family: var(--font-mono); font-size: 13px; }}
        #tracils-panel .panel-divider {{ height: 1px; background: var(--border); margin: 13px 0; }}
        #tracils-panel .legend-row {{
            display: flex; align-items: center; gap: 9px; font-size: 12px;
            color: var(--text-muted); padding: 3px 0;
        }}
        #tracils-panel .dot {{
            width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
            box-shadow: 0 0 4px currentColor;
        }}
    </style>

    <div id="tracils-brand">
        <div class="brand-name">TRACILS</div>
        <div class="brand-sub">Trivandrum International &middot; VOTV</div>
    </div>

    <div id="tracils-status-pill">
        <span class="live-dot" id="live-dot"></span>
        <span id="live-label">Live</span>
        <span class="live-clock" id="live-clock">--:--:--</span>
    </div>

    <div id="feed-banner" style="display:none;position:fixed;top:62px;left:70px;z-index:1001;width:300px;
         background:#2a1a1a;border:1px solid #FF5C5C;border-radius:10px;padding:9px 12px;
         font:12px var(--font-sans);color:#FFB3B3;line-height:1.4;"></div>

    <div id="tracils-atc">
        <div class="atc-title">ARRIVAL SEQUENCE</div>
        <div id="atc-seq"><div class="atc-empty">No arrivals on final</div></div>
        <div id="atc-attn-wrap" style="display:none;">
            <div class="atc-title" style="margin-top:12px;color:#F2A73B;">NEEDS ATTENTION</div>
            <div id="atc-attn"></div>
        </div>
        <div class="atc-foot" id="atc-foot"></div>
    </div>

    <div id="tracils-panel">
        <div class="hero-value" id="landing-count">0</div>
        <div class="hero-label">Landing now</div>

        <div class="panel-row">
            <span class="row-label">Tracked</span>
            <span class="row-value" id="tracked-count">0</span>
        </div>

        <div class="panel-divider"></div>

        <div class="legend-row"><span class="dot" style="background:var(--accent-alert);color:var(--accent-alert)"></span>Act now</div>
        <div class="legend-row"><span class="dot" style="background:var(--accent-ok);color:var(--accent-ok)"></span>Stable on final</div>
        <div class="legend-row"><span class="dot" style="background:var(--accent-approach);color:var(--accent-approach)"></span>Watch / approaching</div>
        <div class="legend-row"><span class="dot" style="background:transparent;border:1.5px dashed #B98CE0;box-shadow:none;"></span>ML flag (ring)</div>
        <div class="legend-row"><span class="dot" style="background:var(--accent-cruise);color:var(--accent-cruise)"></span>Cruising</div>
        <div class="legend-row"><span class="dot" style="background:var(--accent-landed);color:var(--accent-landed)"></span>Landed, last {LANDED_DISPLAY_SECONDS // 60} min</div>
        <div class="legend-row"><span class="dot" style="background:#4B5A67"></span>Runway 14/32</div>

        <div class="panel-divider"></div>

        <div style="font-size:10px;letter-spacing:.08em;color:#B98CE0;margin-bottom:5px;">&#129302; ML MODEL</div>
        <div class="panel-row"><span class="row-label">Type</span><span class="row-value" id="ml-name">&ndash;</span></div>
        <div class="panel-row"><span class="row-label">Trained on</span><span class="row-value" id="ml-train">&ndash;</span></div>
        <div class="panel-row"><span class="row-label">Scoring now</span><span class="row-value" id="ml-scoring">0</span></div>
        <div class="panel-row"><span class="row-label">ML flagged</span><span class="row-value" id="ml-flagged">0</span></div>
        <div id="ml-base" style="font-size:11px;color:#7C8B98;margin-top:4px;"></div>
        <div id="ml-warn" style="display:none;font-size:11px;color:#F2A73B;margin-top:4px;"></div>
        <div id="build-tag" style="font-size:10px;color:#4B5A67;margin-top:6px;"></div>

        <div class="panel-divider"></div>

        <div class="panel-row">
            <span class="row-label">Updated</span>
            <span class="row-value" id="last-update">waiting for data&hellip;</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(chrome))

    map_var = m.get_name()  # folium's auto-generated JS variable, e.g. map_ab12cd

    live_js = f"""
    <script>
        window.addEventListener('load', function() {{
    (function() {{
        var map = {map_var};
        var aircraftLayer = L.layerGroup().addTo(map);
        var landedLayer = L.layerGroup().addTo(map);
        var didInitialFit = false;

        // Independent ticking clock in the status pill -- keeps the HUD
        // feeling alive between poll cycles, separate from data freshness.
        function tickClock() {{
            var el = document.getElementById('live-clock');
            if (el) el.innerText = new Date().toLocaleTimeString();
        }}
        tickClock();
        setInterval(tickClock, 1000);

        // A clean top-down aircraft silhouette (the same style real flight
        // trackers use), not a generic map-pin font glyph. It points due
        // north (0deg) by default, so rotating it by heading/track lines
        // it up correctly on the map without any offset correction.
        var AIRCRAFT_GLYPH =
            '<path d="M21,16v-2l-8-5V3.5C13,2.67,12.33,2,11.5,2S10,2.67,10,3.5V9l-8,5v2l8-2.5V19l-2,1.5V22' +
            'l3.5-1l3.5,1v-1.5L14,19v-5.5L21,16z"/>';

        function planeIcon(color, heading, ring) {{
            var rot = (typeof heading === 'number') ? heading : 0;
            var ringHtml = ring ? '<div style="position:absolute;left:-5px;top:-5px;width:28px;height:28px;' +
                'border-radius:50%;border:1.5px dashed #B98CE0;"></div>' : '';
            return L.divIcon({{
                className: '',
                html: ringHtml + '<div style="width:22px;height:22px;transform:rotate(' + rot + 'deg);' +
                      'filter:drop-shadow(0 1px 2px rgba(0,0,0,0.9));">' +
                      '<svg viewBox="0 0 24 24" width="22" height="22" fill="' + color + '">' +
                      AIRCRAFT_GLYPH + '</svg></div>',
                iconSize: [22, 22],
                iconAnchor: [11, 11]
            }});
        }}

        function landedIcon(rotation) {{
            return L.divIcon({{
                className: '',
                html: '<div style="width:20px;height:20px;transform:rotate(' + rotation + 'deg);' +
                      'filter:drop-shadow(0 1px 2px rgba(0,0,0,0.9));">' +
                      '<svg viewBox="0 0 24 24" width="20" height="20" fill="#B98CE0">' +
                      AIRCRAFT_GLYPH + '</svg></div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            }});
        }}

        function popupShell(title, rows) {{
            return '<div style="font-family:\\'IBM Plex Sans\\',sans-serif;' +
                'background:#0F141A;color:#DCE6ED;width:236px;margin:-9px -13px;' +
                'padding:12px 14px;border-radius:4px;">' +
                '<div style="font-size:13px;font-weight:600;margin-bottom:8px;' +
                'padding-bottom:8px;border-bottom:1px solid #263340;">' + title + '</div>' +
                rows +
                '</div>';
        }}

        function dataRow(label, value) {{
            return '<div style="display:flex;justify-content:space-between;' +
                'font-size:12px;color:#7C8B98;padding:2px 0;">' +
                '<span>' + label + '</span>' +
                '<span style="font-family:\\'IBM Plex Mono\\',monospace;color:#DCE6ED;">' +
                value + '</span></div>';
        }}

        function mlBlock(ac) {{
            var ml = ac.ml;
            if (!ml) return '';
            var divider = '<div style="height:1px;background:#263340;margin:8px 0;"></div>';
            var head = divider + '<div style="font-size:10px;letter-spacing:.08em;color:#B98CE0;' +
                'margin-bottom:4px;">&#129302; ML ANALYSIS &middot; ISOLATION FOREST</div>' +
                (ml.baseline === 'sim' ? '<div style="font-size:10.5px;color:#7C8B98;margin-bottom:3px;">Baseline: simulated approaches</div>' : '');
            if (ml.verdict === 'OUT_OF_SCOPE') {{
                return head + dataRow('ML', 'n/a for this runway') +
                    '<div style="font-size:11px;color:#7C8B98;">' + ml.note + '</div>';
            }}
            if (ml.verdict === 'UNAVAILABLE') {{
                return head + dataRow('ML', 'unavailable');
            }}
            if (ml.verdict === 'COLLECTING') {{
                var pct = Math.round((ml.progress || 0) * 100);
                return head + dataRow('Status', 'collecting ' + ml.samples + '/' + ml.needed + ' fixes') +
                    '<div style="height:4px;background:#263340;border-radius:2px;margin-top:4px;">' +
                    '<div style="height:4px;width:' + pct + '%;background:#B98CE0;border-radius:2px;"></div></div>';
            }}
            var bad = ml.verdict === 'ABNORMAL';
            var sim = ml.baseline === 'sim';
            var col = bad ? (sim ? '#F2A73B' : '#FF5C5C') : '#3ACB7E';
            var label = (bad && sim) ? 'UNUSUAL vs simulated baseline' : ml.verdict;
            var html = head +
                '<div style="font-size:13px;font-weight:600;color:' + col + ';margin:2px 0 4px;">' +
                label + (ml.provisional ? ' <span style="font-size:10px;font-weight:400;color:#7C8B98;">provisional</span>' : '') + '</div>' +
                dataRow('Anomaly score', ml.score) +
                dataRow('Typicality', 'above ' + ml.percentile + '% of training');
            var ruleAlert = ac.status.indexOf('ALERT') >= 0;
            if (bad && !ruleAlert) {{
                html += '<div style="font-size:11px;color:#F2A73B;margin-top:5px;">ML flags a pattern the fixed thresholds did not.</div>';
            }} else if (!bad && ruleAlert) {{
                html += '<div style="font-size:11px;color:#F2A73B;margin-top:5px;">Threshold tripped, but ML sees a typical approach.</div>';
            }} else {{
                html += '<div style="font-size:11px;color:#7C8B98;margin-top:5px;">Rules and ML agree.</div>';
            }}
            if (bad && ml.reasons && ml.reasons.length) {{
                html += '<div style="font-size:11px;color:#7C8B98;margin-top:6px;">Why:</div>';
                ml.reasons.forEach(function(r) {{
                    html += dataRow(r.feature, r.value + ' (' + r.sigma + '&sigma; ' + r.direction + ')');
                }});
            }}
            return html;
        }}

        function esc(t) {{
            return String(t).replace(/[&<>"']/g, function(c) {{
                return {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c];
            }});
        }}

        function atcRows(ac) {{
            var a = ac.atc;
            if (!a) return '';
            var out = dataRow('Sequence', '#' + a.seq_pos + ' for ' + a.runway) +
                dataRow('ETA to touchdown', a.eta_label);
            if (a.gap_nm !== null && a.gap_nm !== undefined) {{
                out += dataRow('Gap to ' + esc(a.leader), a.gap_nm + ' NM' + (a.gap_s ? ' / ' + a.gap_s + ' s' : ''));
            }}
            if (a.headline) out += dataRow('Attention', esc(a.headline));
            return out + '<div style="height:1px;background:#263340;margin:8px 0;"></div>';
        }}

        function renderAtc(data) {{
            var seq = data.sequence || [];
            var box = document.getElementById('atc-seq');
            if (!seq.length) {{
                box.innerHTML = '<div class="atc-empty">No arrivals on final</div>';
            }} else {{
                box.innerHTML = seq.map(function(r) {{
                    var gap = (r.gap_nm === null || r.gap_nm === undefined) ? 'lead' : r.gap_nm + ' NM';
                    return '<div class="atc-row"><span class="n">' + r.pos + '</span>' +
                        '<span class="cs">' + esc(r.callsign) + '<small>' + esc(r.runway.replace('RWY ', '')) + '</small></span>' +
                        '<span class="eta">' + r.eta_label + '</span>' +
                        '<span class="gap">' + gap + '</span>' +
                        '<span class="st" style="background:' + r.color + ';"></span></div>';
                }}).join('');
            }}

            var attn = data.attention || [];
            document.getElementById('atc-attn-wrap').style.display = attn.length ? 'block' : 'none';
            document.getElementById('atc-attn').innerHTML = attn.map(function(c) {{
                var html = '<div class="atc-card" style="border-color:' + c.color + ';">' +
                    '<div class="hd"><span style="color:' + c.color + ';">' + (c.level >= 2 ? 'ACT NOW' : 'WATCH') + ' &middot; ' + esc(c.callsign) + '</span>' +
                    (c.ml_assisted ? '<span class="ml">ML</span>' : '') + '</div>' +
                    '<div class="hl">' + esc(c.headline) + '</div>';
                if (c.detail) html += '<div class="dt">' + esc(c.detail) + '</div>';
                if (c.loc_dots !== null && c.loc_dots !== undefined) {{
                    var d = Math.max(-2.5, Math.min(2.5, c.loc_dots));
                    var pos = 50 - (d / 2.5) * 45;
                    html += '<div class="cdi"><div class="tr"></div>' +
                        '<div class="dot" style="left:10%;"></div><div class="dot" style="left:30%;"></div>' +
                        '<div class="mid"></div>' +
                        '<div class="dot" style="left:70%;"></div><div class="dot" style="left:90%;"></div>' +
                        '<div class="mk" style="left:' + pos + '%;background:' + c.color + ';"></div></div>';
                    var side = c.loc_dots > 0 ? 'left' : 'right';
                    var cap = 'Localizer ' + Math.abs(c.loc_dots).toFixed(1) + ' dots ' + side;
                    if (c.gs_deviation !== null && c.gs_deviation !== undefined) {{
                        cap += ' &middot; glideslope ' + Math.abs(c.gs_deviation).toFixed(1) + '&deg; ' + (c.gs_deviation > 0 ? 'high' : 'low');
                    }}
                    html += '<div class="cdi-cap">' + cap + '</div>';
                }}
                if (c.action) html += '<div class="act"><span>Suggested:</span> ' + esc(c.action) + '</div>';
                return html + '</div>';
            }}).join('');

            var minSp = data.atc_config ? data.atc_config.min_spacing_nm : 3;
            document.getElementById('atc-foot').innerText =
                'Min spacing ' + minSp.toFixed(1) + ' NM. Prototype aid on ~45 s ADS-B data.';
        }}

        function aircraftPopup(ac) {{
            var rows =
                dataRow('Status', ac.status) +
                atcRows(ac) +
                dataRow('Altitude', (ac.altitude_ft ?? 'N/A') + ' ft') +
                dataRow('Heading', (ac.heading ?? 'N/A') + '&deg;') +
                dataRow('Distance from TRV', (ac.distance_nm ?? 'N/A') + ' NM') +
                dataRow('Runway', ac.runway ?? 'N/A') +
                dataRow('Threshold distance', (ac.runway_distance ?? 'N/A') + ' NM') +
                '<div style="height:1px;background:#263340;margin:8px 0;"></div>' +
                dataRow('Localizer deviation', ac.loc_deviation + '&deg; (' + ac.loc_dots + ' dots)') +
                dataRow('Glide slope', ac.gs_angle + '&deg;') +
                dataRow('GS deviation', ac.gs_deviation + '&deg; (' + ac.gs_dots + ' dots)') +
                mlBlock(ac);
            return popupShell(esc(ac.callsign) + (ac.synthetic ? ' <span class="simtag">SIMULATED</span>' : ''), rows);
        }}

        function landedPopup(ev) {{
            var rows =
                dataRow('Runway', ev.runway) +
                dataRow('Landed at', ev.landed_time) +
                dataRow('Altitude before touchdown', ev.altitude + ' ft');
            if (ev.ml && ev.ml.verdict === 'NOT_SCORED') {{
                rows += '<div style="height:1px;background:#263340;margin:8px 0;"></div>' +
                    '<div style="font-size:10px;letter-spacing:.08em;color:#B98CE0;margin-bottom:4px;">&#129302; ML REPORT</div>' +
                    '<div style="font-size:12px;color:#7C8B98;">Not scored: ' + esc(ev.ml.note || 'too little approach data') + '</div>';
            }} else if (ev.ml) {{
                var bad = ev.ml.verdict === 'ABNORMAL';
                var simB = ev.ml.baseline === 'sim';
                rows += '<div style="height:1px;background:#263340;margin:8px 0;"></div>' +
                    '<div style="font-size:10px;letter-spacing:.08em;color:#B98CE0;margin-bottom:4px;">&#129302; ML REPORT &middot; WHOLE APPROACH</div>' +
                    '<div style="font-size:13px;font-weight:600;color:' + (bad ? (simB ? '#F2A73B' : '#FF5C5C') : '#3ACB7E') + ';margin:2px 0 4px;">' +
                        ((bad && simB) ? 'UNUSUAL vs simulated baseline' : ev.ml.verdict) + '</div>' +
                    dataRow('Anomaly score', ev.ml.score) +
                    dataRow('Baseline', ev.ml.baseline === 'sim' ? 'simulated' : 'real landings') +
                    dataRow('Rule-based checker', ev.ml.rule_flagged ? 'flagged' : 'normal') +
                    dataRow('Logged to dataset', ev.synthetic ? 'no (demo)' : 'yes');
                if (bad && ev.ml.reasons && ev.ml.reasons.length) {{
                    rows += '<div style="font-size:11px;color:#7C8B98;margin-top:6px;">Why:</div>';
                    ev.ml.reasons.forEach(function(r) {{
                        rows += dataRow(r.feature, r.value + ' (' + r.sigma + '&sigma; ' + r.direction + ')');
                    }});
                }}
            }}
            return popupShell(esc(ev.callsign) + ' &middot; landed' + (ev.synthetic ? ' <span class="simtag">SIMULATED</span>' : ''), rows);
        }}

        // ------------------------------------------------------------
        // Smoothing between polls -- WITHOUT dead-reckoning
        // ------------------------------------------------------------
        // The first version of this smoothing extrapolated each marker
        // forward every second using its last known heading and ground
        // speed, then snapped it to the real position on the next poll.
        // That's what caused the "going back and forth" bug at landing:
        // near the ground, ADS-B heading/speed fields get noisy or briefly
        // null (reduced squitter rate during flare/rollout), so the
        // extrapolation would guess a position that was wrong, and the
        // next real fix would yank the marker back -- forward, then
        // backward, repeatedly.
        //
        // This version never guesses ahead of confirmed data. It only
        // ever animates BETWEEN two real fixes it already has (the
        // previous poll's position and the new poll's position), over
        // exactly the time that poll interval took. It can lag the true
        // position by up to one poll interval, but it can never overshoot
        // or reverse, because it's not predicting anything -- just
        // smoothly showing the transition between two known truths.
        var aircraftMarkers = {{}};  // callsign -> {{marker, prevLat, prevLon, prevHeading, curLat, curLon, curHeading, pollStart, pollDuration}}

        function lerp(a, b, t) {{ return a + (b - a) * t; }}

        function lerpAngle(a, b, t) {{
            if (typeof a !== 'number') return b;
            if (typeof b !== 'number') return a;
            var diff = ((b - a + 540) % 360) - 180;
            return (a + diff * t + 360) % 360;
        }}

        // Real fixes arrive every poll_interval_s; demo flights update every demo tick.
        function fixDuration(ac, data) {{
            if (ac.synthetic) return ((data.demo && data.demo.tick_s) || 2) * 1000 + 600;
            return data.poll_interval_s ? data.poll_interval_s * 1000 : 30000;
        }}

        function setTip(entry, kind, html, opts) {{
            if (entry.tipKind === kind && entry.marker.getTooltip()) {{
                entry.marker.setTooltipContent(html);
            }} else {{
                entry.marker.unbindTooltip();
                entry.marker.bindTooltip(html, opts);
                entry.tipKind = kind;
            }}
        }}

        function render(data) {{
            var now = Date.now();
            var seenCallsigns = {{}};

            data.aircraft.forEach(function(ac) {{
                seenCallsigns[ac.callsign] = true;
                var entry = aircraftMarkers[ac.callsign];

                if (!entry) {{
                    var marker = L.marker([ac.lat, ac.lon], {{icon: planeIcon(ac.color, ac.heading, ac.atc && ac.atc.ml_assisted)}})
                        .addTo(aircraftLayer);
                    entry = aircraftMarkers[ac.callsign] = {{
                        marker: marker,
                        prevLat: ac.lat, prevLon: ac.lon, prevHeading: ac.heading,
                        curLat: ac.lat, curLon: ac.lon, curHeading: ac.heading,
                        pollStart: now, pollDuration: fixDuration(ac, data),
                        color: ac.color,
                        ring: !!(ac.atc && ac.atc.ml_assisted),
                    }};
                }} else {{
                    // The new confirmed fix becomes the target we animate
                    // TO; the position we were just showing becomes the
                    // start point -- never a guessed/extrapolated point.
                    // Polling is now every 2 s but real fixes only change every ~45 s,
                    // so only restart the glide when the position actually moved.
                    if (ac.lat !== entry.curLat || ac.lon !== entry.curLon) {{
                        entry.prevLat = entry.curLat;
                        entry.prevLon = entry.curLon;
                        entry.prevHeading = entry.curHeading;
                        entry.curLat = ac.lat;
                        entry.curLon = ac.lon;
                        entry.curHeading = ac.heading;
                        entry.pollStart = now;
                        entry.pollDuration = fixDuration(ac, data);
                    }}
                    entry.color = ac.color;
                    entry.ring = !!(ac.atc && ac.atc.ml_assisted);
                }}

                entry.marker.bindPopup(aircraftPopup(ac));
                var simTag = ac.synthetic ? '<span class="simtag">SIM</span>' : '';
                if (ac.atc) {{
                    var rwyShort = (ac.atc.runway || '').replace('RWY ', '');
                    var lbl = '<span style="color:' + ac.atc.color + ';">&#9679;</span> <b>' + esc(ac.callsign) + '</b>' + simTag + ' &middot; ' + esc(rwyShort) + '<br>' +
                        ac.atc.distance_nm + ' NM &middot; ETA ' + ac.atc.eta_label;
                    if (ac.atc.gap_nm !== null && ac.atc.gap_nm !== undefined) {{
                        lbl += '<br><small>gap ' + ac.atc.gap_nm + ' NM' +
                            (ac.atc.gap_s ? ' &middot; ' + ac.atc.gap_s + ' s' : '') + '</small>';
                    }}
                    setTip(entry, 'atc', lbl, {{
                        permanent: true, direction: 'right', offset: [16, 0],
                        className: 'tracils-label'
                    }});
                    entry.labelColor = ac.atc.color;
                }} else {{
                    setTip(entry, 'plain', esc(ac.callsign) + simTag + ' | ' + esc(ac.status), {{}});
                }}
            }});

            // Drop markers for aircraft no longer in this update (landed,
            // suppressed during the recently-landed cooldown, or left
            // coverage) -- these are removed immediately, not animated.
            Object.keys(aircraftMarkers).forEach(function(callsign) {{
                if (!seenCallsigns[callsign]) {{
                    aircraftLayer.removeLayer(aircraftMarkers[callsign].marker);
                    delete aircraftMarkers[callsign];
                }}
            }});

            // Approach lines are cheap to just redraw each poll.
            aircraftLayer.eachLayer(function(layer) {{
                if (layer instanceof L.Polyline) aircraftLayer.removeLayer(layer);
            }});
            data.aircraft.forEach(function(ac) {{
                if (ac.runway_threshold) {{
                    L.polyline([[ac.lat, ac.lon], ac.runway_threshold], {{
                        color: ac.approach_line_color, weight: 3,
                        dashArray: '8,8', opacity: 0.8
                    }}).addTo(aircraftLayer);
                }}
            }});

            var landedSig = data.landed.map(function(e) {{ return e.callsign + '@' + e.landed_time; }}).join('|');
            if (landedSig !== window._tracilsLandedSig) {{
              window._tracilsLandedSig = landedSig;
              landedLayer.clearLayers();
              data.landed.forEach(function(ev) {{
                L.polyline([[ev.last_lat, ev.last_lon], [ev.lat, ev.lon]], {{
                    color: '#B98CE0', weight: 2, opacity: 0.6, dashArray: '4,8'
                }}).addTo(landedLayer);

                L.marker([ev.lat, ev.lon], {{icon: landedIcon(ev.icon_rotation)}})
                    .bindPopup(landedPopup(ev))
                    .bindTooltip(ev.callsign + (ev.synthetic ? ' (SIM)' : '') + ' | LANDED ' + ev.landed_time)
                    .addTo(landedLayer);
              }});
            }}

            var heroEl = document.getElementById('landing-count');
            heroEl.innerText = data.landing_count;
            heroEl.classList.toggle('active', data.landing_count > 0);

            document.getElementById('tracked-count').innerText = data.total;
            renderAtc(data);
            document.getElementById('build-tag').innerText = 'build ' + (data.build || 'unknown (old backend)');

            if (data.ml_model && data.ml_model.available) {{
                document.getElementById('ml-name').innerText = data.ml_model.name;
                document.getElementById('ml-train').innerText = data.ml_model.trained_on;
                var mb = document.getElementById('ml-base');
                if (data.ml_model.tier === 'sim') {{
                    mb.innerText = 'Starter baseline. Switches to real data after ' +
                        data.ml_model.live_needed + ' logged landings (' + data.ml_model.live_rows + ' so far).';
                }} else {{
                    mb.innerText = 'Baseline: real logged landings.';
                }}
                var mw = document.getElementById('ml-warn');
                if (data.ml_model.baseline_warning) {{
                    mw.style.display = 'block';
                    mw.innerText = 'High flag rate (' + Math.round(data.ml_model.recent_flag_rate * 100) +
                        '%): baseline probably does not match real traffic yet.';
                }} else {{
                    mw.style.display = 'none';
                }}
            }} else {{
                document.getElementById('ml-name').innerText = 'unavailable';
            }}
            var mlScoring = 0, mlFlagged = 0;
            data.aircraft.forEach(function(a) {{
                if (a.ml && (a.ml.verdict === 'NORMAL' || a.ml.verdict === 'ABNORMAL')) {{
                    mlScoring++;
                    if (a.ml.verdict === 'ABNORMAL') mlFlagged++;
                }}
            }});
            document.getElementById('ml-scoring').innerText = mlScoring;
            document.getElementById('ml-flagged').innerText = mlFlagged;
            document.getElementById('last-update').innerText =
                data.real_timestamp ? new Date(data.real_timestamp * 1000).toLocaleTimeString() : 'no live feed';

            window._tracilsLastDataTimestamp = data.real_timestamp || null;
            window._tracilsFeedError = data.feed_error || null;
            var fb = document.getElementById('feed-banner');
            var realN = data.real_count || 0;
            if (data.feed_error) {{
                fb.style.display = 'block';
                fb.innerHTML = '<b>Live traffic problem</b><br>' + esc(data.feed_error) +
                    '<br><span style="color:#9a7a7a;">Simulated flights still work.</span>';
            }} else if (data.real_timestamp && realN === 0 && data.aircraft.filter(function(a) {{ return !a.synthetic; }}).length === 0) {{
                fb.style.display = 'block';
                fb.style.borderColor = '#F2A73B'; fb.style.color = '#F2D9A8'; fb.style.background = '#2a2417';
                fb.innerHTML = 'Feed connected, but OpenSky reports 0 aircraft in the area right now.';
            }} else {{
                fb.style.display = 'none';
                fb.style.borderColor = '#FF5C5C'; fb.style.color = '#FFB3B3'; fb.style.background = '#2a1a1a';
            }}
            window._tracilsGotPayload = true;
            window._tracilsPollIntervalS = data.poll_interval_s || 30;

            // Auto-fit to traffic ONLY the very first time data arrives.
            // After that we never touch the view again, so the user's
            // zoom/pan sticks.
            if (!didInitialFit && data.aircraft.length > 0) {{
                var pts = data.aircraft.map(function(ac) {{ return [ac.lat, ac.lon]; }});
                pts.push([{TRV_LAT}, {TRV_LON}]);
                map.fitBounds(pts, {{maxZoom: 9}});
                didInitialFit = true;
            }}
        }}

        function tickPositions() {{
            var now = Date.now();
            Object.keys(aircraftMarkers).forEach(function(callsign) {{
                var e = aircraftMarkers[callsign];
                var t = Math.min(1, (now - e.pollStart) / e.pollDuration);
                var lat = lerp(e.prevLat, e.curLat, t);
                var lon = lerp(e.prevLon, e.curLon, t);
                var heading = lerpAngle(e.prevHeading, e.curHeading, t);
                e.marker.setLatLng([lat, lon]);
                e.marker.setIcon(planeIcon(e.color, heading, e.ring));
            }});
        }}
        setInterval(tickPositions, 200);

        // ------------------------------------------------------------
        // Stale-data indicator
        // ------------------------------------------------------------
        // OpenSky's own credit budget puts a hard floor under how often
        // this can poll (see adsb_receiver.py); if a poll is delayed or
        // fails (e.g. rate-limited), the UI should say so plainly rather
        // than silently keep showing old positions as if they were live.
        function tickStaleness() {{
            var dot = document.getElementById('live-dot');
            var label = document.getElementById('live-label');
            if (!dot || !label || !window._tracilsGotPayload) return;
            if (!window._tracilsLastDataTimestamp) {{
                dot.classList.add('stale');
                label.innerText = 'No live feed';
                label.title = window._tracilsFeedError || 'waiting for first OpenSky response';
                return;
            }}
            label.title = window._tracilsFeedError || '';
            var ageS = (Date.now() / 1000) - window._tracilsLastDataTimestamp;
            var staleAfter = (window._tracilsPollIntervalS || 30) * 2;
            if (ageS > staleAfter) {{
                dot.classList.add('stale');
                label.innerText = 'Delayed ' + Math.round(ageS) + 's';
            }} else {{
                dot.classList.remove('stale');
                label.innerText = 'Live';
            }}
        }}
        setInterval(tickStaleness, 1000);

        function poll() {{
            fetch('/api/data')
                .then(function(r) {{ return r.json(); }})
                .then(render)
                .catch(function(e) {{ console.error('TRACILS refresh failed:', e); }});
        }}

        poll();                  // first load
        setInterval(poll, 2000); // local-server poll only; OpenSky itself is still fetched every ~{poll_interval_seconds} s server-side
    }})();
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(live_js))

    import demo_panel
    m.get_root().html.add_child(folium.Element(demo_panel.render(map_var, TRV_LAT, TRV_LON)))

    m.save(output)
    print(f"Static map shell written to {output} (built once, updates live from now on).")
    return output


if __name__ == "__main__":

    print(
        "Fetching live ADS-B aircraft..."
    )

    planes = get_aircraft()

    draw_map(planes)

    print(
        "TRACILS map updated."
    )