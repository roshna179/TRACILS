"""
atc_metrics.py -- controller-facing metrics for TRACILS.

Turns the raw per-aircraft ILS numbers and ML verdicts into the few things
an approach controller actually needs:

  1. ETA to touchdown            (distance to threshold / ground speed)
  2. Arrival sequence per runway (who is first, second, third)
  3. Spacing to the aircraft ahead, in NM and seconds
  4. ONE merged status per aircraft: stable / watch / act
     (rule-based ILS + ML + spacing folded into a single colour)
  5. A plain-language headline and suggested action for anything that
     is not stable -- so deviation numbers can stay hidden until needed.

Everything here is a decision-SUPPORT aid built on ADS-B data that is
refreshed roughly every 45 s. It is not an operational tool.

TUNABLE, NOT VALIDATED: every threshold below is a starting value chosen
for the prototype, not a figure from ATC regulations. Check them with a
real controller before relying on them.
"""

from ils_checker import heading_difference, MAX_APPROACH_DISTANCE, MAX_LANDING_ALTITUDE

MS_TO_KT = 1.94384

MIN_SPACING_NM = 3.0        # below this gap -> watch
ACT_SPACING_NM = 2.0        # below this gap -> act
ALIGN_TOLERANCE_DEG = 40    # heading vs runway heading to count as "arriving"
MIN_SPEED_KT = 40           # below this, ETA is meaningless (taxi / stopped)
LOC_ACT_DOTS = 2.0          # full-scale localizer deflection
GS_WATCH_DEG = 3.0          # glideslope offset treated as worth watching.
                            # ils_checker's own 0.5 deg limit fires on almost
                            # every approach in the training data (mean |GS
                            # deviation| ~1.75 deg), so using it for the merged
                            # colour would turn nearly everything amber.

COLORS = {"stable": "#3ACB7E", "watch": "#F2A73B", "act": "#FF5C5C"}
LEVEL_NAME = {0: "stable", 1: "watch", 2: "act"}


def eta_seconds(distance_nm, speed_ms):
    if distance_nm is None or speed_ms is None:
        return None
    kt = speed_ms * MS_TO_KT
    if kt < MIN_SPEED_KT:
        return None
    return distance_nm / kt * 3600.0


def fmt_eta(seconds):
    if seconds is None:
        return "--:--"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def is_arriving(ac, runway):
    if not runway or runway["distance"] > MAX_APPROACH_DISTANCE:
        return False
    alt, hdg = ac.get("altitude_ft"), ac.get("heading")
    if alt is None or hdg is None or alt > MAX_LANDING_ALTITUDE:
        return False
    return heading_difference(hdg, runway["heading"]) <= ALIGN_TOLERANCE_DEG


def _side(loc_dev_deg):
    # loc_deviation > 0 means left of the extended centerline as seen by
    # the pilot flying the approach (bearing from threshold rotated
    # clockwise = the pilot's left).
    return "left" if loc_dev_deg > 0 else "right"


def build_atc(entries):
    """
    entries: list of dicts {ac, ils, runway, ml} for every aircraft this
             cycle (ml may be None). Aircraft not arriving are ignored.

    Returns (per_callsign, sequence, attention):
      per_callsign -> {callsign: atc dict attached to that aircraft}
      sequence     -> rows for the arrival-sequence panel
      attention    -> cards for the needs-attention panel
    """
    arriving = [e for e in entries if is_arriving(e["ac"], e["runway"])]

    by_rwy = {}
    for e in arriving:
        by_rwy.setdefault(e["runway"]["runway"], []).append(e)

    per, sequence, attention = {}, [], []

    for rwy, group in by_rwy.items():
        group.sort(key=lambda e: e["runway"]["distance"])
        leader = None
        for pos, e in enumerate(group, start=1):
            ac, ils, ml = e["ac"], e["ils"], e.get("ml") or {}
            callsign = ac.get("callsign", "N/A")
            dist = e["runway"]["distance"]
            eta = eta_seconds(dist, ac.get("speed_ms"))

            gap_nm = gap_s = None
            leader_cs = None
            if leader is not None:
                gap_nm = dist - leader["dist"]
                if eta is not None and leader["eta"] is not None:
                    gap_s = eta - leader["eta"]
                leader_cs = leader["callsign"]

            level, headline, kind = 0, None, None

            def raise_to(lv, text, k):
                nonlocal level, headline, kind
                if lv > level:
                    level, headline, kind = lv, text, k

            loc_dev, loc_dots = ils.get("loc_deviation"), ils.get("loc_dots")
            gs_dev, gs_dots = ils.get("gs_deviation"), ils.get("gs_dots")

            rule_hit = False
            if loc_dev is not None:
                if loc_dots is not None and abs(loc_dots) >= LOC_ACT_DOTS:
                    raise_to(2, f"Far {_side(loc_dev)} of centerline", "loc")
                    rule_hit = True
                elif ils.get("loc_alert"):
                    raise_to(1, f"{_side(loc_dev).capitalize()} of centerline", "loc")
                    rule_hit = True
            if gs_dev is not None and abs(gs_dev) > GS_WATCH_DEG:
                raise_to(1, "High on glidepath" if gs_dev > 0 else "Low on glidepath", "gs")
                rule_hit = True

            # A flag measured against the SIMULATED starter baseline is shown in
            # the popup but does not recolour the aircraft; only a baseline built
            # from real logged landings is trusted to do that.
            ml_flag = ml.get("verdict") == "ABNORMAL" and ml.get("baseline") == "field"
            if ml_flag:
                if rule_hit:
                    raise_to(2, headline or "Unusual approach", "ml+rule")
                else:
                    text = (f"Drifting {_side(loc_dev)} of centerline"
                            if loc_dev is not None and abs(loc_dev) > 0.15 else "Unusual approach pattern")
                    raise_to(1, text, "ml")

            if gap_nm is not None:
                if gap_nm < ACT_SPACING_NM:
                    raise_to(2, f"Too close to {leader_cs}", "spacing")
                elif gap_nm < MIN_SPACING_NM:
                    raise_to(1, f"Closing on {leader_cs}", "spacing")

            if level == 0:
                action = None
            elif kind == "spacing":
                action = "Check spacing; consider speed adjustment" if level == 1 else "Increase spacing or consider go-around"
            elif level == 2:
                action = "Consider go-around or vector"
            elif kind == "ml":
                action = "Monitor; correct heading if drift continues"
            else:
                action = "Monitor"

            status = LEVEL_NAME[level]
            color = COLORS[status]
            ml_only = ml_flag and not rule_hit

            per[callsign] = {
                "runway": rwy,
                "seq_pos": pos,
                "distance_nm": round(dist, 1),
                "eta_s": None if eta is None else round(eta),
                "eta_label": fmt_eta(eta),
                "gap_nm": None if gap_nm is None else round(gap_nm, 1),
                "gap_s": None if gap_s is None else round(gap_s),
                "leader": leader_cs,
                "level": level,
                "status": status,
                "color": color,
                "headline": headline,
                "action": action,
                "ml_assisted": ml_flag,
            }

            sequence.append({
                "pos": pos, "runway": rwy, "callsign": callsign,
                "distance_nm": round(dist, 1), "eta_label": fmt_eta(eta),
                "gap_nm": None if gap_nm is None else round(gap_nm, 1),
                "level": level, "status": status, "color": color,
            })

            if level >= 1:
                attention.append({
                    "callsign": callsign, "runway": rwy, "level": level,
                    "status": status, "color": color, "headline": headline,
                    "detail": ("Rule limits not exceeded; ML sees a pattern unlike normal approaches."
                               if ml_only else None),
                    "loc_dots": loc_dots, "gs_dots": gs_dots,
                    "loc_deviation": loc_dev, "gs_deviation": gs_dev,
                    "action": action, "ml_assisted": ml_flag,
                    "gap_nm": None if gap_nm is None else round(gap_nm, 1),
                })

            leader = {"callsign": callsign, "dist": dist, "eta": eta}

    attention.sort(key=lambda a: -a["level"])
    return per, sequence, attention
