"""
qnh.py -- turn OpenSky's PRESSURE altitude into true altitude (MSL).

OpenSky's baro_altitude is measured against the standard 1013.25 hPa. The
real altimeter setting at VOTV (QNH) is usually lower in the monsoon-
influenced Kerala climate, so pressure altitude reads HIGH. At 1.5 NM on a
perfect 3 deg path (474 ft) a QNH of ~1008 hPa makes an aircraft look
~145 ft high, i.e. ~0.9 deg above the glidepath, which is exactly the
offset seen on normal approaches before this correction.

QNH comes from the raw VOTV METAR (aviationweather.gov), cached for
30 minutes. If it cannot be fetched, NO correction is applied (and one
warning is printed) rather than guessing.
"""
import re, time
import requests

URL = "https://aviationweather.gov/api/data/metar?ids=VOTV&format=raw"
CACHE_S = 1800
FT_PER_HPA = 27.3          # near sea level
STD_HPA = 1013.25

_cache = {"hpa": None, "at": 0.0, "warned": False}


def get_qnh_hpa():
    now = time.time()
    if _cache["hpa"] is not None and now - _cache["at"] < CACHE_S:
        return _cache["hpa"]
    try:
        r = requests.get(URL, timeout=5)
        txt = r.text
        m = re.search(r"\bQ(\d{4})\b", txt)
        if m:
            _cache.update(hpa=float(m.group(1)), at=now, warned=False)
            print(f"  QNH {_cache['hpa']:.0f} hPa (VOTV METAR)")
            return _cache["hpa"]
        a = re.search(r"\bA(\d{4})\b", txt)       # inches of mercury, just in case
        if a:
            _cache.update(hpa=float(a.group(1)) / 100 * 33.8639, at=now, warned=False)
            return _cache["hpa"]
        raise ValueError("no QNH group in METAR")
    except Exception as e:
        if not _cache["warned"]:
            print(f"  (QNH unavailable, altitudes left uncorrected: {e})")
            _cache["warned"] = True
        return _cache["hpa"]        # last good value, or None


def pressure_to_true_ft(pressure_alt_ft):
    hpa = get_qnh_hpa()
    if hpa is None:
        return pressure_alt_ft
    return pressure_alt_ft - (STD_HPA - hpa) * FT_PER_HPA
