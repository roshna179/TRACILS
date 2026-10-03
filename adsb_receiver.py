import requests, json, math, time

# TRV Airport coordinates
TRV_LAT = 8.4821
TRV_LON = 76.9199
RADIUS_NM = 250

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
STATES_URL = "https://opensky-network.org/api/states/all"

BBOX = {"lamin": 4.5, "lomin": 72.9, "lamax": 12.5, "lomax": 80.9}

# --------------------------------------------------------------------
# OpenSky credit economics
# --------------------------------------------------------------------
# /states/all is billed in "credits" based on the bounding box AREA, not
# per-second. The brackets are: <=25 sq deg -> 1 credit, 25-100 -> 2,
# 100-400 -> 3, >400 -> 4. Our TRV box is 64 sq deg -> 2 credits/call.
#
# Daily budgets: 400/day anonymous, 4000/day standard authenticated
# accounts (what credentials.json is set up for), 8000/day if your
# account also feeds OpenSky its own ADS-B data with >=30% uptime.
#
# This means there is a hard floor on how often this can poll and stay
# sustainable for 24/7 operation: 4000 credits / 2 credits-per-call =
# 2000 calls/day = one call roughly every 43 seconds. Polling faster
# than that (like earlier 10s/30s settings) isn't a "the code is slow"
# problem -- it borrows against a budget that runs out partway through
# the day, and once it's spent every call 429s until the daily reset,
# which is what silently froze/staled the data ("lagging again").
#
# If you have an "Active Feeder" account (8000/day), that floor drops
# to ~22s. There's no way to safely poll faster than your account's
# actual credit budget allows without either shrinking the bbox (ours
# is already about as small as it can be for 250nm coverage) or getting
# a paid/licensed tier. This is a real ceiling on how close this can
# get to something like FlightRadar24, which blends thousands of
# proprietary low-latency feeders instead of one public, credit-metered
# endpoint.
CREDITS_PER_CALL = 2                 # matches the 25-100 sq deg bracket above
DAILY_CREDIT_BUDGET = 4000           # set to 8000 if you have an Active Feeder account
MIN_SAFE_INTERVAL_S = math.ceil(86400 / (DAILY_CREDIT_BUDGET / CREDITS_PER_CALL))

_token_cache = {"token": None, "expires_at": 0}

# Tracks OpenSky's own reported remaining balance, when it sends one,
# so callers can see the real budget instead of guessing.
last_rate_limit_info = {"remaining": None, "checked_at": None}


class RateLimitError(Exception):
    """Raised when OpenSky returns 429 -- credit budget exhausted for now."""
    def __init__(self, retry_after_s):
        self.retry_after_s = retry_after_s
        super().__init__(f"OpenSky rate-limited us; retry after {retry_after_s}s")


class OpenSkyAuthError(Exception):
    pass


def get_token():
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 30:
        return _token_cache["token"]

    import os
    cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
    with open(cred_path) as f:
        creds = json.load(f)
    resp = requests.post(
        TOKEN_URL,
        timeout=15,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"]
        }
    )
    if resp.status_code != 200:
        raise OpenSkyAuthError(f"Token request failed ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 1800)
    return _token_cache["token"]


def haversine_nm(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) / 1.852


def _read_rate_limit_headers(resp):
    remaining = resp.headers.get("X-Rate-Limit-Remaining")
    if remaining is not None:
        try:
            last_rate_limit_info["remaining"] = int(remaining)
            last_rate_limit_info["checked_at"] = time.time()
        except ValueError:
            pass


def get_aircraft():
    token = get_token()
    resp = requests.get(
        STATES_URL,
        headers={"Authorization": f"Bearer {token}"},
        params=BBOX,
        timeout=15,
    )

    _read_rate_limit_headers(resp)

    if resp.status_code == 429:
        # OpenSky's own docs note this header is sometimes unreliable;
        # fall back to a conservative default if it's missing.
        retry_after = resp.headers.get("X-Rate-Limit-Retry-After-Seconds")
        try:
            retry_after = int(retry_after)
        except (TypeError, ValueError):
            retry_after = MIN_SAFE_INTERVAL_S
        raise RateLimitError(retry_after)

    if resp.status_code != 200:
        raise RuntimeError(f"OpenSky states/all failed ({resp.status_code}): {resp.text[:200]}")

    states = resp.json().get("states", []) or []

    aircraft = []
    for s in states:
        lat, lon = s[6], s[5]
        if lat is None or lon is None:
            continue
        dist = haversine_nm(TRV_LAT, TRV_LON, lat, lon)
        if dist <= RADIUS_NM:
            alt_m = s[7] or 0
            alt_ft = alt_m * 3.28084
            # OpenSky gives PRESSURE altitude; correct to true altitude (MSL)
            # with the VOTV QNH so glideslope maths is not biased ~100-150 ft high.
            if not s[8]:
                try:
                    import qnh
                    alt_ft = qnh.pressure_to_true_ft(alt_ft)
                except Exception:
                    pass
            aircraft.append({
                "callsign":    (s[1] or "???").strip(),
                "icao24":      s[0],
                "lat":         lat,
                "lon":         lon,
                "altitude_ft": round(alt_ft),
                "heading":     s[10],
                "speed_ms":    s[9],
                "distance_nm": round(dist, 1),
                "on_ground":   bool(s[8]) if len(s) > 8 else False,
                "on_approach": dist < 10 and alt_ft < 3000
            })

    remaining_note = ""
    if last_rate_limit_info["remaining"] is not None:
        remaining_note = f" ({last_rate_limit_info['remaining']} OpenSky credits left today)"
    print(f"✅ {len(aircraft)} aircraft within {RADIUS_NM}nm of TRV{remaining_note}")
    return aircraft


if __name__ == "__main__":
    planes = get_aircraft()
    print(f"\n{'Callsign':<12}{'Lat':>8}{'Lon':>9}{'Alt(ft)':>10}{'Dist(nm)':>10}{'Approach':>10}")
    print("-" * 60)
    for p in sorted(planes, key=lambda x: x["distance_nm"]):
        print(f"  ✈ {p['callsign']:<10}{p['lat']:>8.3f}{p['lon']:>9.3f}"
              f"{p['altitude_ft']:>10}{p['distance_nm']:>10}"
              f"{'YES' if p['on_approach'] else '':>10}")
