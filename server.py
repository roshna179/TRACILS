"""
TRACILS live server
====================
Replaces the old approach of regenerating tracils_map.html and forcing
the browser to reload every 30s with <meta http-equiv="refresh">.

What happens now:
  1. On startup, build_static_map() draws the map ONCE (tiles, runway,
     zones, legend) and saves it to tracils_map.html, along with a
     small JS polling loop baked into the page.
  2. A background thread keeps fetching OpenSky data every
     REFRESH_SECONDS and running the ILS/landing checks, exactly like
     before — it just stores the result in memory instead of writing
     a new HTML file.
  3. The browser's own JS calls GET /api/data every 30s and swaps only
     the aircraft/landed markers. The page never reloads, so your zoom
     and pan stay exactly where you left them.

Run with:  python server.py
Then leave the browser tab open — it updates itself silently.
"""

import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from adsb_receiver import get_aircraft, RateLimitError, OpenSkyAuthError, MIN_SAFE_INTERVAL_S, last_rate_limit_info
from ils_checker import run_ils_checks, check_aircraft
from map_plot import build_static_map, build_live_payload, purge_synthetic
import landing_logger
import demo_flights

# --------------------------------------------------------------------
# Poll interval -- derived from OpenSky's actual credit budget instead
# of a guessed number. See adsb_receiver.py for the math: our bounding
# box costs 2 credits/call, a standard authenticated account gets 4000
# credits/day, so ~43s is the fastest interval that can run 24/7
# without ever hitting a 429. Earlier settings of 10s and 30s both
# eventually exhausted the daily budget (in ~6h and ~17h of continuous
# running respectively) -- every call after that returned 429 and the
# payload silently stopped updating, which is what showed up as
# "lagging again": not the code getting slower, the budget running out
# mid-session.
#
# A few seconds of margin on top of the bare-minimum safe interval so
# a slow request or two doesn't tip it over.
REFRESH_SECONDS = MIN_SAFE_INTERVAL_S + 3
MAP_FILE = "tracils_map.html"
HOST = "127.0.0.1"
PORT = 5000

MAX_BACKOFF_S = 300  # cap how long we'll wait after repeated failures

app = Flask(__name__)

_lock = threading.Lock()          # guards _latest_payload (read by the browser)
_pipe_lock = threading.RLock()    # guards the detection pipeline (poll thread + demo thread both use it)
_real_planes = []                 # last good OpenSky fetch
_real_ts = None                   # when that fetch happened (None = never succeeded)
_feed_error = None                # last OpenSky problem, shown in the UI
_credits = None
_latest_payload = {
    "timestamp": time.time(),
    "real_timestamp": None,
    "feed_error": None,
    "aircraft": [],
    "landed": [],
    "landing_count": 0,
    "total": 0,
    "real_count": 0,
    "poll_interval_s": REFRESH_SECONDS,
    "credits_remaining": None,
}


def run_cycle(fresh):
    """
    Build one payload from (real traffic + demo flights) and publish it.

    fresh=True   : a brand-new OpenSky fetch just arrived (poll thread).
    fresh=False  : a demo tick. The real aircraft are re-sent unchanged, tagged
                   _repeat so landing/stagnation logic does not treat the repeat as
                   new evidence, and the real aircraft are NOT logged a second time.
    Demo flights are always (re)recorded on their own accelerated clock by the
    demo thread, never by the poll thread.
    """
    global _latest_payload
    with _pipe_lock:
        real = [dict(a) for a in _real_planes]
        if not fresh:
            for a in real:
                a["_repeat"] = True
        demo = demo_flights.current_aircraft()
        planes = real + demo

        if fresh:
            run_ils_checks(real)                       # console output/alerts (real traffic only)
            for ac in real:
                landing_logger.record_sample(ac, check_aircraft(ac))
        else:
            for ac in demo:
                landing_logger.record_sample(ac, check_aircraft(ac))

        payload = build_live_payload(planes)
        payload["poll_interval_s"] = REFRESH_SECONDS
        payload["credits_remaining"] = _credits
        payload["real_timestamp"] = _real_ts
        payload["feed_error"] = _feed_error
        payload["demo"] = {"active": demo_flights.active(), "tick_s": demo_flights.TICK_S}
        payload["real_count"] = len(real)
        with _lock:
            _latest_payload = payload


def poll_loop():
    """
    Background thread: fetch real traffic -> run the pipeline -> publish.

    Backoff is explicit: a real rate limit (429) waits for however long OpenSky
    says to (or a safe fallback); other transient errors back off exponentially
    instead of hammering the API every REFRESH_SECONDS while something is broken.
    """
    global _real_planes, _real_ts, _feed_error, _credits

    cycle = 1
    consecutive_failures = 0

    while True:
        print(f"\n{'─'*55}")
        print(f"  Cycle #{cycle} — fetching live data...")
        print(f"{'─'*55}")
        cycle_start = time.time()
        sleep_for = REFRESH_SECONDS

        try:
            planes = get_aircraft()
            _real_planes, _real_ts, _feed_error = planes, time.time(), None
            _credits = last_rate_limit_info["remaining"]
            try:
                run_cycle(fresh=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                _feed_error = f"Processing error (feed OK, {len(planes)} aircraft received): {str(e)[:80]}"
                print(f"  ⚠️  {_feed_error}")

            consecutive_failures = 0
            fetch_ms = round((time.time() - cycle_start) * 1000)
            print(f"  ✅ Live data refreshed silently in {fetch_ms}ms — no page reload.")

        except RateLimitError as e:
            consecutive_failures += 1
            sleep_for = max(e.retry_after_s, REFRESH_SECONDS)
            _feed_error = "OpenSky rate limit reached"
            print(f"  🛑 OpenSky rate-limited us (daily credit budget likely exhausted).")
            print(f"     Backing off {sleep_for}s. The map keeps showing the last good data (marked stale).")

        except FileNotFoundError:
            consecutive_failures += 1
            sleep_for = 60
            _feed_error = "credentials.json not found in tracils_live/ (live traffic off)"
            print("  ⚠️  credentials.json not found next to server.py. Live traffic is OFF;")
            print("     the demo flight button still works. Retrying in 60s.")
        except OpenSkyAuthError as e:
            consecutive_failures += 1
            sleep_for = 60
            _feed_error = f"OpenSky login failed: {str(e)[:90]}"
            print(f"  ⚠️  {e}")
            print("     Check client_id / client_secret in credentials.json. Retrying in 60s.")

        except Exception as e:
            consecutive_failures += 1
            sleep_for = min(REFRESH_SECONDS * (2 ** consecutive_failures), MAX_BACKOFF_S)
            _feed_error = f"OpenSky problem: {str(e)[:80]}"
            print(f"  ⚠️  Poll cycle failed: {e}")
            print(f"     Backing off {sleep_for}s before retrying (failure #{consecutive_failures}).")

        if _feed_error:
            # Publish the problem so the page can show it (otherwise a failed fetch
            # leaves the browser looking at an empty, silent map).
            try:
                run_cycle(fresh=False)
            except Exception:
                pass

        cycle += 1
        time.sleep(sleep_for)


def demo_loop():
    """
    Fast clock for demo flights. Idle (does nothing) until a flight is injected.
    One extra cycle runs after the last demo flight lands so its landing is logged
    to the screen and the display clears.
    """
    was_active = False
    while True:
        time.sleep(demo_flights.TICK_S)
        try:
            with _pipe_lock:
                active = demo_flights.is_active()
                if active:
                    demo_flights.step()
                if active or was_active:
                    run_cycle(fresh=False)
                was_active = active
        except Exception as e:
            print(f"  ⚠️  demo tick failed: {e}")


@app.route("/")
def index():
    return send_from_directory(".", MAP_FILE)


@app.route("/api/data")
def api_data():
    with _lock:
        return jsonify(_latest_payload)


@app.route("/api/inject", methods=["POST"])
def api_inject():
    body = request.get_json(silent=True) or {}
    try:
        with _pipe_lock:
            flights = demo_flights.inject(body.get("scenario", "drift"), body.get("runway", "RWY 32"))
            run_cycle(fresh=False)      # show it immediately, do not wait for the next tick
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    print(f"  🧪 DEMO injected: {', '.join(f['callsign'] for f in flights)} ({body.get('scenario')})")
    return jsonify({"ok": True, "flights": flights})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    with _pipe_lock:
        demo_flights.clear()
        purge_synthetic()
        run_cycle(fresh=False)
    print("  🧪 DEMO flights cleared")
    return jsonify({"ok": True})


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify({k: v for k, v in demo_flights.SCENARIOS.items()})


def run():
    print("=" * 55)
    print("   TRACILS — Live Aircraft & ILS Monitor")
    print("   Trivandrum International Airport (VOTV)")
    print("=" * 55)
    import version, ils_checker, ml_live
    info = ml_live.model_info()
    print(f"   BUILD: {version.BUILD}")
    print(f"   Runway headings (true): RWY14={ils_checker.RWY14_HEADING}  RWY32={ils_checker.RWY32_HEADING}")
    print(f"   ML baseline: {info.get('trained_on', 'unavailable')}")
    print(f"   Background refresh every {REFRESH_SECONDS} seconds")
    print(f"   Serving at http://{HOST}:{PORT}")
    print("   Press Ctrl+C to stop\n")

    build_static_map(MAP_FILE, poll_interval_seconds=REFRESH_SECONDS)  # built ONCE — never rebuilt after this

    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=demo_loop, daemon=True).start()

    try:
        import webbrowser
        webbrowser.open(f"http://{HOST}:{PORT}/")
    except Exception:
        pass

    app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  TRACILS stopped. Goodbye! ✈")
