```python
"""
TRACILS live server
====================

Runs the TRACILS live aircraft + ILS monitoring server.

Features:
  1. Static map is built ONCE at startup.
  2. Browser polls /api/data every REFRESH_SECONDS.
  3. Aircraft markers update without reloading the page.
  4. Zoom and pan remain unchanged.
  5. Background thread fetches OpenSky traffic.
  6. Demo flights continue to work.
  7. Compatible with both local Python and Render/Gunicorn.

Local:
    python server.py

Render:
    gunicorn server:app
"""

import os
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from adsb_receiver import (
    get_aircraft,
    RateLimitError,
    OpenSkyAuthError,
    MIN_SAFE_INTERVAL_S,
    last_rate_limit_info,
)

from ils_checker import run_ils_checks, check_aircraft
from map_plot import build_static_map, build_live_payload, purge_synthetic

import landing_logger
import demo_flights


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

# OpenSky refresh interval.
# Uses the safe interval calculated in adsb_receiver.py plus a small margin.
REFRESH_SECONDS = MIN_SAFE_INTERVAL_S + 3

MAP_FILE = "tracils_map.html"

# Render requires the application to listen on 0.0.0.0.
HOST = "0.0.0.0"

# Render provides the PORT environment variable.
# When running locally, it falls back to port 5000.
PORT = int(os.environ.get("PORT", 5000))

MAX_BACKOFF_S = 300


# --------------------------------------------------------------------
# Flask application
# --------------------------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------

_lock = threading.Lock()

# Guards the detection pipeline.
_pipe_lock = threading.RLock()

_real_planes = []
_real_ts = None
_feed_error = None
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


# --------------------------------------------------------------------
# Build and publish one live payload
# --------------------------------------------------------------------

def run_cycle(fresh):
    """
    Build one payload from real OpenSky traffic + demo flights.

    fresh=True:
        A brand-new OpenSky fetch arrived.

    fresh=False:
        Re-publish the last real aircraft together with demo flights.
        Real aircraft are marked as repeated so that they are not treated
        as new evidence by the landing/stagnation logic.
    """

    global _latest_payload

    with _pipe_lock:

        # Copy real aircraft so the original list is not modified.
        real = [dict(a) for a in _real_planes]

        # If this is not a fresh OpenSky fetch, mark real aircraft as repeats.
        if not fresh:
            for aircraft in real:
                aircraft["_repeat"] = True

        # Get currently active demo aircraft.
        demo = demo_flights.current_aircraft()

        planes = real + demo

        # ------------------------------------------------------------
        # Run ILS / landing processing
        # ------------------------------------------------------------

        if fresh:

            # Real traffic only.
            run_ils_checks(real)

            # Log real aircraft samples.
            for aircraft in real:
                landing_logger.record_sample(
                    aircraft,
                    check_aircraft(aircraft)
                )

        else:

            # Demo flights are processed on their own accelerated clock.
            for aircraft in demo:
                landing_logger.record_sample(
                    aircraft,
                    check_aircraft(aircraft)
                )

        # ------------------------------------------------------------
        # Build browser payload
        # ------------------------------------------------------------

        payload = build_live_payload(planes)

        payload["poll_interval_s"] = REFRESH_SECONDS
        payload["credits_remaining"] = _credits
        payload["real_timestamp"] = _real_ts
        payload["feed_error"] = _feed_error

        payload["demo"] = {
            "active": demo_flights.active(),
            "tick_s": demo_flights.TICK_S,
        }

        payload["real_count"] = len(real)

        # Publish safely.
        with _lock:
            _latest_payload = payload


# --------------------------------------------------------------------
# OpenSky polling thread
# --------------------------------------------------------------------

def poll_loop():
    """
    Background thread.

    Fetches real aircraft from OpenSky, runs the TRACILS pipeline,
    and publishes the latest result.

    Handles:
      - Rate limiting
      - Missing credentials
      - Authentication failures
      - Temporary network/API errors
      - Exponential backoff
    """

    global _real_planes
    global _real_ts
    global _feed_error
    global _credits

    cycle = 1
    consecutive_failures = 0

    while True:

        print("\n" + "─" * 55)
        print(f"  Cycle #{cycle} — fetching live data...")
        print("─" * 55)

        cycle_start = time.time()

        sleep_for = REFRESH_SECONDS

        try:

            # --------------------------------------------------------
            # Fetch live aircraft
            # --------------------------------------------------------

            planes = get_aircraft()

            _real_planes = planes
            _real_ts = time.time()
            _feed_error = None

            _credits = last_rate_limit_info["remaining"]

            # --------------------------------------------------------
            # Process fresh aircraft
            # --------------------------------------------------------

            try:

                run_cycle(fresh=True)

            except Exception as e:

                import traceback

                traceback.print_exc()

                _feed_error = (
                    f"Processing error "
                    f"(feed OK, {len(planes)} aircraft received): "
                    f"{str(e)[:80]}"
                )

                print(f"  ⚠️  {_feed_error}")

            consecutive_failures = 0

            fetch_ms = round(
                (time.time() - cycle_start) * 1000
            )

            print(
                f"  ✅ Live data refreshed silently in "
                f"{fetch_ms}ms — no page reload."
            )

        # ------------------------------------------------------------
        # OpenSky rate limit
        # ------------------------------------------------------------

        except RateLimitError as e:

            consecutive_failures += 1

            sleep_for = max(
                e.retry_after_s,
                REFRESH_SECONDS
            )

            _feed_error = "OpenSky rate limit reached"

            print(
                "  🛑 OpenSky rate-limited us "
                "(daily credit budget likely exhausted)."
            )

            print(
                f"     Backing off {sleep_for}s. "
                "The map keeps showing the last good data."
            )

        # ------------------------------------------------------------
        # Missing credentials
        # ------------------------------------------------------------

        except FileNotFoundError:

            consecutive_failures += 1

            sleep_for = 60

            _feed_error = (
                "credentials.json not found in tracils_live/ "
                "(live traffic off)"
            )

            print(
                "  ⚠️  credentials.json not found next to server.py."
            )

            print(
                "     Live traffic is OFF; demo flight button still works."
            )

            print("     Retrying in 60s.")

        # ------------------------------------------------------------
        # OpenSky authentication failure
        # ------------------------------------------------------------

        except OpenSkyAuthError as e:

            consecutive_failures += 1

            sleep_for = 60

            _feed_error = (
                f"OpenSky login failed: {str(e)[:90]}"
            )

            print(f"  ⚠️  {e}")

            print(
                "     Check client_id / client_secret "
                "in credentials.json. Retrying in 60s."
            )

        # ------------------------------------------------------------
        # Other temporary errors
        # ------------------------------------------------------------

        except Exception as e:

            consecutive_failures += 1

            sleep_for = min(
                REFRESH_SECONDS * (2 ** consecutive_failures),
                MAX_BACKOFF_S
            )

            _feed_error = (
                f"OpenSky problem: {str(e)[:80]}"
            )

            print(
                f"  ⚠️  Poll cycle failed: {e}"
            )

            print(
                f"     Backing off {sleep_for}s "
                f"before retrying "
                f"(failure #{consecutive_failures})."
            )

        # ------------------------------------------------------------
        # Publish errors to browser
        # ------------------------------------------------------------

        if _feed_error:

            try:
                run_cycle(fresh=False)
            except Exception:
                pass

        cycle += 1

        time.sleep(sleep_for)


# --------------------------------------------------------------------
# Demo flight thread
# --------------------------------------------------------------------

def demo_loop():
    """
    Fast clock for demo flights.

    Idle when no demo flight is active.

    When a demo flight is active:
        - Advance the demo flight.
        - Update the browser payload.

    One additional cycle runs after the demo flight finishes so
    that the landing is logged and the display clears.
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

            print(
                f"  ⚠️  demo tick failed: {e}"
            )


# --------------------------------------------------------------------
# Flask routes
# --------------------------------------------------------------------

@app.route("/")
def index():
    """
    Serve the static TRACILS map.
    """

    return send_from_directory(
        ".",
        MAP_FILE
    )


@app.route("/api/data")
def api_data():
    """
    Return the latest aircraft/landing payload.

    The browser calls this endpoint periodically.
    """

    with _lock:
        return jsonify(_latest_payload)


@app.route("/api/inject", methods=["POST"])
def api_inject():
    """
    Inject a demo flight scenario.
    """

    body = request.get_json(
        silent=True
    ) or {}

    try:

        with _pipe_lock:

            flights = demo_flights.inject(
                body.get(
                    "scenario",
                    "drift"
                ),
                body.get(
                    "runway",
                    "RWY 32"
                )
            )

            # Update immediately.
            run_cycle(
                fresh=False
            )

    except ValueError as e:

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400

    print(
        "  🧪 DEMO injected: "
        + ", ".join(
            f["callsign"]
            for f in flights
        )
        + f" ({body.get('scenario')})"
    )

    return jsonify({
        "ok": True,
        "flights": flights
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """
    Clear all demo flights.
    """

    with _pipe_lock:

        demo_flights.clear()

        purge_synthetic()

        run_cycle(
            fresh=False
        )

    print(
        "  🧪 DEMO flights cleared"
    )

    return jsonify({
        "ok": True
    })


@app.route("/api/scenarios")
def api_scenarios():
    """
    Return available demo scenarios.
    """

    return jsonify({
        k: v
        for k, v in demo_flights.SCENARIOS.items()
    })


# --------------------------------------------------------------------
# Background startup
# --------------------------------------------------------------------

_background_started = False
_background_lock = threading.Lock()


def start_background_threads():
    """
    Start the TRACILS background threads exactly once.

    This is important for Gunicorn/Render because:

        gunicorn server:app

    imports the Flask app directly and does NOT execute:

        if __name__ == "__main__":

    Therefore the background workers must be started during app startup.
    """

    global _background_started

    with _background_lock:

        if _background_started:
            return

        _background_started = True

        print(
            "  🚀 Starting TRACILS background workers..."
        )

        threading.Thread(
            target=poll_loop,
            daemon=True,
            name="TRACILS-OpenSky-Poller"
        ).start()

        threading.Thread(
            target=demo_loop,
            daemon=True,
            name="TRACILS-Demo-Loop"
        ).start()


# --------------------------------------------------------------------
# Application initialization
# --------------------------------------------------------------------

def initialize_app():
    """
    Build the static map once and start background processing.

    Safe to call multiple times.
    """

    global _background_started

    # Build map only once.
    if not os.path.exists(MAP_FILE):

        print(
            f"  🗺️  Building static map: {MAP_FILE}"
        )

        build_static_map(
            MAP_FILE,
            poll_interval_seconds=REFRESH_SECONDS
        )

        print(
            "  ✅ Static map built."
        )

    else:

        print(
            f"  🗺️  Using existing {MAP_FILE}"
        )

    start_background_threads()


# --------------------------------------------------------------------
# Local / Render startup
# --------------------------------------------------------------------

def run():

    print("=" * 55)
    print("   TRACILS — Live Aircraft & ILS Monitor")
    print("   Trivandrum International Airport (VOTV)")
    print("=" * 55)

    # Import information modules.
    import version
    import ils_checker
    import ml_live

    info = ml_live.model_info()

    print(
        f"   BUILD: {version.BUILD}"
    )

    print(
        "   Runway headings (true): "
        f"RWY14={ils_checker.RWY14_HEADING}  "
        f"RWY32={ils_checker.RWY32_HEADING}"
    )

    print(
        f"   ML baseline: "
        f"{info.get('trained_on', 'unavailable')}"
    )

    print(
        f"   Background refresh every "
        f"{REFRESH_SECONDS} seconds"
    )

    print(
        f"   Serving at "
        f"http://{HOST}:{PORT}"
    )

    print(
        "   Press Ctrl+C to stop\n"
    )

    # Build map and start background threads.
    initialize_app()

    # Open browser only for local execution.
    try:

        import webbrowser

        webbrowser.open(
            f"http://127.0.0.1:{PORT}/"
        )

    except Exception:
        pass

    # Flask server.
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


# --------------------------------------------------------------------
# Start when executed directly
# --------------------------------------------------------------------

if __name__ == "__main__":

    try:

        run()

    except KeyboardInterrupt:

        print(
            "\n\n  TRACILS stopped. Goodbye! ✈"
        )


# --------------------------------------------------------------------
# Gunicorn / Render initialization
# --------------------------------------------------------------------

# Gunicorn imports "app" directly, so run initialization here.
#
# The environment variable prevents accidental initialization during
# special tooling/import situations.

if os.environ.get("TRACILS_DISABLE_AUTO_INIT") != "1":

    try:

        initialize_app()

    except Exception as e:

        print(
            f"  ⚠️  TRACILS startup initialization failed: {e}"
        )
```
