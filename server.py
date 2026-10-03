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
from ils_checker import run_ils_checks
from map_plot import build_static_map, build_live_payload, purge_synthetic
import landing_logger
import demo_flights

app = Flask(__name__)

REFRESH_SECONDS = MIN_SAFE_INTERVAL_S + 3
MAP_FILE = "tracils_map.html"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
MAX_BACKOFF_S = 300

_threads_started = False


def run_cycle():
    try:
        aircraft = get_aircraft()
        checked = run_ils_checks(aircraft)
        payload = build_live_payload(checked)
        landing_logger.log_aircraft(checked)
        return payload
    except RateLimitError as e:
        print(f"Rate limited: {e}")
        return {"error": "rate_limited", "message": str(e)}
    except OpenSkyAuthError as e:
        print(f"OpenSky authentication error: {e}")
        return {"error": "auth_error", "message": str(e)}
    except Exception as e:
        print(f"Cycle error: {e}")
        return {"error": "cycle_error", "message": str(e)}


def poll_loop():
    backoff = REFRESH_SECONDS

    while True:
        try:
            run_cycle()
            backoff = REFRESH_SECONDS
        except Exception as e:
            print(f"Polling error: {e}")
            backoff = min(backoff * 2, MAX_BACKOFF_S)

        time.sleep(backoff)


def demo_loop():
    while True:
        try:
            demo_flights.update_demo_flights()
        except Exception as e:
            print(f"Demo loop error: {e}")
        time.sleep(5)


@app.route("/")
def index():
    return send_from_directory(".", MAP_FILE)


@app.route("/api/data")
def api_data():
    try:
        aircraft = get_aircraft()
        checked = run_ils_checks(aircraft)
        return jsonify(build_live_payload(checked))
    except RateLimitError as e:
        return jsonify({
            "error": "rate_limited",
            "message": str(e),
            "rate_limit_info": last_rate_limit_info(),
        }), 429
    except OpenSkyAuthError as e:
        return jsonify({
            "error": "auth_error",
            "message": str(e)
        }), 401
    except Exception as e:
        return jsonify({
            "error": "server_error",
            "message": str(e)
        }), 500


@app.route("/api/inject", methods=["POST"])
def api_inject():
    data = request.get_json(silent=True) or {}
    try:
        result = demo_flights.inject(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        purge_synthetic()
        return jsonify({"status": "cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scenarios")
def api_scenarios():
    try:
        return jsonify(demo_flights.get_scenarios())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def start_background_threads():
    global _threads_started

    if _threads_started:
        return

    _threads_started = True

    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=demo_loop, daemon=True).start()

    print("Background threads started.")


def initialize_app():
    if not os.path.exists(MAP_FILE):
        print("Building initial TRACILS map...")
        try:
            build_static_map()
            print("Initial map built successfully.")
        except Exception as e:
            print(f"Could not build initial map: {e}")

    start_background_threads()


if os.environ.get("TRACILS_DISABLE_AUTO_INIT") != "1":
    try:
        initialize_app()
    except Exception as e:
        print(f"Initialization error: {e}")


def run():
    initialize_app()
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    run()
