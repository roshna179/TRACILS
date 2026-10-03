import time
import os
from adsb_receiver import get_aircraft, MIN_SAFE_INTERVAL_S
from ils_checker import run_ils_checks, check_aircraft
from map_plot import draw_map
import landing_logger

# Fastest interval that stays inside your OpenSky daily credit budget
# 24/7 (see the math in adsb_receiver.py) -- not a guess.
REFRESH_SECONDS = MIN_SAFE_INTERVAL_S + 3

def run():
    print("=" * 55)
    print("   TRACILS — Live Aircraft & ILS Monitor")
    print("   Trivandrum International Airport (VOTV)")
    print("=" * 55)
    print(f"   Refreshing every {REFRESH_SECONDS} seconds")
    print("   Press Ctrl+C to stop\n")

    cycle = 1
    while True:
        print(f"\n{'─'*55}")
        print(f"  Cycle #{cycle} — fetching live data...")
        print(f"{'─'*55}")

        # Step 1 — Get aircraft
        planes = get_aircraft()

        # Step 2 — Run ILS checks
        run_ils_checks(planes)

        # Feed this cycle's readings to the landing logger (see
        # landing_logger.py) so it has a full history to summarize
        # whenever draw_map finalizes a landing.
        for ac in planes:
            landing_logger.record_sample(ac, check_aircraft(ac))

        # Step 3 — Update map
        draw_map(planes)
        print(f"\n  ✅ Map updated! Open tracils_map.html in browser.")
        print(f"  ⏳ Next refresh in {REFRESH_SECONDS} seconds... (Ctrl+C to stop)")

        cycle += 1
        time.sleep(REFRESH_SECONDS)

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  TRACILS stopped. Goodbye! ✈")