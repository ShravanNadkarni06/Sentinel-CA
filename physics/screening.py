from propagate import load_satellite, get_position
from fetch_tle import get_session, fetch_tle
from datetime import datetime, timedelta, timezone
import numpy as np

def get_clean_tle(session, norad_id):
    raw = fetch_tle(session, norad_id)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if not lines or len(lines) < 2:
        raise ValueError(f"No TLE data returned for NORAD {norad_id}")
    return lines[0], lines[1]


def screen_pair(sat_a, sat_b, start_time, hours=24, step_minutes=1):
    min_dist = float("inf")
    min_time = None
    for i in range(0, hours * 60, step_minutes):
        t = start_time + timedelta(minutes=i)
        try:
            pos_a, _ = get_position(sat_a, t)
            pos_b, _ = get_position(sat_b, t)
        except RuntimeError:
            continue
        dist = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
        if dist < min_dist:
            min_dist = dist
            min_time = t
    return min_dist, min_time

if __name__ == "__main__":
    session = get_session()

    # A small starter list of LEO objects in similar altitude bands —
    # you'll expand this list later with real debris-heavy regions
    candidates = {
    "ISS": 25544,
    "Cartosat-3": 44804,
    "Fengyun-1C debris": 29656,
    "Cosmos-2251 debris": 34427,   # from the 2009 Iridium-Cosmos collision, well documented
}
    

    ids = list(candidates.items())
    now = datetime.now(timezone.utc)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            name_a, id_a = ids[i]
            name_b, id_b = ids[j]

            l1a, l2a = get_clean_tle(session, id_a)
            l1b, l2b = get_clean_tle(session, id_b)
            sat_a = load_satellite(l1a, l2a)
            sat_b = load_satellite(l1b, l2b)

            min_dist, min_time = screen_pair(sat_a, sat_b, now)
            print(f"{name_a} vs {name_b}: min distance {min_dist:.2f} km at {min_time}")