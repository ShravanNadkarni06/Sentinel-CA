from propagate import load_satellite, get_position
from fetch_tle import get_session, fetch_tle
from datetime import datetime, timezone
import numpy as np

def get_clean_tle(session, norad_id):
    raw = fetch_tle(session, norad_id)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    return lines[0], lines[1]

def relative_state(pos1, vel1, pos2, vel2):
    rel_pos = np.array(pos2) - np.array(pos1)
    rel_vel = np.array(vel2) - np.array(vel1)
    miss_distance = np.linalg.norm(rel_pos)
    relative_speed = np.linalg.norm(rel_vel)
    return rel_pos, rel_vel, miss_distance, relative_speed

if __name__ == "__main__":
    session = get_session()

    # pick two real, currently-active objects for now (not yet a known close-approach pair)
    id_a, id_b = 25544, 44804  # ISS and Cartosat-3, just as a working example

    l1a, l2a = get_clean_tle(session, id_a)
    l1b, l2b = get_clean_tle(session, id_b)

    sat_a = load_satellite(l1a, l2a)
    sat_b = load_satellite(l1b, l2b)

    now = datetime.now(timezone.utc)
    pos_a, vel_a = get_position(sat_a, now)
    pos_b, vel_b = get_position(sat_b, now)

    rel_pos, rel_vel, miss_dist, rel_speed = relative_state(pos_a, vel_a, pos_b, vel_b)

    print(f"Miss distance right now: {miss_dist:.2f} km")
    print(f"Relative speed: {rel_speed:.2f} km/s")