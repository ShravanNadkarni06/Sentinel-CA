from propagate import load_satellite, get_position
from fetch_tle import get_session, fetch_tle
from datetime import datetime, timedelta, timezone
import numpy as np

def get_clean_tle(session, norad_id):
    raw = fetch_tle(session, norad_id)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    return lines[0], lines[1]

def distance_at(sat_a, sat_b, t):
    pos_a, vel_a = get_position(sat_a, t)
    pos_b, vel_b = get_position(sat_b, t)
    rel_pos = np.array(pos_b) - np.array(pos_a)
    rel_vel = np.array(vel_b) - np.array(vel_a)
    return np.linalg.norm(rel_pos), rel_pos, rel_vel, pos_a, vel_a, pos_b, vel_b

def refine_tca(sat_a, sat_b, coarse_time, window_minutes=2, step_seconds=1):
    best_dist = float("inf")
    best_time = None
    best_state = None
    start = coarse_time - timedelta(minutes=window_minutes/2)
    total_steps = int(window_minutes * 60 / step_seconds)

    for i in range(total_steps):
        t = start + timedelta(seconds=i * step_seconds)
        dist, rel_pos, rel_vel, pos_a, vel_a, pos_b, vel_b = distance_at(sat_a, sat_b, t)
        if dist < best_dist:
            best_dist = dist
            best_time = t
            best_state = (rel_pos, rel_vel, pos_a, vel_a, pos_b, vel_b)

    return best_dist, best_time, best_state

if __name__ == "__main__":
    session = get_session()

    l1a, l2a = get_clean_tle(session, 25544)   # ISS
    l1b, l2b = get_clean_tle(session, 44804)   # Cartosat-3
    sat_a = load_satellite(l1a, l2a)
    sat_b = load_satellite(l1b, l2b)

    # paste the coarse TCA timestamp your screening.py found
    coarse_time = datetime(2026, 7, 17, 7, 58, 9, tzinfo=timezone.utc)

    best_dist, best_time, best_state = refine_tca(sat_a, sat_b, coarse_time)

    print(f"Refined TCA: {best_time}")
    print(f"Refined miss distance: {best_dist:.4f} km")

    rel_pos, rel_vel, pos_a, vel_a, pos_b, vel_b = best_state
    print("Relative position (km):", rel_pos)
    print("Relative velocity (km/s):", rel_vel)