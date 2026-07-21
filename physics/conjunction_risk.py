import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.stats import norm

from fetch_tle import get_session, fetch_tle
from propagate import load_satellite, get_position


def get_clean_tle(session, norad_id):
    raw = fetch_tle(session, norad_id)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError(f"No valid TLE returned for NORAD {norad_id}")
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
            min_dist, min_time = dist, t
    return min_dist, min_time


def refine_tca(sat_a, sat_b, coarse_time, window_minutes=2, step_seconds=1):
    best_dist = float("inf")
    best_time = None
    best_state = None
    start = coarse_time - timedelta(minutes=window_minutes / 2)
    total_steps = int(window_minutes * 60 / step_seconds)

    for i in range(total_steps):
        t = start + timedelta(seconds=i * step_seconds)
        pos_a, vel_a = get_position(sat_a, t)
        pos_b, vel_b = get_position(sat_b, t)
        rel_pos = np.array(pos_b) - np.array(pos_a)
        rel_vel = np.array(vel_b) - np.array(vel_a)
        dist = np.linalg.norm(rel_pos)
        if dist < best_dist:
            best_dist, best_time = dist, t
            best_state = (rel_pos, rel_vel)

    return best_dist, best_time, best_state


def build_encounter_plane_basis(rel_vel):
    rel_vel_unit = rel_vel / np.linalg.norm(rel_vel)
    arbitrary = np.array([1, 0, 0]) if abs(rel_vel_unit[0]) < 0.9 else np.array([0, 1, 0])
    y_axis = np.cross(rel_vel_unit, arbitrary)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, rel_vel_unit)
    return x_axis, y_axis


def probability_of_collision(x, y, sigma_km, hbr_km):
    grid_n = 400
    span = 5 * sigma_km + hbr_km
    xs = np.linspace(x - span, x + span, grid_n)
    ys = np.linspace(y - span, y + span, grid_n)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys)
    density = norm.pdf(X, loc=0, scale=sigma_km) * norm.pdf(Y, loc=0, scale=sigma_km)
    mask = np.sqrt((X - x) ** 2 + (Y - y) ** 2) <= hbr_km
    return float(np.sum(density[mask]) * dx * dy)


def compute_conjunction_risk(
    norad_id_a: int,
    norad_id_b: int,
    sigma_a_km: float = 0.5,
    sigma_b_km: float = 0.5,
    hbr_km: float = 0.06,
    screen_hours: int = 24,
) -> dict:
    """
    Full pipeline: fetch TLEs -> propagate -> screen for closest approach ->
    refine TCA -> estimate probability of collision.

    NOTE: sigma_a_km / sigma_b_km are ASSUMED position-uncertainty values
    (real CDM covariance is not public data) — flagged clearly as an
    estimate, not an operational-grade number. See project README for
    justification.
    """
    session = get_session()

    l1a, l2a = get_clean_tle(session, norad_id_a)
    l1b, l2b = get_clean_tle(session, norad_id_b)
    sat_a = load_satellite(l1a, l2a)
    sat_b = load_satellite(l1b, l2b)

    now = datetime.now(timezone.utc)

    coarse_dist, coarse_time = screen_pair(sat_a, sat_b, now, hours=screen_hours)
    if coarse_time is None:
        raise RuntimeError("No valid closest-approach time found in screening window")

    refined_dist, tca, (rel_pos, rel_vel) = refine_tca(sat_a, sat_b, coarse_time)

    x_axis, y_axis = build_encounter_plane_basis(rel_vel)
    x = float(np.dot(rel_pos, x_axis))
    y = float(np.dot(rel_pos, y_axis))

    combined_sigma = float(np.sqrt(sigma_a_km ** 2 + sigma_b_km ** 2))
    pc = probability_of_collision(x, y, combined_sigma, hbr_km)

    return {
        "object_a_norad_id": norad_id_a,
        "object_b_norad_id": norad_id_b,
        "tca_utc": tca.isoformat(),
        "miss_distance_km": round(refined_dist, 4),
        "relative_velocity_km_s": round(float(np.linalg.norm(rel_vel)), 4),
        "encounter_plane_x_km": round(x, 4),
        "encounter_plane_y_km": round(y, 4),
        "assumed_sigma_km": round(combined_sigma, 4),
        "hard_body_radius_km": hbr_km,
        "probability_of_collision": pc,
        "risk_tier": (
            "HIGH" if pc > 1e-4 else "MEDIUM" if pc > 1e-6 else "LOW"
        ),
    }


if __name__ == "__main__":
    result = compute_conjunction_risk(25544, 44804)  # ISS vs Cartosat-3
    import json
    print(json.dumps(result, indent=2))
