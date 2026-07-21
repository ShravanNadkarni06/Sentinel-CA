import numpy as np
from scipy.stats import norm

def build_encounter_plane_basis(rel_vel):
    """Build 2D basis vectors perpendicular to relative velocity."""
    rel_vel_unit = rel_vel / np.linalg.norm(rel_vel)
    arbitrary = np.array([1, 0, 0]) if abs(rel_vel_unit[0]) < 0.9 else np.array([0, 1, 0])
    y_axis = np.cross(rel_vel_unit, arbitrary)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, rel_vel_unit)
    return x_axis, y_axis

def project_to_encounter_plane(rel_pos, x_axis, y_axis):
    x = np.dot(rel_pos, x_axis)
    y = np.dot(rel_pos, y_axis)
    return x, y

def combined_covariance_2d(sigma_a_km, sigma_b_km):
    """
    Simplified isotropic covariance assumption (documented limitation):
    real CDMs give full 3x3 covariance per object; here we assume a
    reasonable circular position-uncertainty radius for each object,
    typical of TLE-based tracking (a few hundred meters to ~1km for
    well-tracked LEO objects).
    """
    sigma_combined = np.sqrt(sigma_a_km**2 + sigma_b_km**2)
    return np.array([[sigma_combined**2, 0], [0, sigma_combined**2]])

def probability_of_collision(x, y, cov_2d, hbr_km):
    """
    2D Gaussian integral over a circular hard-body region, using the
    standard Foster max-probability approximation via numerical grid
    integration for clarity (closed-form uses error functions).
    """
    sigma_x = np.sqrt(cov_2d[0, 0])
    sigma_y = np.sqrt(cov_2d[1, 1])

    grid_n = 400
    span = 5 * max(sigma_x, sigma_y) + hbr_km
    xs = np.linspace(x - span, x + span, grid_n)
    ys = np.linspace(y - span, y + span, grid_n)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    X, Y = np.meshgrid(xs, ys)
    density = norm.pdf(X, loc=0, scale=sigma_x) * norm.pdf(Y, loc=0, scale=sigma_y)

    dist_from_relpos = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist_from_relpos <= hbr_km

    pc = np.sum(density[mask]) * dx * dy
    return pc

if __name__ == "__main__":
    # From your refine_tca.py output
    rel_pos = np.array([-128.7863347, -87.55390918, -0.57208931])
    rel_vel = np.array([-6.47356872, 9.13327697, 5.47899569])

    x_axis, y_axis = build_encounter_plane_basis(rel_vel)
    x, y = project_to_encounter_plane(rel_pos, x_axis, y_axis)

    # Assumed uncertainty — typical published TLE-based position error
    # for well-tracked LEO objects is roughly 0.1-1 km; CLEARLY LABEL
    # this as an assumption in your report, not real CDM covariance
    sigma_a = 0.5  # km, assumed for ISS (well-tracked, frequent updates)
    sigma_b = 0.5  # km, assumed for Cartosat-3

    cov_2d = combined_covariance_2d(sigma_a, sigma_b)

    # Combined hard-body radius: ISS ~ 55m wingspan-ish, Cartosat-3 ~ 2-3m
    # use conservative combined estimate
    hbr_km = 0.06  # 60 meters combined radius, conservative

    pc = probability_of_collision(x, y, cov_2d, hbr_km)

    print(f"Encounter plane position: x={x:.4f} km, y={y:.4f} km")
    print(f"Miss distance in-plane: {np.sqrt(x**2 + y**2):.4f} km")
    print(f"Assumed combined sigma: {np.sqrt(cov_2d[0,0]):.4f} km")
    print(f"Estimated Probability of Collision: {pc:.2e}")