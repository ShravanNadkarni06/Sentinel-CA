from sgp4.api import Satrec, jday
from datetime import datetime, timezone
from fetch_tle import get_session, fetch_tle

def load_satellite(tle_line1, tle_line2):
    return Satrec.twoline2rv(tle_line1, tle_line2)

def get_position(satrec, dt: datetime):
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    error_code, position, velocity = satrec.sgp4(jd, fr)
    if error_code != 0:
        raise RuntimeError(f"SGP4 propagation error code {error_code}")
    return position, velocity

if __name__ == "__main__":
    session = get_session()
    raw_tle = fetch_tle(session, 25544)

    # Split into clean lines, strip whitespace, remove any blank lines
    lines = [l.strip() for l in raw_tle.strip().splitlines() if l.strip()]
    print("Parsed lines:")
    for l in lines:
        print(repr(l))  # repr shows hidden characters/spacing issues clearly

    line1, line2 = lines[0], lines[1]

    sat = load_satellite(line1, line2)
    now = datetime.now(timezone.utc)
    pos, vel = get_position(sat, now)

    print("Time (UTC):", now)
    print("Position (km):", pos)
    print("Velocity (km/s):", vel)