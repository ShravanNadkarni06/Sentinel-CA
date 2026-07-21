import requests
import os
from dotenv import load_dotenv

load_dotenv()

SPACETRACK_URL = "https://www.space-track.org"
LOGIN_URL = f"{SPACETRACK_URL}/ajaxauth/login"

def get_session():
    session = requests.Session()
    payload = {
        "identity": os.getenv("SPACETRACK_USER"),
        "password": os.getenv("SPACETRACK_PASS")
    }
    print("USER:", os.getenv("SPACETRACK_USER"))
    print("PASS SET:", bool(os.getenv("SPACETRACK_PASS")))

    resp = session.post(LOGIN_URL, data=payload)
    print("Status:", resp.status_code)
    print("Response body:", resp.text)

    resp.raise_for_status()
    return session

def fetch_tle(session, norad_id):
    query_url = (
        f"{SPACETRACK_URL}/basicspacedata/query/class/gp/"
        f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/1/format/tle"
    )
    resp = session.get(query_url)
    print("Query Status:", resp.status_code)
    print("Query Response:", resp.text[:300])
    resp.raise_for_status()
    return resp.text

if __name__ == "__main__":
    session = get_session()
    tle = fetch_tle(session, 25544)  # ISS as a test case
    print(tle)