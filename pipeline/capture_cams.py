"""Periodically snapshot every Manhattan NYC DOT traffic camera (for offline pose estimation)."""
import json, sys, time, datetime, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/cam_frames"
S = requests.Session()
S.headers["User-Agent"] = "cityviz/0.1 (https://github.com/kevmo314/cityviz)"

def grab(cam, stamp):
    try:
        r = S.get(cam["imageUrl"], timeout=20)
        if r.ok and r.headers.get("content-type", "").startswith("image") and len(r.content) > 2000:
            d = OUT / cam["id"]; d.mkdir(parents=True, exist_ok=True)
            (d / f"{stamp}.jpg").write_bytes(r.content)
    except requests.RequestException:
        pass

def main(interval=1200, hours=20):
    cams = [c for c in requests.get("https://webcams.nyctmc.org/api/cameras", timeout=60).json() if c["area"] == "Manhattan"]
    (ROOT / "data/raw/traffic_cams.json").write_text(json.dumps(cams))
    end = time.time() + hours * 3600
    while time.time() < end:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda c: grab(c, stamp), cams))
        print(stamp, "captured", file=sys.stderr, flush=True)
        time.sleep(interval)

if __name__ == "__main__":
    main()
