"""Download Manhattan building footprints (with roof heights) from NYC Open Data."""
import json, sys, requests
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data/raw/buildings_manhattan.geojson"
URL = "https://data.cityofnewyork.us/resource/5zhs-2jue.geojson"

feats, offset = [], 0
while True:
    r = requests.get(URL, params={"$where": "starts_with(base_bbl,'1')", "$limit": 20000,
                                  "$offset": offset, "$order": "objectid"}, timeout=600)
    r.raise_for_status()
    batch = r.json()["features"]
    feats += batch
    print(f"{len(feats)} features", file=sys.stderr)
    if len(batch) < 20000:
        break
    offset += 20000
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
