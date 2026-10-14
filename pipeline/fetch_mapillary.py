"""Index Mapillary street-level photos in a lon/lat box and sample a well-spread subset for pose solving.

Each record carries Mapillary's own pose (SfM-computed position and compass angle) and calibrated focal length, which
localize_photos.py uses as tight priors. Sampling keeps one image per SAMPLE_CELL-metre cell per heading bin, skipping
dashcams (windshield glare, hood in frame) and panoramas (would need cutting into faces first), preferring recent
captures and phones / action cams over unknown cameras.

usage: fetch_mapillary.py <name> [<lon0> <lat0> <lon1> <lat1>]   (all of Manhattan without a box; token in
~/.mapillary_token)
writes data/raw/mapillary_<name>.jsonl (all indexed images) and data/work/candidates_mly_<name>.jsonl (sample, in the
Commons candidate schema: pageid = "mly_<id>", lat, lon, heading, width, height, focal (width-normalised), ...)
"""
import json, math, re, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import geo

ROOT = Path(__file__).resolve().parent.parent
TOKEN = (Path.home() / ".mapillary_token").read_text().strip()
FIELDS = ("id,captured_at,camera_type,camera_parameters,computed_compass_angle,computed_geometry,width,height,"
          "sequence,creator,make,model")
DASHCAM = re.compile(r"blackvue|garmin|nextbase|viofo|vantrue|thinkware|dashcam|dash cam|dr\d{3}", re.I)
SAMPLE_CELL, HEADING_BINS, MIN_YEAR = 20.0, 8, 2016
GRID = 0.004  # initial query cell (degrees)
S = requests.Session()


def query(bbox):
    """images in a box, or None when the API refuses it (large boxes time out server-side)"""
    for k in range(3):
        try:
            r = S.get("https://graph.mapillary.com/images", timeout=45,
                      params=dict(access_token=TOKEN, fields=FIELDS, bbox=",".join(f"{v:.6f}" for v in bbox), limit=2000))
            if r.status_code == 200:
                return r.json().get("data", [])
        except requests.RequestException:
            pass
        time.sleep(3 * (k + 1))
    return None


def index(bbox, depth=0, out=None):
    """quadtree over the box: any cell that hits the 2000-result cap, or that the API refuses, is split"""
    out = {} if out is None else out
    data = query(bbox)
    if data is None and depth >= 10:
        raise RuntimeError(f"query failed for {bbox}")
    if (data is None or len(data) >= 2000) and depth < 10:
        x0, y0, x1, y1 = bbox
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        for b in ((x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)):
            index(b, depth + 1, out)
    else:
        for d in data:
            out[d["id"]] = d
    return out


def to_record(d):
    w, h = d["width"], d["height"]
    lon, lat = d["computed_geometry"]["coordinates"]
    cp = d.get("camera_parameters") or []
    date = datetime.fromtimestamp(d["captured_at"] / 1000, timezone.utc)
    user = (d.get("creator") or {}).get("username") or "Mapillary contributor"
    return {
        "pageid": f"mly_{d['id']}", "mly_id": d["id"], "source": "mapillary",
        "title": f"Mapillary · {user} · {date:%Y-%m-%d}",
        "url": f"https://www.mapillary.com/app/?pKey={d['id']}",
        "lat": lat, "lon": lon, "heading": d["computed_compass_angle"], "width": w, "height": h,
        # Mapillary focal is relative to the longer side; the solver's is relative to the width
        "focal": cp[0] * max(w, h) / w if cp and 0.2 < cp[0] < 3 else None,
        "date": f"{date:%Y-%m-%d}", "artist": user, "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "camera": f"{d.get('make') or ''} {d.get('model') or ''}".strip(), "sequence": d.get("sequence"),
    }


def usable(d):
    return (d.get("camera_type") == "perspective" and d.get("computed_geometry") and d.get("computed_compass_angle") is not None
            and d.get("width") and d.get("height") and min(d["width"], d["height"]) >= 700
            and not DASHCAM.search(f"{d.get('make') or ''} {d.get('model') or ''}")
            and datetime.fromtimestamp(d["captured_at"] / 1000, timezone.utc).year >= MIN_YEAR)


def cell_key(r):
    x, y = map(float, geo.to_local(r["lon"], r["lat"]))
    return int(x // SAMPLE_CELL), int(y // SAMPLE_CELL), int(r["heading"] % 360 // (360 / HEADING_BINS))


def sample(recs, taken=frozenset()):
    """one image per (cell, heading bin) not already in `taken`: most recent first, known camera with calibrated focal
    preferred"""
    cells = defaultdict(list)
    for r in recs:
        k = cell_key(r)
        if k not in taken:
            cells[k].append(r)
    out = []
    for rs in cells.values():
        rs.sort(key=lambda r: (r["focal"] is not None, bool(r["camera"]), r["date"]), reverse=True)
        out.append(rs[0])
    return out


def coarse_to_fine(cand, levels=(160.0, 80.0, 40.0)):
    """order candidates so any prefix covers the area evenly: one per 160 m cell first, then per 80 m, 40 m, then
    the rest (solving the full list takes days; it can be stopped and published at any point)"""
    import random
    rng = random.Random(0)
    rest, order = list(cand), []
    for size in levels:
        seen, keep = set(), []
        rng.shuffle(rest)
        for r in rest:
            x, y = map(float, geo.to_local(r["lon"], r["lat"]))
            k = (int(x // size), int(y // size))
            (keep if k in seen else order).append(r)
            seen.add(k)
        rest = keep
    rng.shuffle(rest)
    return order + rest


def manhattan():
    from shapely.geometry import shape
    from shapely.ops import unary_union
    feats = json.loads((ROOT / "data/raw/boroughs.geojson").read_text())["features"]
    return unary_union([shape(f["geometry"]) for f in feats if f["properties"].get("boroname") == "Manhattan"])


def main():
    """usage: fetch_mapillary.py <name> [lon0 lat0 lon1 lat1]   (no box: all of Manhattan)"""
    from concurrent.futures import ThreadPoolExecutor
    from shapely.geometry import Point, box as sbox
    name, *bx = sys.argv[1:]
    area = manhattan()
    x0, y0, x1, y1 = map(float, bx) if bx else area.bounds
    # ~400 m starting cells (the API times out on large boxes long before it would return the 2000 cap), only those
    # touching Manhattan: the city model has no buildings elsewhere
    nx, ny = math.ceil((x1 - x0) / GRID), math.ceil((y1 - y0) / GRID)
    cells = [(x0 + i * (x1 - x0) / nx, y0 + j * (y1 - y0) / ny, x0 + (i + 1) * (x1 - x0) / nx, y0 + (j + 1) * (y1 - y0) / ny)
             for i in range(nx) for j in range(ny)]
    cells = [c for c in cells if area.intersects(sbox(*c))]
    print(f"{len(cells)} cells", file=sys.stderr, flush=True)
    found = {}
    with ThreadPoolExecutor(4) as ex:
        for n, part in enumerate(ex.map(index, cells)):
            found.update(part)
            if n % 20 == 0:
                print(f"{n + 1}/{len(cells)} cells: {len(found)} indexed", file=sys.stderr, flush=True)
    raw = ROOT / f"data/raw/mapillary_{name}.jsonl"
    with open(raw, "w") as f:
        for d in found.values():
            f.write(json.dumps(d) + "\n")
    recs = [to_record(d) for d in found.values() if usable(d)]
    recs = [r for r in recs if area.contains(Point(r["lon"], r["lat"]))]
    # cells sampled by earlier runs (other candidate lists) are not sampled again
    taken = {cell_key(json.loads(l)) for f in (ROOT / "data/work").glob("candidates_mly_*.jsonl")
             if f.name != f"candidates_mly_{name}.jsonl" for l in open(f)}
    cand = coarse_to_fine(sample(recs, taken))
    out = ROOT / f"data/work/candidates_mly_{name}.jsonl"
    with open(out, "w") as f:
        for r in cand:
            f.write(json.dumps(r) + "\n")
    print(f"{len(found)} indexed, {len(recs)} usable in Manhattan, {len(cand)} sampled -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
