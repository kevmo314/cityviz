"""Batch: download Commons photos, segment, solve 7-DoF pose against the city model, export accepted ones for the web."""
import io, json, os, re, sys, time, random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import geo

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
RESULTS = ROOT / "data/work/photo_poses.jsonl"
IMGDIR = ROOT / "data/work/photos"
UA = "cityviz/0.1 (https://github.com/kevmo314/cityviz; kevmo314@gmail.com)"

BAD = re.compile(r"interior|inside|lobby|museum|exhibit|gallery|painting|sculpture|statue|subway|station platform|"
                 r"food|menu|portrait|concert|stage|performance|aerial|drone|helicopter|map of|diagram|logo|plaque|"
                 r"sign\b|ceiling|stained glass|mural|graffiti|panorama|360", re.I)


def year(r):
    m = re.search(r"(19|20)\d\d", str(r.get("date") or ""))
    return int(m.group(0)) if m else None


def candidates(limit, per_cell=6, cell=75.0):
    recs = [json.loads(l) for l in open(RAW / "commons.jsonl")]
    done = set()
    if RESULTS.exists():
        done = {json.loads(l)["pageid"] for l in open(RESULTS)}
    cells = defaultdict(list)
    for r in recs:
        if r["pageid"] in done or r["heading"] is None or not r.get("thumb"):
            continue
        if min(r["width"], r["height"]) < 700 or max(r["width"], r["height"]) / min(r["width"], r["height"]) > 2.2:
            continue
        y = year(r)
        if y is not None and y < 2008:
            continue
        if BAD.search(r["title"]) or any(BAD.search(c) for c in r.get("categories", [])):
            continue
        x, yy = geo.to_local(r["lon"], r["lat"])
        r["_quality"] = (bool(r.get("focal35")), y or 2000)
        cells[(int(x // cell), int(yy // cell))].append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["_quality"], reverse=True)
        cells[k] = cells[k][:per_cell]
    keys = list(cells)
    random.Random(0).shuffle(keys)
    out, depth = [], 0
    while len(out) < limit and any(len(cells[k]) > depth for k in keys):
        out += [cells[k][depth] for k in keys if len(cells[k]) > depth]
        depth += 1
    return out[:limit]


_W = {}


def init_worker():
    import torch
    torch.set_num_threads(int(os.environ.get("SEG_THREADS", "3")))
    import render, segment
    _W["city"] = render.City()
    _W["seg"] = segment.Segmenter()
    _W["http"] = requests.Session()
    _W["http"].headers["User-Agent"] = UA


def fetch(url):
    for attempt in range(5):
        r = _W["http"].get(url, timeout=60)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1)); continue
        r.raise_for_status()
        return r.content
    raise RuntimeError("rate limited")


def solve(rec):
    import localize, segment
    city, seg = _W["city"], _W["seg"]
    out = {"pageid": rec["pageid"], "title": rec["title"]}
    t0 = time.time()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(fetch(rec["thumb"])))).convert("RGB")
    except Exception as e:
        return {**out, "status": f"download: {e}"}
    lab = seg.labels(img)
    n = lab.size
    frac = {k: float((lab == v).sum() / n) for k, v in (("sky", segment.SKY), ("ground", segment.GROUND), ("building", segment.BUILDING))}
    out["frac"] = frac
    if frac["sky"] < 0.02 or frac["building"] < 0.12 or frac["sky"] + frac["building"] + frac["ground"] < 0.45:
        return {**out, "status": "unsuitable"}

    x, y = map(float, geo.to_local(rec["lon"], rec["lat"]))
    g = city.ground(x, y)
    W, H = img.size
    fov0 = localize.fov_from_exif(rec, W, H)
    fovs = [fov0 * k for k in (0.93, 1.0, 1.07)] if fov0 else [30, 40, 50, 60, 70]
    h0 = geo.true_heading_to_grid(rec["heading"])
    L = localize.Localizer(city, lab)
    heights = [g + 1.6]
    # Camera inside a footprint: likely an elevated vantage point (observation deck, roof, window).
    prim, t, hit = city.cast(np.array([[x, y, 2000.0]]), np.array([[0, 0, -1.0]]))
    if hit[0] and city.label[prim[0]] == 3 and 2000 - t[0] > g + 10:
        roof = 2000 - t[0]
        heights = [roof + 1.6, g + 0.9 * (roof - g), g + 0.8 * (roof - g)]
    pos = [(x + dx, y + dy, z) for z in heights for dx in (-15, 0, 15) for dy in (-15, 0, 15)]
    cands = L.coarse(pos, np.arange(h0 - 30, h0 + 30.1, 1.0), np.arange(-15, 36, 2.5), fovs)
    if not cands:
        return {**out, "status": "no-candidates"}
    best = None
    for s, p in cands[:4]:
        zlo, zhi = (g + 0.5, g + 6) if p[2] < g + 5 else (p[2] - 20, p[2] + 10)
        b = [(p[0] - 25, p[0] + 25), (p[1] - 25, p[1] + 25), (zlo, zhi), (p[3] - 12, p[3] + 12), (-35, 60), (-8, 8),
             (fovs[0] * 0.85, fovs[-1] * 1.15)]
        r = L.refine(np.array(p), b)
        if best is None or r[0] > best[0]:
            best = r
    # ambiguity: best coarse candidate that differs substantially in yaw or position
    alt = [s for s, p in cands if abs(((p[3] - best[1][3] + 180) % 360) - 180) > 8 or np.hypot(p[0] - best[1][0], p[1] - best[1][1]) > 12]
    IMGDIR.mkdir(parents=True, exist_ok=True)
    img.save(IMGDIR / f"{rec['pageid']}.jpg", quality=90)
    return {**out, "status": "ok", "score": best[0], "pose": [round(v, 3) for v in best[1]], "alt": alt[0] if alt else None,
            "coarse": cands[0][0], "w": W, "h": H, "fov_exif": fov0, "secs": round(time.time() - t0, 1)}


def main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 2000, workers=int(os.environ.get("WORKERS", "4"))):
    todo = candidates(limit)
    print(f"{len(todo)} candidates", file=sys.stderr, flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    with Pool(workers, initializer=init_worker) as pool, open(RESULTS, "a") as f:
        for i, r in enumerate(pool.imap_unordered(solve, todo)):
            f.write(json.dumps(r) + "\n"); f.flush()
            ok += r.get("status") == "ok" and r["score"] > 0.85
            if i % 10 == 0:
                print(f"{i}/{len(todo)} good={ok} last={r.get('status')} {r.get('score')}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
