"""Export accepted photo / camera poses and imagery into web/."""
import json, re, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import camera as C

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data/work"
WEB = ROOT / "web"
EDGE_MAX_PX = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
CAM_EDGE_MAX_PX = float(sys.argv[2]) if len(sys.argv) > 2 else 9.0


def accepted_photos():
    meta = {}
    for l in open(ROOT / "data/raw/commons.jsonl"):
        r = json.loads(l)
        meta[r["pageid"]] = r
    for f in WORK.glob("candidates_mly_*.jsonl"):  # Mapillary candidates carry their own credit fields
        for l in open(f):
            r = json.loads(l)
            meta[r["pageid"]] = r
    best = {}
    # LOD2-model solves (pulled from the compute nodes) supersede the earlier LOD1 runs entirely
    sources = sorted(WORK.glob("nodes/poses_shard*.jsonl")) or [WORK / "photo_poses.jsonl", *sorted(WORK.glob("cloud/*/results/*.jsonl"))]
    sources += sorted(WORK.glob("nodes/mly_*.jsonl"))
    for l in (l for src in sources if src.exists() for l in open(src)):
        r = json.loads(l)
        if r.get("status") != "ok" or r.get("edge_med_px") is None:
            continue
        if r["edge_med_px"] > EDGE_MAX_PX or r["edge_support"] < 0.5 or r["iou"] < 0.85:
            continue
        if r.get("margin") is not None and r["margin"] < 0.005:
            continue  # ambiguous: another distinct pose explains the image about as well
        best[r["pageid"]] = r
    return [(r, meta[pid]) for pid, r in best.items() if pid in meta]


MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}


def capture_year(date):
    """Decimal capture year from Commons' free-text date ('2022-04-16 12:58', 'Taken on 2 March 2011', ...)."""
    s = str(date or "").lower()
    m = re.search(r"\b(19[5-9]\d|20[0-3]\d)-(\d\d)(?:-(\d\d))?", s)
    if m:
        return round(int(m[1]) + (int(m[2]) - 1 + (int(m[3] or 15) - 1) / 31) / 12, 2)
    y = re.search(r"\b(19[5-9]\d|20[0-3]\d)\b", s)
    if not y:
        return None
    mo = re.search(r"\b(" + "|".join(MONTHS) + r")[a-z]*\b", s)
    return round(int(y[1]) + ((MONTHS[mo[1]] - 0.5) / 12 if mo else 0.5), 2)


def banner_frac(path, lo=0.085):
    """Height of the burned-in timestamp banner: the run of top rows that are mostly near-black."""
    a = np.asarray(Image.open(path).convert("L"), np.float32)
    dark = (a < 40).mean(1)
    k = 0
    while k < len(dark) * 0.2 and dark[k] > 0.5:
        k += 1
    return max(lo, (k + 2) / len(dark))


def undistort(im, theta):
    """Resample to an ideal pinhole image (same size, same focal length and principal point, k1 = 0)."""
    k1 = theta[9]
    if abs(k1) < 1e-4:
        return im
    W, H = im.size
    xo, yo = C.pixel_grid(W, H)
    xd, yd = C.distort_norm(xo, yo, k1)
    mx = (xd * W + W / 2 - 0.5).astype(np.float32)
    my = (yd * W + H / 2 - 0.5).astype(np.float32)
    a = cv2.remap(np.asarray(im), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(a)


def is_dark(path):
    im = Image.open(path).convert("L")
    im.thumbnail((128, 128))
    import numpy as np
    return float(np.asarray(im).mean()) < 70


def main():
    (WEB / "photos/t").mkdir(parents=True, exist_ok=True)
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    photos = []
    for r, m in accepted_photos():
        src = next((q for q in [WORK / f"photos/{r['pageid']}.jpg", *WORK.glob(f"cloud/*/photos/{r['pageid']}.jpg")]
                    if q.exists()), None)
        if src is None:
            continue
        if is_dark(src):
            continue  # night shots segment poorly and paint the city black
        dst, th = WEB / f"photos/{r['pageid']}.jpg", WEB / f"photos/t/{r['pageid']}.jpg"
        if not dst.exists():
            im = undistort(Image.open(src).convert("RGB"), r["theta"])
            im.thumbnail((960, 960), Image.LANCZOS)
            im.save(dst, quality=74, optimize=True, progressive=True)
            t = im.copy()
            t.thumbnail((320, 168), Image.LANCZOS)
            t.save(th, quality=75)
        im = Image.open(dst)
        photos.append({
            "id": r["pageid"], "title": m["title"], "url": m["url"], "img": f"photos/{r['pageid']}.jpg",
            "thumb": f"photos/t/{r['pageid']}.jpg", "w": im.width, "h": im.height, "pose": [round(v, 3) for v in r["theta"][:6]],
            "K": [round(v, 5) for v in r["theta"][6:9]], "edge_px": round(r["edge_med_px"], 2), "artist": m.get("artist", "")[:120], "license": m.get("license"),
            "license_url": m.get("license_url"), "date": m.get("date"),
            "year": capture_year(m.get("date")),
            **({"source": "mapillary"} if m.get("source") == "mapillary" else {}),
        })
    for p in photos:
        if (WEB / f"depth/{p['id']}.bin").exists():
            p["depth"] = f"depth/{p['id']}.bin"
    photos.sort(key=lambda p: p["id"])
    (WEB / "data/photos.json").write_text(json.dumps(photos, separators=(",", ":")))

    cams = []
    if (WORK / "cam_poses.jsonl").exists():
        best = {}
        for l in open(WORK / "cam_poses.jsonl"):
            r = json.loads(l)
            if r.get("status") == "ok" and (r.get("edge_med_px") or 99) <= CAM_EDGE_MAX_PX and r["iou"] >= 0.85 \
                    and (r.get("margin") is None or r["margin"] >= 0.005):
                best[r["id"]] = r
        cams = [{"id": r["id"], "name": r["name"], "pose": [round(v, 3) for v in r["theta"][:6]],
                 "K": [round(v, 5) for v in r["theta"][6:9]], "k1": round(r["theta"][9], 5), "w": r["w"], "h": r["h"],
                 "rect": [0, 0, 1, round(1 - banner_frac(WORK / f"cam_ref/{r['id']}.jpg"), 3)], "edge_px": round(r["edge_med_px"], 2)} for r in best.values()]
    (WEB / "data/cams.json").write_text(json.dumps(cams, separators=(",", ":")))
    print(len(photos), "photos,", len(cams), "cams")


if __name__ == "__main__":
    main()
