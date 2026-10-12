"""Export accepted photo / camera poses and imagery into web/."""
import json, sys
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
CAM_EDGE_MAX_PX = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0


def accepted_photos():
    meta = {}
    for l in open(ROOT / "data/raw/commons.jsonl"):
        r = json.loads(l)
        meta[r["pageid"]] = r
    best = {}
    for l in open(WORK / "photo_poses.jsonl"):
        r = json.loads(l)
        if r.get("status") != "ok" or r.get("edge_med_px") is None:
            continue
        if r["edge_med_px"] > EDGE_MAX_PX or r["edge_support"] < 0.5 or r["iou"] < 0.85:
            continue
        if r.get("margin") is not None and r["margin"] < 0.005:
            continue  # ambiguous: another distinct pose explains the image about as well
        best[r["pageid"]] = r
    return [(r, meta[pid]) for pid, r in best.items() if pid in meta]


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
        src = WORK / f"photos/{r['pageid']}.jpg"
        if not src.exists():
            continue
        if is_dark(src):
            continue  # night shots segment poorly and paint the city black
        dst, th = WEB / f"photos/{r['pageid']}.jpg", WEB / f"photos/t/{r['pageid']}.jpg"
        if not dst.exists():
            im = undistort(Image.open(src).convert("RGB"), r["theta"])
            im.thumbnail((1024, 1024), Image.LANCZOS)
            im.save(dst, quality=82, optimize=True, progressive=True)
            t = im.copy()
            t.thumbnail((320, 168), Image.LANCZOS)
            t.save(th, quality=75)
        im = Image.open(dst)
        photos.append({
            "id": r["pageid"], "title": m["title"], "url": m["url"], "img": f"photos/{r['pageid']}.jpg",
            "thumb": f"photos/t/{r['pageid']}.jpg", "w": im.width, "h": im.height, "pose": [round(v, 3) for v in r["theta"][:6]],
            "K": [round(v, 5) for v in r["theta"][6:9]], "edge_px": round(r["edge_med_px"], 2), "artist": m.get("artist", "")[:120], "license": m.get("license"),
            "license_url": m.get("license_url"), "date": m.get("date"),
        })
    photos.sort(key=lambda p: p["id"])
    (WEB / "data/photos.json").write_text(json.dumps(photos, separators=(",", ":")))

    cams = []
    if (WORK / "cam_poses.jsonl").exists():
        best = {}
        for l in open(WORK / "cam_poses.jsonl"):
            r = json.loads(l)
            if r.get("status") == "ok" and (r.get("edge_med_px") or 99) <= CAM_EDGE_MAX_PX and r["iou"] >= 0.8:
                best[r["id"]] = r
        cams = [{"id": r["id"], "name": r["name"], "pose": [round(v, 3) for v in r["theta"][:6]],
                 "K": [round(v, 5) for v in r["theta"][6:9]], "k1": round(r["theta"][9], 5), "w": r["w"], "h": r["h"],
                 "rect": r["rect"], "edge_px": round(r["edge_med_px"], 2)} for r in best.values()]
    (WEB / "data/cams.json").write_text(json.dumps(cams, separators=(",", ":")))
    print(len(photos), "photos,", len(cams), "cams")


if __name__ == "__main__":
    main()
