"""Export accepted photo / camera poses and imagery into web/."""
import json, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data/work"
WEB = ROOT / "web"
PHOTO_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
CAM_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80


def accepted_photos():
    meta = {}
    for l in open(ROOT / "data/raw/commons.jsonl"):
        r = json.loads(l)
        meta[r["pageid"]] = r
    best = {}
    for l in open(WORK / "photo_poses.jsonl"):
        r = json.loads(l)
        if r.get("status") != "ok" or r["score"] < PHOTO_MIN:
            continue
        if r.get("alt") is not None and r["score"] - r["alt"] < 0.015:
            continue  # ambiguous: another distinct pose explains the image about as well
        best[r["pageid"]] = r
    return [(r, meta[pid]) for pid, r in best.items() if pid in meta]


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
            im = Image.open(src).convert("RGB")
            im.thumbnail((1024, 1024), Image.LANCZOS)
            im.save(dst, quality=82, optimize=True, progressive=True)
            t = im.copy()
            t.thumbnail((320, 168), Image.LANCZOS)
            t.save(th, quality=75)
        im = Image.open(dst)
        photos.append({
            "id": r["pageid"], "title": m["title"], "url": m["url"], "img": f"photos/{r['pageid']}.jpg",
            "thumb": f"photos/t/{r['pageid']}.jpg", "w": im.width, "h": im.height, "pose": r["pose"],
            "score": round(r["score"], 3), "artist": m.get("artist", "")[:120], "license": m.get("license"),
            "license_url": m.get("license_url"), "date": m.get("date"),
        })
    photos.sort(key=lambda p: p["id"])
    (WEB / "data/photos.json").write_text(json.dumps(photos, separators=(",", ":")))

    cams = []
    if (WORK / "cam_poses.jsonl").exists():
        best = {}
        for l in open(WORK / "cam_poses.jsonl"):
            r = json.loads(l)
            if r.get("status") == "ok" and r["score"] >= CAM_MIN:
                best[r["id"]] = r
        cams = [{"id": r["id"], "name": r["name"], "pose": r["pose"], "w": r["w"], "h": r["h"], "rect": r["rect"],
                 "score": round(r["score"], 3)} for r in best.values()]
    (WEB / "data/cams.json").write_text(json.dumps(cams, separators=(",", ":")))
    print(len(photos), "photos,", len(cams), "cams")


if __name__ == "__main__":
    main()
