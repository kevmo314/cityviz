"""Find the car hood at the bottom of dashboard-mounted Mapillary photos, so it isn't painted onto the street.

Scans up from the bottom row: a row belongs to the hood while most of its central band segments as car (ADE20k). The
crop is stored as a texture rect [x0, y0, x1, y1] (image fractions, y down), the same mechanism that hides the
timestamp banner on traffic cams.

usage: photo_hood.py <results.jsonl>...   (GPU node, cityviz .venv) -> data/work/photo_rect.jsonl (appends new ids)
"""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import segment

ROOT = Path(__file__).resolve().parent.parent
CAR = 20          # ADE20k "car"
MIN_HOOD = 0.03   # ignore thinner bands (a car parked across the bottom edge, noise)
MARGIN = 0.01     # extra rows cropped above the detected hood edge


def hood_frac(probs):
    lab = probs.argmax(0)
    h, w = lab.shape
    band = lab[:, int(w * 0.2):int(w * 0.8)] == CAR
    rows = band.mean(1)
    k = 0
    while k < h // 2 and rows[h - 1 - k] > 0.5:
        k += 1
    f = k / h
    return f + MARGIN if f >= MIN_HOOD else 0.0


def accepted(r):
    """the export gates (export_web.accepted_photos): no point checking photos that won't be published"""
    return (r.get("status") == "ok" and r.get("edge_med_px") is not None and r["edge_med_px"] <= 5.0
            and r["edge_support"] >= 0.5 and r["iou"] >= 0.85 and (r.get("margin") is None or r["margin"] >= 0.005))


def main():
    out = ROOT / "data/work/photo_rect.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    ids = []
    for p in sys.argv[1:]:
        for l in open(p):
            r = json.loads(l)
            if accepted(r) and str(r["pageid"]).startswith("mly_") and r["pageid"] not in done:
                ids.append(r["pageid"])
    seg = segment.Segmenter()
    with open(out, "a") as f:
        for pid in ids:
            src = ROOT / f"data/work/photos/{pid}.jpg"
            if not src.exists():
                continue
            hf = hood_frac(seg.probs(Image.open(src).convert("RGB"), size=512))
            f.write(json.dumps({"id": pid, "rect": [0, 0, 1, round(1 - hf, 3)]}) + "\n")
    print(len(ids), "photos checked")


if __name__ == "__main__":
    main()
