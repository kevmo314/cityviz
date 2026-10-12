"""Stage A of per-photo 2.5D layers: monocular depth (Depth Anything 3, Apache-licensed checkpoint) for exported photos.

Runs in the Depth-Anything-3 environment (GPU). Reads undistorted pinhole images from web/photos and writes
data/work/depth_pred/<id>.npz with depth (relative/metric, aligned later) and confidence.
usage: depth_predict.py [--model depth-anything/DA3METRIC-LARGE] [ids...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--res", type=int, default=756)
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args()
    photos = json.loads((ROOT / "web/data/photos.json").read_text())
    if a.ids:
        photos = [p for p in photos if str(p["id"]) in set(a.ids)]
    out = ROOT / "data/work/depth_pred"
    out.mkdir(parents=True, exist_ok=True)
    model = DepthAnything3.from_pretrained(a.model).to("cuda").eval()
    for p in photos:
        dst = out / f"{p['id']}.npz"
        if dst.exists():
            continue
        img = Image.open(ROOT / "web" / p["img"]).convert("RGB")
        with torch.no_grad():
            pred = model.inference([np.asarray(img)], process_res=a.res)
        np.savez_compressed(dst, depth=pred.depth[0].astype(np.float32),
                            conf=pred.conf[0].astype(np.float32) if pred.conf is not None else None)
        print(p["id"], pred.depth.shape, flush=True)


if __name__ == "__main__":
    main()
