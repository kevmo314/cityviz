"""Render model-edge overlays for solved cameras into a contact sheet for eyeballing accuracy.

usage: inspect_poses.py results.jsonl out.png min_edge_px max_edge_px [n] [imgdir] [key]
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import render, solver, camera as C

ROOT = Path(__file__).resolve().parent.parent


def theta_of(r):
    return np.array(r["theta"]) if "theta" in r else C.from_legacy(r["pose"])


def edges(city, theta, W, H):
    d = C.ray_dirs(theta, W, H)
    o = np.broadcast_to(np.asarray(theta[:3], np.float32), d.shape)
    prim, t, hit = city.cast(o, d)
    lab = np.zeros(len(d), np.uint8)
    lab[hit] = city.label[prim[hit]]
    _, e = solver.model_edges(lab.reshape(H, W), t.reshape(H, W), np.where(hit, prim, -1).reshape(H, W), city.normal)
    return e


def main(src, out, lo, hi, n=16, imgdir="photos", key="pageid"):
    city = render.City()
    rs = [json.loads(l) for l in open(src)]
    rs = [r for r in rs if r.get("status") == "ok" and r.get("edge_med_px") is not None and lo <= r["edge_med_px"] < hi][:n]
    tiles = []
    for r in rs:
        im = Image.open(ROOT / f"data/work/{imgdir}/{r[key]}.jpg").convert("RGB")
        im.thumbnail((400, 400))
        e = edges(city, theta_of(r), im.width, im.height)
        a = np.asarray(im).copy()
        a[e] = [255, 40, 40]
        t = Image.fromarray(a)
        ImageDraw.Draw(t).text((4, 4), f"{r['edge_med_px']:.1f}px iou {r['iou']:.2f} sup {r['edge_support']:.2f}", fill=(255, 255, 0))
        tiles.append(t)
    cols = 4
    rows = max(1, (len(tiles) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * 400, rows * 400), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 400, (i // cols) * 400))
    sheet.save(out)
    print(len(tiles))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], float(a[3]), float(a[4]), *(a[5:6] and [int(a[5])]), *a[6:])
