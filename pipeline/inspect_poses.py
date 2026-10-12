"""Render model-edge overlays for solved photo poses into a contact sheet for eyeballing accuracy."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import render, localize

ROOT = Path(__file__).resolve().parent.parent


def edges(city, pose, W, H):
    lab, dep, prim = render.render(city, pose, W, H)
    m = localize.model_classes(lab)
    nrm = city.normal[np.maximum(prim, 0)]
    ld = np.log(np.nan_to_num(dep, posinf=1e5) + 1)
    e = np.zeros((H, W), bool)
    e[:, 1:] |= (m[:, 1:] != m[:, :-1]) | (np.abs(ld[:, 1:] - ld[:, :-1]) > 0.15) | ((nrm[:, 1:] * nrm[:, :-1]).sum(-1) < 0.8)
    e[1:] |= (m[1:] != m[:-1]) | (np.abs(ld[1:] - ld[:-1]) > 0.15) | ((nrm[1:] * nrm[:-1]).sum(-1) < 0.8)
    return e


def main(src, out, lo, hi, n=16, imgdir="photos", key="pageid"):
    city = render.City()
    rs = [json.loads(l) for l in open(src)]
    rs = [r for r in rs if r.get("status") == "ok" and lo <= r["score"] < hi][:n]
    tiles = []
    for r in rs:
        im = Image.open(ROOT / f"data/work/{imgdir}/{r[key]}.jpg").convert("RGB")
        im.thumbnail((400, 400))
        e = edges(city, r["pose"], im.width, im.height)
        a = np.asarray(im).copy()
        a[e] = [255, 40, 40]
        t = Image.fromarray(a)
        ImageDraw.Draw(t).text((4, 4), f"{r['score']:.3f}", fill=(255, 255, 0))
        tiles.append(t)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 400, rows * 400), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 400, (i // cols) * 400))
    sheet.save(out)
    print(len(tiles))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], float(a[3]), float(a[4]), *(a[5:6] and [int(a[5])]), *a[6:])
