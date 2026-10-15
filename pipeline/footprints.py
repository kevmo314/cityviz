"""Where each published photo / traffic cam lands on the city: a coarse grid of image rays cast into the city mesh.

The viewer projects these footprints into the current view to pick which projectors are worth a texture slot (the
ones covering the most of the screen that nothing else covers), instead of guessing from distance alone.

usage: footprints.py   (node with embreex; reads web/data/photos.json and cams.json)
writes web/data/footprints.bin:
  b"CVF1", uint32 n_photos, uint32 n_cams, uint8 GW, uint8 GH, 2 pad bytes,
  then int16 [n_photos + n_cams][GH][GW][3]: hit point minus camera position (x east, y north, z up, metres), in
  photos.json then cams.json order; x = -32768 where the ray hits nothing within MAX_DIST
"""
import json, struct, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import render

ROOT = Path(__file__).resolve().parent.parent
GW, GH = 6, 4
MAX_DIST = 2000.0
MISS = -32768


def rays(item):
    """grid of ray directions over the image's texture rect (same camera model as web/src/projectors.js; lens
    distortion is ignored, it moves the coarse grid by well under a cell)"""
    f, cx, cy = item["K"]
    a = item["w"] / item["h"]
    x0, y0, x1, y1 = item.get("rect") or [0, 0, 1, 1]
    fw, rt, up = render.basis(*item["pose"][3:6])
    u = np.linspace(x0, x1, GW) - 0.5                 # width units, right +
    v = (np.linspace(y0, y1, GH) - 0.5) / a           # width units, down +
    U, V = np.meshgrid(u, v)
    d = fw[None, None] + ((U - cx) / f)[..., None] * rt + ((cy - V) / f)[..., None] * up
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)


def main():
    items = [json.loads((ROOT / f"web/data/{n}.json").read_text()) for n in ("photos", "cams")]
    city = render.City()
    out = np.full((sum(map(len, items)), GH * GW, 3), MISS, np.int16)
    k = 0
    for group in items:
        for it in group:
            d = rays(it)
            o = np.broadcast_to(np.asarray(it["pose"][:3], np.float32), d.shape)
            _, t, hit = city.cast(o, d)
            hit &= t < MAX_DIST
            out[k, hit] = np.round(d[hit] * t[hit, None]).astype(np.int16)
            k += 1
    with open(ROOT / "web/data/footprints.bin", "wb") as fh:
        fh.write(b"CVF1" + struct.pack("<IIBBxx", len(items[0]), len(items[1]), GW, GH))
        fh.write(out.astype("<i2").tobytes())
    print(k, "footprints,", round(float((out[..., 0] != MISS).mean()), 3), "of rays hit")


if __name__ == "__main__":
    main()
