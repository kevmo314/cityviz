"""Stage B of per-photo 2.5D layers: align predicted depth to the city model and export a compact depth grid.

For each exported photo with a stage-A prediction:
  1. render the model's z-depth from the solved (pinhole) camera at grid resolution
  2. fit a robust scale between predicted and model depth on building/ground pixels
  3. fuse: keep predicted depth where something stands in front of the model (trees, awnings, vehicles,
     street furniture) or where the model is empty (e.g. newer buildings); never push surfaces behind
     the model's facade; drop sky
  4. write web/depth/<id>.bin:  'CVD2' | uint16 GW, GH | float32 zmax | deflate(uint16[GH*GW] log-encoded z, 0 = empty)
The viewer back-projects the grid through the photo's camera to build a textured mesh.
"""
import json, struct, sys, zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import camera as C
import render
import segment

ROOT = Path(__file__).resolve().parent.parent
GRID_W = 192
ZMAX = 2000.0


def encode(z):
    q = np.zeros(z.shape, np.uint16)
    ok = (z > 0.5) & np.isfinite(z)
    q[ok] = np.clip(np.round(np.log(z[ok]) / np.log(ZMAX) * 65534) + 1, 1, 65535).astype(np.uint16)
    return q


def theta_of(p):
    return np.array([*p["pose"], *p["K"], 0.0])


def layer(city, seg, p, pred):
    W, H = p["w"], p["h"]
    gw, gh = GRID_W, max(4, round(GRID_W * H / W))
    th = theta_of(p)
    # model z-depth at grid resolution
    d = C.ray_dirs(th, gw, gh)
    o = np.broadcast_to(np.asarray(th[:3], np.float32), d.shape)
    prim, t, hit = city.cast(o, d)
    fwd, _, _ = C.basis(th[3], th[4], th[5])
    zm = (t * (d @ fwd)).reshape(gh, gw)
    hit = hit.reshape(gh, gw)
    img = Image.open(ROOT / "web" / p["img"]).convert("RGB")
    lab = seg.labels(img)
    lab = cv2.resize(lab, (gw, gh), interpolation=cv2.INTER_NEAREST)
    zp = cv2.resize(pred["depth"], (gw, gh), interpolation=cv2.INTER_AREA)
    conf = cv2.resize(pred["conf"], (gw, gh), interpolation=cv2.INTER_AREA) if pred.get("conf") is not None else np.ones_like(zp)
    fit = hit & np.isin(lab, [segment.BUILDING, segment.GROUND]) & (zm < 400) & (zp > 0) & (conf >= np.percentile(conf, 30))
    if fit.sum() < 50:
        return None, None
    r = np.log(zm[fit] / zp[fit])
    for _ in range(3):  # trimmed median in log space
        m = np.median(r)
        r = r[np.abs(r - m) < max(0.15, 2.5 * np.median(np.abs(r - m)))]
    s = float(np.exp(np.median(r)))
    z = zp * s
    resid = float(np.median(np.abs(np.log(zm[fit] / z[fit]))))
    z = np.where(hit, np.minimum(z, zm * 0.995), z)  # never behind the modelled facade
    # Keep only residual geometry: what stands in front of the model (trees, awnings, vehicles, fences) or where
    # the model is empty (e.g. newer buildings). Surfaces the model already explains are painted, not layered.
    z[hit & (np.abs(np.log(np.maximum(z, 1e-3) / zm)) < 0.05)] = 0
    z[lab == segment.SKY] = 0
    z[(z > ZMAX) | ~np.isfinite(z)] = 0
    # drop points lying on the street surface (already painted); keeps upright things like cars and trees
    P = th[:3] + d.reshape(gh, gw, 3) * (z / np.maximum((d @ fwd).reshape(gh, gw), 1e-3))[..., None]
    gz = np.vectorize(city.ground)(P[..., 0], P[..., 1])
    z[(z > 0) & (P[..., 2] < gz + 0.4)] = 0
    keep = cv2.morphologyEx((z > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    z[~keep] = 0
    return z, dict(scale=s, resid=resid, coverage=float((z > 0).mean()))


def main():
    photos = json.loads((ROOT / "web/data/photos.json").read_text())
    city = render.City()
    seg = segment.Segmenter()
    out = ROOT / "web/depth"
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for p in photos:
        src = ROOT / f"data/work/depth_pred/{p['id']}.npz"
        if not src.exists() or "K" not in p:
            continue
        pred = dict(np.load(src, allow_pickle=True))
        if pred.get("conf") is not None and pred["conf"].dtype == object:
            pred["conf"] = None
        z, st = layer(city, seg, p, pred)
        if z is None or st["resid"] > 0.25:
            continue
        gh, gw = z.shape
        with open(out / f"{p['id']}.bin", "wb") as f:
            f.write(b"CVD2" + struct.pack("<HHf", gw, gh, ZMAX))
            f.write(zlib.compress(encode(z).tobytes(), 9))
        stats[p["id"]] = st
        print(p["id"], {k: round(v, 3) for k, v in st.items()}, flush=True)
    (ROOT / "data/work/depth_layers.json").write_text(json.dumps(stats))


if __name__ == "__main__":
    main()
