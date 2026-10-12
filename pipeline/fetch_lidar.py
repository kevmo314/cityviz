"""Fetch the USGS NYC LiDAR point cloud (Entwine/EPT on AWS public data) for a lon/lat box and bin it into
local-frame tiles: data/work/lidar/<tx>_<ty>.bin, one record per point (see DT).

usage: fetch_lidar.py [lon0 lat0 lon1 lat1]   (default: Manhattan)
"""
import io, json, os, sys, urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
import geo

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_NewYorkCity/"
OUT = ROOT / "data/work/lidar"
TILE = 250.0
DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("cls", "u1"), ("ret", "u1"), ("nret", "u1"),
               ("i", "<u2"), ("rgb", "u1", 3)])
_merc = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def get(url, tries=4):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except Exception:
            if k == tries - 1:
                raise


def hierarchy(key="0-0-0-0"):
    h = json.loads(get(BASE + f"ept-hierarchy/{key}.json"))
    out = {}
    subs = [k for k, v in h.items() if v == -1]
    out.update({k: v for k, v in h.items() if v > 0})
    with ThreadPoolExecutor(16) as ex:
        for sub in ex.map(hierarchy, subs):
            out.update(sub)
    return out


def node_bounds(key, cube):
    d, x, y, z = map(int, key.split("-"))
    s = (cube[3] - cube[0]) / 2 ** d
    return cube[0] + x * s, cube[1] + y * s, cube[0] + (x + 1) * s, cube[1] + (y + 1) * s


def fetch_node(args):
    key, box = args
    las = laspy.read(io.BytesIO(get(BASE + f"ept-data/{key}.laz")))
    mx, my = np.asarray(las.x), np.asarray(las.y)
    m = (mx >= box[0]) & (mx <= box[2]) & (my >= box[1]) & (my <= box[3])
    if not m.any():
        return {}
    lon, lat = _merc.transform(mx[m], my[m])
    x, y = geo.to_local(lon, lat)
    a = np.empty(int(m.sum()), DT)
    a["x"], a["y"], a["z"] = x, y, np.asarray(las.z)[m]
    a["cls"] = np.asarray(las.classification)[m]
    a["ret"], a["nret"] = np.asarray(las.return_number)[m], np.asarray(las.number_of_returns)[m]
    a["i"] = np.asarray(las.intensity)[m]
    if "red" in las.point_format.dimension_names:
        a["rgb"] = np.stack([np.asarray(las.red)[m], np.asarray(las.green)[m], np.asarray(las.blue)[m]], 1) >> 8
    else:
        a["rgb"] = 0
    tx, ty = np.floor(a["x"] / TILE).astype(int), np.floor(a["y"] / TILE).astype(int)
    out = {}
    for t in np.unique(np.stack([tx, ty], 1), axis=0):
        out[f"{t[0]}_{t[1]}"] = a[(tx == t[0]) & (ty == t[1])].tobytes()
    return out


def main():
    lon0, lat0, lon1, lat1 = map(float, sys.argv[1:5]) if len(sys.argv) >= 5 else (-74.025, 40.695, -73.905, 40.88)
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bx0, by0 = fwd.transform(lon0, lat0)
    bx1, by1 = fwd.transform(lon1, lat1)
    box = (bx0, by0, bx1, by1)
    ept = json.loads(get(BASE + "ept.json"))
    cube = ept["bounds"]
    h = hierarchy()
    keys = [k for k in h if (lambda b: b[0] < box[2] and b[2] > box[0] and b[1] < box[3] and b[3] > box[1])(node_bounds(k, cube))]
    print(len(h), "nodes;", len(keys), "intersect;", sum(h[k] for k in keys), "points", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.bin"):
        f.unlink()
    n = 0
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "16"))) as ex:
        for i, parts in enumerate(ex.map(fetch_node, [(k, box) for k in keys], chunksize=4)):
            for t, b in parts.items():
                with open(OUT / f"{t}.bin", "ab") as f:
                    f.write(b)
                n += len(b) // DT.itemsize
            if i % 200 == 0:
                print(i, "/", len(keys), n, "points", flush=True)
    print("done", n, "points in", len(list(OUT.glob("*.bin"))), "tiles")


if __name__ == "__main__":
    main()
