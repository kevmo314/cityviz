"""Street detail from LiDAR: geometry the city model doesn't have (trees, elevated rail, scaffolding, bridges, rooftop
water tanks, newer buildings), meshed and written as streamable detail tiles; plus a 1 m bare-earth DTM.

For each 250 m LiDAR tile (pipeline/fetch_lidar.py):
  1. 1 m DTM from ground returns (classes 2/18), holes filled from the nearest ground cell
  2. candidates: non-ground returns (1/17) more than MIN_H above the DTM (drops cars, people)
  3. drop returns the model already explains: within NEAR m of a model surface along any axis, or inside a building
  4. voxelize at VOX m, remove isolated voxels, smooth, marching cubes, decimate
Outputs: web/tiles/d_<i>_<j>.bin (500 m tiles, CVT1 format) + index.json "detail" list, data/work/dtm/<tile>.npz,
data/work/detail_mesh.npz (all detail geometry, for localization occluders).
"""
import json, os, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from fetch_lidar import DT, TILE as LT
from build_city import write_tile, TILE

ROOT = Path(__file__).resolve().parent.parent
LIDAR = ROOT / "data/work/lidar"
WORK = ROOT / "data/work"
OUT = ROOT / "web/tiles"
VOX = 1.5
MIN_H = 2.2
NEAR = 2.5        # horizontal: facade detail (cornices, balconies) and model offsets
NEAR_DOWN = 2.5   # above a roof: parapets, mechanical units, roof offsets in the 2014 model
MARGIN = 4.0
ZGAP = 3
KEEP = float(os.environ.get("DECIMATE_KEEP", "0.15"))
_W = {}


def init_worker():
    import render
    if hasattr(os, "sched_setaffinity") and os.environ.get("CORES_PER_WORKER"):
        from localize_photos import pin_worker
        pin_worker()
    _W["city"] = render.City()
    # Only Manhattan has a current building model; elsewhere every building would count as "residual".
    from shapely import unary_union, prepared
    from build_city import local_poly
    feats = json.loads((ROOT / "data/raw/boroughs.geojson").read_text())["features"]
    _W["area"] = unary_union([q for f in feats if f["properties"].get("boroname") == "Manhattan"
                              for q in local_poly(f["geometry"])]).buffer(150)


def load(tx, ty):
    parts = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            f = LIDAR / f"{tx + dx}_{ty + dy}.bin"
            if f.exists():
                parts.append(np.fromfile(f, DT))
    a = np.concatenate(parts) if parts else np.empty(0, DT)
    x0, y0 = tx * LT - MARGIN, ty * LT - MARGIN
    m = (a["x"] >= x0) & (a["x"] < x0 + LT + 2 * MARGIN) & (a["y"] >= y0) & (a["y"] < y0 + LT + 2 * MARGIN)
    return a[m]


def dtm(a, x0, y0, n):
    g = a[np.isin(a["cls"], (2, 18))]
    z = np.full((n, n), np.nan, np.float32)
    if len(g) == 0:
        return z
    i = np.clip(((g["y"] - y0) / VOX).astype(int), 0, n - 1)
    j = np.clip(((g["x"] - x0) / VOX).astype(int), 0, n - 1)
    # per-cell low-ish ground (min of returns rejects nothing useful; use mean of the lower half via sort trick)
    order = np.lexsort((g["z"], i * n + j))
    k = (i * n + j)[order]
    first = np.r_[True, k[1:] != k[:-1]]
    z.flat[k[first]] = g["z"][order][first]
    return z


def fill(z):
    bad = np.isnan(z)
    if bad.all():
        return z
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    z = z[tuple(idx)]
    return ndimage.median_filter(z, 3)


HDIRS = [(np.cos(a), np.sin(a), 0.0) for a in np.radians(np.arange(0, 360, 22.5))]


def explained(city, p):
    """True where a point lies within NEAR (horizontally) / NEAR_DOWN (above a roof) of a model surface, or inside a
    closed building."""
    o = np.ascontiguousarray(p, np.float32)
    near = np.zeros(len(p), bool)
    for d in HDIRS + [(0.0, 0.0, -1.0)]:
        dv = np.tile(np.float32(d), (len(p), 1))
        prim, t, hit = city.cast(o, dv)
        near |= hit & (t < (NEAR_DOWN if d[2] < 0 else NEAR))
    prim, t, hit = city.cast(o, np.tile(np.float32([0, 0, 1]), (len(p), 1)))
    # ray going up hits the underside of a roof facing up: we're inside the building volume
    inside = hit & (city.label[np.maximum(prim, 0)] != 0) & (city.normal[np.maximum(prim, 0), 2] > 0.3)
    return near | inside


def tile(key):
    import fast_simplification
    from skimage.measure import marching_cubes
    tx, ty = key
    city = _W["city"]
    a = load(tx, ty)
    if len(a) < 100:
        return key, None, None
    x0, y0 = tx * LT - MARGIN, ty * LT - MARGIN
    n = int((LT + 2 * MARGIN) / VOX)
    ground = fill(dtm(a, x0, y0, n))
    if np.isnan(ground).all():
        return key, None, None
    ng = a[np.isin(a["cls"], (1, 17))]
    from shapely import contains_xy
    ng = ng[contains_xy(_W["area"], ng["x"].astype(np.float64), ng["y"].astype(np.float64))]
    i = np.clip(((ng["y"] - y0) / VOX).astype(int), 0, n - 1)
    j = np.clip(((ng["x"] - x0) / VOX).astype(int), 0, n - 1)
    h = ng["z"] - ground[i, j]
    ng, h, i, j = ng[h > MIN_H], h[h > MIN_H], i[h > MIN_H], j[h > MIN_H]
    res = dict(dtm=ground.astype(np.float32))
    if len(ng) < 50:
        return key, None, res
    p = np.stack([ng["x"], ng["y"], ng["z"]], 1)
    keep = ~explained(city, p)
    ng, p, i, j = ng[keep], p[keep], i[keep], j[keep]
    if len(p) < 50:
        return key, None, res
    # 3D occupancy. LiDAR at ~5 pts/m2 samples crowns sparsely, so voxels are closed vertically across gaps of up to
    # ZGAP (solid crowns, decks) and horizontally across single empty cells, without bridging a low tree to a roof edge.
    zb = np.floor(p[:, 2].min()) - 3
    nz = int(np.ceil((p[:, 2].max() - zb) / VOX)) + 4
    k = ((p[:, 2] - zb) / VOX).astype(int)
    cnt = np.zeros((n, n, nz), np.int16)
    np.add.at(cnt, (j, i, k), 1)
    multi = np.zeros((n, n, nz), np.int16)  # multiple-return pulses: vegetation (walls and roofs return once)
    np.add.at(multi, (j, i, k), (ng["nret"] > 1).astype(np.int16))
    occ = cnt > 0
    occ &= ndimage.uniform_filter(occ.astype(np.float32), 3) * 27 >= 3  # isolated returns: wires, birds, noise
    occ = ndimage.binary_closing(occ, np.ones((1, 1, ZGAP + 1), bool))
    occ = ndimage.binary_closing(occ, np.ones((3, 3, 1), bool))
    occ |= np.roll(occ, -1, axis=2)  # one voxel thicker downward (LiDAR sees surfaces from above)
    lab, nlab = ndimage.label(occ)
    if nlab == 0:
        return key, None, res
    ids = np.arange(1, nlab + 1)
    vol = ndimage.sum(occ, lab, ids)
    veg = ndimage.sum(multi, lab, ids) / np.maximum(ndimage.sum(cnt, lab, ids), 1)
    # small single-return pieces are facade/model mismatches; keep vegetation and sizeable structures
    occ &= np.isin(lab, ids[(vol >= 4) & ((veg >= 0.15) | (vol >= 40))])
    if occ.sum() < 8:
        return key, None, res
    occ = occ.astype(np.float32)
    f = ndimage.gaussian_filter(occ, 0.9)
    if f.max() <= 0.35:
        return key, None, res
    v, faces, _, _ = marching_cubes(f, 0.35)
    v = v * VOX + np.array([x0 + VOX / 2, y0 + VOX / 2, zb + VOX / 2])
    # crop to the tile proper (the margin only exists for seamless smoothing)
    c = v[faces].mean(1)
    inside = (c[:, 0] >= tx * LT) & (c[:, 0] < (tx + 1) * LT) & (c[:, 1] >= ty * LT) & (c[:, 1] < (ty + 1) * LT)
    faces = faces[inside]
    if len(faces) == 0:
        return key, None, res
    used = np.unique(faces)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    v, faces = v[used], remap[faces]
    if len(faces) > 200:
        v, faces = fast_simplification.simplify(v.astype(np.float32), faces.astype(np.int32), 1 - KEEP)
    # marching cubes winds faces toward decreasing values; flip so normals face outward (toward empty space)
    faces = faces[:, [0, 2, 1]]
    return key, (v.astype(np.float32), faces.astype(np.int32)), res


def main():
    keys = sorted({tuple(map(int, f.stem.split("_"))) for f in LIDAR.glob("*.bin")})
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        keys = [k for k in keys if abs(k[0] - int(sys.argv[2])) <= 1 and abs(k[1] - int(sys.argv[3])) <= 1]
    (WORK / "dtm").mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: ([], []))
    nv = defaultdict(int)
    with Pool(int(os.environ.get("WORKERS", "8")), initializer=init_worker) as pool:
        for c, (key, mesh, res) in enumerate(pool.imap_unordered(tile, keys)):
            if res is not None:
                np.savez_compressed(WORK / f"dtm/{key[0]}_{key[1]}.npz", z=res["dtm"],
                                    x0=key[0] * LT - MARGIN, y0=key[1] * LT - MARGIN, res=VOX)
            if mesh is not None:
                v, fc = mesh
                g = (int(np.floor(key[0] * LT / TILE)), int(np.floor(key[1] * LT / TILE)))
                groups[g][0].append(v)
                groups[g][1].append(fc + nv[g])
                nv[g] += len(v)
            if c % 50 == 0:
                print(c, "/", len(keys), "tiles", sum(nv.values()), "verts", flush=True)
    idx_path = OUT / "index.json"
    index = json.loads(idx_path.read_text())
    for f in OUT.glob("d_*.bin"):
        f.unlink()
    index["detail"] = []
    allv, allf, n = [], [], 0
    for (i, j), (vs, fs) in sorted(groups.items()):
        v, fc = np.concatenate(vs), np.concatenate(fs)
        fn = f"d_{i}_{j}.bin"
        write_tile(OUT / fn, v, fc)
        index["detail"].append({"file": fn, "i": i, "j": j, "min": v.min(0).round(1).tolist(),
                                "max": v.max(0).round(1).tolist(), "tris": int(len(fc))})
        allv.append(v); allf.append(fc + n); n += len(v)
    idx_path.write_text(json.dumps(index))
    if allv:
        np.savez_compressed(WORK / "detail_mesh.npz", v=np.concatenate(allv), f=np.concatenate(allf))
    print("detail tiles", len(index["detail"]), "tris", sum(t["tris"] for t in index["detail"]))


if __name__ == "__main__":
    main()
