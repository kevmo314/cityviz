"""Extrude building footprints + terrain into streamable binary tiles and a full mesh for localization.

Tile format (little endian):
  char[4] 'CVT1' | uint32 nverts | uint32 nindices | float32 ox, oy, oz, scale
  int16[nverts*3] quantized local positions (x east, y north, z up) = origin + q*scale
  uint32[nindices] triangle indices
"""
import json, struct, sys
from collections import defaultdict
from pathlib import Path

import mapbox_earcut as earcut
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from shapely.geometry import shape, box, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep

sys.path.insert(0, str(Path(__file__).parent))
import geo

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
OUT = ROOT / "web/tiles"
WORK = ROOT / "data/work"
TILE = 500.0
SCALE = 0.05


def local_poly(geom):
    def ring(coords):
        c = np.asarray(coords)[:, :2]
        x, y = geo.to_local(c[:, 0], c[:, 1])
        return np.column_stack([x, y])
    polys = []
    g = shape(geom)
    for p in (g.geoms if isinstance(g, MultiPolygon) else [g]):
        lp = Polygon(ring(p.exterior.coords), [ring(i.coords) for i in p.interiors])
        if not lp.is_valid:
            lp = lp.buffer(0)
        for q in (lp.geoms if isinstance(lp, MultiPolygon) else [lp]):
            if isinstance(q, Polygon) and q.area > 1:
                polys.append(q)
    return polys


class MeshBuilder:
    def __init__(self):
        self.v, self.i, self.n = [], [], 0

    def add(self, verts, idx):
        self.v.append(np.asarray(verts, np.float64))
        self.i.append(np.asarray(idx, np.int64) + self.n)
        self.n += len(verts)

    def arrays(self):
        if not self.v:
            return np.zeros((0, 3)), np.zeros(0, np.int64)
        return np.concatenate(self.v), np.concatenate(self.i)


def rings_of(poly):
    rings = [np.asarray(poly.exterior.coords)[:-1, :2]]
    rings += [np.asarray(r.coords)[:-1, :2] for r in poly.interiors]
    return [r for r in rings if len(r) >= 3]


def triangulate(poly):
    rings = rings_of(poly)
    if not rings:
        return None, None
    pts = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = earcut.triangulate_float64(pts, ends)
    return pts, np.asarray(tri).reshape(-1, 3)


def extrude(mb, poly, z0, z1):
    poly = poly.simplify(0.2, preserve_topology=True)
    if poly.is_empty or not isinstance(poly, Polygon):
        return
    pts, tri = triangulate(poly)
    if pts is None or len(tri) == 0:
        return
    n = len(pts)
    top = np.column_stack([pts, np.full(n, z1)])
    bot = np.column_stack([pts, np.full(n, z0)])
    verts = np.concatenate([top, bot])
    idx = [tri]  # roof, CCW from above (earcut keeps input winding; fixed below)
    start = 0
    for r in rings_of(poly):
        k = len(r)
        a = np.arange(start, start + k)
        b = np.roll(a, -1)
        idx.append(np.column_stack([a, a + n, b + n]))
        idx.append(np.column_stack([a, b + n, b]))
        start += k
    idx = np.concatenate(idx)
    # Make every triangle face outward: roof up, walls away from the footprint.
    p = verts[idx]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    cen = p.mean(1)
    roof = np.abs(nrm[:, 2]) > 1e-3 * np.linalg.norm(nrm, axis=1)
    flip = np.zeros(len(idx), bool)
    flip[roof] = nrm[roof, 2] < 0
    walls = ~roof
    if walls.any():
        # outward test: step from wall centroid along normal; outward if that point is outside polygon
        from shapely import contains_xy
        step = cen[walls, :2] + 0.05 * nrm[walls, :2] / (np.linalg.norm(nrm[walls, :2], axis=1, keepdims=True) + 1e-12)
        flip[walls] = contains_xy(poly, step[:, 0], step[:, 1])
    idx[flip] = idx[flip][:, [0, 2, 1]]
    mb.add(verts, idx)


def load_lod2():
    """BIN -> {tiles: {tile_key: (verts, faces)}, cx, cy, top} from parsed CityGML (ground surfaces dropped)."""
    out = {}
    for fn in sorted((WORK / "lod2").glob("DA*.npz")):
        d = np.load(fn)
        v, f, kind, bins = d["v"].astype(np.float64), d["f"], d["kind"], d["bin"]
        keep = kind != 0
        f, bins = f[keep], bins[keep]
        order = np.argsort(bins, kind="stable")
        f, bins = f[order], bins[order]
        split = np.flatnonzero(np.diff(bins)) + 1
        for bb, ff in zip(np.split(bins, split), np.split(f, split)):
            b = int(bb[0])
            if b <= 0 or b in out:
                continue
            tri = v[ff]
            cen = tri.mean(1)
            keys = np.floor(cen[:, :2] / TILE).astype(np.int64)
            tl = {}
            for k in {tuple(x) for x in keys}:
                m = (keys[:, 0] == k[0]) & (keys[:, 1] == k[1])
                used, inv = np.unique(ff[m], return_inverse=True)
                tl[(int(k[0]), int(k[1]))] = (v[used], inv.reshape(-1, 3))
            out[b] = {"tiles": tl, "cx": float(cen[:, 0].mean()), "cy": float(cen[:, 1].mean()),
                      "top": float(tri[:, :, 2].max()), "bottom": float(tri[:, :, 2].min())}
    return out


def lod2_matches(b, polys, h):
    """Same building still standing: centroid within 5 m and roof height within 20% (or 4 m)."""
    if b["cx"] is None or h <= 2:
        return False
    c = polys[0].centroid
    if np.hypot(b["cx"] - c.x, b["cy"] - c.y) > 5 + 0.2 * np.sqrt(polys[0].area):
        return False
    h2 = b["top"] - b["bottom"]
    return abs(h2 - h) <= max(4.0, 0.2 * h)


def write_tile(path, verts, idx):
    origin = np.floor(verts.min(0))
    scale = max(SCALE, float(np.ceil((verts - origin).max() / 32000 * 100) / 100))
    q = np.round((verts - origin) / scale).astype(np.int16)
    # weld duplicate vertices (CityGML polygons each carry their own copies)
    q, inv = np.unique(q, axis=0, return_inverse=True)
    idx = inv.reshape(-1)[idx]
    idx = idx[(idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])]
    with open(path, "wb") as f:
        f.write(b"CVT1" + struct.pack("<II4f", len(q), len(idx) * 3, *origin, scale))
        f.write(q.tobytes())
        f.write(idx.astype(np.uint32).tobytes())


def main():
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    feats = json.loads((RAW / "buildings_manhattan.geojson").read_text())["features"]
    print(len(feats), "buildings")

    # --- ground elevation field from building ground elevations
    gpts, gz, blds = [], [], []
    for f in feats:
        p = f["properties"]
        if not f.get("geometry"):
            continue
        polys = local_poly(f["geometry"])
        if not polys:
            continue
        ge = float(p.get("ground_elevation") or 0) * geo.FT
        h = float(p.get("height_roof") or 0) * geo.FT
        if 0 < ge < 100:
            c = polys[0].centroid
            gpts.append((c.x, c.y)); gz.append(ge)
        blds.append((polys, ge, h, int(p.get("bin") or 0)))
    gpts, gz = np.array(gpts), np.array(gz)

    # --- land polygons (Manhattan + neighbors) in local frame
    bbox = box(-9000, -9000, 11000, 17000)
    land = []
    for f in json.loads((RAW / "boroughs.geojson").read_text())["features"]:
        for q in local_poly(f["geometry"]):
            q = q.intersection(bbox)
            if not q.is_empty:
                land.append(q)
    land = unary_union(land).buffer(0)
    manhattan = unary_union([q for f in json.loads((RAW / "boroughs.geojson").read_text())["features"]
                             if f["properties"].get("boroname") == "Manhattan" for q in local_poly(f["geometry"])])

    # heightfield grid (5 m resolution field, used for terrain + camera heights)
    G = 10.0
    gx = np.arange(-9000, 11000 + G, G)
    gy = np.arange(-9000, 17000 + G, G)
    XX, YY = np.meshgrid(gx, gy)
    hf = griddata(gpts, gz, (XX, YY), method="linear")
    near = griddata(gpts, gz, (XX, YY), method="nearest")
    # outside Manhattan's building cloud fall back to a flat 3 m
    from shapely import contains_xy
    inman = contains_xy(manhattan.buffer(50), XX, YY)
    hf = np.where(np.isnan(hf), near, hf)
    hf = np.where(inman, hf, 3.0)
    hf = gaussian_filter(hf, 2.0)
    np.savez_compressed(WORK / "heightfield.npz", hf=hf.astype(np.float32), x0=gx[0], y0=gy[0], res=G)

    def ground(x, y):
        i = np.clip(((np.asarray(y) - gy[0]) / G).astype(int), 0, len(gy) - 1)
        j = np.clip(((np.asarray(x) - gx[0]) / G).astype(int), 0, len(gx) - 1)
        return hf[i, j]

    # --- buildings into tiles
    tiles = defaultdict(MeshBuilder)
    lod1 = defaultdict(MeshBuilder)
    full = MeshBuilder()
    lod2 = load_lod2()
    n_lod2 = 0
    for polys, ge, h, bin_ in blds:
        use2 = lod2.get(bin_)
        if use2 is not None and lod2_matches(use2, polys, h):
            for k, (vv, ff) in use2["tiles"].items():
                tiles[k].add(vv, ff)
            use2["used"] = True
            n_lod2 += 1
        for poly in polys:
            c = poly.centroid
            g0 = float(ground(c.x, c.y))
            hh = h if h > 2 else 6.0
            key = (int(np.floor(c.x / TILE)), int(np.floor(c.y / TILE)))
            if not (use2 is not None and use2.get("used")):
                extrude(tiles[key], poly, g0 - 2.0, g0 + hh)
            if hh >= 25:
                big = poly.simplify(1.5, preserve_topology=True)
                extrude(lod1[(key[0] // 4, key[1] // 4)], big, g0 - 2.0, g0 + hh)

    # Outside Manhattan we have no current footprints, so LOD2 is used as-is (BIN borough prefix != 1).
    for bin_, b in lod2.items():
        if bin_ // 1000000 != 1 and bin_ % 1000000 != 0:
            for k, (vv, ff) in b["tiles"].items():
                tiles[k].add(vv, ff)
    print(f"LOD2 used for {n_lod2} Manhattan buildings; extruded footprints elsewhere")

    # --- terrain: land clipped to 50 m cells, triangulated, z from heightfield
    terrain = defaultdict(MeshBuilder)
    land_p = prep(land)
    T = 50.0
    minx, miny, maxx, maxy = land.bounds
    for x in np.arange(np.floor(minx / T) * T, maxx, T):
        for y in np.arange(np.floor(miny / T) * T, maxy, T):
            cell = box(x, y, x + T, y + T)
            if not land_p.intersects(cell):
                continue
            part = cell if land_p.contains(cell) else land.intersection(cell)
            for q in (part.geoms if hasattr(part, "geoms") else [part]):
                if not isinstance(q, Polygon) or q.area < 1:
                    continue
                pts, tri = triangulate(q)
                if pts is None or len(tri) == 0:
                    continue
                z = ground(pts[:, 0], pts[:, 1])
                verts = np.column_stack([pts, z])
                p = verts[tri]
                nz = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
                tri = np.where((nz < 0)[:, None], tri[:, [0, 2, 1]], tri)
                key = (int(np.floor((x + 1) / (TILE * 4))), int(np.floor((y + 1) / (TILE * 4))))
                terrain[key].add(verts, tri)

    index = {"tileSize": TILE, "lod1Size": TILE * 4, "tiles": [], "lod1": [], "terrain": []}
    allv, alli, allk = [], [], []
    nv = 0
    for name, group, lst in (("b", tiles, "tiles"), ("l", lod1, "lod1"), ("t", terrain, "terrain")):
        for (i, j), mb in sorted(group.items()):
            v, ix = mb.arrays()
            if len(ix) == 0:
                continue
            fn = f"{name}_{i}_{j}.bin"
            write_tile(OUT / fn, v, ix)
            lo, hi = v.min(0), v.max(0)
            index[lst].append({"file": fn, "i": i, "j": j, "min": lo.round(1).tolist(), "max": hi.round(1).tolist(),
                               "tris": int(len(ix))})
            if name in ("b", "t"):
                allv.append(v); alli.append(ix + nv); nv += len(v)
                allk.append(np.full(len(ix), name == "t", np.uint8))
    (OUT / "index.json").write_text(json.dumps(index))
    np.savez_compressed(WORK / "city_mesh.npz", v=np.concatenate(allv).astype(np.float32),
                        f=np.concatenate(alli).astype(np.int32), terrain=np.concatenate(allk))
    for k in ("tiles", "lod1", "terrain"):
        print(k, len(index[k]), sum(t["tris"] for t in index[k]), "tris")


if __name__ == "__main__":
    main()
