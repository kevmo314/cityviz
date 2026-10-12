"""Parse NYC's CityGML LOD2 building model (DA_WISE_GMLs) into triangle meshes in the local frame.

Each delivery-area file becomes data/work/lod2/DA<n>.npz with
  v (N,3) float32 local meters, f (M,3) int32, kind (M,) uint8 (0 ground, 1 wall, 2 roof), bin (M,) int64
usage: parse_citygml.py DA12_3D_Buildings_Merged.gml [...]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mapbox_earcut as earcut
import numpy as np
from lxml import etree
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
import geo

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/work/lod2"
NS = {"gml": "http://www.opengis.net/gml", "bldg": "http://www.opengis.net/citygml/building/1.0",
      "gen": "http://www.opengis.net/citygml/generics/1.0"}
KINDS = {"GroundSurface": 0, "WallSurface": 1, "RoofSurface": 2}
_sp = Transformer.from_crs("EPSG:2263", "EPSG:32618", always_xy=True)


def ring_coords(el):
    vals = np.array(el.text.split(), dtype=np.float64)
    pts = vals.reshape(-1, 3)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def triangulate_polygon(rings):
    """Triangulate a planar 3D polygon (exterior + holes) by projecting onto its dominant plane."""
    allp = np.concatenate(rings)
    if len(allp) < 3:
        return None
    # Newell normal of the exterior ring
    e = rings[0]
    n = np.zeros(3)
    for i in range(len(e)):
        a, b = e[i], e[(i + 1) % len(e)]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    if np.linalg.norm(n) < 1e-9:
        return None
    drop = int(np.argmax(np.abs(n)))
    keep = [i for i in range(3) if i != drop]
    p2 = allp[:, keep]
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = np.asarray(earcut.triangulate_float64(p2, ends), np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return None
    # earcut output winding follows the 2D projection; restore the ring's 3D orientation
    t = allp[tri]
    tn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (tn @ n) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return allp, tri


def parse(path):
    path = Path(path)
    V, F, K, B = [], [], [], []
    nv = 0
    for _, bld in etree.iterparse(str(path), tag="{%s}Building" % NS["bldg"], huge_tree=True):
        bin_el = bld.find("gen:stringAttribute[@name='BIN']/gen:value", NS)
        try:
            bin_ = int(bin_el.text) if bin_el is not None else -1
        except ValueError:
            bin_ = -1
        for surf_name, kind in KINDS.items():
            for surf in bld.iter("{%s}%s" % (NS["bldg"], surf_name)):
                for poly in surf.iter("{%s}Polygon" % NS["gml"]):
                    ext = poly.find("gml:exterior//gml:posList", NS)
                    if ext is None:
                        continue
                    rings = [ring_coords(ext)] + [ring_coords(r) for r in poly.findall("gml:interior//gml:posList", NS)]
                    rings = [r for r in rings if len(r) >= 3]
                    if not rings:
                        continue
                    res = triangulate_polygon(rings)
                    if res is None:
                        continue
                    pts, tri = res
                    V.append(pts); F.append(tri + nv); nv += len(pts)
                    K.append(np.full(len(tri), kind, np.uint8)); B.append(np.full(len(tri), bin_, np.int64))
        bld.clear()
        while bld.getprevious() is not None:
            del bld.getparent()[0]
    v = np.concatenate(V)
    x, y = _sp.transform(v[:, 0], v[:, 1])  # EPSG:2263 input is in US survey feet
    v = np.column_stack([x - geo.OX, y - geo.OY, v[:, 2] * 0.3048006096]).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (path.name.split("_")[0] + ".npz")
    np.savez_compressed(out, v=v, f=np.concatenate(F).astype(np.int32), kind=np.concatenate(K), bin=np.concatenate(B))
    return out.name, len(v), sum(len(f) for f in F)


if __name__ == "__main__":
    with ProcessPoolExecutor(len(sys.argv) - 1) as ex:
        for r in ex.map(parse, sys.argv[1:]):
            print(*r, flush=True)
