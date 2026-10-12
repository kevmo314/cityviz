"""Ground/roof base texture: NYS ITS orthoimagery (2022-2025 statewide mosaic, ~0.3 m) per 500 m city tile,
reprojected server-side into our UTM 18N grid so each image is axis-aligned with the local frame.

writes web/ortho/<i>_<j>.jpg (ORTHO_PX square, north up) and web/ortho/index.json
"""
import io, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image
from shapely import box, unary_union

sys.path.insert(0, str(Path(__file__).parent))
import geo
from build_city import local_poly, TILE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web/ortho"
URL = "https://orthos.its.ny.gov/arcgis/rest/services/wms/Latest/MapServer/export"
ORTHO_PX = 1024


def fetch(key):
    i, j = key
    dst = OUT / f"{i}_{j}.jpg"
    if dst.exists():
        return key, True
    x0, y0 = geo.OX + i * TILE, geo.OY + j * TILE
    params = dict(bbox=f"{x0},{y0},{x0 + TILE},{y0 + TILE}", bboxSR=32618, imageSR=32618,
                  size=f"{ORTHO_PX},{ORTHO_PX}", format="jpg", f="image")
    for k in range(4):
        try:
            r = requests.get(URL, params=params, timeout=120)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            im.save(dst, quality=80, optimize=True, progressive=True)
            return key, True
        except Exception as e:
            time.sleep(2 + 4 * k)
    return key, False


def main():
    feats = json.loads((ROOT / "data/raw/boroughs.geojson").read_text())["features"]
    area = unary_union([q for f in feats if f["properties"].get("boroname") == "Manhattan"
                        for q in local_poly(f["geometry"])]).buffer(100)
    minx, miny, maxx, maxy = area.bounds
    keys = [(i, j) for i in range(int(minx // TILE), int(maxx // TILE) + 1)
            for j in range(int(miny // TILE), int(maxy // TILE) + 1)
            if area.intersects(box(i * TILE, j * TILE, (i + 1) * TILE, (j + 1) * TILE))]
    OUT.mkdir(parents=True, exist_ok=True)
    print(len(keys), "tiles", flush=True)
    ok = []
    with ThreadPoolExecutor(4) as ex:
        for n, (key, good) in enumerate(ex.map(fetch, keys)):
            if good:
                ok.append(key)
            if n % 25 == 0:
                print(n, len(ok), flush=True)
    (OUT / "index.json").write_text(json.dumps({"tile": TILE, "px": ORTHO_PX, "tiles": sorted(ok),
                                                "credit": "Orthoimagery: NYS ITS Geospatial Services"}))
    print(len(ok), "/", len(keys))


if __name__ == "__main__":
    main()
