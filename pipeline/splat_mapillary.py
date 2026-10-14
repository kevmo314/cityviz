"""Collect Mapillary street-level photos around one splat scene (dense, overlapping sequences: what SfM needs and
Commons lacks). Panoramas are cut into four 90-degree faces; dashcams are skipped (windshield glare, hood in frame and
near-duplicate frames reconstruct as collapsed, mis-scaled sequences that drag whole blocks out of place).

usage: splat_mapillary.py <name> <lon> <lat> <radius_m>      (token in ~/.mapillary_token)
writes data/work/splat/<name>/images/mly_<id>[_f<k>].jpg and mly.jsonl (id, lon, lat, compass, captured_at, sequence,
creator, face)
"""
import json, math, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TOKEN = (Path.home() / ".mapillary_token").read_text().strip()
FIELDS = "id,captured_at,camera_type,computed_compass_angle,compass_angle,computed_geometry,geometry,sequence,creator,make,model,thumb_2048_url"
S = requests.Session()
FACE = 1024
DASHCAM = re.compile(r"blackvue|garmin|nextbase|viofo|vantrue|thinkware|dashcam|dash cam|dr\d{3}", re.I)


def search(lon, lat, radius):
    """bbox queries return at most 2000 images, so tile the area into ~100 m cells"""
    dlat = radius / 111320.0
    dlon = radius / (111320.0 * math.cos(math.radians(lat)))
    n = max(1, math.ceil(radius / 50))
    out = {}
    for i in range(n):
        for j in range(n):
            x0, y0 = lon - dlon + 2 * dlon * i / n, lat - dlat + 2 * dlat * j / n
            bbox = f"{x0},{y0},{x0 + 2 * dlon / n},{y0 + 2 * dlat / n}"
            for k in range(4):
                try:
                    r = S.get("https://graph.mapillary.com/images", timeout=60,
                              params=dict(access_token=TOKEN, fields=FIELDS, bbox=bbox, limit=2000))
                    r.raise_for_status()
                    break
                except Exception:
                    time.sleep(3 * (k + 1))
            for d in r.json().get("data", []):
                out[d["id"]] = d
    # keep the scene circle
    keep = {}
    for k, d in out.items():
        g = (d.get("computed_geometry") or d["geometry"])["coordinates"]
        if math.hypot((g[0] - lon) * 111320 * math.cos(math.radians(lat)), (g[1] - lat) * 111320) <= radius:
            keep[k] = d
    return list(keep.values())


def pano_faces(img):
    """equirectangular -> four horizontal 90-degree pinhole faces (focal = FACE / 2), yaw 0/90/180/270 from the
    panorama's centre column"""
    a = np.asarray(img)
    H, W = a.shape[:2]
    u = (np.arange(FACE) + 0.5 - FACE / 2) / (FACE / 2)
    X, Y = np.meshgrid(u, u)
    faces = []
    for k in range(4):
        yaw = k * math.pi / 2
        lon = np.arctan2(X, 1.0) + yaw
        lat = -np.arctan2(Y, np.hypot(X, 1.0))
        px = ((lon / (2 * math.pi) + 0.5) % 1.0) * W
        py = (0.5 - lat / math.pi) * H
        faces.append(Image.fromarray(a[py.astype(int).clip(0, H - 1), px.astype(int).clip(0, W - 1)]))
    return faces


def download(d, out):
    pano = d.get("camera_type") in ("spherical", "equirectangular")
    names = [f"mly_{d['id']}_f{k}.jpg" for k in range(4)] if pano else [f"mly_{d['id']}.jpg"]
    if all((out / n).exists() for n in names):
        return names
    for k in range(4):
        try:
            r = S.get(d["thumb_2048_url"], timeout=60)
            r.raise_for_status()
            break
        except Exception:
            time.sleep(2 * (k + 1))
    else:
        return []
    from io import BytesIO
    img = Image.open(BytesIO(r.content)).convert("RGB")
    if pano:
        for n, f in zip(names, pano_faces(img)):
            f.save(out / n, quality=92)
    else:
        img.save(out / names[0], quality=92)
    return names


def main():
    name, lon, lat, radius = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    out = ROOT / f"data/work/splat/{name}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    found = search(lon, lat, radius)
    found = [d for d in found if d.get("thumb_2048_url") and d.get("camera_type") != "fisheye"]
    dash = [d for d in found if DASHCAM.search(f"{d.get('make', '')} {d.get('model', '')}")]
    found = [d for d in found if d not in dash]
    print(len(found), "images,", len(dash), "dashcam frames skipped", flush=True)
    with ThreadPoolExecutor(8) as ex:
        names = list(ex.map(lambda d: download(d, out / "images"), found))
    n = 0
    with open(out / "mly.jsonl", "w") as f:
        for d, ns in zip(found, names):
            g = (d.get("computed_geometry") or d["geometry"])["coordinates"]
            for k, nm in enumerate(ns):
                f.write(json.dumps(dict(file=nm, id=d["id"], lon=g[0], lat=g[1], sequence=d.get("sequence"),
                                        compass=d.get("computed_compass_angle", d.get("compass_angle")),
                                        captured_at=d.get("captured_at"), creator=(d.get("creator") or {}).get("username"),
                                        camera=f"{d.get('make', '')} {d.get('model', '')}".strip(),
                                        face=k if len(ns) > 1 else None)) + "\n")
                n += 1
    print(n, "files")


if __name__ == "__main__":
    main()
