"""SfM for one splat scene (pipeline/splat_collect.py) and georegistration into the city's local frame.

1. pycolmap: SIFT, matching, incremental mapping (one SIMPLE_RADIAL camera per photo). Without Mapillary images
   (splat_mapillary.py) matching is exhaustive; with them it is a pair list: Mapillary images with their neighbours
   within PAIR_RADIUS, Commons photos with each other and with a spatial subsample of Mapillary images
2. similarity transform SfM -> local frame from camera centres: photos solved against the city model (photos.json)
   are trusted most, then Mapillary positions, then Commons geotags; RANSAC, heights from the ground model
   With Mapillary images their positions are pose priors in mapping; every model is then levelled from its cameras
   and fitted to those positions in heading, scale and offset, and the well-aligned ones are all kept (select_models)
3. writes undist/<model>/ (pinhole images + COLMAP model in local-frame metres) and selection.json (models, images)

usage: splat_sfm.py <name>      (runs in ~/splat-venv on the GPU node)
"""
import json, sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).parent))
import geo

ROOT = Path(__file__).resolve().parent.parent
PAIR_RADIUS, PAIR_K, HUB_CELL = 60.0, 40, 12.0
MODEL_ERR_MAX, IMAGE_ERR_MAX, MODEL_MIN_EXTENT = 2.0, 4.0, 20.0
WALL_TILT_MAX = 3.0  # median facade-normal tilt after levelling (deg); ~2 is the noise floor
PRIOR_SIGMA = (4.0, 3.0)  # Mapillary computed position, horizontal / vertical (m)


def ground(x, y):
    """bare-earth height (NAVD88 m) from the 250 m DTM tiles (data/work/dtm/<tx>_<ty>.npz)"""
    f = ROOT / f"data/work/dtm/{int(np.floor(x / 250))}_{int(np.floor(y / 250))}.npz"
    if not f.exists():
        return np.nan
    z = np.load(f)
    i, j = int((y - z["y0"]) / z["res"]), int((x - z["x0"]) / z["res"])
    g = z["z"][max(0, i - 2):i + 3, max(0, j - 2):j + 3]
    return float(np.nanmin(g)) if g.size else np.nan


def write_priors(db_path, mly):
    """replace EXIF GPS priors with Mapillary positions in the local frame (eye level above the ground model)"""
    db = pycolmap.Database.open(db_path)
    db.clear_pose_priors()
    by_name = {im.name: im for im in db.read_all_images()}
    cov = np.diag([PRIOR_SIGMA[0] ** 2, PRIOR_SIGMA[0] ** 2, PRIOR_SIGMA[1] ** 2])
    n = 0
    for r in mly:
        im = by_name.get(r["file"])
        if im is None:
            continue
        x, y = map(float, geo.to_local(r["lon"], r["lat"]))
        z = ground(x, y) + 1.8
        if np.isnan(z):
            continue
        db.write_pose_prior(pycolmap.PosePrior(corr_data_id=im.data_id, position=np.array([x, y, z]),
                                               position_covariance=cov,
                                               coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN))
        n += 1
    db.close()
    print(n, "pose priors", flush=True)


def write_pairs(d, mly, path):
    """pair list for imported matching (see module doc)"""
    names = sorted(f.name for f in (d / "images").glob("*.jpg"))
    com = [n for n in names if not n.startswith("mly_")]
    pos = {}
    for r in mly:
        x, y = geo.to_local(r["lon"], r["lat"])
        pos[r["file"]] = (float(x), float(y))
    ml = [n for n in names if n in pos]
    P = np.array([pos[n] for n in ml])
    pairs = set()
    for a in range(len(com)):
        for b in range(a + 1, len(com)):
            pairs.add((com[a], com[b]))
    for a, n in enumerate(ml):
        dd = np.linalg.norm(P - P[a], axis=1)
        for b in np.argsort(dd)[1:PAIR_K + 1]:
            if dd[b] <= PAIR_RADIUS:
                pairs.add(tuple(sorted((n, ml[b]))))
    # one Mapillary hub per HUB_CELL grid cell (non-pano first) matched against every Commons photo
    hubs = {}
    for n in sorted(ml, key=lambda n: "_f" in n):
        hubs.setdefault((int(pos[n][0] // HUB_CELL), int(pos[n][1] // HUB_CELL)), n)
    for h in hubs.values():
        for c in com:
            pairs.add(tuple(sorted((h, c))))
    with open(path, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    print(len(names), "images,", len(ml), "Mapillary,", len(hubs), "hubs,", len(pairs), "pairs", flush=True)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / (s0 ** 2).sum(1).mean()
    return scale, R, mu_d - scale * R @ mu_s


def ransac_sim3(src, dst, w, thresh, iters=4000, seed=0):
    rng = np.random.default_rng(seed)
    best, best_in = None, None
    for _ in range(iters):
        k = rng.choice(len(src), 3, replace=False, p=w / w.sum())
        try:
            s, R, t = umeyama(src[k], dst[k])
        except np.linalg.LinAlgError:
            continue
        if not (0.01 < s < 100):
            continue
        err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        inl = err < thresh
        if best_in is None or (w[inl].sum() > w[best_in].sum()):
            best, best_in = (s, R, t), inl
    s, R, t = umeyama(src[best_in], dst[best_in])
    return s, R, t, best_in


def georegister(d, rec):
    """Commons-only scenes: similarity transform of the largest model from solved poses and geotags"""
    sparse = d / "sparse"
    print("largest model:", rec.num_reg_images(), "images,", rec.num_points3D(), "points", flush=True)

    # anchors: solved poses (accurate, w 10) and Commons geotags (w 1, eye level above the bare-earth ground)
    solved = {str(p["id"]): p["pose"][:3] for p in json.loads((ROOT / "web/data/photos.json").read_text())}
    meta = {str(json.loads(l)["pageid"]): json.loads(l) for l in open(ROOT / "data/raw/commons.jsonl")}
    src, dst, w = [], [], []
    for im in rec.images.values():
        pid = Path(im.name).stem
        c = im.projection_center()
        if pid in solved:
            src.append(c); dst.append(solved[pid]); w.append(10.0)
            continue
        if pid in meta:
            x, y = map(float, geo.to_local(float(meta[pid]["lon"]), float(meta[pid]["lat"])))
            src.append(c); dst.append([x, y, ground(x, y) + 1.8]); w.append(1.0)
    src, dst, w = np.array(src), np.array(dst), np.array(w)
    ok = ~np.isnan(dst[:, 2])
    src, dst, w = src[ok], dst[ok], w[ok]
    print(len(src), "anchors,", int((w > 1).sum()), "solved", flush=True)
    s, R, t, inl = ransac_sim3(src, dst, w, thresh=12.0)
    k = inl & (w > 1)
    if k.sum() >= 3:  # refine on the trusted inliers
        s, R, t = umeyama(src[k], dst[k])
    err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    med = lambda m: f"{np.median(err[m]):.1f}" if m.any() else "-"
    print(f"sim3 scale {s:.3f}; inliers {inl.sum()}/{len(src)}; err median solved {med(w > 1)} m, "
          f"geotag {med(w == 1)} m", flush=True)
    # gravity check: the SfM 'up' should map to +z
    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
    out = sparse / "0"
    out.mkdir(parents=True, exist_ok=True)
    rec.write(out)
    json.dump(dict(scale=s, R=R.tolist(), t=t.tolist(), n_images=rec.num_reg_images()), open(d / "georef.json", "w"))
    # pinhole images + model for the splat trainer
    (d / "undist").mkdir(exist_ok=True)
    pycolmap.undistort_images(d / "undist/0", out, d / "images")
    json.dump(dict(models=["0"], images=sorted(im.name for im in rec.images.values())), open(d / "selection.json", "w"))


def gravity(rec):
    """world 'up' of a model from its cameras: image x axes are horizontal whatever the pitch (people level the horizon,
    panorama faces are cut level), so up is the direction most perpendicular to all of them; falls back to the mean
    image-up vector when the cameras all face one way and the x axes don't span the horizontal plane"""
    R = np.array([im.cam_from_world().rotation.matrix() for im in rec.images.values()])
    right, up = R[:, 0], -R[:, 1]
    w, V = np.linalg.eigh(right.T @ right)
    g = V[:, 0] if w[1] > 0.1 * len(R) else up.mean(0)
    g /= np.linalg.norm(g)
    return g if g @ up.mean(0) > 0 else -g


def wall_normals(rec):
    """unit normals of planar patches of the sparse points (16-NN PCA on well-tracked points)"""
    P = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 4]).reshape(-1, 3)
    if len(P) < 200:
        return np.zeros((0, 3))
    from scipy.spatial import cKDTree
    _, idx = cKDTree(P).query(P, k=16)
    Q = P[idx] - P[idx].mean(1, keepdims=True)
    w, V = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))
    return V[w[:, 0] < 0.05 * w[:, 1], :, 0]


def level(rec):
    """world 'up' of a model: start from the cameras (gravity()), then refine so that facade normals are horizontal.
    Walls facing two directions (the street grid) fix both tilt axes; walls facing one way fix only the tilt about
    the horizontal axis along them, and the camera estimate keeps the other."""
    g = gravity(rec)
    n = wall_normals(rec)
    for _ in range(3):
        walls = n[np.abs(n @ g) < np.sin(np.radians(25))]
        if len(walls) < 100:
            break
        w, V = np.linalg.eigh(walls.T @ walls)
        if w[1] > 0.15 * w[2]:
            g_new = V[:, 0]
        else:  # one wall direction: remove the component of g along it
            g_new = g - (g @ V[:, 2]) * V[:, 2]
        g_new /= np.linalg.norm(g_new)
        g = g_new if g_new @ g > 0 else -g_new
    return g


def rot_to_z(g):
    """smallest rotation taking unit vector g to +z"""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(g, z), float(g @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def level_and_fit(rec, anchors):
    """Sim3 for one model: level it (level()), then a 2D similarity (scale, heading, xy) to the Mapillary
    positions by RANSAC, and the height offset that puts cameras at eye level. Position priors alone leave the tilt of
    street-level models free (all cameras at nearly one height), which tilted whole models by up to 40 degrees."""
    R1 = rot_to_z(level(rec))
    names = [n for n in anchors if n in {im.name for im in rec.images.values()}]
    cen = {im.name: im.projection_center() for im in rec.images.values()}
    a = np.array([R1 @ cen[n] for n in names])
    b = np.array([anchors[n] for n in names])
    best = None
    rng = np.random.default_rng(0)
    for _ in range(2000):
        k = rng.choice(len(a), 2, replace=False)
        da, db = a[k[1], :2] - a[k[0], :2], b[k[1], :2] - b[k[0], :2]
        if np.linalg.norm(da) < 1e-3 or np.linalg.norm(db) < 2.0:
            continue
        sc = np.linalg.norm(db) / np.linalg.norm(da)
        th = np.arctan2(db[1], db[0]) - np.arctan2(da[1], da[0])
        Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        t = b[k[0], :2] - sc * Rz @ a[k[0], :2]
        inl = np.linalg.norm((sc * (Rz @ a[:, :2].T)).T + t - b[:, :2], axis=1) < IMAGE_ERR_MAX
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        return None
    # least squares 2D similarity on the inliers
    A, B = a[best, :2], b[best, :2]
    ma, mb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    Rz = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    sc = S.sum() / ((A - ma) ** 2).sum()
    R = np.eye(3)
    R[:2, :2] = Rz
    R = R @ R1
    t = np.zeros(3)
    t[:2] = mb - sc * Rz @ ma
    t[2] = np.median(b[best, 2] - sc * a[best, 2])
    return sc, R, t


def select_models(d, mly):
    """Level and georegister every model (level_and_fit), keep those whose Mapillary images then sit near their
    recorded positions (median <= MODEL_ERR_MAX) and that span at least MODEL_MIN_EXTENT (smaller ones have no
    usable scale) and whose facades stand within WALL_TILT_MAX of vertical, drop single images further off than IMAGE_ERR_MAX (Commons photos carry no position and are
    trusted with their model), and undistort each kept model into undist/<k>."""
    anchors = {}
    for r in mly:
        x, y = map(float, geo.to_local(r["lon"], r["lat"]))
        anchors[r["file"]] = np.array([x, y, ground(x, y) + 1.8])
    anchors = {k: v for k, v in anchors.items() if not np.isnan(v[2])}
    sel, keep = [], []
    for p in sorted((d / "sparse/raw").iterdir(), key=lambda p: int(p.name)):
        rec = pycolmap.Reconstruction(p)
        named = [im.name for im in rec.images.values() if im.name in anchors]
        span = np.ptp(np.array([anchors[n][:2] for n in named]), 0).max() if len(named) >= 3 else 0.0
        fit = level_and_fit(rec, anchors) if span >= MODEL_MIN_EXTENT else None
        if fit is None:
            print(f"model {p.name}: {rec.num_reg_images()} images, GPS span {span:.0f} m, drop", flush=True)
            continue
        sc, R, t = fit
        rec.transform(pycolmap.Sim3d(sc, pycolmap.Rotation3d(R), t))
        err = {im.name: float(np.linalg.norm(im.projection_center()[:2] - anchors[im.name][:2]))
               for im in rec.images.values() if im.name in anchors}
        med = np.median(list(err.values()))
        nz = np.abs(wall_normals(rec)[:, 2])
        tilt = np.degrees(np.arcsin(np.median(nz[nz < 0.5]))) if (nz < 0.5).sum() >= 100 else np.nan
        ok = [im.name for im in rec.images.values() if err.get(im.name, 0.0) <= IMAGE_ERR_MAX]
        good = med <= MODEL_ERR_MAX and not tilt > WALL_TILT_MAX
        print(f"model {p.name}: {rec.num_reg_images()} images, GPS span {span:.0f} m, scale {sc:.3f}, "
              f"err median {med:.1f} m, wall tilt {tilt:.1f} deg, {'keep ' + str(len(ok)) if good else 'drop'}", flush=True)
        if not good:
            continue
        out = d / f"sparse/aligned/{p.name}"
        out.mkdir(parents=True, exist_ok=True)
        rec.write(out)
        sel.append(p.name)
        keep += ok
        (d / "undist").mkdir(exist_ok=True)
        pycolmap.undistort_images(d / f"undist/{p.name}", out, d / "images")
    json.dump(dict(models=sel, images=sorted(keep)), open(d / "selection.json", "w"))
    print(len(sel), "models,", len(keep), "images selected", flush=True)

def main():
    name = sys.argv[1]
    d = ROOT / f"data/work/splat/{name}"
    db, sparse = d / "database.db", d / "sparse"
    mly = [json.loads(l) for l in open(d / "mly.jsonl")] if (d / "mly.jsonl").exists() else []
    if not (d / "matched.ok").exists():
        db.unlink(missing_ok=True)
        opts = pycolmap.FeatureExtractionOptions()
        opts.sift.max_num_features = 8192
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        pycolmap.extract_features(db, d / "images", camera_mode=pycolmap.CameraMode.PER_IMAGE,
                                  reader_options=reader, extraction_options=opts)
        if mly:
            write_pairs(d, mly, d / "pairs.txt")
            po = pycolmap.ImportedPairingOptions()
            po.match_list_path = d / "pairs.txt"
            pycolmap.match_image_pairs(db, pairing_options=po)
        else:
            pycolmap.match_exhaustive(db)
        (d / "matched.ok").touch()
    if not (sparse / "raw" / "0").exists():
        mapper = pycolmap.IncrementalPipelineOptions()
        if mly:
            # only Commons photos and the Mapillary images mly.jsonl still lists (the database may hold dropped ones)
            keep = {r["file"] for r in mly}
            mapper.image_names = sorted(f.name for f in (d / "images").glob("*.jpg")
                                        if f.name in keep or not f.name.startswith("mly_"))
            # Mapillary positions as priors keep look-alike Manhattan blocks from being merged into one street
            write_priors(db, mly)
            mapper.use_prior_position = True
            mapper.use_robust_loss_on_prior_position = True
        (sparse / "raw").mkdir(parents=True, exist_ok=True)
        recs = pycolmap.incremental_mapping(db, d / "images", sparse / "raw", options=mapper)
        print({k: r.num_reg_images() for k, r in recs.items()}, flush=True)
    if mly:
        select_models(d, mly)
    else:
        georegister(d, max((pycolmap.Reconstruction(p) for p in (sparse / "raw").iterdir() if p.is_dir()),
                           key=lambda r: r.num_reg_images()))


if __name__ == "__main__":
    main()
