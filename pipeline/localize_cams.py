"""Solve a fixed 7-DoF pose for each NYC DOT traffic camera by aligning its (multi-frame, daylight) segmentation
with ray-cast renders of the city model. Cameras are static, so this runs once per camera."""
import json, math, os, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import geo

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data/raw/cam_frames"
OUT = ROOT / "data/work/cam_poses.jsonl"
TOP_CROP = 0.085  # timestamp banner
# NYC DOT cams sit at wide zoom (~60 deg horizontal). Focal length is barely observable from the solver's cost (a
# narrow lens further back fits the far street canyon almost as well), so without this prior they came out 15-30 deg.
CAM_F_PRIOR = 0.5 / math.tan(math.radians(60) / 2)
DIRS = {"north": 0, "northeast": 45, "east": 90, "southeast": 135, "south": 180, "southwest": 225, "west": 270, "northwest": 315}

_W = {}


def init_worker():
    from localize_photos import pin_worker
    pin_worker()
    import torch
    torch.set_num_threads(int(os.environ.get("SEG_THREADS", "3")))
    import render, segment
    from rapidocr_onnxruntime import RapidOCR
    _W.update(city=render.City(), seg=segment.Segmenter(), ocr=RapidOCR())


def daylight_frames(cam_id, n=6):
    fs = sorted((FRAMES / cam_id).glob("*.jpg"))
    scored = []
    for f in fs:
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        a = np.asarray(im, np.float32)
        h = a.shape[0]
        body = a[int(h * TOP_CROP):]
        lum = body.mean()
        # skip night frames and frozen/blank frames
        if lum > 70 and body.std() > 25:
            scored.append((lum, f, im))
    # the feed's resolution occasionally changes; keep frames at the most common size
    sizes = [s[2].size for s in scored]
    size = max(set(sizes), key=sizes.count) if sizes else None
    scored = [s for s in scored if s[2].size == size]
    scored.sort(key=lambda s: -s[0])
    return [s[2] for s in scored[:n]]


def ocr_heading(im):
    try:
        res, _ = _W["ocr"](np.asarray(im.crop((0, 0, im.width, int(im.height * 0.12)))))
    except Exception:
        return None
    text = " ".join(r[1] for r in res or []).lower().replace(" ", "")
    m = re.search(r"facing(north|south)?(east|west)?", text)
    if not m or not (m.group(1) or m.group(2)):
        return None
    return DIRS[(m.group(1) or "") + (m.group(2) or "")]


def solve(cam):
    import segment, solver
    city, seg = _W["city"], _W["seg"]
    out = {"id": cam["id"], "name": cam["name"]}
    frames = daylight_frames(cam["id"])
    if len(frames) < 2:
        return {**out, "status": "no-daylight-frames"}
    t0 = time.time()
    # Per-pixel median over frames gives a traffic-free background plate; class probabilities are averaged too.
    plate = Image.fromarray(np.median(np.stack([np.asarray(f) for f in frames]), axis=0).astype(np.uint8))
    probs = np.mean([seg.probs(f) for f in frames + [plate]], axis=0)
    lab = segment.collapse(probs.argmax(0))
    lab[: int(lab.shape[0] * TOP_CROP) + 1] = segment.IGNORE
    frac = {k: float((lab == v).mean()) for k, v in (("sky", segment.SKY), ("ground", segment.GROUND), ("building", segment.BUILDING))}
    out["frac"] = frac
    if frac["building"] + frac["sky"] < 0.08 or frac["ground"] < 0.05:
        return {**out, "status": "unsuitable"}
    ocr = ocr_heading(frames[0])
    out["ocr_heading"] = ocr

    x, y = map(float, geo.to_local(cam["longitude"], cam["latitude"]))
    g = city.ground(x, y)
    # Mask the timestamp banner out of the line detector by blanking it.
    a = np.asarray(plate).copy()
    a[: int(a.shape[0] * TOP_CROP) + 1] = a[int(a.shape[0] * TOP_CROP) + 2]
    prior = solver.Prior(x=x, y=y, ground=g, pos_sigma=12.0,
                         heading=geo.true_heading_to_grid(ocr) if ocr is not None else None, heading_sigma=60.0,
                         heights=[5.0, 9.0], z_range=(2.5, 16.0), pitch_range=(-50.0, 10.0),
                         f=CAM_F_PRIOR, f_sigma=0.25)
    obs = solver.Observation(Image.fromarray(a), lab)
    res = solver.Solver(city, obs, prior).solve()
    if res is None:
        return {**out, "status": "no-candidates"}
    theta, J, parts = res
    (ROOT / "data/work/cam_ref").mkdir(parents=True, exist_ok=True)
    plate.save(ROOT / f"data/work/cam_ref/{cam['id']}.jpg")
    w, h = plate.size
    return {**out, "status": "ok", "theta": [round(float(v), 5) for v in theta], "J": float(J), **parts,
            "w": w, "h": h, "rect": [0, 0, 1, round(1 - TOP_CROP, 3)], "secs": round(time.time() - t0, 1)}


def main(workers=int(os.environ.get("WORKERS", "4"))):
    cams = json.loads((ROOT / "data/raw/traffic_cams.json").read_text())
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in open(OUT) if json.loads(l).get("status") == "ok"}
    cams = [c for c in cams if c["id"] not in done]
    print(len(cams), "cameras", file=sys.stderr, flush=True)
    with Pool(workers, initializer=init_worker) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(solve, cams)):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(i, r.get("name"), r.get("status"), r.get("score"), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
