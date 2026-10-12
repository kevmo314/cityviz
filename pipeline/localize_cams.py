"""Solve a fixed 7-DoF pose for each NYC DOT traffic camera by aligning its (multi-frame, daylight) segmentation
with ray-cast renders of the city model. Cameras are static, so this runs once per camera."""
import json, os, re, sys, time
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
DIRS = {"north": 0, "northeast": 45, "east": 90, "southeast": 135, "south": 180, "southwest": 225, "west": 270, "northwest": 315}

_W = {}


def init_worker():
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
    import localize, segment
    city, seg = _W["city"], _W["seg"]
    out = {"id": cam["id"], "name": cam["name"]}
    frames = daylight_frames(cam["id"])
    if len(frames) < 2:
        return {**out, "status": "no-daylight-frames"}
    t0 = time.time()
    # Average class probabilities over frames so moving traffic washes out.
    probs = np.mean([seg.probs(f) for f in frames], axis=0)
    lab = segment.collapse(probs.argmax(0))
    lab[: int(lab.shape[0] * TOP_CROP) + 1] = segment.IGNORE
    frac = {k: float((lab == v).mean()) for k, v in (("sky", segment.SKY), ("ground", segment.GROUND), ("building", segment.BUILDING))}
    out["frac"] = frac
    if frac["building"] + frac["sky"] < 0.08 or frac["ground"] < 0.05:
        return {**out, "status": "unsuitable"}
    prior = ocr_heading(frames[0])
    out["ocr_heading"] = prior

    x, y = map(float, geo.to_local(cam["longitude"], cam["latitude"]))
    g = city.ground(x, y)
    L = localize.Localizer(city, lab)
    pos = [(x + dx, y + dy, g + h) for h in (5.0, 9.0) for dx in (-12, 0, 12) for dy in (-12, 0, 12)]
    cands = L.coarse(pos, np.arange(0, 360, 2.0), np.arange(-30, 6, 3.0), [32, 42, 52, 62, 75], keep=12)

    def prior_ok(p):
        return prior is None or abs(((geo.true_heading_to_grid(prior) - p[3] + 180) % 360) - 180) <= 70
    # The burned-in compass label only breaks ties between near-equal hypotheses (e.g. up vs down an avenue).
    top = cands[0][0]
    pool_ = [c for c in cands if c[0] >= top - 0.04 and prior_ok(c[1])] or cands
    best = None
    for s, p in pool_[:4]:
        b = [(p[0] - 20, p[0] + 20), (p[1] - 20, p[1] + 20), (g + 2.5, g + 16), (p[3] - 12, p[3] + 12), (-45, 15), (-6, 6), (20, 95)]
        r = L.refine(np.array(p), b, iters=350)
        if best is None or r[0] > best[0]:
            best = r
    w, h = frames[0].size
    for i, f in enumerate(frames[:1]):
        (ROOT / "data/work/cam_ref").mkdir(parents=True, exist_ok=True)
        f.save(ROOT / f"data/work/cam_ref/{cam['id']}.jpg")
    return {**out, "status": "ok", "score": best[0], "pose": [round(v, 3) for v in best[1]], "w": w, "h": h,
            "rect": [0, 0, 1, round(1 - TOP_CROP, 3)], "prior_used": prior is not None and pool_ is not cands,
            "secs": round(time.time() - t0, 1)}


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
