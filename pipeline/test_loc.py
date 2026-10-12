import sys, time, json; sys.path.insert(0, "pipeline")
import numpy as np, render, geo, segment, localize
from PIL import Image, ImageOps
S = sys.argv[1]
rec = json.loads(sys.argv[2])
img = ImageOps.exif_transpose(Image.open(rec["path"]))
seg = segment.Segmenter().labels(img)
city = render.City()
x, y = map(float, geo.to_local(rec["lon"], rec["lat"]))
g = city.ground(x, y)
L = localize.Localizer(city, seg)
fov0 = localize.fov_from_exif(rec, *img.size)
fovs = [fov0 * k for k in (0.92, 1.0, 1.08)] if fov0 else [35, 45, 55, 65, 75]
h0 = geo.true_heading_to_grid(rec["heading"]) if rec.get("heading") is not None else None
yaws = np.arange(h0 - 30, h0 + 30.1, 1.0) if h0 is not None else np.arange(0, 360, 1.0)
pos = [(x + dx, y + dy, g + 1.6) for dx in (-15, 0, 15) for dy in (-15, 0, 15)]
t = time.time()
cands = L.coarse(pos, yaws, np.arange(-15, 31, 2.5), fovs)
print("coarse", time.time() - t, cands[:3])
best = None
for s, p in cands[:4]:
    b = [(p[0]-25, p[0]+25), (p[1]-25, p[1]+25), (g + 0.5, g + 6), (p[3]-15, p[3]+15), (-30, 50), (-8, 8), (fovs[0]*0.85, fovs[-1]*1.15)]
    r = L.refine(np.array(p), b)
    if best is None or r[0] > best[0]: best = r
print("refined", time.time() - t, best, "evals", L.evals)
pal = np.zeros((256, 3), np.uint8); pal[0] = [150, 190, 240]; pal[1] = [90, 90, 90]; pal[2] = [200, 170, 140]; pal[255] = [255, 0, 255]
H, W = seg.shape
lab, _, _ = render.render(city, best[1], W, H)
m = localize.model_classes(lab)
im = np.asarray(img.convert("RGB").resize((W, H))).astype(float)
_, dep, prim = render.render(city, best[1], W, H)
nrm = city.normal[np.maximum(prim, 0)]
e = np.zeros((H, W), bool)
e[:, 1:] |= (m[:, 1:] != m[:, :-1]) | (np.abs(np.log(dep[:, 1:] + 1) - np.log(dep[:, :-1] + 1)) > 0.15) | ((nrm[:, 1:] * nrm[:, :-1]).sum(-1) < 0.8)
e[1:, :] |= (m[1:, :] != m[:-1, :]) | (np.abs(np.log(dep[1:] + 1) - np.log(dep[:-1] + 1)) > 0.15) | ((nrm[1:] * nrm[:-1]).sum(-1) < 0.8)
over = im.copy(); over[e] = [255, 40, 40]; over = over.astype(np.uint8)
Image.fromarray(np.concatenate([pal[seg], over], 1)).save(S)
