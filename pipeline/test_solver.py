import sys, json, time; sys.path.insert(0, "pipeline")
import numpy as np, render, geo, segment, solver, camera as C
from PIL import Image, ImageOps
import cv2
pid, out = int(sys.argv[1]), sys.argv[2]
rec = [json.loads(l) for l in open("data/raw/commons.jsonl") if f'"pageid": {pid},' in l][0]
img = Image.open(f"data/work/photos/{pid}.jpg").convert("RGB")
seg = segment.Segmenter().labels(img)
city = render.City()
x, y = map(float, geo.to_local(rec["lon"], rec["lat"]))
g = city.ground(x, y)
W, H = img.size
f35 = rec.get("focal35")
fexif = (np.hypot(W, H) * f35 / 43.27) / W if f35 and 8 < f35 < 400 else None
pr = solver.Prior(x=x, y=y, ground=g, heading=geo.true_heading_to_grid(rec["heading"]) if rec.get("heading") is not None else None, f=fexif)
obs = solver.Observation(img, seg)
print("segments", len(obs.segs), "vertical", None if obs.cl["vertical"] is None else obs.cl["vertical"][1].sum(), "hclusters", len(obs.cl["horizontal"]))
S = solver.Solver(city, obs, pr)
t = time.time()
rot = S.init_rotation(); print("rot hyps", rot, time.time() - t)
best, J, parts = S.solve()
print("time", time.time() - t, "evals", S.evals)
print("theta", np.round(best, 4).tolist(), "hfov", C.f_to_hfov(best[6]), parts)
# overlay
im = img.copy(); im.thumbnail((900, 900)); w, h = im.size
lab, dep, prim = S.render(best, w) if False else (None, None, None)
d = C.ray_dirs(best, w, h)
o = np.broadcast_to(np.asarray(best[:3], np.float32), d.shape)
pr_, t_, hit = city.cast(o, d)
lab = np.zeros(len(d), np.uint8); lab[hit] = city.label[pr_[hit]]
_, e = solver.model_edges(lab.reshape(h, w), t_.reshape(h, w), np.where(hit, pr_, -1).reshape(h, w), city.normal)
a = np.asarray(im).copy(); a[e] = [255, 40, 40]
Image.fromarray(a).save(out)
