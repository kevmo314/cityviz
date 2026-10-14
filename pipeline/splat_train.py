"""Train a Gaussian splat for one scene from in-the-wild photos (gsplat), in the city's local frame.

Inputs (data/work/splat/<name>/): selection.json + undist/<model>/ (pinhole images + COLMAP models in local-frame
metres, see splat_sfm.py), masks/ (splat_masks.py).
In-the-wild handling: per-photo 3x4 colour transform (exposure / white balance / season), transient masks, sky kept
transparent. Output: web/splats/<name>.splat (32 B/Gaussian: xyz f32, scale f32x3, rgba u8, quat u8) + .json bounds.

usage: splat_train.py <name> [steps]      (runs in ~/splat-venv on the GPU node)
"""
import json, math, sys, time
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

ROOT = Path(__file__).resolve().parent.parent
MAX_SIDE = 1100
C0 = 0.28209479177387814


def load(name):
    """views and initial points from every model in selection.json (all already in the local frame)"""
    d = ROOT / f"data/work/splat/{name}"
    sel = json.loads((d / "selection.json").read_text())
    keep = set(sel["images"])
    views, pts, rgb = [], [], []
    for k in sel["models"]:
        v, p, c = load_model(d, d / f"undist/{k}", keep)
        views += v; pts.append(p); rgb.append(c)
    return views, np.concatenate(pts), np.concatenate(rgb)


def load_model(d, u, keep):
    rec = pycolmap.Reconstruction(u / "sparse")
    views, ids = [], set()
    for im in rec.images.values():
        if im.name not in keep:
            continue
        ids.add(im.image_id)
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4] if len(cam.params) >= 4 else (cam.params[0], cam.params[0], *cam.params[1:3])
        img = Image.open(u / "images" / im.name).convert("RGB")
        s = min(1.0, MAX_SIDE / max(img.size))
        W, H = round(img.width * s), round(img.height * s)
        img = img.resize((W, H), Image.LANCZOS)
        mp = d / "masks" / (Path(im.name).stem + ".png")
        mask = np.array(Image.open(mp).resize((W, H), Image.NEAREST)) if mp.exists() else np.ones((H, W), np.uint8)
        K = np.array([[fx * W / cam.width, 0, cx * W / cam.width], [0, fy * H / cam.height, cy * H / cam.height], [0, 0, 1]])
        vm = np.eye(4)
        vm[:3] = im.cam_from_world().matrix()
        views.append(dict(name=im.name, img=torch.from_numpy(np.asarray(img)), mask=torch.from_numpy(mask), K=K, vm=vm,
                          W=W, H=H, center=im.projection_center()))
    # points seen by at least 3 kept images (the dropped, misplaced ones leave floaters otherwise)
    good = [p for p in rec.points3D.values() if sum(e.image_id in ids for e in p.track.elements) >= 3]
    return views, np.array([p.xyz for p in good]).reshape(-1, 3), np.array([p.color for p in good]).reshape(-1, 3) / 255.0


def ssim_map(x, y):
    # x, y: [1, 3, H, W]
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    f = lambda t: F.conv2d(t, w, padding=5, groups=3)
    mx, my = f(x), f(y)
    sx, sy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))


def main():
    name = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    dev = "cuda"
    views, pts, rgb = load(name)
    centers = np.array([v["center"] for v in views])
    scene_center = np.median(centers, 0)
    scene_scale = float(np.percentile(np.linalg.norm(centers - scene_center, axis=1), 90)) * 1.1
    print(len(views), "views,", len(pts), "points, scene scale", round(scene_scale, 1), flush=True)

    # init from SfM points
    p = torch.tensor(pts, dtype=torch.float32, device=dev)
    d2 = torch.cdist(p, p).topk(4, largest=False).values[:, 1:] if len(p) < 60000 else None
    dist = (d2 ** 2).mean(1).sqrt().clamp(min=0.05) if d2 is not None else torch.full((len(p),), 0.3, device=dev)
    N = len(p)
    sh_deg = 3
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(p),
        "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(N, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(rgb, dtype=torch.float32, device=dev) - 0.5) / C0)[:, None]),
        "shN": torch.nn.Parameter(torch.zeros(N, (sh_deg + 1) ** 2 - 1, 3, device=dev)),
    })
    lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    # per-photo colour transform, initialised to identity
    app = torch.nn.Parameter(torch.eye(3, 4, device=dev).repeat(len(views), 1, 1))
    app_opt = torch.optim.Adam([app], lr=1e-3)
    strategy = DefaultStrategy(refine_stop_iter=int(steps * 0.5), reset_every=3000, absgrad=True, grow_grad2d=0.0008)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)
    decay = 0.01 ** (1 / steps)

    t0 = time.time()
    rng = np.random.default_rng(0)
    for step in range(steps):
        idx = int(rng.integers(len(views)))
        v = views[idx]
        gt = v["img"].to(dev).float() / 255.0
        m = v["mask"].to(dev)
        vm = torch.tensor(v["vm"], dtype=torch.float32, device=dev)[None]
        K = torch.tensor(v["K"], dtype=torch.float32, device=dev)[None]
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        out, alpha, info = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                         torch.sigmoid(params["opacities"]), colors, vm, K, v["W"], v["H"],
                                         sh_degree=min(step // 1000, sh_deg), absgrad=True, packed=False)
        strategy.step_pre_backward(params, opts, state, step, info)
        A = app[idx]
        pred = out[0] @ A[:, :3].T + A[:, 3]
        static = (m == 1).float()[..., None]
        sky = (m == 2).float()[..., None]
        l1 = ((pred - gt).abs() * static).sum() / (static.sum() * 3 + 1)
        ss = ssim_map(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
        lss = ((1 - ss) * static).sum() / (static.sum() * 3 + 1)
        lsky = (alpha[0] * sky).sum() / (sky.sum() + 1)
        lopaque = ((1 - alpha[0]) * static).sum() / (static.sum() + 1)
        lapp = ((app - torch.eye(3, 4, device=dev)) ** 2).mean()
        loss = 0.8 * l1 + 0.2 * lss + 0.1 * lsky + 0.01 * lopaque + 1.0 * lapp
        loss.backward()
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        app_opt.step(); app_opt.zero_grad(set_to_none=True)
        for g in opts["means"].param_groups:
            g["lr"] *= decay
        if step % 1000 == 0:
            psnr = -10 * math.log10(float((((pred - gt) ** 2) * static).sum() / (static.sum() * 3 + 1)) + 1e-10)
            print(f"step {step} loss {loss.item():.4f} psnr {psnr:.2f} gaussians {len(params['means'])} "
                  f"{time.time() - t0:.0f}s", flush=True)

    export(name, params, scene_center, scene_scale)
    torch.save({k: v.detach().cpu() for k, v in params.items()}, ROOT / f"data/work/splat/{name}/gaussians.pt")


def export(name, params, center, radius):
    with torch.no_grad():
        xyz = params["means"].cpu().numpy()
        sc = torch.exp(params["scales"]).cpu().numpy()
        op = torch.sigmoid(params["opacities"]).cpu().numpy()
        col = (params["sh0"][:, 0].cpu().numpy() * C0 + 0.5).clip(0, 1)
        q = F.normalize(params["quats"], dim=-1).cpu().numpy()
    # keep the scene itself: drop faint, huge and far-away Gaussians (background sky/skyline blobs)
    keep = (op > 0.03) & (sc.max(1) < 8.0) & (np.linalg.norm(xyz[:, :2] - center[:2], axis=1) < max(250.0, radius * 2))
    keep &= (xyz[:, 2] < center[2] + 400) & (xyz[:, 2] > center[2] - 30)
    xyz, sc, op, col, q = xyz[keep], sc[keep], op[keep], col[keep], q[keep]
    order = np.argsort(-(op * sc.prod(1)))  # important first (progressive loading)
    xyz, sc, op, col, q = xyz[order], sc[order], op[order], col[order], q[order]
    rec = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"], rec["s"] = xyz, sc
    rec["c"] = np.concatenate([col, op[:, None]], 1).__mul__(255).round().clip(0, 255)
    rec["r"] = (q * 128 + 128).round().clip(0, 255)  # wxyz
    out = ROOT / "web/splats"
    out.mkdir(parents=True, exist_ok=True)
    rec.tofile(out / f"{name}.splat")
    json.dump({"name": name, "file": f"splats/{name}.splat", "count": int(len(rec)), "center": [float(c) for c in center],
               "bmin": xyz.min(0).round(1).tolist(), "bmax": xyz.max(0).round(1).tolist()},
              open(out / f"{name}.json", "w"))
    print("exported", len(rec), "gaussians")


if __name__ == "__main__":
    main()
