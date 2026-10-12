"""Pose refinement by aligning semantic segmentation (sky/ground/building) with ray-cast renders of the city model.

Stage 1: for a grid of candidate positions, ray-cast one label panorama each and exhaustively score
         (yaw, pitch, fov) by resampling the panorama into the image frame.
Stage 2: Nelder-Mead over (x, y, z, yaw, pitch, roll, fov) with direct ray-cast renders.
"""
import numpy as np
from scipy.optimize import minimize

import render
from segment import SKY, GROUND, BUILDING, IGNORE

PANO_W, PANO_H, PANO_PITCH = 1440, 640, (-80.0, 80.0)


def model_classes(lab):
    """Render labels -> segmentation classes."""
    out = np.full(lab.shape, BUILDING, np.uint8)
    out[lab == render.SKY] = SKY
    out[lab == render.GROUND] = GROUND
    return out


def score_maps(pred, target, weights=(1.0, 0.5, 1.0)):
    """Weighted mean IoU over sky/ground/building for valid target pixels. pred: (..., N), target: (N,)."""
    valid = target != IGNORE
    t = target[valid]
    p = pred[..., valid]
    tot, wsum = 0.0, 0.0
    for c, w in zip((SKY, GROUND, BUILDING), weights):
        tc = t == c
        if tc.sum() < 0.01 * len(t):
            continue
        pc = p == c
        inter = (pc & tc).sum(-1)
        union = (pc | tc).sum(-1)
        tot = tot + w * inter / np.maximum(union, 1)
        wsum += w
    return tot / max(wsum, 1e-9)


def fov_from_exif(rec, W, H):
    """Horizontal FOV (deg) of the oriented image from EXIF 35mm-equivalent focal length if present."""
    f35 = rec.get("focal35")
    if not f35 or f35 < 8 or f35 > 400:
        return None
    fpx = np.hypot(W, H) * f35 / 43.27
    return float(np.degrees(2 * np.arctan(W / 2 / fpx)))


class Localizer:
    def __init__(self, city, target, w=96):
        """target: segmentation label image (H, W) of the photo."""
        self.city = city
        H0, W0 = target.shape
        self.W, self.H = w, max(8, round(w * H0 / W0))
        ys = ((np.arange(self.H) + 0.5) * H0 / self.H).astype(int)
        xs = ((np.arange(self.W) + 0.5) * W0 / self.W).astype(int)
        self.target = target[np.ix_(ys, xs)].reshape(-1)
        self.evals = 0

    def coarse(self, positions, yaws, pitches, fovs, rolls=(0.0,), keep=8):
        results = []
        for pos in positions:
            pano, _ = render.panorama(self.city, pos, PANO_W, PANO_H, PANO_PITCH)
            pano = model_classes(pano)
            for fov in fovs:
                for roll in rolls:
                    for pitch in pitches:
                        # yaw-independent camera ray angles; then shift yaw
                        d = render.ray_dirs([0, 0, 0, 0.0, pitch, roll, fov], self.W, self.H)
                        az = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
                        el = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
                        row = np.clip(((PANO_PITCH[1] - el) / (PANO_PITCH[1] - PANO_PITCH[0]) * PANO_H).astype(int), 0, PANO_H - 1)
                        yaws_a = np.asarray(yaws)
                        col = (((az[None, :] + yaws_a[:, None]) % 360) / 360 * PANO_W).astype(int) % PANO_W
                        pred = pano[row[None, :], col]
                        s = score_maps(pred, self.target)
                        for yi in np.argsort(-s)[:3]:
                            results.append((float(s[yi]), [*map(float, pos), float(yaws_a[yi]), float(pitch), float(roll), float(fov)]))
        results.sort(key=lambda r: -r[0])
        out = []
        for s, p in results:  # non-max suppression in pose space
            if all(abs(((p[3] - q[3] + 180) % 360) - 180) > 4 or np.linalg.norm(np.subtract(p[:3], q[:3])) > 5 for _, q in out):
                out.append((s, p))
            if len(out) >= keep:
                break
        return out

    def score(self, pose):
        self.evals += 1
        lab, _, _ = render.render(self.city, pose, self.W, self.H)
        return float(score_maps(model_classes(lab).reshape(-1), self.target))

    def refine(self, pose, bounds, iters=250):
        lo, hi = np.array(bounds).T
        scale = np.array([4, 4, 2, 2, 2, 1.5, 4.0])

        def f(z):
            p = np.clip(pose + z * scale, lo, hi)
            return -self.score(p)
        r = minimize(f, np.zeros(7), method="Nelder-Mead",
                     options=dict(maxfev=iters, xatol=0.02, fatol=1e-4, initial_simplex=np.vstack([np.zeros(7), np.eye(7)])))
        best = np.clip(pose + r.x * scale, lo, hi)
        return -r.fun, best.tolist()
