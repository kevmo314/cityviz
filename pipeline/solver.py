"""Full camera solve (pose + intrinsics + distortion) of an image against the city model.

Stages
  1. rotation/intrinsics init from vanishing points (vertical VP -> pitch/roll; horizontal VPs on the horizon -> f)
  2. coarse search: label panoramas on a position grid x yaw x focal hypotheses, scored by class IoU
  3. joint refinement of all 10 camera parameters, minimizing
        - class IoU between segmentation and render
        - truncated chamfer distance from rendered model edges to image edges
        - vanishing point consistency (vertical lines -> world up, horizontal clusters -> horizon)
        - weak priors (GPS, compass, EXIF focal)
     first at 128 px, then at 256 px
  4. verification metrics (edge residual in pixels, edge support, IoU)
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import minimize

import camera as C
import lines as LN
import render
from segment import SKY, GROUND, BUILDING, IGNORE

PANO_W, PANO_H, PANO_PITCH = 1440, 640, (-80.0, 80.0)


def model_classes(lab):
    out = np.full(lab.shape, BUILDING, np.uint8)
    out[lab == render.SKY] = SKY
    out[lab == render.GROUND] = GROUND
    return out


def class_iou(pred, target, weights=(1.0, 0.5, 1.0)):
    valid = target != IGNORE
    t, p = target[valid], pred[..., valid]
    tot, wsum = 0.0, 0.0
    for c, w in zip((SKY, GROUND, BUILDING), weights):
        tc = t == c
        if tc.sum() < 0.01 * len(t):
            continue
        pc = p == c
        tot = tot + w * (pc & tc).sum(-1) / np.maximum((pc | tc).sum(-1), 1)
        wsum += w
    return tot / max(wsum, 1e-9)


@dataclass
class Prior:
    x: float
    y: float
    ground: float
    heading: float | None = None      # grid-frame degrees
    f: float | None = None            # EXIF focal / width
    f_sigma: float = 0.25             # prior sigma of log f
    pos_sigma: float = 25.0
    heading_sigma: float = 45.0
    heights: list = field(default_factory=lambda: [1.6])   # meters above ground for coarse search
    z_range: tuple = (0.5, 6.0)       # meters above ground (refinement bounds)
    pitch_range: tuple = (-35.0, 60.0)


class Observation:
    """Everything derived from the image that the solver compares against."""

    def __init__(self, img, seg_labels):
        self.W, self.H = img.size
        self.aspect = self.H / self.W
        self.seg = seg_labels
        self.segs = LN.detect(img)
        self.cl = LN.clusters(self.segs)
        L = np.hypot(self.segs[:, 2] - self.segs[:, 0], self.segs[:, 3] - self.segs[:, 1])
        self.seg_len = L
        self._cache = {}

    def at(self, w):
        if w in self._cache:
            return self._cache[w]
        h = max(8, round(w * self.aspect))
        H0, W0 = self.seg.shape
        ys = ((np.arange(h) + 0.5) * H0 / h).astype(int)
        xs = ((np.arange(w) + 0.5) * W0 / w).astype(int)
        target = self.seg[np.ix_(ys, xs)]
        # image edges: long line segments + sky/ground boundaries of the segmentation
        e = LN.rasterize(self.segs, w, h) > 0
        t = target
        b = np.zeros_like(e)
        for a, c in ((t[:, 1:], t[:, :-1]), (t[1:], t[:-1])):
            diff = (a != c) & (a != IGNORE) & (c != IGNORE) & ((a == SKY) | (c == SKY) | (a == GROUND) | (c == GROUND))
            if a.shape[1] != t.shape[1]:
                b[:, 1:] |= diff
            else:
                b[1:] |= diff
        edges = e | b
        dt = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
        self._cache[w] = (h, target.reshape(-1), dt, edges)
        return self._cache[w]


def model_edges(lab, dep, prim, normals):
    m = model_classes(lab)
    nrm = normals[np.maximum(prim, 0)]
    ld = np.log(np.nan_to_num(dep, posinf=1e5) + 1)
    e = np.zeros(lab.shape, bool)
    for sl_a, sl_b, tgt in (((slice(None), slice(1, None)), (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                            ((slice(1, None), slice(None)), (slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        a_m, b_m = m[sl_a], m[sl_b]
        both_ground = (a_m == GROUND) & (b_m == GROUND)
        d = (a_m != b_m) | ((np.abs(ld[sl_a] - ld[sl_b]) > 0.12) & ~both_ground) | \
            (((nrm[sl_a] * nrm[sl_b]).sum(-1) < 0.85) & (a_m != SKY) & ~both_ground)
        e[tgt] |= d
    return m, e


class Solver:
    def __init__(self, city, obs: Observation, prior: Prior):
        self.city, self.obs, self.prior = city, obs, prior
        self.evals = 0

    # ---------------------------------------------------------------- rendering
    def render(self, theta, w):
        h = self.obs.at(w)[0]
        d = C.ray_dirs(theta, w, h)
        o = np.broadcast_to(np.asarray(theta[:3], np.float32), d.shape)
        prim, t, hit = self.city.cast(o, d)
        lab = np.zeros(len(d), np.uint8)
        lab[hit] = self.city.label[prim[hit]]
        return lab.reshape(h, w), t.reshape(h, w), np.where(hit, prim, -1).reshape(h, w)

    # ---------------------------------------------------------------- line terms
    def vp_errors(self, theta):
        """(vertical mean err deg, horizon err deg) for the current camera; None when unavailable."""
        f = theta[6]
        out = [None, None]
        cl = self.obs.cl
        if cl["vertical"] is not None:
            segs = self._undistort_segs(cl["vertical"][1], theta)
            d = C.world_to_cam(theta, np.array([[0, 0, 1.0]]))[0]
            vp = np.array([f * d[0], f * d[1], d[2]])
            err = np.minimum(LN.vp_angle_error(segs, vp), 5.0)
            L = self.obs.seg_len[cl["vertical"][1]]
            out[0] = float((err * L).sum() / L.sum())
        if cl["horizontal"]:
            fw, r, u = C.basis(theta[3], theta[4], theta[5])
            errs = []
            for _, mask in cl["horizontal"]:
                segs = self._undistort_segs(mask, theta)
                v = LN.fit_vp(segs, self.obs.seg_len[mask])
                ray = v[0] * r - v[1] * u + f * v[2] * fw
                errs.append(abs(np.degrees(np.arcsin(abs(ray[2]) / (np.linalg.norm(ray) + 1e-12)))))
            out[1] = float(np.mean(np.minimum(errs, 10.0)))
        return out

    def _undistort_segs(self, mask, theta):
        s = self.obs.segs[mask]
        x1, y1 = C.undistort_norm(s[:, 0], s[:, 1], theta)
        x2, y2 = C.undistort_norm(s[:, 2], s[:, 3], theta)
        return np.column_stack([x1, y1, x2, y2])

    # ---------------------------------------------------------------- objective
    def prior_cost(self, theta):
        p = self.prior
        c = (np.hypot(theta[0] - p.x, theta[1] - p.y) / p.pos_sigma) ** 2
        if p.heading is not None:
            c += (((theta[3] - p.heading + 180) % 360 - 180) / p.heading_sigma) ** 2
        if p.f is not None:
            c += (np.log(theta[6] / p.f) / p.f_sigma) ** 2
        c += (theta[7] / 0.06) ** 2 + (theta[8] / 0.06) ** 2 + (theta[9] / 0.3) ** 2 + (theta[5] / 10) ** 2
        return 0.002 * c

    def cost(self, theta, w, parts=False):
        self.evals += 1
        h, target, dt, _ = self.obs.at(w)
        lab, dep, prim = self.render(theta, w)
        m, e = model_edges(lab, dep, prim, self.city.normal)
        iou = float(class_iou(m.reshape(-1), target))
        tau = 0.04 * w
        ch = float(np.minimum(dt[e], tau).mean() / tau) if e.any() else 1.0
        ve, he = self.vp_errors(theta)
        J = -iou + 0.35 * ch + 0.03 * (ve or 0) + 0.03 * (he or 0) + self.prior_cost(theta)
        if parts:
            return J, dict(iou=iou, chamfer=ch, vert_deg=ve, horizon_deg=he)
        return J

    # ---------------------------------------------------------------- stage 1: VP init
    def init_rotation(self):
        """Hypotheses (f, pitch, roll) from vanishing points; empty if the image has no usable verticals."""
        if self.obs.cl["vertical"] is None:
            return []
        hyps = []
        f_list = [0.55, 0.7, 0.85, 1.0, 1.25, 1.6, 2.2]
        if self.prior.f:
            f_list = sorted(set(f_list + [self.prior.f]))
        has_h = bool(self.obs.cl["horizontal"])

        def vp_cost(z, f_fixed):
            pitch, roll = z[:2]
            f = f_fixed if f_fixed else np.exp(z[2])
            th = np.array([0, 0, 0, 0, pitch, roll, f, 0, 0, 0.0])
            ve, he = self.vp_errors(th)
            return (ve or 0) + (he or 0) * (1 if has_h else 0)

        for f0 in f_list:
            best = None
            for p0 in (-10, 5, 20, 35):
                r = minimize(vp_cost, [p0, 0, np.log(f0)], args=(None if has_h else f0,), method="Nelder-Mead",
                             options=dict(maxfev=300, xatol=0.01, fatol=1e-4))
                if best is None or r.fun < best.fun:
                    best = r
            f = float(np.exp(best.x[2])) if has_h else f0
            if 0.35 < f < 4 and abs(best.x[1]) < 15:
                hyps.append((float(best.fun), f, float(best.x[0]), float(best.x[1])))
        hyps.sort()
        # keep distinct focal hypotheses
        out = []
        for e, f, p, r in hyps:
            if all(abs(np.log(f / g)) > 0.08 for _, g, _, _ in out):
                out.append((e, f, p, r))
        out = out[:3] if has_h else out
        # Always try the prior focal length too: vanishing points from small, noisy frames lean telephoto, and a
        # narrow lens further back matches the far street canyon about as well as the true wide one, so the search
        # never recovers if no hypothesis starts near the right focal length.
        if self.prior.f and all(abs(np.log(self.prior.f / g)) > 0.08 for _, g, _, _ in out):
            best = min((minimize(vp_cost, [p0, 0, np.log(self.prior.f)], args=(self.prior.f,), method="Nelder-Mead",
                                 options=dict(maxfev=300, xatol=0.01, fatol=1e-4)) for p0 in (-10, 5, 20, 35)),
                       key=lambda r: r.fun)
            if abs(best.x[1]) < 15:
                out.append((float(best.fun), self.prior.f, float(best.x[0]), float(best.x[1])))
        return out

    # ---------------------------------------------------------------- stage 2: coarse panorama search
    def coarse(self, rot_hyps, keep=6):
        p = self.prior
        w = 96
        h, target, _, _ = self.obs.at(w)
        if p.heading is not None:
            yaws = np.arange(p.heading - 90, p.heading + 90.1, 1.5)
        else:
            yaws = np.arange(0, 360, 1.5)
        if rot_hyps:
            combos = [(f, pt + dp, r) for _, f, pt, r in rot_hyps for dp in (-2, 0, 2)]
        else:
            fs = [p.f * k for k in (0.9, 1.0, 1.1)] if p.f else [0.6, 0.8, 1.0, 1.3, 1.7]
            combos = [(f, pt, 0.0) for f in fs for pt in np.arange(-15, 36, 2.5)]
        positions = [(p.x + dx, p.y + dy, p.ground + hh) for hh in p.heights for dx in (-15, 0, 15) for dy in (-15, 0, 15)]
        results = []
        for pos in positions:
            pano, _ = render.panorama(self.city, pos, PANO_W, PANO_H, PANO_PITCH)
            pano = model_classes(pano)
            for f, pitch, roll in combos:
                d = C.ray_dirs(np.array([0, 0, 0, 0, pitch, roll, f, 0, 0, 0]), w, h)
                az = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
                el = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
                row = np.clip(((PANO_PITCH[1] - el) / (PANO_PITCH[1] - PANO_PITCH[0]) * PANO_H).astype(int), 0, PANO_H - 1)
                col = (((az[None, :] + yaws[:, None]) % 360) / 360 * PANO_W).astype(int) % PANO_W
                s = class_iou(pano[row[None, :], col], target)
                for yi in np.argsort(-s)[:3]:
                    results.append((float(s[yi]), np.array([*pos, yaws[yi] % 360, pitch, roll, f, 0, 0, 0.0])))
        results.sort(key=lambda r: -r[0])
        out = []
        for s, th in results:
            if all(abs((th[3] - q[3] + 180) % 360 - 180) > 5 or np.linalg.norm(th[:3] - q[:3]) > 6 for _, q in out):
                out.append((s, th))
            if len(out) >= keep:
                break
        return out

    # ---------------------------------------------------------------- stage 3: joint refinement
    def bounds(self, th):
        p = self.prior
        return np.array([
            (p.x - 2.5 * p.pos_sigma, p.x + 2.5 * p.pos_sigma), (p.y - 2.5 * p.pos_sigma, p.y + 2.5 * p.pos_sigma),
            (p.ground + p.z_range[0], p.ground + p.z_range[1]), (th[3] - 25, th[3] + 25), p.pitch_range, (-15, 15),
            (0.3, 5.0), (-0.15, 0.15), (-0.15, 0.15), (-0.6, 0.6)])

    def refine(self, th, w, iters):
        lo, hi = self.bounds(th).T
        scale = np.array([3, 3, 1.5, 1.5, 1.5, 1.0, 0.05 * th[6], 0.01, 0.01, 0.05])

        def f(z):
            return self.cost(np.clip(th + z * scale, lo, hi), w)
        r = minimize(f, np.zeros(C.NPARAM), method="Nelder-Mead",
                     options=dict(maxfev=iters, xatol=0.02, fatol=1e-5, adaptive=True,
                                  initial_simplex=np.vstack([np.zeros(C.NPARAM), np.eye(C.NPARAM)])))
        return r.fun, np.clip(th + r.x * scale, lo, hi)

    def solve(self, n_refine=3):
        rot = self.init_rotation()
        cands = self.coarse(rot)
        if not cands:
            return None
        refined = []
        for s, th in cands[:n_refine]:
            J, t1 = self.refine(th, 128, 450)
            refined.append((J, t1))
        refined.sort(key=lambda r: r[0])
        J, best = self.refine(refined[0][1], 256, 300)
        J, parts = self.cost(best, 256, parts=True)
        parts.update(self.verify(best))
        # ambiguity: best competing hypothesis that differs in yaw/position
        alt = [Jr for Jr, t in refined[1:] if abs((t[3] - best[3] + 180) % 360 - 180) > 8 or np.hypot(*(t[:2] - best[:2])) > 12]
        parts["margin"] = float(min(alt) - refined[0][0]) if alt else None
        parts["n_vertical"] = int(self.obs.cl["vertical"][1].sum()) if self.obs.cl["vertical"] is not None else 0
        parts["n_horizontal_clusters"] = len(self.obs.cl["horizontal"])
        parts["vp_init"] = bool(rot)
        return best, J, parts

    # ---------------------------------------------------------------- stage 4: verification
    def verify(self, theta, w=384):
        """Median distance (px at 1000 px width) from model edges to image edges, and edge support fraction."""
        h, _, dt, _ = self.obs.at(w)
        lab, dep, prim = self.render(theta, w)
        _, e = model_edges(lab, dep, prim, self.city.normal)
        if not e.any():
            return dict(edge_med_px=None, edge_support=0.0)
        d = dt[e] * (1000.0 / w)
        return dict(edge_med_px=float(np.median(d)), edge_support=float((d < 6).mean()))
