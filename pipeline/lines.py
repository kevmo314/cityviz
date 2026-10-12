"""Line segments and vanishing-point clusters.

Segments are returned in width-normalized image coordinates (see camera.py) so they can be undistorted
with the current camera hypothesis inside the solver.
"""
import cv2
import numpy as np


def detect(img_rgb, max_side=1024, min_len=0.025):
    a = np.asarray(img_rgb)
    H, W = a.shape[:2]
    s = min(1.0, max_side / max(W, H))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if s < 1:
        g = cv2.resize(g, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(g)[0]
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4) / s
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[L > min_len * max(W, H)]
    # pixel -> width-normalized
    out = segs.copy()
    out[:, [0, 2]] = (segs[:, [0, 2]] - W / 2) / W
    out[:, [1, 3]] = (segs[:, [1, 3]] - H / 2) / W
    return out


def _homog_lines(segs):
    p1 = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], np.ones(len(segs))])
    l = np.cross(p1, p2)
    return l / (np.linalg.norm(l[:, :2], axis=1, keepdims=True) + 1e-12)


def vp_angle_error(segs, vp):
    """Angle (deg) between each segment and the direction from its midpoint to the vanishing point (homogeneous)."""
    m = (segs[:, :2] + segs[:, 2:]) / 2
    d = segs[:, 2:] - segs[:, :2]
    t = vp[None, :2] - m * vp[2]
    c = np.abs((d * t).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def fit_vp(segs, w=None):
    """Least-squares vanishing point (homogeneous, unit norm) of a set of segments."""
    l = _homog_lines(segs)
    if w is not None:
        l = l * w[:, None]
    _, _, vt = np.linalg.svd(l)
    v = vt[-1]
    return v / np.linalg.norm(v)


def ransac_vp(segs, cand_mask, iters=400, thresh=1.5, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = np.flatnonzero(cand_mask)
    if len(idx) < 3:
        return None, np.zeros(len(segs), bool)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    lines = _homog_lines(segs)
    best, best_score = None, 0
    for _ in range(iters):
        i, j = rng.choice(idx, 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if np.linalg.norm(v) < 1e-12:
            continue
        v /= np.linalg.norm(v)
        inl = cand_mask & (vp_angle_error(segs, v) < thresh)
        sc = L[inl].sum()
        if sc > best_score:
            best, best_score = v, sc
    if best is None:
        return None, np.zeros(len(segs), bool)
    inl = cand_mask & (vp_angle_error(segs, best) < thresh)
    v = fit_vp(segs[inl], L[inl])
    inl = cand_mask & (vp_angle_error(segs, v) < thresh)
    return v, inl


def clusters(segs, min_support=4):
    """Return dict(vertical=(vp, mask), horizontal=[(vp, mask), ...])."""
    out = {"vertical": None, "horizontal": []}
    if len(segs) < 3:
        return out
    d = segs[:, 2:] - segs[:, :2]
    ang = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))  # 0 = image vertical
    v, inl = ransac_vp(segs, ang < 25)
    used = np.zeros(len(segs), bool)
    if v is not None and inl.sum() >= min_support:
        out["vertical"] = (v, inl)
        used |= inl
    for _ in range(2):
        cand = ~used & (ang > 30)
        h, inl = ransac_vp(segs, cand, thresh=1.2)
        if h is None or inl.sum() < min_support:
            break
        out["horizontal"].append((h, inl))
        used |= inl
    return out


def rasterize(segs, W, H, thickness=1):
    """Draw width-normalized segments into a W x H uint8 mask (in distorted image space)."""
    m = np.zeros((H, W), np.uint8)
    p = segs.copy()
    p[:, [0, 2]] = p[:, [0, 2]] * W + W / 2
    p[:, [1, 3]] = p[:, [1, 3]] * W + H / 2
    for x1, y1, x2, y2 in p:
        cv2.line(m, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, thickness)
    return m
