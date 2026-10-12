"""Camera model shared by the solver, the exporter and (mirrored in JS) the viewer.

theta = [x, y, z, yaw, pitch, roll, f, cx, cy, k1]
  x, y, z      position, local meters (x east, y north, z up)
  yaw          degrees clockwise from grid north; pitch up +; roll degrees
  f            focal length / image width
  cx, cy       principal point offset from the image center, in units of image width (y down)
  k1           division-model radial distortion on width-normalized coordinates:
               undistorted = distorted / (1 + k1 * r_d^2)

Image coordinates are "width-normalized": xn = (u - W/2) / W, yn = (v - H/2) / W, with u, v pixel coords.
"""
import numpy as np

NPARAM = 10


def basis(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    f = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(y), -np.sin(y), 0.0])
    up = np.cross(right, f)
    rr = np.cos(r) * right + np.sin(r) * up
    ru = np.cos(r) * up - np.sin(r) * right
    return f, rr, ru


def hfov_to_f(hfov_deg):
    return 0.5 / np.tan(np.radians(hfov_deg) / 2)


def f_to_hfov(f):
    return float(np.degrees(2 * np.arctan(0.5 / f)))


def pixel_grid(W, H):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    return (u - W / 2) / W, (v - H / 2) / W


def undistort_norm(xn, yn, theta):
    """Distorted width-normalized coords -> ideal pinhole coords relative to the principal point."""
    _, _, _, _, _, _, f, cx, cy, k1 = theta
    s = 1.0 / (1.0 + k1 * (xn * xn + yn * yn))
    return xn * s - cx, yn * s - cy


def distort_norm(xu, yu, k1):
    """Inverse of the division model for points relative to the image center (closed form)."""
    ru = np.hypot(xu, yu)
    if abs(k1) < 1e-9:
        return xu, yu
    disc = np.clip(1 - 4 * k1 * ru * ru, 0, None)
    rd = np.where(ru > 1e-12, (1 - np.sqrt(disc)) / (2 * k1 * np.maximum(ru, 1e-12)), 0.0)
    s = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
    return xu * s, yu * s


def ray_dirs(theta, W, H, xn=None, yn=None):
    """World-space unit ray directions for every pixel (or for given normalized coords)."""
    if xn is None:
        xn, yn = pixel_grid(W, H)
    xu, yu = undistort_norm(xn, yn, theta)
    fw, r, u = basis(theta[3], theta[4], theta[5])
    f = theta[6]
    d = fw[None] * f + xu.reshape(-1, 1) * r[None] - yu.reshape(-1, 1) * u[None]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def world_to_cam(theta, d):
    """World directions (N,3) -> camera coords (x right, y down, z forward)."""
    fw, r, u = basis(theta[3], theta[4], theta[5])
    return np.stack([d @ r, -(d @ u), d @ fw], -1)


def from_legacy(pose7):
    """[x, y, z, yaw, pitch, roll, hfov] -> theta."""
    return np.array([*pose7[:6], hfov_to_f(pose7[6]), 0.0, 0.0, 0.0])
