"""CPU ray-cast renderer of the city mesh for pose search.

Pose: x, y, z (local meters), yaw (deg clockwise from grid north), pitch (deg, up +), roll (deg), hfov (deg).
Labels: 0 sky, 1 ground, 2 building wall, 3 roof.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from embreex import rtcore_scene as rtcs
from embreex.mesh_construction import TriangleMesh

ROOT = Path(__file__).resolve().parent.parent
SKY, GROUND, WALL, ROOF = 0, 1, 2, 3


class City:
    def __init__(self):
        d = np.load(ROOT / "data/work/city_mesh.npz")
        self.v, self.f = d["v"].astype(np.float32), d["f"]
        tri = self.v[self.f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.normal = (n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        self.label = np.where(d["terrain"] == 1, GROUND, np.where(self.normal[:, 2] > 0.7, ROOF, WALL)).astype(np.uint8)
        self.scene = rtcs.EmbreeScene()
        TriangleMesh(self.scene, tri)
        hf = np.load(ROOT / "data/work/heightfield.npz")
        self.hf, self.hx0, self.hy0, self.hres = hf["hf"], float(hf["x0"]), float(hf["y0"]), float(hf["res"])

    def ground(self, x, y):
        i = int(np.clip((y - self.hy0) / self.hres, 0, self.hf.shape[0] - 1))
        j = int(np.clip((x - self.hx0) / self.hres, 0, self.hf.shape[1] - 1))
        return float(self.hf[i, j])

    def cast(self, origins, dirs):
        res = self.scene.run(np.ascontiguousarray(origins, np.float32), np.ascontiguousarray(dirs, np.float32), output=1)
        prim, t = res["primID"], res["tfar"]
        hit = prim >= 0
        return prim, np.where(hit, t, np.inf), hit


def basis(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    f = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(y), -np.sin(y), 0.0])
    up = np.cross(right, f)
    rr = np.cos(r) * right + np.sin(r) * up
    ru = np.cos(r) * up - np.sin(r) * right
    return f, rr, ru


def ray_dirs(pose, W, H):
    f, r, u = basis(pose[3], pose[4], pose[5])
    fpx = (W / 2) / np.tan(np.radians(pose[6]) / 2)
    uu, vv = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
    d = f[None, None] + (uu / fpx)[..., None] * r + (-vv / fpx)[..., None] * u
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(-1, 3)


def render(city, pose, W, H):
    d = ray_dirs(pose, W, H)
    o = np.broadcast_to(np.asarray(pose[:3], np.float32), d.shape)
    prim, t, hit = city.cast(o, d)
    lab = np.zeros(len(d), np.uint8)
    lab[hit] = city.label[prim[hit]]
    return lab.reshape(H, W), t.reshape(H, W), np.where(hit, prim, -1).reshape(H, W)


def panorama(city, pos, W=1440, H=720, pitch_range=(-60, 60)):
    """Equirectangular label/depth panorama around pos covering yaw 0..360, pitch in pitch_range."""
    yaw = (np.arange(W) + 0.5) / W * 360.0
    pit = pitch_range[1] - (np.arange(H) + 0.5) / H * (pitch_range[1] - pitch_range[0])
    Y, P = np.meshgrid(np.radians(yaw), np.radians(pit))
    d = np.stack([np.sin(Y) * np.cos(P), np.cos(Y) * np.cos(P), np.sin(P)], -1).reshape(-1, 3)
    o = np.broadcast_to(np.asarray(pos, np.float32), d.shape)
    prim, t, hit = city.cast(o, d)
    lab = np.zeros(len(d), np.uint8)
    lab[hit] = city.label[prim[hit]]
    return lab.reshape(H, W), t.reshape(H, W)
