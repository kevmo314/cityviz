// Picks which photos / cams get a projector slot: the ones that paint the most of the current view that nothing
// better already paints.
//
// Each item carries a footprint (pipeline/footprints.py): a GW x GH grid of image rays cast into the city mesh. Every
// pick, footprint quads are projected into the view and rasterized onto a coarse grid of screen cells; a low-res
// distance render of the view drops cells where something nearer hides the wall. Greedy max coverage then picks items
// by the screen area they add, with already-covered cells counting less, so overlapping photos of one facade don't
// crowd out the building next to it.
import * as THREE from 'three';

const CW = 48, CH = 27;               // screen cells (and distance-buffer resolution)
const OVERLAP = [1, 0.3, 0.08];      // value of a cell already covered by 0, 1, 2+ picked items
const MISS = -32768;

// footprints.bin -> item.fp (Float32Array of three-frame points, NaN where the ray missed) and item.fpSphere
export async function loadFootprints(url, photos, cams) {
  const buf = await fetch(url).then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); });
  const dv = new DataView(buf);
  if (new TextDecoder().decode(buf.slice(0, 4)) !== 'CVF1') throw new Error('bad footprints');
  const nP = dv.getUint32(4, true), nC = dv.getUint32(8, true), gw = dv.getUint8(12), gh = dv.getUint8(13);
  if (nP !== photos.length || nC !== cams.length) throw new Error('footprints out of date');
  const raw = new Int16Array(buf, 16), n = gw * gh;
  [...photos, ...cams].forEach((it, k) => {
    const fp = new Float32Array(n * 3), pts = [];
    for (let i = 0; i < n; i++) {
      const o = (k * n + i) * 3;
      if (raw[o] === MISS) { fp.fill(NaN, i * 3, i * 3 + 3); continue; }
      // pipeline offset (x east, y north, z up) -> three (X = x, Y = z, Z = -y)
      const p = new THREE.Vector3(it.pos.x + raw[o], it.pos.y + raw[o + 2], it.pos.z - raw[o + 1]);
      p.toArray(fp, i * 3);
      pts.push(p);
    }
    it.fp = fp;
    it.fpSphere = pts.length ? new THREE.Sphere().setFromPoints(pts) : null;
  });
  return { gw, gh };
}

export class CoverageSelector {
  constructor(proj, grid) {
    this.proj = proj;
    this.gw = grid.gw; this.gh = grid.gh;
    this.rt = new THREE.WebGLRenderTarget(CW, CH, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: true });
    this.dist = new Float32Array(CW * CH * 4);
    this.frustum = new THREE.Frustum();
    this.vp = new THREE.Matrix4();
    this.occlusion = false;  // until the first distance readback lands
    this.reading = false;
  }

  // Distance render of the view, read back without stalling the GPU: picks use the latest completed readback (at most
  // one pick interval old).
  readDistance(camera) {
    if (this.reading || this.failed) return;
    this.reading = true;
    try {
      this.proj.renderDistance(camera, this.rt);
      const buf = new Float32Array(CW * CH * 4);
      this.proj.renderer.readRenderTargetPixelsAsync(this.rt, 0, 0, CW, CH, buf)
        .then(() => { this.dist = buf; this.occlusion = true; })
        .catch(() => { this.failed = true; this.occlusion = false; })
        .finally(() => { this.reading = false; });
    } catch (e) { this.failed = true; this.reading = false; }
  }

  // Ordered list of up to `budget` items for `camera` among `items`; `bonus(item)` scales an item's value.
  pick(camera, items, budget, bonus) {
    camera.updateMatrixWorld();
    this.vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.vp);
    this.readDistance(camera);
    const cands = [];
    for (const it of items) {
      if (!it.fp || !it.fpSphere || !this.frustum.intersectsSphere(it.fpSphere)) continue;
      const cells = this.footprintCells(it, camera.position);
      if (cells.size) cands.push({ it, cells, b: bonus(it) });
    }
    // greedy max coverage with lazy gain updates (gains only shrink as cells fill up)
    const cover = new Uint8Array(CW * CH), out = [];
    const gain = c => { let g = 0; for (const [i, w] of c.cells) g += w * OVERLAP[Math.min(cover[i], 2)]; return g * c.b; };
    for (const c of cands) c.g = gain(c);
    while (out.length < budget && cands.length) {
      cands.sort((a, b) => b.g - a.g);
      const top = cands[0];
      top.g = gain(top);
      if (cands.length > 1 && top.g < cands[1].g) continue;
      if (top.g < 0.5) break;  // adds less than half a cell
      out.push(top.it);
      for (const i of top.cells.keys()) cover[i]++;
      cands.shift();
    }
    return out;
  }

  // screen cell index -> value (visible area x sharpness) of the item's footprint in this view
  footprintCells(it, eye) {
    const { gw, gh } = this, fp = it.fp, m = this.vp.elements, n = gw * gh;
    if (!this.sx || this.sx.length < n) {
      this.sx = new Float32Array(n); this.sy = new Float32Array(n); this.d = new Float32Array(n); this.dp = new Float32Array(n);
      this.ok = new Uint8Array(n);
    }
    const { sx, sy, d, dp, ok } = this, X = [0, 0, 0, 0], Y = [0, 0, 0, 0], q = [0, 0, 0, 0];
    let any = false;
    for (let i = 0; i < n; i++) {
      ok[i] = 0;
      const x = fp[i * 3], y = fp[i * 3 + 1], z = fp[i * 3 + 2];
      if (Number.isNaN(x)) continue;
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (w <= 0.5) continue;
      sx[i] = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * CW;
      sy[i] = ((m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5 + 0.5) * CH;
      d[i] = Math.hypot(x - eye.x, y - eye.y, z - eye.z);
      dp[i] = Math.hypot(x - it.pos.x, y - it.pos.y, z - it.pos.z);
      ok[i] = 1; any = true;
    }
    const cells = new Map();
    if (!any) return cells;
    const add = (i, v) => { const o = cells.get(i); if (o === undefined || v > o) cells.set(i, v); };
    for (let r = 0; r + 1 < gh; r++) {
      for (let c = 0; c + 1 < gw; c++) {
        q[0] = r * gw + c; q[1] = q[0] + 1; q[3] = q[0] + gw; q[2] = q[3] + 1;
        if (!(ok[q[0]] && ok[q[1]] && ok[q[2]] && ok[q[3]])) continue;
        // sharpness: how close the photo was to this wall compared with the viewer (texels per screen pixel)
        let dv = 0, dps = 0, dmin = Infinity, xa = Infinity, xb = -Infinity, ya = Infinity, yb = -Infinity;
        for (let k = 0; k < 4; k++) {
          const i = q[k];
          dv += d[i]; dps += dp[i]; if (d[i] < dmin) dmin = d[i];
          X[k] = sx[i]; Y[k] = sy[i];
          if (X[k] < xa) xa = X[k]; if (X[k] > xb) xb = X[k]; if (Y[k] < ya) ya = Y[k]; if (Y[k] > yb) yb = Y[k];
        }
        const x0 = Math.max(0, Math.floor(xa)), x1 = Math.min(CW - 1, Math.floor(xb));
        const y0 = Math.max(0, Math.floor(ya)), y1 = Math.min(CH - 1, Math.floor(yb));
        if (x0 > x1 || y0 > y1) continue;
        const sharp = Math.sqrt(Math.min(1, Math.max(0.05, dv / Math.max(dps, 1))));
        const area = Math.abs((X[0] - X[2]) * (Y[1] - Y[3]) - (X[1] - X[3]) * (Y[0] - Y[2])) / 2;
        const far = dmin * 0.92 - 2;
        if (area < 1) {
          // smaller than a cell: credit its area to the cell under its centre
          const cx = Math.floor((X[0] + X[1] + X[2] + X[3]) / 4), cy = Math.floor((Y[0] + Y[1] + Y[2] + Y[3]) / 4);
          if (cx < 0 || cy < 0 || cx >= CW || cy >= CH) continue;
          const i = cy * CW + cx;
          if (!this.occlusion || this.dist[i * 4] >= far) add(i, area * sharp);
          continue;
        }
        for (let y = y0; y <= y1; y++) {
          for (let x = x0; x <= x1; x++) {
            const i = y * CW + x;
            if ((!this.occlusion || this.dist[i * 4] >= far) && inQuad(x + 0.5, y + 0.5, X, Y)) add(i, sharp);
          }
        }
      }
    }
    return cells;
  }
}

// point inside a (possibly non-convex after projection) quad: test both triangles
function inQuad(px, py, X, Y) {
  return inTri(px, py, X[0], Y[0], X[1], Y[1], X[2], Y[2]) || inTri(px, py, X[0], Y[0], X[2], Y[2], X[3], Y[3]);
}
function inTri(px, py, ax, ay, bx, by, cx, cy) {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
}
