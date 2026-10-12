// Aerial base texture (pipeline/fetch_ortho.py): 500 m tiles, axis-aligned with the local frame, streamed into a
// texture array around the focus point. A small index texture maps tile (i, j) -> array layer for the city shader.
import * as THREE from 'three';

const LAYERS = 36;
const RES = 1024;
export const IDX_N = 64; // index covers tiles i, j in [-32, 32)
const IDX_OFF = IDX_N / 2;

export class OrthoCache {
  constructor(renderer, proj) {
    this.renderer = renderer;
    this.proj = proj; // reuses its fullscreen blit
    this.rt = new THREE.WebGLArrayRenderTarget(RES, RES, LAYERS, { depthBuffer: false });
    Object.assign(this.rt.texture, {
      generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, anisotropy: 8,
    });
    this.idxData = new Uint8Array(IDX_N * IDX_N * 4);
    this.idx = new THREE.DataTexture(this.idxData, IDX_N, IDX_N, THREE.RGBAFormat);
    this.idx.minFilter = this.idx.magFilter = THREE.NearestFilter;
    this.idx.needsUpdate = true;
    this.layers = new Array(LAYERS).fill(null); // tile key per layer
    this.loading = new Set();
    this.available = new Set();
    proj.uniforms.uOrtho.value = this.rt.texture;
    proj.uniforms.uOrthoIdx.value = this.idx;
  }

  async init(base = 'ortho/') {
    this.base = base;
    try {
      const idx = await (await fetch(base + 'index.json')).json();
      this.tile = idx.tile;
      for (const [i, j] of idx.tiles) this.available.add(`${i}_${j}`);
      this.proj.uniforms.uOrthoTile.value = idx.tile;
    } catch { /* no aerial imagery published */ }
  }

  setIndex(key, layer) {
    const [i, j] = key.split('_').map(Number);
    const u = i + IDX_OFF, v = j + IDX_OFF;
    if (u < 0 || v < 0 || u >= IDX_N || v >= IDX_N) return;
    this.idxData[(v * IDX_N + u) * 4] = layer + 1;
    this.idx.needsUpdate = true;
  }

  // focus: THREE world position (X = x, Z = -y)
  update(focus, radius = 1500) {
    if (!this.tile) return;
    const T = this.tile;
    const fx = focus.x, fy = -focus.z;
    const want = [];
    const r = Math.ceil(radius / T);
    const ci = Math.floor(fx / T), cj = Math.floor(fy / T);
    for (let i = ci - r; i <= ci + r; i++) for (let j = cj - r; j <= cj + r; j++) {
      const key = `${i}_${j}`;
      if (!this.available.has(key)) continue;
      const d = Math.hypot((i + 0.5) * T - fx, (j + 0.5) * T - fy);
      if (d < radius + T) want.push([d, key]);
    }
    want.sort((a, b) => a[0] - b[0]);
    const keep = new Set(want.slice(0, LAYERS).map(w => w[1]));
    for (const [, key] of want.slice(0, LAYERS)) {
      if (this.layers.includes(key) || this.loading.has(key)) continue;
      let slot = this.layers.indexOf(null);
      if (slot < 0) slot = this.layers.findIndex(k => k && !k.startsWith('loading:') && !keep.has(k));
      if (slot < 0) break;
      if (this.layers[slot]) this.setIndex(this.layers[slot], -1);
      const tag = 'loading:' + key;
      this.layers[slot] = tag; // reserved
      this.loading.add(key);
      new THREE.TextureLoader().loadAsync(this.base + key + '.jpg').then(tex => {
        this.loading.delete(key);
        if (this.layers[slot] !== tag) { tex.dispose(); return; } // slot was reassigned meanwhile
        tex.colorSpace = THREE.NoColorSpace;
        this.proj.blit(this.rt, slot, tex);
        tex.dispose();
        this.layers[slot] = key;
        this.setIndex(key, slot);
      }).catch(e => { console.warn('ortho', key, e); this.loading.delete(key); if (this.layers[slot] === tag) this.layers[slot] = null; });
    }
  }
}
