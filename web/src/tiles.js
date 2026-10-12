// Streams city tiles (see pipeline/build_city.py for the binary format).
// Local frame: x east, y north, z up (meters). Three.js frame: X = x, Y = z, Z = -y.
import * as THREE from 'three';

const MAGIC = 0x31545643; // 'CVT1'

export function decodeTile(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('bad tile');
  const nv = dv.getUint32(4, true), ni = dv.getUint32(8, true);
  const ox = dv.getFloat32(12, true), oy = dv.getFloat32(16, true), oz = dv.getFloat32(20, true), s = dv.getFloat32(24, true);
  const q = new Int16Array(buf, 28, nv * 3);
  const idxOff = 28 + nv * 6;
  const idx = new Uint32Array(buf.slice(idxOff, idxOff + ni * 4));
  const pos = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    pos[i * 3] = ox + q[i * 3] * s;
    pos[i * 3 + 1] = oz + q[i * 3 + 2] * s;
    pos[i * 3 + 2] = -(oy + q[i * 3 + 1] * s);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeBoundingSphere();
  g.computeBoundingBox();
  return g;
}

export class TileStreamer {
  constructor(scene, materials, base = 'tiles/') {
    this.scene = scene;
    this.mat = materials;
    this.base = base;
    this.blocks = new Map(); // lod1 block key -> {lod1Mesh, children:[tile entries], loaded}
    this.loading = 0;
    this.onChange = () => {};
    this.group = new THREE.Group();
    scene.add(this.group);
  }

  async init() {
    const idx = await (await fetch(this.base + 'index.json')).json();
    this.index = idx;
    const k = idx.lod1Size / idx.tileSize;
    for (const t of idx.tiles) {
      const key = `${Math.floor(t.i / k)}_${Math.floor(t.j / k)}`;
      if (!this.blocks.has(key)) this.blocks.set(key, { key, tiles: [], lod1: null, state: 'coarse' });
      const b = this.blocks.get(key);
      t.center = new THREE.Vector3((t.min[0] + t.max[0]) / 2, 0, -(t.min[1] + t.max[1]) / 2);
      b.tiles.push(t);
    }
    for (const b of this.blocks.values()) {
      const c = new THREE.Vector3();
      for (const t of b.tiles) c.add(t.center);
      b.center = c.divideScalar(b.tiles.length);
    }
    // Terrain and coarse level load immediately.
    await Promise.all([
      ...idx.terrain.map(t => this.load(t.file).then(g => this.addMesh(g, this.mat.terrain))),
      ...idx.lod1.map(t => this.load(t.file).then(g => {
        const b = this.blocks.get(`${t.i}_${t.j}`);
        const m = this.addMesh(g, this.mat.building);
        if (b) b.lod1 = m;
      })),
    ]);
    this.onChange();
  }

  async load(file) {
    this.loading++;
    try {
      const r = await fetch(this.base + file);
      return decodeTile(await r.arrayBuffer());
    } finally {
      this.loading--;
    }
  }

  addMesh(g, mat) {
    const m = new THREE.Mesh(g, mat);
    m.matrixAutoUpdate = false;
    this.group.add(m);
    return m;
  }

  // Swap blocks between the coarse (tall buildings only) and full level based on distance to the focus point.
  update(focus, nearDist = 2600, farDist = 3600) {
    for (const b of this.blocks.values()) {
      const d = Math.hypot(b.center.x - focus.x, b.center.z - focus.z);
      if (d < nearDist && b.state === 'coarse') {
        b.state = 'loading';
        Promise.all(b.tiles.map(t => this.load(t.file))).then(gs => {
          b.meshes = gs.map(g => this.addMesh(g, this.mat.building));
          if (b.lod1) b.lod1.visible = false;
          b.state = 'full';
          if (b.wantCoarse) this.unload(b);
          this.onChange();
        });
      } else if (d > farDist) {
        if (b.state === 'full') this.unload(b);
        else if (b.state === 'loading') b.wantCoarse = true;
      } else {
        b.wantCoarse = false;
      }
    }
  }

  unload(b) {
    for (const m of b.meshes || []) { this.group.remove(m); m.geometry.dispose(); }
    b.meshes = null;
    b.wantCoarse = false;
    if (b.lod1) b.lod1.visible = true;
    b.state = 'coarse';
    this.onChange();
  }
}
