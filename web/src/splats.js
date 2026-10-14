// Gaussian splats reconstructed from photos (pipeline/splat_train.py), rendered with Spark and composited with the
// city mesh through the shared depth buffer. Splats are in the local frame (z up) and load when the view is nearby.
import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

export class Splats {
  constructor(renderer, scene) {
    this.scene = scene;
    this.spark = new SparkRenderer({ renderer });
    scene.add(this.spark);
    this.items = [];
    this.visible = true;
  }

  async init(base, list) {
    this.base = base;
    this.items = list.map(s => ({ ...s, focus: new THREE.Vector3(s.center[0], s.center[2], -s.center[1]), mesh: null }));
  }

  update(focus, radius = 900) {
    for (const s of this.items) {
      const near = this.visible && s.focus.distanceTo(focus) < radius;
      if (near && !s.mesh) {
        s.mesh = new SplatMesh({ url: this.base + s.file });
        s.mesh.rotation.x = -Math.PI / 2; // local z-up -> three y-up (x, y, z) -> (x, z, -y)
        this.scene.add(s.mesh);
      } else if (!near && s.mesh && s.focus.distanceTo(focus) > radius * 1.5) {
        this.scene.remove(s.mesh);
        s.mesh.dispose?.();
        s.mesh = null;
      }
      if (s.mesh) s.mesh.visible = this.visible;
    }
  }
}
