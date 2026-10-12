// Per-photo 2.5D layers: a depth grid (pipeline/depth_layers.py) back-projected through the photo's solved camera
// and textured with the photo itself. Gives trees, awnings, vehicles and facade relief real parallax.
import * as THREE from 'three';
import { orientCamera, intrinsics } from './projectors.js';

const vert = /* glsl */`
  varying vec2 vUv;
  varying float vFog;
  void main() {
    vUv = uv;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vFog = -mv.z;
    gl_Position = projectionMatrix * mv;
  }`;
const frag = /* glsl */`
  uniform sampler2D map;
  uniform float uAlpha;
  uniform vec3 uFogColor;
  uniform float uFogDensity;
  varying vec2 vUv;
  varying float vFog;
  void main() {
    // dithered transparency: no sorting, depth stays correct
    float h = fract(sin(dot(floor(gl_FragCoord.xy), vec2(12.9898, 78.233))) * 43758.5453);
    if (h > uAlpha) discard;
    vec3 c = texture2D(map, vUv).rgb;
    float fog = 1.0 - exp(-pow(uFogDensity * vFog, 2.0));
    gl_FragColor = vec4(mix(c, uFogColor, fog), 1.0);
  }`;

export async function buildLayer(item, texture, fogUniforms) {
  const buf = await (await fetch(item.depth)).arrayBuffer();
  const dv = new DataView(buf);
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'CVD2') throw new Error('bad depth layer');
  const gw = dv.getUint16(4, true), gh = dv.getUint16(6, true), zmax = dv.getFloat32(8, true);
  const raw = new Response(new Blob([buf.slice(12)]).stream().pipeThrough(new DecompressionStream('deflate')));
  const q = new Uint16Array(await raw.arrayBuffer());
  const lz = Math.log(zmax);
  const z = new Float32Array(gw * gh);
  for (let k = 0; k < z.length; k++) z[k] = q[k] ? Math.exp((q[k] - 1) / 65534 * lz) : 0;

  const cam = orientCamera(item.pose, new THREE.PerspectiveCamera());
  const [f, cx, cy] = intrinsics(item);
  const A = item.h / item.w;
  const pos = new Float32Array(gw * gh * 3), uv = new Float32Array(gw * gh * 2);
  const v = new THREE.Vector3();
  for (let i = 0; i < gh; i++) {
    for (let j = 0; j < gw; j++) {
      const k = i * gw + j, d = z[k] || 1;
      const xu = (j + 0.5) / gw - 0.5 - cx, yu = ((i + 0.5) / gh - 0.5) * A - cy;
      v.set(xu / f * d, -yu / f * d, -d).applyMatrix4(cam.matrixWorld);
      pos.set([v.x, v.y, v.z], k * 3);
      uv.set([(j + 0.5) / gw, 1 - (i + 0.5) / gh], k * 2);
    }
  }
  const idx = [];
  const eye = cam.position;
  const pa = new THREE.Vector3(), pb = new THREE.Vector3(), pc = new THREE.Vector3(), n = new THREE.Vector3(), ray = new THREE.Vector3();
  const ok = (...ks) => {
    let lo = Infinity, hi = 0;
    for (const k of ks) { if (!z[k]) return false; lo = Math.min(lo, z[k]); hi = Math.max(hi, z[k]); }
    if (hi / lo > 1.12) return false; // depth discontinuity
    // drop triangles seen edge-on from the photo (rubber-sheet streaks along camera rays)
    pa.fromArray(pos, ks[0] * 3); pb.fromArray(pos, ks[1] * 3); pc.fromArray(pos, ks[2] * 3);
    n.subVectors(pb, pa).cross(ray.subVectors(pc, pa)).normalize();
    ray.subVectors(pa, eye).normalize();
    return Math.abs(n.dot(ray)) > 0.2;
  };
  for (let i = 0; i < gh - 1; i++) {
    for (let j = 0; j < gw - 1; j++) {
      const a = i * gw + j, b = a + 1, c = a + gw, d = c + 1;
      if (ok(a, b, c)) idx.push(a, c, b);
      if (ok(b, c, d)) idx.push(b, c, d);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeBoundingSphere();
  const mat = new THREE.ShaderMaterial({
    vertexShader: vert, fragmentShader: frag, side: THREE.DoubleSide,
    uniforms: { map: { value: texture }, uAlpha: { value: 0 }, ...fogUniforms },
  });
  const mesh = new THREE.Mesh(g, mat);
  mesh.userData.noDepth = true; // keep projector depth maps built from the city model only
  mesh.frustumCulled = true;
  return mesh;
}

// How strongly to show a layer given the current viewing camera: aligned direction and nearby position.
export function layerAlpha(item, camera) {
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const align = THREE.MathUtils.smoothstep(fwd.dot(item.fwd), Math.cos(THREE.MathUtils.degToRad(55)), Math.cos(THREE.MathUtils.degToRad(15)));
  const near = 1 - THREE.MathUtils.smoothstep(camera.position.distanceTo(item.pos), 120, 450);
  return align * near;
}
