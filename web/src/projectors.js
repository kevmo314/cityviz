// Projective texturing of geo-registered photos / live camera frames onto the city mesh.
// Each active projector owns one layer in a color texture array and one layer in a depth (distance) texture array.
import * as THREE from 'three';

export const SLOTS = 16;
const COLOR_RES = 1024;
const DEPTH_RES = 512;

// pose = [x, y, z, yaw, pitch, roll, hfov] in the pipeline frame (x east, y north, z up; yaw clockwise from north).
export function poseCamera(pose, aspect, cam = new THREE.PerspectiveCamera()) {
  const [x, y, z, yawD, pitchD, rollD, hfovD] = pose;
  const yaw = THREE.MathUtils.degToRad(yawD), p = THREE.MathUtils.degToRad(pitchD), r = THREE.MathUtils.degToRad(rollD);
  // pipeline-frame basis -> three frame (X=x, Y=z, Z=-y)
  const T = v => new THREE.Vector3(v[0], v[2], -v[1]);
  const f = [Math.sin(yaw) * Math.cos(p), Math.cos(yaw) * Math.cos(p), Math.sin(p)];
  const right = [Math.cos(yaw), -Math.sin(yaw), 0];
  const up = [right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2], right[0] * f[1] - right[1] * f[0]];
  const ru = up.map((u, i) => Math.cos(r) * u - Math.sin(r) * right[i]);
  cam.position.copy(T([x, y, z]));
  cam.up.copy(T(ru));
  cam.lookAt(cam.position.clone().add(T(f)));
  const hfov = THREE.MathUtils.degToRad(hfovD);
  cam.aspect = aspect;
  cam.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(hfov / 2) / aspect));
  cam.updateMatrixWorld(true);
  cam.updateProjectionMatrix();
  return cam;
}

const vert = /* glsl */`
  varying vec3 vWorld;
  varying float vFog;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    vec4 mv = viewMatrix * w;
    vFog = -mv.z;
    gl_Position = projectionMatrix * mv;
  }`;

const frag = /* glsl */`
  precision highp sampler2DArray;
  uniform sampler2DArray uPhotos;
  uniform sampler2DArray uDepth;
  uniform mat4 uVP[${SLOTS}];
  uniform vec3 uPos[${SLOTS}];
  uniform vec4 uRect[${SLOTS}];
  uniform float uW[${SLOTS}];
  uniform float uPaint;
  uniform float uTerrain;
  uniform vec3 uFogColor;
  uniform float uFogDensity;
  varying vec3 vWorld;
  varying float vFog;
  void main() {
    vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    vec3 L = normalize(vec3(0.45, 0.8, 0.35));
    float diff = 0.62 + 0.38 * max(dot(n, L), 0.0);
    vec3 base = uTerrain > 0.5 ? vec3(0.36, 0.38, 0.38)
      : (abs(n.y) > 0.7 ? vec3(0.80, 0.80, 0.82) : vec3(0.68, 0.69, 0.72)) * diff;
    vec4 acc = vec4(0.0);
    for (int i = 0; i < ${SLOTS}; i++) {
      if (uW[i] <= 0.0) continue;
      vec4 c = uVP[i] * vec4(vWorld, 1.0);
      if (c.w <= 0.0) continue;
      vec2 ndc = c.xy / c.w;
      if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) continue;
      vec2 uv = ndc * 0.5 + 0.5;
      vec4 rc = uRect[i];
      if (uv.x < rc.x || uv.y < rc.y || uv.x > rc.z || uv.y > rc.w) continue;
      vec3 d = vWorld - uPos[i];
      float dist = length(d);
      float facing = dot(n, -d / dist);
      if (facing < 0.03) continue;
      float dm = texture(uDepth, vec3(uv, float(i))).r;
      if (dist > dm * 1.004 + 0.8) continue;
      vec2 e = min(uv - rc.xy, rc.zw - uv) / (rc.zw - rc.xy);
      float edge = smoothstep(0.0, 0.08, min(e.x, e.y));
      float w = uW[i] * edge * smoothstep(0.03, 0.3, facing) / (1.0 + dist / 120.0);
      acc += vec4(texture(uPhotos, vec3(uv, float(i))).rgb * w, w);
    }
    vec3 col = base;
    if (acc.w > 0.0) col = mix(base, acc.rgb / acc.w, clamp(acc.w * 6.0, 0.0, 1.0) * uPaint);
    float fog = 1.0 - exp(-pow(uFogDensity * vFog, 2.0));
    gl_FragColor = vec4(mix(col, uFogColor, fog), 1.0);
  }`;

const depthVert = /* glsl */`
  varying vec3 vWorld;
  void main() { vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const depthFrag = /* glsl */`
  uniform vec3 uEye;
  varying vec3 vWorld;
  void main() { gl_FragColor = vec4(length(vWorld - uEye), 0.0, 0.0, 1.0); }`;

export class ProjectorSystem {
  constructor(renderer, scene, fogColor) {
    this.renderer = renderer;
    this.scene = scene;
    // WebGLArrayRenderTarget replaces its texture with a bare DataArrayTexture, so texture options are set directly.
    this.colorRT = new THREE.WebGLArrayRenderTarget(COLOR_RES, COLOR_RES, SLOTS, { depthBuffer: false });
    Object.assign(this.colorRT.texture, {
      generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, anisotropy: 4,
    });
    this.depthRT = new THREE.WebGLArrayRenderTarget(DEPTH_RES, DEPTH_RES, SLOTS, { depthBuffer: true });
    Object.assign(this.depthRT.texture, {
      type: THREE.FloatType, format: THREE.RedFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false,
    });
    this.uniforms = {
      uPhotos: { value: this.colorRT.texture },
      uDepth: { value: this.depthRT.texture },
      uVP: { value: Array.from({ length: SLOTS }, () => new THREE.Matrix4()) },
      uPos: { value: Array.from({ length: SLOTS }, () => new THREE.Vector3()) },
      uRect: { value: Array.from({ length: SLOTS }, () => new THREE.Vector4(0, 0, 1, 1)) },
      uW: { value: new Array(SLOTS).fill(0) },
      uPaint: { value: 1 },
      uFogColor: { value: fogColor },
      uFogDensity: { value: 0.00012 },
    };
    const mk = terrain => new THREE.ShaderMaterial({
      vertexShader: vert, fragmentShader: frag,
      uniforms: { ...this.uniforms, uTerrain: { value: terrain ? 1 : 0 } },
    });
    this.materials = { building: mk(false), terrain: mk(true) };
    this.depthMat = new THREE.ShaderMaterial({ vertexShader: depthVert, fragmentShader: depthFrag, uniforms: { uEye: { value: new THREE.Vector3() } } });
    this.blitScene = new THREE.Scene();
    this.blitCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.blitMat = new THREE.MeshBasicMaterial({ depthTest: false, depthWrite: false });
    this.blitScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.blitMat));
    this.slots = new Array(SLOTS).fill(null); // slot -> projector item
    this.fade = new Array(SLOTS).fill(0);
    this.cam = new THREE.PerspectiveCamera(50, 1, 0.5, 6000);
    this.depthDirty = new Set();
  }

  // Draw an image/texture into a color layer.
  drawColor(slot, texture) {
    const r = this.renderer;
    const prev = r.getRenderTarget();
    this.blitMat.map = texture;
    this.blitMat.needsUpdate = true;
    r.setRenderTarget(this.colorRT, slot);
    r.render(this.blitScene, this.blitCam);
    r.setRenderTarget(prev);
  }

  drawDepth(slot) {
    const item = this.slots[slot];
    if (!item) return;
    const r = this.renderer;
    const cam = poseCamera(item.pose, item.aspect, this.cam);
    this.depthMat.uniforms.uEye.value.copy(cam.position);
    const prev = r.getRenderTarget();
    const prevClear = r.getClearColor(new THREE.Color()), prevAlpha = r.getClearAlpha();
    const prevBg = this.scene.background, prevOverride = this.scene.overrideMaterial, prevFog = this.scene.fog;
    this.scene.background = null;
    this.scene.fog = null;
    this.scene.overrideMaterial = this.depthMat;
    const hidden = [];
    this.scene.traverse(o => { if (o.userData.noDepth && o.visible) { o.visible = false; hidden.push(o); } });
    r.setClearColor(new THREE.Color(1e7, 0, 0), 1);
    r.setRenderTarget(this.depthRT, slot);
    r.clear();
    r.render(this.scene, cam);
    r.setRenderTarget(prev);
    r.setClearColor(prevClear, prevAlpha);
    hidden.forEach(o => (o.visible = true));
    this.scene.background = prevBg;
    this.scene.overrideMaterial = prevOverride;
    this.scene.fog = prevFog;
    const u = this.uniforms;
    u.uVP.value[slot].multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    u.uPos.value[slot].copy(cam.position);
    const rc = item.rect || [0, 0, 1, 1];
    u.uRect.value[slot].set(rc[0], rc[1], rc[2], rc[3]);
  }

  invalidateDepth() {
    this.slots.forEach((s, i) => s && this.depthDirty.add(i));
  }

  // Assign the wanted items (ordered by priority) to slots, loading textures as needed.
  assign(wanted) {
    const want = wanted.slice(0, SLOTS);
    const keep = new Set(want);
    for (let i = 0; i < SLOTS; i++) {
      const s = this.slots[i];
      if (s && !keep.has(s)) { this.release(i); }
    }
    for (const item of want) {
      if (this.slots.includes(item)) continue;
      const free = this.slots.indexOf(null);
      if (free < 0) break;
      this.slots[free] = item;
      this.fade[free] = 0;
      item.slot = free;
      item.ready = false;
      item.load().then(tex => {
        if (this.slots[free] !== item) return;
        this.drawColor(free, tex);
        this.drawDepth(free);
        item.ready = true;
      }).catch(() => {});
    }
  }

  release(i) {
    const s = this.slots[i];
    if (s) { s.slot = -1; s.ready = false; s.unload?.(); }
    this.slots[i] = null;
    this.uniforms.uW.value[i] = 0;
  }

  // Called every frame.
  update(dt, weightOf) {
    for (const i of this.depthDirty) this.drawDepth(i);
    this.depthDirty.clear();
    for (let i = 0; i < SLOTS; i++) {
      const s = this.slots[i];
      const target = s && s.ready ? weightOf(s) : 0;
      this.fade[i] += (target - this.fade[i]) * Math.min(1, dt * 3);
      this.uniforms.uW.value[i] = this.fade[i] > 0.005 ? this.fade[i] : 0;
    }
  }
}
