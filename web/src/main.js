import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { TileStreamer } from './tiles.js';
import { ProjectorSystem, poseCamera, orientCamera, imageFrustum, setFrustum, cameraFrustum } from './projectors.js';
import { CAM_PROXY, CAM_REFRESH_MS } from './config.js';
import { buildLayer, layerAlpha } from './layers.js';

const $ = id => document.getElementById(id);
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // photos are blended in their native (sRGB) encoding

const SKY = new THREE.Color(0.72, 0.80, 0.88);
const scene = new THREE.Scene();
scene.background = SKY;

const camera = new THREE.PerspectiveCamera(55, 1, 1, 30000);
camera.position.set(-1900, 1300, 2300);
const controls = new MapControls(camera, canvas);
controls.target.set(-200, 0, -300);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 20;
controls.maxDistance = 12000;
controls.screenSpacePanning = false;

const proj = new ProjectorSystem(renderer, scene, SKY);
const tiles = new TileStreamer(scene, proj.materials);
tiles.onChange = () => proj.invalidateDepth();

// Water
const water = new THREE.Mesh(new THREE.PlaneGeometry(60000, 60000), new THREE.ShaderMaterial({
  uniforms: { uColor: { value: new THREE.Color(0.30, 0.42, 0.52) }, uFogColor: proj.uniforms.uFogColor, uFogDensity: proj.uniforms.uFogDensity },
  vertexShader: `varying float vFog; void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vFog = -mv.z; gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `uniform vec3 uColor; uniform vec3 uFogColor; uniform float uFogDensity; varying float vFog;
    void main() { float f = 1.0 - exp(-pow(uFogDensity * vFog, 2.0)); gl_FragColor = vec4(mix(uColor, uFogColor, f), 1.0); }`,
}));
water.rotation.x = -Math.PI / 2;
water.position.y = -0.5;
water.userData.noDepth = true;
scene.add(water);

// ---------------------------------------------------------------- data
const state = { items: [], photos: [], cams: [], selected: null, mode: 'map', showPhotos: true, showCams: true, showLayers: true };
const toThree = p => new THREE.Vector3(p[0], p[2], -p[1]);

function forwardOf(pose) {
  const y = THREE.MathUtils.degToRad(pose[3]), p = THREE.MathUtils.degToRad(pose[4]);
  return new THREE.Vector3(Math.sin(y) * Math.cos(p), Math.sin(p), -Math.cos(y) * Math.cos(p));
}

const texLoader = new THREE.TextureLoader();
texLoader.setCrossOrigin('anonymous');
function loadTexture(url) {
  return new Promise((res, rej) => texLoader.load(url, t => { t.colorSpace = THREE.NoColorSpace; res(t); }, undefined, rej));
}

function makePhoto(p) {
  const item = {
    kind: 'photo', ...p, aspect: p.w / p.h, rect: [0, 0, 1, 1],
    pos: toThree(p.pose), fwd: forwardOf(p.pose),
    async load() {
      this.tex = await loadTexture(p.img);
      if (p.depth) {
        buildLayer(this, this.tex, { uFogColor: proj.uniforms.uFogColor, uFogDensity: proj.uniforms.uFogDensity })
          .then(m => { if (this.slot >= 0 && this.tex) { this.layer = m; scene.add(m); } else { m.geometry.dispose(); m.material.dispose(); } })
          .catch(() => {});
      }
      return this.tex;
    },
    unload() {
      if (this.layer) { scene.remove(this.layer); this.layer.geometry.dispose(); this.layer.material.dispose(); this.layer = null; }
      this.tex?.dispose(); this.tex = null;
    },
  };
  item.look = item.pos.clone().addScaledVector(item.fwd, 40);
  return item;
}

function camUrl(id) { return `${CAM_PROXY}/cam/${id}?t=${Math.floor(Date.now() / CAM_REFRESH_MS)}`; }

function makeCam(c) {
  const item = {
    kind: 'cam', ...c, aspect: c.w / c.h, rect: c.rect || [0, 0, 1, 1],
    pos: toThree(c.pose), fwd: forwardOf(c.pose), title: c.name,
    async load() { this.tex = await loadTexture(camUrl(c.id)); this.lastFrame = performance.now(); return this.tex; },
    unload() { this.tex?.dispose(); this.tex = null; },
    async refresh() {
      if (this.busy) return;
      this.busy = true;
      try {
        const t = await loadTexture(camUrl(c.id));
        if (this.slot >= 0) { proj.drawColor(this.slot, t); this.tex?.dispose(); this.tex = t; } else t.dispose();
      } catch (e) { /* camera offline */ }
      this.busy = false;
      this.lastFrame = performance.now();
    },
  };
  item.look = item.pos.clone().addScaledVector(item.fwd, 40);
  return item;
}

// ---------------------------------------------------------------- markers
let markers;
function buildMarkers() {
  const n = state.items.length;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const cP = new THREE.Color(0xffb454), cC = new THREE.Color(0x4fd1c5);
  state.items.forEach((it, i) => {
    pos.set([it.pos.x, it.pos.y + 1, it.pos.z], i * 3);
    (it.kind === 'cam' ? cC : cP).toArray(col, i * 3);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  markers = new THREE.Points(g, new THREE.PointsMaterial({ size: 7, sizeAttenuation: false, vertexColors: true, depthTest: true }));
  markers.userData.noDepth = true;
  markers.renderOrder = 2;
  scene.add(markers);
}

const frustum = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
frustum.userData.noDepth = true;
frustum.renderOrder = 3;
frustum.visible = false;
scene.add(frustum);
function showFrustum(item) {
  if (!item) { frustum.visible = false; return; }
  const c = poseCamera(item);
  const d = 25, fr = imageFrustum(item);
  const corners = [[fr.l, fr.b], [fr.r, fr.b], [fr.r, fr.t], [fr.l, fr.t]]
    .map(([x, y]) => new THREE.Vector3(x * d, y * d, -d).applyMatrix4(c.matrixWorld));
  const o = c.position;
  const pts = [];
  corners.forEach((p, i) => { pts.push(o, p, p, corners[(i + 1) % 4]); });
  frustum.geometry.setFromPoints(pts);
  frustum.material.color.set(item.kind === 'cam' ? 0x4fd1c5 : 0xffb454);
  frustum.visible = true;
}

// Screen-space picking of markers.
function pick(clientX, clientY, radius = 12) {
  if (!markers?.visible) return null;
  const v = new THREE.Vector3();
  let best = null, bd = radius * radius;
  for (const it of state.items) {
    if (!visibleKind(it)) continue;
    v.copy(it.pos).project(camera);
    if (v.z > 1 || v.z < -1) continue;
    const x = (v.x + 1) / 2 * innerWidth, y = (1 - v.y) / 2 * innerHeight;
    const d = (x - clientX) ** 2 + (y - clientY) ** 2;
    if (d < bd) { bd = d; best = it; }
  }
  return best;
}
const visibleKind = it => (it.kind === 'cam' ? state.showCams : state.showPhotos);

// ---------------------------------------------------------------- projector scheduling
function focusPoint() {
  if (state.mode !== 'map' && state.selected) return state.selected.look;
  return controls.target;
}

function scheduleProjectors() {
  const f = focusPoint();
  const cand = state.items.filter(it => visibleKind(it) && it.look.distanceTo(f) < 900);
  cand.sort((a, b) => a.look.distanceToSquared(f) - b.look.distanceToSquared(f));
  // Keep a mix: up to 10 photos and 6 cams nearest the focus.
  const photos = cand.filter(c => c.kind === 'photo').slice(0, 10);
  const cams = cand.filter(c => c.kind === 'cam').slice(0, 6);
  const wanted = [...photos, ...cams].sort((a, b) => a.look.distanceToSquared(f) - b.look.distanceToSquared(f));
  if (state.selected) {
    const i = wanted.indexOf(state.selected);
    if (i >= 0) wanted.splice(i, 1);
    wanted.unshift(state.selected);
  }
  proj.assign(wanted);
}

function weightOf(it) {
  if (state.selected === it) return 4;
  if (state.mode === 'photo') return 0.25;
  return it.kind === 'cam' ? 1.5 : 1;
}

// ---------------------------------------------------------------- photo mode
function applyFrustum(fr) {
  camera.userData.frustum = fr;
  setFrustum(camera, fr, 1, 30000);
}

const overlay = $('overlay');
let anim = null;

// Window frustum that shows the image letterboxed in the center of the viewport, with its exact intrinsics.
function viewParams(item) {
  const A = innerWidth / innerHeight;
  const c = orientCamera(item.pose, new THREE.PerspectiveCamera());
  const fr = imageFrustum(item);
  const w = A > item.aspect ? innerHeight * item.aspect : innerWidth;
  const h = w / item.aspect;
  const sx = innerWidth / w, sy = innerHeight / h;
  const mx = (fr.l + fr.r) / 2, my = (fr.t + fr.b) / 2, hx = (fr.r - fr.l) / 2 * sx, hy = (fr.t - fr.b) / 2 * sy;
  return { c, frustum: { l: mx - hx, r: mx + hx, t: my + hy, b: my - hy }, w, h };
}

function placeOverlay(item) {
  const { w, h } = viewParams(item);
  Object.assign(overlay.style, { width: `${w}px`, height: `${h}px`, left: `${(innerWidth - w) / 2}px`, top: `${(innerHeight - h) / 2}px` });
}

function enterPhoto(item) {
  state.selected = item;
  state.mode = 'flying';
  controls.enabled = false;
  overlay.style.opacity = 0;
  scheduleProjectors();
  const { c, frustum } = viewParams(item);
  anim = {
    t: 0, dur: Math.min(2.6, 0.9 + camera.position.distanceTo(c.position) / 1500),
    p0: camera.position.clone(), q0: camera.quaternion.clone(), f0: state.mode === 'photo' && camera.userData.frustum || cameraFrustum(camera),
    p1: c.position.clone(), q1: c.quaternion.clone(), f1: frustum,
    lift: Math.min(300, camera.position.distanceTo(c.position) * 0.25),
  };
  showFrustum(null);
  updatePhotobar(item);
  renderStrip(true);
}

function arrive(item) {
  state.mode = 'photo';
  if (item.kind === 'photo') {
    overlay.src = item.img;
    placeOverlay(item);
    overlay.style.opacity = $('pOpacity').value;
  }
}

function exitPhoto() {
  if (state.mode === 'map') return;
  anim = null;
  state.mode = 'map';
  state.selected = null;
  overlay.style.opacity = 0;
  $('photobar').hidden = true;
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const tgt = camera.position.clone().addScaledVector(fwd, 60);
  if (tgt.y < 0) tgt.y = 0;
  controls.target.copy(tgt);
  camera.fov = 55;
  camera.near = 1; camera.far = 30000;
  camera.userData.frustum = null;
  camera.updateProjectionMatrix();
  controls.enabled = true;
  renderStrip(true);
}

function updatePhotobar(item) {
  $('photobar').hidden = false;
  const t = $('pTitle');
  t.textContent = item.kind === 'cam' ? `● LIVE · ${item.name}` : item.title.replace(/^File:/, '').replace(/\.[a-z]+$/i, '');
  t.href = item.kind === 'cam' ? 'https://webcams.nyctmc.org/' : item.url;
  const cr = $('pCredit');
  cr.textContent = '';
  if (item.kind === 'cam') cr.textContent = 'NYC DOT traffic camera · pose solved against the 3D model';
  else {
    cr.append(`${item.artist || 'Unknown author'} · `);
    const a = document.createElement('a');
    a.href = item.license_url || item.url; a.target = '_blank'; a.rel = 'noopener';
    a.textContent = item.license || 'see source';
    cr.append(a, ` · Wikimedia Commons${item.date ? ' · ' + String(item.date).slice(0, 10) : ''}`);
  }
}

function neighbors(item) {
  return state.items.filter(it => it !== item && visibleKind(it))
    .map(it => ({ it, d: it.pos.distanceTo(item.pos) + 30 * (1 - it.fwd.dot(item.fwd)) }))
    .sort((a, b) => a.d - b.d).slice(0, 12).map(x => x.it);
}
function step(dir) {
  if (!state.selected) return;
  // Choose the nearest neighbor to the left/right of the current view direction.
  const cur = state.selected;
  const right = new THREE.Vector3().crossVectors(cur.fwd, new THREE.Vector3(0, 1, 0)).normalize();
  const cands = neighbors(cur).filter(it => Math.sign(it.look.clone().sub(cur.look).dot(right)) === dir);
  const next = cands[0] || neighbors(cur)[0];
  if (next) enterPhoto(next);
}

// ---------------------------------------------------------------- thumbnail strip
let stripFocus = null;
function renderStrip(force) {
  const f = focusPoint();
  if (!force && stripFocus && stripFocus.distanceTo(f) < 60) return;
  stripFocus = f.clone();
  const strip = $('strip');
  const near = state.items.filter(visibleKind).map(it => ({ it, d: it.look.distanceTo(f) })).sort((a, b) => a.d - b.d).slice(0, 24);
  strip.textContent = '';
  for (const { it } of near) {
    const div = document.createElement('div');
    div.className = 'th' + (it.kind === 'cam' ? ' cam' : '') + (it === state.selected ? ' sel' : '');
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = it.kind === 'cam' ? camUrl(it.id) : it.thumb;
    img.alt = it.title || it.name;
    div.append(img);
    if (it.kind === 'cam') { const l = document.createElement('span'); l.className = 'live'; l.textContent = 'LIVE'; div.append(l); }
    div.onclick = () => enterPhoto(it);
    div.onmouseenter = () => state.mode === 'map' && showFrustum(it);
    div.onmouseleave = () => state.mode === 'map' && showFrustum(null);
    strip.append(div);
  }
}

// ---------------------------------------------------------------- input
const tooltip = $('tooltip');
let down = null;
canvas.addEventListener('pointerdown', e => {
  down = { x: e.clientX, y: e.clientY };
  if (state.mode !== 'map') exitPhoto();
});
canvas.addEventListener('pointerup', e => {
  if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
  const it = pick(e.clientX, e.clientY);
  if (it) enterPhoto(it);
});
canvas.addEventListener('pointermove', e => {
  if (state.mode !== 'map' || e.buttons) { tooltip.hidden = true; return; }
  const it = pick(e.clientX, e.clientY);
  showFrustum(it);
  if (!it) { tooltip.hidden = true; canvas.style.cursor = ''; return; }
  canvas.style.cursor = 'pointer';
  tooltip.hidden = false;
  tooltip.querySelector('img').src = it.kind === 'cam' ? camUrl(it.id) : it.thumb;
  tooltip.querySelector('.t').textContent = it.kind === 'cam' ? `LIVE · ${it.name}` : it.title.replace(/^File:/, '');
  tooltip.style.left = `${Math.min(e.clientX + 14, innerWidth - 230)}px`;
  tooltip.style.top = `${Math.min(e.clientY + 14, innerHeight - 200)}px`;
});
addEventListener('keydown', e => {
  if (e.key === 'Escape') exitPhoto();
  if (e.key === 'ArrowRight') step(1);
  if (e.key === 'ArrowLeft') step(-1);
});
$('pExit').onclick = exitPhoto;
$('pNext').onclick = () => step(1);
$('pPrev').onclick = () => step(-1);
$('pOpacity').oninput = e => { if (state.mode === 'photo') overlay.style.opacity = e.target.value; };
$('tPhotos').onchange = e => { state.showPhotos = e.target.checked; renderStrip(true); };
$('tCams').onchange = e => { state.showCams = e.target.checked; renderStrip(true); };
$('tLayers').onchange = e => { state.showLayers = e.target.checked; };
$('tMarkers').onchange = e => { if (markers) markers.visible = e.target.checked; };

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  if (state.mode === 'photo' && state.selected) { applyFrustum(viewParams(state.selected).frustum); placeOverlay(state.selected); }
}
addEventListener('resize', resize);
resize();

// ---------------------------------------------------------------- deep links
// #p=<photo or camera id> while viewing an image; #v=camX,camY,camZ,targetX,targetY,targetZ (meters) on the map.
let lastHash = location.hash;
function writeHash() {
  if (!state.ready || state.mode === 'flying') return;
  let h;
  if (state.mode === 'photo' && state.selected) h = `#p=${state.selected.id}`;
  else h = '#v=' + [...camera.position.toArray(), ...controls.target.toArray()].map(v => Math.round(v)).join(',');
  if (h !== lastHash) { lastHash = h; history.replaceState(null, '', h); }
}

function applyHash(initial) {
  lastHash = location.hash;
  const h = decodeURIComponent(location.hash.slice(1));
  const params = new URLSearchParams(h.includes('=') ? h : `p=${h}`);
  const id = params.get('p');
  if (id) {
    const it = state.items.find(x => String(x.id) === id);
    if (it) { enterPhoto(it); if (initial) { anim.t = 1; } }
    return;
  }
  const v = (params.get('v') || '').split(',').map(Number);
  if (v.length === 6 && v.every(Number.isFinite)) {
    exitPhoto();
    camera.position.set(v[0], v[1], v[2]);
    controls.target.set(v[3], v[4], v[5]);
    controls.update();
  }
}

// ---------------------------------------------------------------- loop
const clock = new THREE.Clock();
let schedT = 0;
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (anim) {
    anim.t += dt / anim.dur;
    const k = Math.min(anim.t, 1), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    camera.position.lerpVectors(anim.p0, anim.p1, e);
    camera.position.y += Math.sin(Math.PI * e) * anim.lift;
    camera.quaternion.slerpQuaternions(anim.q0, anim.q1, e);
    const L = k => anim.f0[k] + (anim.f1[k] - anim.f0[k]) * e;
    applyFrustum({ l: L('l'), r: L('r'), t: L('t'), b: L('b') });
    if (k >= 1) { const it = state.selected; anim = null; arrive(it); }
  } else if (state.mode === 'map') {
    controls.update();
  }
  schedT -= dt;
  if (schedT <= 0) {
    schedT = 0.5;
    tiles.update(focusPoint());
    scheduleProjectors();
    if (state.mode === 'map') renderStrip(false);
    writeHash();
    const st = tiles.loading ? `streaming tiles (${tiles.loading})…` : '';
    $('status').textContent = st;
  }
  const now = performance.now();
  for (const s of proj.slots) if (s?.kind === 'cam' && s.ready && now - s.lastFrame > CAM_REFRESH_MS) s.refresh();
  proj.update(dt, weightOf);
  for (const s of proj.slots) {
    if (!s?.layer) continue;
    const target = !state.showLayers || !state.showPhotos ? 0
      : state.mode !== 'map' && state.selected === s ? 1 : state.mode === 'map' ? layerAlpha(s, camera) : 0;
    const u = s.layer.material.uniforms.uAlpha;
    u.value += (target - u.value) * Math.min(1, dt * 4);
    s.layer.visible = u.value > 0.01;
  }
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

async function main() {
  $('status').textContent = 'loading city…';
  const [photos, cams] = await Promise.all([
    fetch('data/photos.json').then(r => r.json()).catch(() => []),
    fetch('data/cams.json').then(r => r.json()).catch(() => []),
    tiles.init(),
  ]);
  state.photos = photos.map(makePhoto);
  state.cams = cams.map(makeCam);
  state.items = [...state.photos, ...state.cams];
  $('nphotos').textContent = state.photos.length.toLocaleString();
  $('ncams').textContent = state.cams.length.toLocaleString();
  buildMarkers();
  applyHash(true);
  state.ready = true;
  addEventListener('hashchange', () => { if (location.hash !== lastHash) applyHash(false); });
  renderStrip(true);
}

window.cityviz = { state, proj, tiles, camera, renderer, controls, enterPhoto, exitPhoto };
requestAnimationFrame(frame);
main();
