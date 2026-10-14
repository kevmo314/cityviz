// Minimal CORS proxy for NYC DOT traffic camera snapshots, plus read-only access to the site's static assets in GCS.
// Only known camera ids are served, and each frame is cached briefly so any number of viewers costs one upstream fetch.
import http from 'node:http';

const UPSTREAM = 'https://webcams.nyctmc.org/api/cameras';
const TTL_MS = 2500;
const ID_RE = /^\/cam\/([0-9a-f-]{36})$/;

let known = new Set();
async function refreshList() {
  try {
    const r = await fetch(UPSTREAM);
    if (r.ok) known = new Set((await r.json()).map(c => c.id));
  } catch (e) { console.error('camera list', e.message); }
}
await refreshList();
setInterval(refreshList, 3600_000);

// --- static assets: /a/<path> -> gs://$BUCKET/site/<path>, using the service account's metadata-server token
const BUCKET = process.env.BUCKET || 'kevmo314-cityviz';
const ASSET_RE = /^\/a\/((?:photos|depth|ortho|tiles|splats)\/[A-Za-z0-9_\-./]+)$/;
const TYPES = { jpg: 'image/jpeg', json: 'application/json', bin: 'application/octet-stream', splat: 'application/octet-stream' };
let token = { value: null, exp: 0 };
async function accessToken() {
  if (token.value && Date.now() < token.exp) return token.value;
  const r = await fetch('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
    { headers: { 'Metadata-Flavor': 'Google' } });
  const j = await r.json();
  token = { value: j.access_token, exp: Date.now() + (j.expires_in - 60) * 1000 };
  return token.value;
}
async function asset(path, res) {
  const r = await fetch(`https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${encodeURIComponent('site/' + path)}?alt=media`,
    { headers: { Authorization: `Bearer ${await accessToken()}` } });
  if (!r.ok) { res.writeHead(r.status === 404 ? 404 : 502); res.end(); return; }
  const ext = path.split('.').pop();
  res.writeHead(200, {
    'Content-Type': TYPES[ext] || 'application/octet-stream',
    'Cache-Control': ext === 'json' ? 'public, max-age=300' : 'public, max-age=86400',
    ...(r.headers.get('content-length') ? { 'Content-Length': r.headers.get('content-length') } : {}),
  });
  for await (const chunk of r.body) res.write(chunk);
  res.end();
}

const cache = new Map(); // id -> {t, buf, type, pending}
async function frame(id) {
  const c = cache.get(id);
  if (c?.buf && Date.now() - c.t < TTL_MS) return c;
  if (c?.pending) return c.pending;
  const pending = (async () => {
    const r = await fetch(`${UPSTREAM}/${id}/image`, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) throw new Error(`upstream ${r.status}`);
    const entry = { t: Date.now(), buf: Buffer.from(await r.arrayBuffer()), type: r.headers.get('content-type') || 'image/jpeg' };
    cache.set(id, entry);
    return entry;
  })();
  cache.set(id, { ...(c || {}), pending });
  try { return await pending; } catch (e) { if (c?.buf) { cache.set(id, c); return c; } cache.delete(id); throw e; }
}

http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const url = new URL(req.url, 'http://x');
  if (url.pathname === '/healthz') { res.end('ok'); return; }
  const a = ASSET_RE.exec(url.pathname);
  if (a && req.method === 'GET' && !a[1].includes('..')) {
    try { await asset(a[1], res); } catch (e) { if (!res.headersSent) res.writeHead(502); res.end(); }
    return;
  }
  const m = ID_RE.exec(url.pathname);
  if (req.method !== 'GET' || !m || !known.has(m[1])) { res.writeHead(404); res.end(); return; }
  try {
    const f = await frame(m[1]);
    res.writeHead(200, { 'Content-Type': f.type, 'Cache-Control': 'public, max-age=2', 'Content-Length': f.buf.length });
    res.end(f.buf);
  } catch (e) {
    res.writeHead(502); res.end();
  }
}).listen(process.env.PORT || 8080);

// Drop stale frames so memory stays bounded.
setInterval(() => { const now = Date.now(); for (const [k, v] of cache) if (!v.pending && now - v.t > 60_000) cache.delete(k); }, 30_000);
