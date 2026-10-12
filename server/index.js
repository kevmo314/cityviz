// Minimal CORS proxy for NYC DOT traffic camera snapshots.
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
