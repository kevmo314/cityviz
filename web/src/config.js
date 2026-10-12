// Live camera frames come through a tiny CORS proxy (server/) because NYC DOT does not send CORS headers.
export const CAM_PROXY = 'https://cityviz-camproxy-272419586516.us-east4.run.app';
export const CAM_REFRESH_MS = 3000;
// Heavy static assets (photos, depth layers, city tiles, aerial imagery) live in a GCS bucket served through the same
// Cloud Run service (/a/...); a local dev server serves them from web/ directly.
export const ASSET_BASE = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) ? '' : `${CAM_PROXY}/a/`;
