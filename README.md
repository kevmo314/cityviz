# CityViz NYC

A 3D model of Manhattan painted with geo-registered photographs and live traffic cameras, in the spirit of
Photo Tourism / Photosynth. Every photo and camera is placed at a solved 7-DoF pose (position, yaw, pitch, roll,
field of view) and projected onto the buildings, so the imagery "paints" the city.

**Live:** https://kevmo314.github.io/cityviz/

## How it works

```
NYC building footprints ──► extrude + terrain ──► streamed binary tiles ─────────────┐
(NYC Open Data, roof heights)       │                                                 │
                                    ▼                                                 ▼
Wikimedia Commons photos ──► SegFormer sky/ground/building ──► pose search ──► three.js viewer
(camera GPS, compass, EXIF)         ▲                     against ray-cast      (projective texturing
NYC DOT traffic cams ───────────────┘                     renders of the model   with per-projector
(multi-frame daylight median)                                                     depth maps)
```

1. **City model** (`pipeline/build_city.py`): ~45k Manhattan footprints with roof heights are extruded on a
   terrain surface interpolated from building ground elevations. Output is 500 m tiles (full detail) and 2 km
   blocks (tall buildings only), quantized to int16. The viewer streams full-detail blocks around the view.
2. **Photos** (`pipeline/fetch_commons.py`): geotagged Commons files whose coordinates are the *camera*
   location, with compass heading (from `{{Location|…|heading:}}` or EXIF `GPSImgDirection`) and focal length.
3. **Localization** (`pipeline/localize.py`): GPS and compass are only accurate to tens of meters and
   degrees, which is not enough to paint with. Each image is segmented into sky / ground / building, and the
   pose is solved by matching that against the model:
   - Coarse: for a grid of candidate positions, ray-cast one label panorama, then score every
     (yaw, pitch, FOV) by resampling it into the image frame.
   - Fine: Nelder-Mead over all 7 DoF with direct ray-cast renders, maximizing class IoU.
   Only confident, unambiguous solutions are kept.
4. **Traffic cams** (`pipeline/localize_cams.py`): cameras are static, so each one is solved once offline.
   Class probabilities are averaged over several daylight frames so traffic washes out, and the full
   360° of yaw is searched. The camera's burned-in "Facing X" label (OCR) is used only to break ties
   between near-equal hypotheses.
5. **Viewer** (`web/`): plain ES modules + three.js, no build step. Up to 16 projectors are active at once
   (texture arrays for color and per-projector distance maps for occlusion). Live cameras refresh every
   3 s through a tiny caching CORS proxy (`server/`, Cloud Run) because NYC DOT sends no CORS headers.

## Running the pipeline

```sh
uv venv .venv && uv pip install --python .venv numpy scipy shapely pyproj trimesh mapbox_earcut embreex \
  pillow requests torch transformers opencv-python-headless rapidocr-onnxruntime
.venv/bin/python pipeline/fetch_buildings.py
.venv/bin/python pipeline/build_city.py
.venv/bin/python pipeline/fetch_commons.py
.venv/bin/python pipeline/localize_photos.py 3000
.venv/bin/python pipeline/capture_cams.py      # leave running through daylight
.venv/bin/python pipeline/localize_cams.py
.venv/bin/python pipeline/export_web.py
cd web && python3 -m http.server
```

## Credits

Photos: Wikimedia Commons contributors, under the licenses linked in the viewer. Building footprints and
borough boundaries: NYC Open Data. Traffic cameras: NYC DOT.
