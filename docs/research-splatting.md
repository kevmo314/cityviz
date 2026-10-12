# Beyond projective texturing: city-scale splats, world models, and live layers

Research notes (Oct 2026) on moving CityViz from photos projected onto LOD1 boxes toward a learned
representation (Gaussian splats / neural fields, plus generative completion) that still supports a live layer
from traffic cameras. Items marked **[unverified]** could not be confirmed; **[estimate]** numbers are ours.

## TL;DR

- City-scale 3DGS methods (Hierarchical 3DGS, CityGaussian v2, VastGaussian, Octree-GS, DOGS) all assume
  dense capture: hundreds to thousands of images per block. Our Commons coverage is far below that except
  at landmarks.
- In-the-wild appearance and transient handling is solved enough. WildGaussians bakes back to standard 3DGS.
  **Sparsity is the real problem**, so geometry has to come from outside: NYC LiDAR, LOD2 meshes, and
  feed-forward depth models conditioned on our solved cameras.
- Web: **Spark (sparkjs) 2.x** is the best fit for three.js. It has a paged LoD splat format (`.rad`), loads
  SPZ/SOG/PLY, and splats depth-test against opaque meshes (but do not write depth).
- Live: no dynamic/4D GS method works from fixed low-res cameras in real time. The practical answer is a
  **static learned background plus a live projective/compositional layer**, with optional per-camera affine
  appearance updates and detected-actor proxies.
- Generative world models (Marble, Lyra, GEN3C, Cosmos-Transfer, Difix3D+) can fill gaps but are not metric or
  georeferenced on their own. Use them only conditioned on our scaffold, and mark synthesized content as such.

## Recommended architecture (tiers)

| Tier | Where | What | Source of truth |
|---|---|---|---|
| Scaffold | everywhere | LOD2 buildings (TUM NYC CityGML) + LiDAR depth proxy | NYC Open Data, USGS 3DEP EPT |
| A: real splats | dense blocks (≳30 photos **[estimate]**) | WildGaussians-style splats, LiDAR init + depth loss | photos |
| B: 2.5D per photo | 1–30 photos | MapAnything / Depth Anything 3 depth with our solved cameras, aligned to LiDAR, textured from the photo | photos |
| C: synthesized | gaps | GEN3C / Cosmos-Transfer conditioned on scaffold depth + nearby photos, distilled to splats, Difix cleanup; flagged and toggleable | generative |
| D: live | traffic cams | background plate + foreground projection, per-camera appearance transform, detected-actor proxies | live frames |

## Staged plan

1. **Scaffold upgrade (no GPU)**: swap LOD1 for LOD2. Crop LiDAR from `s3://usgs-lidar-public/NY_NewYorkCity/`
   with PDAL. Use it as the projection and occlusion proxy. Add the traffic-cam background plate and
   foreground layer.
2. **Tier B**: per-photo dense depth from our solved cameras (MapAnything-Apache or DA3-Base/Small for
   permissive licences), scale/shift aligned to the LiDAR, streamed as photo splats or small meshes.
   About 1–3 s per image on one GPU **[estimate]**.
3. **Tier A** on 5–20 landmark blocks: gsplat/WildGaussians, LiDAR init, export SPZ, then Spark `build-lod --rad-chunked`.
   About 0.5–2 GPU-hours per 100 m block; about 100–400 MB/km² **[estimate]**.
4. **Live appearance + continual retraining**: per-camera affine color fits applied via Spark `SplatEdit`/`dyno`;
   re-train only the affected block from checkpoint when new photos arrive.
5. **Tier C experiments**: generative fill, visibly marked as synthesized.

## Key references

### City-scale splats
- Hierarchical 3DGS: [2406.12080](https://arxiv.org/abs/2406.12080) ([code](https://github.com/graphdeco-inria/hierarchical-3d-gaussians), research licence)
- CityGaussianV2: [2411.00771](https://arxiv.org/abs/2411.00771)
- VastGaussian: [2402.17427](https://arxiv.org/abs/2402.17427)
- Octree-GS: [2403.17898](https://arxiv.org/abs/2403.17898)
- DOGS: [2405.13943](https://arxiv.org/abs/2405.13943)
- BlitzGS: [2605.13794](https://arxiv.org/abs/2605.13794) (self-reported numbers)

### In the wild
- WildGaussians: [2407.08447](https://arxiv.org/abs/2407.08447)
- Splatfacto-W: [2407.12306](https://arxiv.org/abs/2407.12306)
- GS-W: [2403.15704](https://arxiv.org/abs/2403.15704)
- SWAG: [2403.10427](https://arxiv.org/abs/2403.10427)
- RobustSplat++: [2512.04815](https://arxiv.org/abs/2512.04815)
- SparseGS-W: [2503.19452](https://arxiv.org/abs/2503.19452)

### Geometry
- NYC Topobathymetric LiDAR 2017 (≥8 pts/m²): [data.gov](https://catalog.data.gov/dataset/topobathymetric-lidar-data-2017)
- USGS EPT copy: [ept.json](https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_NewYorkCity/ept.json) (year inferred as 2017 **[unverified]**)
- TUM NYC CityGML LOD2: [GitHub](https://github.com/tum-gis/3d-model-new-york-city)

### Feed-forward depth
- MapAnything (Apache, accepts known intrinsics and poses): [repo](https://github.com/facebookresearch/map-anything)
- Depth Anything 3 (Base/Small Apache): [repo](https://github.com/ByteDance-Seed/Depth-Anything-3)
- VGGT (commercial checkpoint available): [repo](https://github.com/facebookresearch/vggt)
- DUSt3R/MASt3R are non-commercial.

### Web delivery
- Spark: [repo](https://github.com/sparkjsdev/spark) (MIT; RAD paged LoD; depthTest on, depthWrite off)
- PlayCanvas streamed SOG: [docs](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/)
- SPZ: [repo](https://github.com/nianticlabs/spz)
- glTF `KHR_gaussian_splatting` (release candidate, not ratified): [Khronos](https://www.khronos.org/news/permalink/khronos-announces-gltf-gaussian-splatting-extension)

### Dynamic / continual
- OmniRe: [2408.16760](https://arxiv.org/abs/2408.16760)
- Street Gaussians: [2401.01339](https://arxiv.org/abs/2401.01339)
- StreamSplat: [2506.08862](https://arxiv.org/abs/2506.08862)
- CL-Splats: [2506.21117](https://arxiv.org/abs/2506.21117)
- GaussianUpdate: [2508.08867](https://arxiv.org/abs/2508.08867)

### Generative
- World Labs Marble: [blog](https://www.worldlabs.ai/blog/marble-world-model) (closed, SPZ export)
- NVIDIA Lyra: [repo](https://github.com/nv-tlabs/lyra)
- GEN3C: [HF](https://huggingface.co/nvidia/GEN3C-Cosmos-7B)
- Cosmos Predict 2.5: [repo](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- Difix3D+: [2503.01774](https://arxiv.org/abs/2503.01774)
- Skyfall-GS: [2510.15869](https://arxiv.org/abs/2510.15869)
- CityDreamer4D: [2501.08983](https://arxiv.org/abs/2501.08983)

## Risks

- Commons photos are mostly CC BY-SA. A splat trained to reproduce them is plausibly an adaptation, so ship
  splat tiles CC BY-SA with per-tile attribution manifests (judgement, not legal advice).
- NYC DOT camera reuse terms are unclear.
- Model licences vary: DUSt3R/MASt3R and the large DA3 checkpoints are non-commercial, and the NVIDIA OML has
  guardrail clauses.
- Photos span years. Storefronts and buildings change, so prefer recent photos and use appearance embeddings.
- Sparse blocks produce floaters. Gate Tier A per block on coverage metrics and fall back to Tier B or the scaffold.
- The glTF splat extension is still a release candidate, so prefer Spark-native RAD for now.
