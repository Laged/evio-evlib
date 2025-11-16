# 09 – Drone & Fan Detection Pipelines (“Irene’s Magic”)

This explains how the fan RPM and drone propeller detectors work today, from the legacy scripts to the evlib/Polars demos and the `run-mvp-demo` UI.

## Data prep (inputs)
- Legacy `.dat` (custom, not EVT3) must be converted to `_legacy.h5` via `convert-legacy-dat-to-hdf5` / `convert-all-legacy-to-hdf5` (see `08_data_conversion.md`).
- EVT3 `.raw` (IDS) are different recordings; evlib can load them directly but they’re not parity with legacy `.dat`.

## Fan RPM pipeline
Applies in `evio/scripts/fan_detector_demo.py` (legacy) and `workspace/tools/fan-rpm-demo` (evlib/Polars).
1) Window events (e.g., 30 ms): legacy uses numpy; evlib version uses Polars (`detector_commons.get_window_evlib`).
2) Accumulate frame: event counts per pixel (legacy `build_accum_frame`; evlib `build_accum_frame_evlib`), clip to uint8 for viz.
3) Ellipse fit (Pass 1): threshold/morphology → largest contour → ellipse `(cx, cy, a, b, phi)` with prior fallback.
4) Ring filter + DBSCAN (Pass 2): select events near the ellipse radius, cluster blades (`cluster_blades_dbscan_elliptic`).
5) Angle tracking → RPM: unwrap angles over time, regress angle vs time → RPM; draw ellipse/center/clusters.

Noise handling: geometry gating + density clustering suppress background (trees/birds/airplanes); two-pass keeps shape stable and motion sharp.

## Drone propeller pipeline (multi-ellipse)
Applies in `evio/scripts/drone_detector_demo.py` (legacy) and `workspace/tools/drone-detector-demo` (evlib/Polars).
1) Window events: coarse (e.g., 30 ms) and fine (e.g., 0.5 ms).
2) Accumulate + threshold: normalize, threshold (tunable pre-threshold) to get blobs.
3) Multi-ellipse detect (Pass 1): up to N ellipses (`propeller_mask_from_frame`), store per-window geometry; optional overlay.
4) Per-propeller clustering (Pass 2): pick nearest ellipses in time (`pick_propellers_at_time`), ring-filter events, DBSCAN per propeller.
5) Per-propeller RPM: track a blade per prop, unwrap angles, fit slope → RPM; draw centers/ellipses/clusters.

## Legacy vs evlib implementations
- Legacy (`evio/scripts/*_detector_demo.py`): custom `.dat` loader, numpy filtering; slower and tied to proprietary `.dat`.
- evlib/Polars (`workspace/tools/fan-rpm-demo`, `workspace/tools/drone-detector-demo`): evlib loading, Polars windowing (Duration vs int64), shared helpers (`detector_commons`), format-flexible; driven via flake aliases (`run-fan-rpm-demo`, `run-drone-detector-demo`).
- UI: `run-mvp-demo` wraps detectors/datasets into an OpenCV menu; still uses legacy script paths, but detection logic matches the evlib pipelines.

## Key tunables
- Window sizes: coarse for ellipse fit; fine for blade clustering.
- DBSCAN: `eps`, `min_samples`, ring bounds (`r_min`, `r_max`).
- Thresholds: pre-threshold and Otsu/blur for stability.
- Geometry smoothing: previous-ellipse fallback to avoid jitter on weak contours.

## Outputs
- Overlays: ellipse(s), centers, clusters; RPM text/print.
- For storage/ML (future): `{t, cx, cy, a, b, phi, rpm, prop_idx}` plus event stats; storage sink not implemented yet (see `03_architecture.md` TODO).

## Commands
```bash
nix develop --command unzip-datasets           # or download-datasets
nix develop --command convert-all-legacy-to-hdf5
nix develop --command run-mvp-demo
nix develop --command run-fan-rpm-demo evio/data/fan/fan_const_rpm_legacy.h5
nix develop --command run-drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5
```
