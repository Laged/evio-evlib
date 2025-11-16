# Plan – Drone Detector Migration to evlib (Legacy HDF5 → evlib → Plugin)

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Context:** `drone-example-detector.py` (legacy loader) + architecture in `docs/architecture.md` and the fan detector migration plan.

---

## Current Drone Detector (what it does)
- **Two-pass pipeline** similar structure to the fan script but tuned for dual propellers:
  - **Pass 1 (coarse ~30 ms):** accumulate events → threshold/morph → find *all* prop-like contours (not just largest) → filter by area/orientation → fit up to two ellipses → store timestamps + ellipse params.
  - **Pass 2 (fine ~0.5–2 ms):** for each time window, pick closest prop ellipses from Pass 1, filter events in an elliptical ring, run DBSCAN to find blade clusters, compute per-prop angles, unwrap and estimate per-frame and running RPM. Adds HUD (“WARNING: DRONE DETECTED”, RPM text).
- **Visualization:** stacked views (events + overlay), CV/DBSCAN overlays, optional RPM readouts.
- **Dependencies:** numpy, OpenCV, scikit-learn (DBSCAN), matplotlib; legacy `DatFileSource` with manual bit-unpack; hardcoded 1280×720.

---

## Differences vs Fan Detector
- Supports **multiple propellers**: Pass 1 detects up to `top_k=2` ellipses; Pass 2 maintains angle histories per prop.
- Uses **orientation filtering** (reject near-vertical ellipses) and area-based filtering per contour, instead of just taking the largest blob.
- Dual warning/RPM overlays (per prop and running mean).
- Same legacy loader and windowing assumptions; same manual accumulation/decoding.

---

## Migration Goals (align with fan plan)
1. **Shared evlib ingestion**: load `*_legacy.h5` (converted legacy `.dat`) via `evlib.load_events`, infer resolution from data.
2. **Shared windowing/rendering**: use Polars filters per window; optionally use evlib representations (stacked histogram/timesurface) for accumulation.
3. **Plugin-ize**: expose a Plugin/Detector API so fan/drone detectors run as “external plugins” sharing the same evlib/HDF5 pipeline.
4. **Data selector UX**: allow choosing fan vs drone datasets, with looped playback and overlays; standardize the HDF5 source per selector.

---

## Implementation Steps

1) **Swap loader/windowing to evlib**
   - Input: `evio/data/drone_idle_legacy.h5` (from `convert-legacy-dat-to-hdf5`).
   - Load via `evlib.load_events`, compute `width/height` from `x.max()+1`, `y.max()+1`.
   - Replace `DatFileSource.ranges()` with Polars time filtering (`(t >= t0) & (t < t1)`) for coarse/fine windows.

2) **Acceleration (optional, mirrored from fan plan)**
   - Pass 1 accumulation: use `evlib.representations.create_stacked_histogram` (single bin) instead of manual per-pixel counts.
   - ROI filtering: use Polars masks instead of numpy to prefilter ring points before DBSCAN.

3) **Keep problem-specific logic**
   - Ellipse fits, orientation/area filtering for multiple props; DBSCAN clustering; RPM estimation with angle unwrapping. These remain as custom code.

4) **Wire as plugin**
   - Create a drone plugin under `workspace/plugins/drone-detector` (or similar) exposing a `process(events)` function returning RPM/warnings plus overlay metadata.
   - Align with fan plugin API so both can be launched by the same runner/UI.

5) **Data selector & demo loop**
   - Implement a simple selector (CLI flag or minimal UI) to pick dataset: `fan_const_rpm_legacy.h5`, `drone_idle_legacy.h5`, etc.
   - Use a shared “player” harness (evlib loader + window iterator) to feed either plugin and render looped playback with overlays.

---

## Next Steps for RVT Alignment
- After evlib migration, add an RVT smoke path: take the same evlib events, generate RVT inputs (voxel/time surface), and validate tensor creation for a short clip of the drone dataset.
- Keep RVT hook optional/skipped in CI; document commands once ready.

---

## Commands to standardize (post-migration)
- Prepare data: `convert-legacy-dat-to-hdf5 evio/data/drone_idle.dat` (and other drone clips).
- Demos: extend `run-demo-*` or add `run-demo-drone-ev3` pointing at the HDF5 and evlib-based player.
- Tests: mirror the parity tests if needed (legacy → HDF5 → evlib) for drone datasets.

This keeps the drone detector in sync with the fan migration: one evlib/HDF5 ingestion path, interchangeable plugins, and a data selector for the right recordings.***
