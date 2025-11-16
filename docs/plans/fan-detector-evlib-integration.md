# Plan – Fan Detector Migration to evlib (Legacy HDF5 → evlib → RVT)

**Owner:** Codex (handoff for Claude)  
**Date:** 2025-11-16  
**Context:** `fan-example-detector.py` (legacy loader) + review in `docs/plans/2025-11-16-fan-detector-evlib-integration-review.md`.

---

## TL;DR
- `fan-example-detector.py` is a working ellipse + blade tracker built on the legacy `DatFileSource` and manual event decoding.
- The `.raw → _evt3.dat` route is deprecated (IDS recordings differ); the canonical evlib input is now `*_legacy.h5` produced by `convert-legacy-dat-to-hdf5`.
- We can adapt the detector to evlib by swapping the loader/windowing to `evlib.load_events` + Polars filtering, keeping the OpenCV/DBSCAN/RPM logic intact.
- Target landing spot: a workspace plugin (e.g., `workspace/plugins/fan-rpm`) using evlib ingestion so it fits the architecture.

---

## What the Detector Does (quick recap)
- **Pass 1 (coarse window):** accumulate events → threshold/morph → `cv2.fitEllipse` on largest contour → center/axes/angle.
- **Pass 2 (fine window):** using that ellipse, filter events in an elliptical ring → DBSCAN to find blade clusters → track blade angle over time → unwrap/fit for RPM.
- **Visualization:** overlays ellipse/centers; optional matplotlib plots of angle/RPM.
- **Deps:** numpy, cv2, sklearn (DBSCAN), matplotlib; legacy `DatFileSource` (bit-unpack of event words).

---

## Integration Plan (evlib)

1) **Switch ingestion to evlib HDF5**
   - Input: `evio/data/fan/fan_const_rpm_legacy.h5` (or other `_legacy.h5`).
   - Use `evlib.load_events(...)` → Polars LazyFrame; infer `width/height` from `x.max()+1`, `y.max()+1`.

2) **Windowing via Polars (no DatFileSource)**
   - Replace `time_order` slicing with Polars filters per window:
     ```python
     events_window = events.filter(
         (pl.col("t") >= t0) & (pl.col("t") < t1)
     ).collect()
     ```
   - Keep window durations the same as the legacy script (coarse vs fine).

3) **Representations (optional speedup)**
   - Pass 1: use `evlib.representations.create_stacked_histogram` (single bin) instead of manual accumulation for a ~50x speed boost.
   - Pass 2 filtering: use Polars masks (as above) instead of numpy bit slicing.

4) **Keep geometry/DBSCAN logic**
   - `cv2.fitEllipse`, DBSCAN, angle unwrapping remain unchanged (no evlib equivalent).

5) **Plugin-ize**
   - New workspace plugin (e.g., `workspace/plugins/fan-rpm`) with a `DetectorPlugin`-style API:
     ```python
     def process(events: pl.DataFrame) -> dict:
         return {"rpm": rpm, "omega": omega, "ellipse": (cx, cy, a, b, phi)}
     ```
   - Wire into existing app/dash UI if needed.

6) **Demos**
   - Update/add a CLI: `uv run --package evio python evio/scripts/play_evlib.py evio/data/fan/fan_const_rpm_legacy.h5 --window-ms ...` (or a new `fan_evlib_demo.py`) to show the ellipse/blade overlay powered by evlib.

---

## Benefits vs Legacy Path
- evlib handles ingestion, format detection, lazy scanning.
- Faster accumulation/ROI filtering using Polars/evlib representations.
- Works on the real legacy data (`*_legacy.h5`) with larger windows if needed.
- Aligns with RVT prep: same evlib events can feed RVT preprocessing later.

---

## RVT Next Steps (after evlib migration)
1. Reuse the evlib-loaded HDF5 to compute RVT inputs (voxel grid/time surface) for a short clip.
2. Add a smoke test that runs RVT preprocessing on `fan_const_rpm_legacy.h5` and logs tensor shapes/timing.
3. Optionally compare a small batch of tensors between legacy path (if any) and evlib path to ensure consistency.

---

## Commands (current state)
- Prepare data: `convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat`
- Demos: `run-demo-fan` (legacy) / `run-demo-fan-ev3` (evlib on HDF5)
- Tests: `run-evlib-tests` (legacy → HDF5 → evlib parity)

Once migrated, add an evlib-based detector demo (ellipse/blade overlay) to replace `fan-example-detector.py`’s legacy loader.
