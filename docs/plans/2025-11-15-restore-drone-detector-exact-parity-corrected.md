# Restore Drone Detector Parity (Corrected Plan)

**Why a corrected version?** The original 2025-11-15 plan pointed at files/lines that no longer match the current implementation. The active demo is `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py` (evlib/HDF5), not the legacy script. Also note: the original `example-drone-original.py` scaled the stacked view to 70%, so keep that scale unless you intentionally expose a toggle.

---

## Current implementation (run-drone-detector-demo)
- Entry: `uv run drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5`
- Files: `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py` and `geometry.py` (uses `detector_commons` helpers: `load_legacy_h5`, `get_window_evlib`, `build_accum_frame_evlib`, etc.).
- Notable deviations from the legacy/original script:
  - Pass 2 view downscaled (`cv2.resize` to 70%) — matches original; keep or gate via flag.
  - Pass 1 uses uint16 accumulation and relaxed thresholds: `pre_threshold=50`, `min_area=100`, min contour points likely lower.
  - DBSCAN params for clusters are hardcoded (eps=5, min_samples=15) instead of using CLI defaults.
  - HUD labels/placement differ (top-left, “Frame/Global mean”).

---

## Corrected Fix Plan

1) **Rendering scale**
   - The original uses 70% scaling. Leave the resize in place for parity; optionally gate it via a CLI flag if you want to experiment.

2) **Accumulation mode**
   - Use uint8 clipping for pass 1 accumulation (matches original behavior that normalizes into 0–255).
   - Ensure pass 2 already uses `clip_to_uint8=True` (it does).

3) **Detection parameters (propeller_mask_from_frame)**
   - Raise `pre_threshold` back toward the original (~250) to suppress noise.
   - Raise `min_area` to ~145 (original) and require ~50 contour points for fitEllipse to match prior accuracy.
   - Remove/avoid any aggressive filtering that’s not in the original (e.g., hot-pixel logic).

4) **DBSCAN tuning**
   - Either use the CLI params (`--dbscan-eps`, `--dbscan-min-samples`) instead of hardcoded 5/15, or align defaults with the original script’s values.
   - Keep `r_min/r_max` consistent with the original ring filter (0.8–1.2 seems fine).

5) **HUD/UI parity**
   - Restore labels/placement closer to the original (RPM/Avg RPM, bottom-right overlay) and avoid covering detections.

6) **Windowing defaults**
   - Keep coarse window ~30 ms, fine window ~0.5–2 ms; allow CLI overrides.
   - Confirm width/height from HDF5 (1280×720) or infer from data.

---

## Acceptance checks
- Full-size view (no unintended downscale).
- Propeller detection rate comparable to the original (visual + logs).
- RPM estimates stable and in the expected range.
- Overlays readable and aligned with detections.

---

## Reference
- Active code: `workspace/tools/drone-detector-demo/src/drone_detector_demo/main.py`
- Helpers: `geometry.py`, `detector_commons` (evlib loaders, frame builders).
