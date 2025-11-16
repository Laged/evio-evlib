# Plan – Fix `run-drone-detector-demo` (parity with original demo)

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Goal:** Bring the current `run-drone-detector-demo` (evlib/HDF5 path) back to parity with the original `drone-example-detector.py`: correct propeller detection, full-size rendering in pass 2, and stable RPM estimates.

---

## Symptoms (current demo)
- Pass 2 window appears “tiny” (scaled down) vs original.
- Propeller detection/tracking less reliable than the original standalone script.
- RPM values differ run-to-run; overlays not as clear.

---

## Suspected differences vs original
1. **Rendering scale:** Current script downsamples the stacked view (`combined_small = resize(..., fx=0.7, fy=0.7)`), making the window look small and potentially obscuring details.
2. **Windowing/morph params:** Thresholding and morphology (area filters, orientation filters) may differ from the original; fine-window duration defaults might be too small (0.5 ms).
3. **Loader/path mismatch:** Need to ensure evlib/HDF5 path matches legacy `.dat` geometry (1280×720) and that width/height are inferred from data, not hardcoded incorrectly.
4. **DBSCAN tuning:** eps/min_samples may not match original; ring range (r_min/r_max) might be too tight for noisier prop clusters.

---

## Fix Plan

### Task 1 – Rendering parity
- Remove or make optional the downscale in pass 2:
  ```python
  combined = np.vstack([vis, overlay])
  cv2.imshow("...", combined)  # no resize, or gate with a flag
  ```
- Ensure `vis`/`overlay` are kept at full resolution (no premature resize/hstack).

### Task 2 – Loader & resolution sanity
- Confirm `run-drone-detector-demo` uses the evlib HDF5 (`drone_idle_legacy.h5`) or the legacy `.dat` consistently.
- Infer width/height from data (max(x)+1, max(y)+1) after load; avoid hardcoded 1280×720 if the data differs.
- Verify pass 1/2 window durations match the original: coarse 30 ms, fine 0.5–2 ms (allow CLI override).

### Task 3 – Detection tuning (match original)
- Revisit `propeller_mask_from_frame` params:
  - Threshold strategy (Otsu vs fixed), morphology kernel sizes.
  - Area/angle filters; top_k logic.
- DBSCAN params: expose CLI overrides for `eps`, `min_samples`, `r_min`, `r_max`; set defaults to original values.
- If available, diff against `example-drone-original.py` to reapply its constants.

### Task 4 – UI clarity
- Add optional `--show-pass1` to visualize coarse detections (already present per logs; ensure it renders full size).
- HUD tweaks: keep a single text panel (RPM, mean RPM) with consistent font; avoid overdraw that obscures detections.

### Task 5 – Regression checks
- Manual check: run legacy script vs demo on the same HDF5 and compare:
  - Detection rate (% of windows with props)
  - RPM stability
  - Visual overlay correctness (ellipses centered, blades clustered)
- If needed, add a small smoke test that runs a few windows and asserts a nonzero detection count.

---

## Commands (expected)
```bash
nix develop
unzip-datasets
convert-legacy-dat-to-hdf5 evio/data/drone_idle.dat

# After fixes
run-drone-detector-demo                     # normal (full-size view)
run-drone-detector-demo --show-pass1        # coarse view for debugging
run-drone-detector-demo --cluster-window-ms 1.0 --dbscan-eps 10 --dbscan-min-samples 15
```

---

## Acceptance Criteria
- Pass 2 renders at full resolution (no unintended downscale).
- Propeller detection rate and RPM estimates are on par with the original script.
- Overlays (ellipses, warning, RPM text) are readable and correctly positioned.
- Defaults can be overridden via CLI for fine-tuning, matching the original behavior when needed.
