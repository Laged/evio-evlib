# Plan – Fix EVT3 Conversion vs Legacy Data Parity

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16  
**Goal:** Ensure the evlib `_evt3.dat` files represent the *same recordings* as the legacy `.dat` files so demos/tests compare apples to apples.

---

## 1. Findings

1. **Resolution mismatch:** `run-demo-fan-ev3` reports 2040×1793 while the legacy demo hardcodes 1280×720. The converted `_evt3.dat` currently visualizes a different recording (IDS export), not the legacy Sensofusion `.dat`.
2. **Evidence:** `workspace/libs/evio-core/tests/test_evlib_comparison.py` shows `_evt3.dat` has `x_max=2039`, `y_max=1792`, whereas the legacy `.dat` stays within 1280×720. Event counts differ as well (30.4 M vs 26.4 M).
3. **Root cause:** Our conversion pipeline re-encodes `.raw` files that are not the same capture as the legacy `.dat`. To prove parity, we must ingest the legacy `.dat` via its original loader and export *that data* into an evlib-friendly container (HDF5/EVT3).

---

## 2. Plan Summary

1. **Inventory & manifest:** Document every dataset trio (`legacy .dat`, `.raw`, `_evt3.dat`) and their stats so we know which recordings match.  
2. **Legacy export helper:** Implement the HDF5 exporter (per `docs/plans/2025-11-15-legacy-evlib-parity-hdf5.md`) to convert the **legacy loader’s output** into an evlib-readable file.  
3. **Update demos/tests:** Point `run-demo-fan-ev3` and parity tests at the converted legacy artifact instead of the unrelated `.raw` export.  
4. **Visual smoke test:** Once the data matches, run frame-by-frame comparisons between `run-demo-fan` and `run-demo-fan-ev3` to ensure the experience is identical.

---

## 3. Detailed Tasks

### Task A – Dataset Manifest & Sanity Checks
- Script: `scripts/dataset_manifest.py` (or simple Markdown table in `docs/data/datasets.md`).
- For each dataset stem (`fan_const_rpm`, `drone_idle`, etc.):
  1. Log paths to `*.dat` (legacy), `*.raw`, `*_evt3.dat`.
  2. Record event counts + x/y min/max using `evio.core.recording` and `evlib`.
  3. Flag discrepancies (counts or resolutions differ) so we know which pairs are mismatched.

### Task B – Legacy → HDF5 Export (from plan 2025-11-15-legacy-evlib-parity-hdf5.md)
- Implement `export_legacy_to_hdf5(dat_path, width, height, out_path)` using the legacy loader.
- Write unit tests (mock recording) to verify HDF5 schema and polarity mapping.
- This HDF5 file is the canonical “evlib-ready” representation of the legacy `.dat`.

### Task C – Update Tests
- Modify `test_evlib_comparison.py` to:
  - Load legacy `.dat` via `open_dat` → export to temp HDF5 → load with evlib → compare stats (event count, x/y ranges, polarity counts).
  - This ensures the same recording is used on both sides.

### Task D – Demo Alias Fix
- Once Task B/C are done, update `run-demo-fan-ev3` to load the converted legacy artifact (HDF5 or re-encoded EVT3) so it matches `run-demo-fan`.
- Document the prerequisite: run the legacy→HDF5 conversion (or `convert-all-datasets` extended to handle legacy `.dat`).

### Task E – Visual Smoke Test
- Follow `docs/plans/2025-11-16-evlib-visual-smoke-test.md`:
  - Capture frames from both demos.
  - Run `scripts/compare_frames.py` to produce difference metrics/images.
  - Attach findings to `docs/evlib-integration.md`.

---

## 4. Acceptance Criteria

1. Manifest clearly states which legacy `.dat` corresponds to which evlib-ready file; event counts/resolutions match.
2. Parity tests compare legacy data vs evlib on the *same* recording and pass.
3. `run-demo-fan-ev3` plays the converted legacy dataset and visually matches the original demo.
4. Visual smoke-test report captured and linked in docs.

Once these steps are complete, we can confidently retire the legacy loader knowing evlib reproduces the same data and user experience.***
