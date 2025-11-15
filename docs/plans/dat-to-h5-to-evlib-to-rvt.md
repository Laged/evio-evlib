# Plan – Legacy .dat → HDF5 → evlib → RVT

**Owner:** Codex (handoff for Claude)  
**Date:** 2025-11-16  
**Goal:** Document the current legacy export → evlib flow and outline next steps to validate RVT on the same data.

---

## TL;DR – Current Workflow

1) **Unzip original data**
```bash
nix develop
unzip-datasets          # or download-datasets
```

2) **Convert legacy `.dat` → HDF5 (evlib-ready)**
```bash
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat --force
# Batch for all legacy: convert-all-legacy-to-hdf5
```

3) **Render with evlib pipeline**
```bash
run-demo-fan            # legacy loader on .dat
run-demo-fan-ev3        # evlib on fan_const_rpm_legacy.h5
run-evlib-tests         # parity tests: legacy → HDF5 → evlib
```
Files involved: `evio/src/evio/core/legacy_export.py`, `scripts/convert_legacy_dat_to_hdf5.py`, `scripts/convert_all_legacy_to_hdf5.sh`, alias `run-demo-fan-ev3` (points at `fan_const_rpm_legacy.h5`).

---

## Next Steps – RVT Validation Plan

1) **Data plumbing**
   - Ensure RVT preprocessing consumes the same HDF5 exported from the legacy `.dat`.
   - Add a small loader wrapper in `evio-core` to feed HDF5 → evlib → RVT without touching the legacy loader.

2) **Minimal RVT smoke test**
   - Pick `fan_const_rpm_legacy.h5`.
   - Run a short RVT pipeline (voxel/timesurface) on a limited time window (e.g., first 1–2 s) to confirm tensors are produced without errors.
   - Log tensor shapes and timing; no need for model accuracy yet.

3) **Frame/tensor parity check**
   - Reuse the visual smoke test to capture frames from `run-demo-fan-ev3`.
   - For RVT inputs, generate a small batch of tensors (e.g., stacked histogram) from both legacy RAM path (if any) and evlib path, compare shapes/summary stats to ensure consistency.

4) **Document commands**
   - Add an RVT “quick start” snippet to `docs/evlib-integration.md` or RVT-specific docs:
     ```bash
     # assuming HDF5 export already exists
     uv run --package evio-core python -m evio_core.rvt_smoke evio/data/fan/fan_const_rpm_legacy.h5
     ```
   - Note prerequisites: `nix develop`, `convert-legacy-dat-to-hdf5`, and `uv sync`.

5) **CI/pytest hook (optional)**
   - Add a lightweight RVT smoke test (skip by default) that loads a tiny clip from `fan_const_rpm_legacy.h5` and runs the preprocessing step to catch integration breakage.

---

## Acceptance Criteria

- Clear docs for unzip → legacy-to-HDF5 → evlib demo/tests.
- RVT smoke test strategy defined and commands recorded.
- Optional: a minimal RVT preprocessing script exists and can run on `fan_const_rpm_legacy.h5` without error (even if skipped in CI by default).

This keeps the legacy migration path aligned with RVT without reintroducing the broken `.raw → _evt3.dat` pipeline.
