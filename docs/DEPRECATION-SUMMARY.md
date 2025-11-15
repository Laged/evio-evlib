# Raw-to-EVT3 Pipeline Deprecation Summary

**Date:** 2025-11-16

## What Was Deprecated

The `convert-evt3-raw-to-dat` pipeline that assumed .raw files were conversions of legacy .dat files.

## Why

Investigation proved .raw files are separate IDS camera recordings with:
- Different hardware (IDS vs Sensofusion)
- Different resolutions (2040×1793 vs 1280×720)
- Different durations (682s vs 9.5s)
- Broken polarity (0 OFF events)

See `docs/plans/raw-to-evt3-deprecation.md` for diagnostic evidence.

## What Replaced It

**Legacy export pipeline:** `convert-legacy-dat-to-hdf5`
- Exports actual Sensofusion .dat files to evlib HDF5
- Enables true legacy vs evlib parity testing
- All tests pass (16/16)

## What Was Preserved

1. **Working tools (KEEP USING):**
   - `convert-legacy-dat-to-hdf5` - Legacy export to HDF5
   - `convert-all-legacy-to-hdf5` - Batch legacy export
   - `workspace/tools/evlib-examples/*` - IDS .raw experimentation
   - All parity tests in `test_evlib_comparison.py`

2. **Experimental tools (LIMITED USE):**
   - `convert-evt3-raw-to-dat` - IDS .raw → .dat (experimental)
   - Diagnostic scripts in `scripts/` (historical reference)

## Migration Guide

**Old (WRONG):**
```bash
convert-all-datasets  # Thought this gave us legacy data in EVT3 format
run-demo-fan-ev3      # Thought this played the same recording as run-demo-fan
```

**New (CORRECT):**
```bash
convert-all-legacy-to-hdf5  # Export actual legacy data to HDF5
run-demo-fan-ev3             # Now plays the SAME recording via evlib
```

## For Future Developers

- Use `*_legacy.h5` files for all legacy parity work
- Use `.raw` files only for IDS-specific experiments
- Never assume .raw = legacy .dat without verification
- Read `docs/data/datasets.md` for complete dataset manifest
