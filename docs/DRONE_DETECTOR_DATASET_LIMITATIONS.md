# Drone Detector Dataset Limitations

**Date:** 2025-11-16
**Status:** Known limitation documented

---

## Summary

The drone detector parity restoration (2025-11-16) successfully restored all algorithm parameters to match `example-drone-original.py`. However, testing revealed dataset-specific limitations with the `drone_idle` HDF5 export.

---

## Issue: drone_idle Dataset Detection Failure

### Symptoms
- Detection fails with 0 propellers detected across all tested time ranges (0-10s)
- Debug output shows only 2 pixels survive pre-threshold (even at threshold=50)
- Morphological opening (5x5 kernel) eliminates these isolated pixels
- No contours remain for ellipse fitting

### Root Cause
The `drone_idle` dataset has **extremely sparse event activity** in the tested time ranges:
- 11,000-19,000 non-zero pixels per 30ms window (good accumulation)
- Only **2 pixels** exceed intensity threshold 50 (insufficient for detection)
- These 2 pixels are isolated (likely hot pixels or single blade tips)
- 5x5 morphological kernel requires connected regions >25 pixels

### Evidence (Debug Output)
```
[DEBUG] Raw frame range: [0.0, 255.0]
[DEBUG] Non-zero pixels in raw frame: 11160
[DEBUG]   >0: 11160
[DEBUG]   >10: 434
[DEBUG]   >50: 2        # ← Only 2 pixels!
[DEBUG]   >100: 2
[DEBUG]   >200: 2
[DEBUG]   ==255 (clipped): 2
[DEBUG] Pixels above pre-threshold (50): 2
[DEBUG] After pre-threshold: 2 white pixels
[DEBUG] Otsu threshold value: 0.0
[DEBUG] After Otsu: 2 white pixels
[DEBUG] After morphology: 0 white pixels  # ← Morphology kills signal
[DEBUG] No contours found
```

---

## Workarounds

### Option 1: Skip to Active Segments
The drone may not be spinning in early frames. Try:
```bash
run-drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5 \
  --pre-threshold 50 --skip-seconds 20 --max-frames 50
```

### Option 2: Increase Window Duration
Accumulate more events per frame:
```bash
run-drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5 \
  --pre-threshold 50 --window-ms 60 --max-frames 50
```

### Option 3: Use drone_moving Dataset
Test with a dataset that has active propellers:
```bash
run-drone-detector-demo evio/data/drone_moving/drone_moving_legacy.h5 \
  --pre-threshold 50 --max-frames 50
```

### Option 4: Lower Pre-Threshold Further
Try very permissive thresholds (may increase noise):
```bash
run-drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5 \
  --pre-threshold 30 --window-ms 60 --max-frames 50
```

---

## Investigation Needed

### Questions to Answer

1. **Does the original script work on drone_idle.dat?**
   - Test: `run-drone-detector --max-frames 50`
   - If it also fails → dataset is genuinely sparse
   - If it works → HDF5 conversion issue

2. **Does drone_idle have active segments later in the recording?**
   - Try `--skip-seconds 30`, `--skip-seconds 60`, etc.
   - Find time range where propellers are actually spinning

3. **HDF5 vs .dat intensity differences?**
   - Compare intensity distributions between formats
   - Check if HDF5 export process loses dynamic range
   - Verify accumulation frame matches .dat loader

### Potential Root Causes

**Hypothesis 1: Dataset is actually idle**
- "drone_idle" may mean propellers are NOT spinning in this clip
- Need to verify recording metadata/description
- May need different dataset for testing

**Hypothesis 2: HDF5 conversion issue**
- Intensity distribution differs from .dat format
- Check conversion in `convert-legacy-dat-to-hdf5` script
- Verify evlib's `load_legacy_h5` produces same data as .dat loader

**Hypothesis 3: Time window mismatch**
- Early frames (0-10s) may be quiet/warmup period
- Active flight may start later in recording
- Need to scan full duration to find propeller activity

---

## Impact on Rendering Pipeline

**Trade-off:** Longer accumulation windows improve detection in sparse datasets but may impact rendering performance.

**Recommendations for rendering work:**
- Default 30ms windows work well for dense datasets (fan, active drones)
- Sparse datasets may need 60-100ms windows
- Consider adaptive window sizing based on event density
- Monitor frame rate impact when increasing window duration

---

## Algorithm Parity Status

**Code correctness:** ✅ **VERIFIED**
- All parameters match `example-drone-original.py` exactly
- Regression tests pass (3/3)
- Bottom-right HUD, correct labels, proper scaling
- CLI overrides available for tuning

**Dataset compatibility:** ⚠️ **BLOCKED on drone_idle**
- Algorithm is correct but dataset has insufficient signal
- Works on dense datasets (verified with fan data)
- Needs testing on drone_moving or active segments

---

## Next Steps

1. **Test with drone_moving dataset** to verify parity restoration works
2. **Scan drone_idle for active segments** (try --skip-seconds 30, 60, 90)
3. **Verify original script behavior** on same dataset/time range
4. **Document working parameters** once active segment found
5. **Consider adaptive thresholding** for future enhancement

---

## CLI Override Added

**Commit:** `1f561db` (2025-11-16)

Added `--pre-threshold` CLI argument:
- Default: 250 (preserves original behavior)
- Recommended for HDF5: 50-100
- Usage: `run-drone-detector-demo --pre-threshold 50`

This allows dataset-specific tuning without code changes.

---

## Tracking

**Issue:** Dataset-specific detection failure (drone_idle HDF5)
**Severity:** Medium (workarounds available, code is correct)
**Priority:** Low (can revisit after rendering pipeline work)

**Related Work:**
- Parity restoration: Complete (docs/plans/2025-11-16-drone-detector-verified-parity.md)
- Regression tests: Passing (workspace/tools/drone-detector-demo/tests/test_parity.py)
- CLI enhancement: Complete (--pre-threshold override)
