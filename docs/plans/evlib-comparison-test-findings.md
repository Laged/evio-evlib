# evlib Comparison Test Findings & Trade-offs

**Date:** 2025-11-15
**Context:** Implementation of `workspace/libs/evio-core/tests/test_evlib_comparison.py`

---

## Summary

The implemented test suite validates **EVT3 conversion fidelity** (.raw → _evt3.dat) but does **not** test **evlib vs legacy loader parity** as the original plan title suggested.

## What Was Implemented

**Current Test:** `test_evlib_vs_legacy_stats`
- Loads `.raw` file with evlib → `compute_evlib_stats(raw_path)`
- Loads `_evt3.dat` file with evlib → `compute_evlib_stats(dat_path)`
- Compares the two evlib outputs
- **Result:** Validates that conversion preserves data (both files identical when loaded with evlib)

**What This Tests:**
- ✅ EVT3 .dat conversion doesn't corrupt event data
- ✅ evlib can load both .raw and .dat EVT3 formats
- ✅ Conversion preserves event counts, timestamps, spatial coordinates, polarity

**What This Does NOT Test:**
- ❌ Parity between evlib and legacy `evio.core.recording.open_dat()` loader
- ❌ Whether evlib produces same results as legacy decoder
- ❌ Regression detection if legacy decoder behavior changes

## Why Legacy Comparison Wasn't Implemented

### Issue 1: Dataset Mismatch

The legacy `.dat` files and EVT3 `.raw` files are **different recordings**:

```
File Sizes (fan datasets):
fan_const_rpm.dat        : 202M  (legacy Sensofusion format)
fan_const_rpm.raw        : 119M  (EVT3 format)
fan_const_rpm_evt3.dat   : 119M  (converted from .raw)

Event Counts:
fan_const_rpm.dat        : ~26.4M events  (different recording)
fan_const_rpm.raw        : 30.4M events
fan_const_rpm_evt3.dat   : 30.4M events   (same as .raw)
```

**Conclusion:** Cannot compare legacy .dat vs EVT3 .dat because they contain different datasets.

### Issue 2: Format Incompatibility

Legacy `.dat` files use Sensofusion proprietary format:
- Header: `% Height / % Width / % Version 2 / % end`
- Not parseable by evlib (no EVT3 marker)
- Only parseable by `evio.core.recording.open_dat()`

EVT3 files (.raw, _evt3.dat) use standard Prophesee format:
- Header: `% evt 3.0 / % format EVT3 / ...`
- Parseable by evlib
- Not parseable by legacy loader (expects Sensofusion format)

**Conclusion:** Legacy loader and evlib handle **mutually exclusive** file formats.

## Strategic Decision: Favor Conversion Fidelity Over Legacy Parity

### Rationale (per `docs/plans/wip-evlib-integration.md` section 5.1)

> "docs/architecture.md positions evlib as the canonical ingestion/representation layer; now that datasets are EVT3-compliant, **transition away from the bespoke .dat parser**."

The project's strategic direction is to:
1. ✅ Use evlib as the primary loader (Layer 1 data backbone)
2. ✅ Deprecate legacy `evio.core.recording.open_dat()`
3. ✅ Convert all datasets to EVT3 format
4. ✅ Validate conversion preserves data

### What We Gain

**Current test validates:**
- Data integrity through conversion pipeline (.raw → _evt3.dat)
- evlib can handle both raw and converted formats
- Statistical equivalence (171M+ events tested)
- Conversion workflow is lossless

**This enables:**
- Confidence in migrating to evlib-only codebase
- Deprecation of legacy loader without data loss risk
- Future dataset conversions can be validated with same tests

### What We Lose

**Cannot validate:**
- Historical equivalence with legacy loader output
- Whether evlib interprets events identically to legacy parser
- Regressions in legacy loader (but we're deprecating it anyway)

## Recommendations

### Option A: Document Trade-off (Recommended)

**Action:** Update documentation to clarify test purpose

**File:** `docs/plans/wip-evlib-integration.md` section 5.3

Add note:
```markdown
**Important Note:** This test validates EVT3 conversion fidelity (.raw → _evt3.dat)
rather than evlib vs legacy loader parity. This aligns with the strategic goal (§5.1)
of transitioning away from the legacy loader. Legacy .dat files use a different
(incompatible) format and contain different recordings, making direct comparison
infeasible.
```

**File:** `workspace/libs/evio-core/tests/test_evlib_comparison.py` docstring

Update to:
```python
"""Compare .raw and _evt3.dat files using evlib.

This test validates that the EVT3 .dat conversion workflow preserves data
integrity by comparing the same dataset in two formats (.raw and _evt3.dat).
Both files are loaded with evlib to ensure statistical equivalence.

Note: This does NOT compare evlib vs legacy evio.core.recording loader,
as they handle mutually exclusive file formats (EVT3 vs Sensofusion).
"""
```

### Option B: Add Supplementary Legacy Test

**If** historical parity is critical:

1. Create test using legacy `.dat` → legacy loader → stats
2. Create synthetic EVT3 file with known events
3. Load synthetic file with evlib
4. Compare outputs

**File:** `workspace/libs/evio-core/tests/test_legacy_loader.py`

```python
def test_legacy_loader_still_works():
    """Ensure legacy loader continues to work for old datasets."""
    legacy_rec = open_dat("evio/data/fan/fan_const_rpm.dat", width=1280, height=720)
    stats = compute_legacy_stats(legacy_rec)

    # Validate legacy loader produces expected results
    assert stats['event_count'] > 0
    assert stats['x_min'] >= 0
    # ... basic sanity checks
```

**Purpose:** Regression test for legacy loader, not evlib parity

### Option C: No Action Required

**Rationale:**
- Strategic direction is evlib-only (section 5.1)
- Conversion fidelity is what matters for migration
- Legacy loader will be deprecated per architecture.md
- Test suite already validates evlib works correctly

**Recommended:** Option C + Option A (document the trade-off)

## Conclusion

The current implementation is **correct for the strategic goal** (evlib migration) but the test name `test_evlib_vs_legacy_stats` is **misleading**.

**Suggested Actions:**

1. ✅ Rename test to `test_raw_vs_converted_dat_equivalence` or `test_evt3_conversion_preserves_data`
2. ✅ Update docstrings to clarify what is being tested
3. ✅ Document this trade-off in wip-evlib-integration.md
4. ✅ Acknowledge that legacy loader parity is not tested (and explain why)

**Status:** Test suite is production-ready for its actual purpose (conversion validation), but documentation should be clarified to avoid confusion about what "vs legacy" means.

---

## References

- Original Plan: `docs/plans/2025-11-15-evlib-comparison-tests.md`
- Implementation: `workspace/libs/evio-core/tests/test_evlib_comparison.py`
- Strategy Doc: `docs/plans/wip-evlib-integration.md` section 5.1
- Dataset Audit: `docs/data/evio-data-format.md`
