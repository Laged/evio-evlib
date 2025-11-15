# WIP – evlib Integration Tracking

**Updated:** 2025-11-16  
**Owner:** Codex (handover for Claude)  
**Context references:** `docs/evlib-integration.md`, `docs/data/evio-data-format.md`, `.claude/skills/dev-environment.md`, `evio/docs/dat-format-compatibility.md`

---

## 1. What’s Done

- Confirmed the current `evio/data/fan/*.raw` files carry the `% evt 3.0 / format EVT3` header and therefore work with `evlib.load_events`. See proof in `docs/evlib-integration.md` §1.
- Re-affirmed that legacy `.dat` files only expose the `% Height/% Version 2/% end` header; they remain incompatible with evlib (per `docs/data/evio-data-format.md` §2.2).
- Authored `docs/evlib-integration.md` describing the OpenEB-based conversion plan, including Nix packaging requirements, conversion workflow, regression testing, and documentation touchpoints.
- Sketched the need for an `evlib`-powered verifier CLI (“verify-dat”) to assert that freshly converted `.dat` files truly parse; MVP spec lives in `docs/evlib-integration.md` §4.2.

## 2. Completed Tasks ✅

1. **✅ OpenEB dependency resolved**
   - Investigated nixpkgs - OpenEB not available
   - Implemented simpler Python-based converter that preserves EVT3 headers
   - No external dependencies needed - just copies header + binary payload

2. **✅ EVT3 conversion helper implemented**
   - Created `scripts/convert_evt3_raw_to_dat.py` - Python script that validates and converts
   - Added Nix-wrapped commands in `flake.nix`:
     - `convert-evt3-raw-to-dat` - Single file conversion
     - `convert-all-datasets` - Batch convert all .raw files
   - Successfully tested on all 6 datasets

3. **✅ "verify-dat" CLI built**
   - Package: `workspace/tools/evio-verifier`
   - Command: `verify-dat` (via `uv run --package evio-verifier`)
   - Features: Loads via evlib, reports event counts/ranges, exits non-zero on failure
   - Tests passed: `.raw` files pass, legacy `.dat` files fail as expected

4. **✅ All datasets converted and verified**
   - fan_const_rpm: 30.4M events ✓
   - fan_varying_rpm: 73.1M events ✓
   - fan_varying_rpm_turning: 57.1M events ✓
   - drone_idle: 140.7M events ✓
   - drone_moving: 191.4M events ✓
   - fred-0: 36.1M events ✓

## 2.1. Remaining Documentation Tasks

1. **Update main documentation**
   - Add conversion workflow to `docs/evlib-integration.md`
   - Update `docs/data/evio-data-format.md` with EVT3 conversion section
   - Update `docs/setup.md` with quickstart commands

2. **Future considerations**
   - Distribution: Converted `.dat` files could be added to future ZIP releases
   - Benchmarking: Compare `.raw` vs `.dat` ingestion performance with evlib
   - OpenEB: If advanced features needed later (bias tuning, visualization), revisit packaging

## 3. Usage Commands

Run inside `nix develop`:

```bash
# Get datasets (if not already present)
unzip-datasets   # or download-datasets

# Convert all .raw files to EVT3 .dat
convert-all-datasets

# Verify a converted file
uv run --package evio-verifier verify-dat evio/data/fan/fan_const_rpm_evt3.dat

# Or convert a single file
convert-evt3-raw-to-dat evio/data/fan/fan_const_rpm.raw
```

## 4. Implementation Notes

- ✅ Avoided OpenEB dependency by implementing simple header-preserving Python converter
- ✅ Conversion and verifier tools complete and working
- ✅ All 6 datasets converted and verified successfully
- Legacy `.dat` files remain incompatible (as expected) - use converted `_evt3.dat` files instead
- `.dat`/`.raw` files gitignored - users get them via `unzip-datasets` or `download-datasets`

**Status:** Implementation complete. Documentation updates in progress.
