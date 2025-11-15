# Raw-to-EVT3 Pipeline Deprecation Report

**Author:** Codex  
**Date:** 2025-11-16  
**Audience:** Claude / maintainers intending to remove the `.raw → _evt3.dat` tooling.

---

## 1. Context

- Original assumption: `.raw` files inside `evio/data/*` were EVT3 exports derived from the legacy Sensofusion `.dat`.  
- Goal was to copy `.raw` → `_evt3.dat` and feed those into evlib as “the same dataset”.
- Reality (per `scripts/compare_all_fan_files.py`, `scripts/diagnose_fan_data.py`, and demo observations): the `.raw` files are **independent IDS camera captures** from Nov 2025, not conversions of the historical `.dat`.
- They differ drastically (event counts, resolution, duration, polarity), so the conversion pipeline is misleading and insufficient for legacy migration.

---

## 2. Evidence Summary

| Dataset | Legacy `.dat` | `.raw` / `_evt3.dat` (after conversion) | Notes |
| --- | --- | --- | --- |
| `fan_const_rpm` | 26.4 M events, 1280×720, ~9.5 s, balanced polarity (~50/50) | 30.3 M events, 2040×1793, ~682 s, polarity skewed (ON≈200 k vs OFF≈30 M) | Different recording entirely |
| `fan_varying_rpm` | 64.1 M events, 1280×720, ~22 s | 73.3 M events, 2027×1793, ~699 s, polarity skewed | Same pattern |
| `fan_varying_rpm_turning` | 48.1 M events, 1280×720, ~24 s | 60.2 M events, 2074×1793, ~717 s, polarity skewed | Same pattern |
| `drone_*` | Hardcoded 1280×720 legacy format | `.raw` conversion fails header patch (IDs format different), stats show 1936–2040 width and hundreds of seconds duration | Clear mismatch |

Development evidence:
- `scripts/compare_all_fan_files.py` prints side-by-side comparisons showing mismatches across event counts, x/y ranges, durations, and polarity counts.
- `scripts/diagnose_fan_data.py` logs the header fields (header says “height=720;width=1280”, but payload coordinates reach >2000), further proving data inconsistency.
- Demos: `run-demo-fan` vs `run-demo-fan-ev3` (when `_evt3.dat` was used) showed completely different visuals. Only after converting legacy `.dat` to HDF5 did the demos match.

Conclusion: `.raw` files are not replacements for legacy `.dat`; their conversion pipeline should be marked experimental/optional.

---

## 3. Current Raw→EVT3 Tooling

Files/commands involved:
- `scripts/convert_evt3_raw_to_dat.py`: copies `.raw` header/event payload, now patches metadata and logs stats.
- `convert-evt3-raw-to-dat`, `convert-all-datasets`: nix aliases wrapping the script.
- `workspace/tools/evlib-examples/*`: demo scripts to explore `.raw` with evlib (`run-evlib-raw-demo`, `run-evlib-raw-player`).
- Documentation references (`docs/plans/wip-evlib-integration.md`, `docs/evlib-integration.md`) previously suggested using `_evt3.dat` for evlib parity, which is incorrect.

Issues:
1. Misleading assumption that `_evt3.dat` equals the legacy `.dat`. They are different recordings; parity tests/demos cannot rely on them.
2. Metadata mismatch: header says 1280×720 while events extend to 2040×1793. We patched the header but the payload is still different.
3. Polarity skew indicates broken encoding or IDS-specific output; not representative of the Sensofusion data.

---

## 4. Deprecation Plan

### Step 1 – Update Documentation
- In `docs/evlib-integration.md` & `docs/data/evio-data-format.md`, add an explicit warning: `.raw → _evt3.dat` pipeline operates on separate IDS recordings and cannot be used for legacy parity.
- Move the instructions for `convert-evt3-raw-to-dat` into an “Experimental / IDS data” appendix.
- Emphasize the new canonical path: `convert-legacy-dat-to-hdf5` (legacy loader → HDF5) for real Sensofusion data.

### Step 2 – Adjust Developer Messaging
- Update `flake.nix` banner to:
  - Keep `convert-evt3-raw-to-dat` but label it “IDS raw experiments (not legacy)”.
  - Highlight `convert-legacy-dat-to-hdf5` / `convert-all-legacy-to-hdf5` as the authoritative commands for legacy migration.
- Ensure `run-demo-fan-ev3` references only the HDF5 output.

### Step 3 – Code Cleanup (optional / future)
- Either remove `convert-evt3-raw-to-dat` entirely or move it under `workspace/tools/evlib-examples` as a standalone script, to avoid confusion in core scripts.
- Remove `_evt3.dat` references from parity tests and docs so they aren’t mistaken for legacy data.

### Step 4 – Archive Diagnostic Evidence
- Keep `scripts/compare_all_fan_files.py` and `scripts/diagnose_fan_data.py` (or summarize them in docs) so future contributors know why the raw pipeline was deprecated.

---

## 5. Guidance for Future Developers
- Use `convert-legacy-dat-to-hdf5` / `convert-all-legacy-to-hdf5` for any work that needs the historical Sensofusion datasets. This ensures evlib sees the true data.
- Treat the `.raw` IDS files as optional samples for evlib experimentation (higher resolution, longer duration). They are good for performance testing but not for regression comparisons against the legacy system.
- Whenever `run-demo-fan-ev3` or parity tests are mentioned, ensure they point at `*_legacy.h5`, not `_evt3.dat`.

With these steps, we can safely retire the raw-to-EVT3 pipeline from the critical path while still allowing curious developers to explore the IDS captures if they choose.
