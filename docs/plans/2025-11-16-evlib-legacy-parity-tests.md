# Plan – Legacy Loader vs evlib Parity Tests

**Owner:** Codex (handoff for Claude)  
**Date:** 2025-11-16  
**Goal:** Produce an automated test suite that proves evlib reproduces the legacy `evio.core.recording.open_dat()` loader outputs for the Sensofusion datasets, even though the original `.dat` format is custom.

---

## 1. Rationale & Scope

- `docs/architecture.md` + `docs/evlib-integration.md` state that evlib becomes the canonical ingestion layer once we trust it on our historical data.
- Current “evlib comparison tests” only confirm `.raw → _evt3.dat` conversion fidelity; they never compare against the legacy loader because the formats differ.
- We must still demonstrate “same events in, same events out” to de-risk migration before removing `evio.core.recording`.

**Requirements:**
1. Use the **legacy loader as source of truth** (reads original custom `.dat` files).
2. Convert legacy output into an evlib-supported container **inside the test pipeline**, not by re-recording.
3. Load the converted artifact via **evlib** and compare key statistics and sampled payloads against the legacy data.
4. Automate via pytest; runnable with `run-evlib-tests`.

---

## 2. High-Level Flow

1. **Legacy extraction:** use `evio.core.recording.open_dat(path, width, height)` to stream events from the original `.dat`.
2. **Intermediary export:** write those events to a temporary HDF5 file (or EVT3 `.dat`) matching evlib’s schema.
3. **evlib load:** call `evlib.load_events(temp_path)` to obtain a LazyFrame.
4. **Comparison:** compute the same statistics on both sources and assert they match within tight tolerance; optionally compare hashes of the first N events.
5. **Cleanup:** remove temporary export.

---

## 3. Detailed Task Breakdown

### Task A – Create Legacy → HDF5 Export Helper
- Location: `workspace/libs/evio-core/tests/helpers/legacy_export.py`.
- Steps:
  1. Function `export_legacy_to_h5(dat_path: Path, width: int, height: int, out_path: Path) -> ExportStats`.
  2. Uses `evio.core.recording.open_dat` to obtain `recording.events` / `timestamps`.
  3. Writes to HDF5 layout:
     ```
     /events/t (int64 microseconds)
     /events/x (uint16)
     /events/y (uint16)
     /events/p (int8)  # map polarity to {-1, 1}
     attrs: width, height, source="legacy_dat"
     ```
  4. Returns counts/min/max to avoid re-reading for legacy stats.
- Testing: add unit test with small synthetic `MockRecording` to ensure HDF5 output schema matches expectations.

### Task B – evlib Loader Helper
- Reuse `compute_evlib_stats` but add ability to load from an in-memory HDF5 path (evlib handles it transparently).
- Ensure timestamp dtype normalization (Duration vs Int64).

### Task C – Comparison Utilities
- Extend `assert_within_tolerance` to accept both absolute and relative tolerances (timestamps might need absolute µs tolerance for rounding).
- Add `compare_sample_events(legacy_arrays, evlib_df, sample_size=1000)`:
  - Collect first `sample_size` events from each source, compare tuples `(t, x, y, p)`.
  - Hash if full equality is too heavy.

### Task D – Parametrized Parity Tests
- File: `workspace/libs/evio-core/tests/test_evlib_legacy_parity.py`.
- Parametrize over at least two datasets (`fan_const_rpm`, `drone_idle`).
- Steps per dataset:
  1. Skip test if source `.dat` missing (still leverage `unzip-datasets`).
  2. Call export helper to create temp HDF5 (use `tmp_path_factory`).
  3. Compute legacy stats from helper return; compute evlib stats by loading temp file.
  4. Assertions:
     - Event count exact match.
     - Polarity counts exact match.
     - Timestamp min/max within ≤5 µs (to account for rounding).
     - X/Y min/max exact.
     - Optional: verify HDF5 metadata (width/height).
  5. Call sample comparison helper.

### Task E – CLI Integration
- Update `flake.nix` shellHook to mention the new parity test command (same `run-evlib-tests` alias runs both suites).
- Document prerequisites: `unzip-datasets`, no need for `_evt3.dat` conversions because we start from original `.dat`.

### Task F – Documentation
- Update `docs/plans/wip-evlib-integration.md` “Next Actions” to include this parity plan.
- Add summary + usage instructions to `workspace/libs/evio-core/README.md`.
- Log test results (event counts, runtimes) after first successful execution.

---

## 4. Risk & Mitigations

| Risk | Mitigation |
| --- | --- |
| Legacy loader requires width/height; wrong values break conversion | Store constants per dataset in the test file; fail fast if metadata missing |
| HDF5 export increases disk usage | Write to `tmp_path_factory`; delete after test |
| Test runtime (hundreds of millions of events) | Use streaming writes/reads; only collect stats and limited samples |
| evlib dependency on HDF5 libs | Always run inside `nix develop`; mention in README |

---

## 5. Acceptance Criteria

1. Running `run-evlib-tests` executes both conversion fidelity tests **and** the new legacy parity tests, all green.
2. Failure in either dataset (counts mismatch, timestamp drift, sample mismatch) raises pytest assertion with clear dataset label.
3. Documentation explains the purpose and requirements of the parity suite.
4. Temporary files never linger after pytest completion.

Once these criteria are met, we can safely retire the bespoke `.dat` parser knowing evlib reproduces its output.***
