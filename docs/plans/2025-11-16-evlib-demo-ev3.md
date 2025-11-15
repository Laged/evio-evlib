# Plan – run-demo-fan-ev3 Alias & Workflow

**Owner:** Codex (handoff for Claude)**  
**Date:** 2025-11-16

**Objective:** Provide a first-class demo command that replays the converted EVT3 dataset through the evlib-backed path so contributors can quickly verify the new ingestion flow. We will keep the existing `run-demo-fan` (legacy `.dat`) for comparison but add `run-demo-fan-ev3` that exercises the `_evt3.dat` file and evlib loaders.

---

## 1. Requirements

1. **Reuse existing tooling** in `flake.nix` shellHook (alias definitions, banner messaging).
2. **Leverage evlib** for file loading—no fallback to `evio.core.recording`.
3. **Keep parity** with the current UX (command runs from repo root, `uv run --package evio ...`).
4. **Document prerequisites:** datasets extracted + converted (`convert-all-datasets` run).

---

## 2. Implementation Steps

### Step A – Identify Playback Entry Point

- Confirm `evio/scripts/play_dat.py` already accepts `.dat` paths that evlib can read (via our conversion). If it still calls the legacy loader internally, add a flag (`--use-evlib` or auto-detect by header) so `_evt3.dat` runs through evlib.
- Alternatively, create a thin wrapper (e.g., `workspace/libs/evio-core/tools/play_evlib.py`) that mirrors `play_dat.py` behavior but uses `evlib.load_events`.

### Step B – Add Alias in `flake.nix`

- In the shellHook demo section, define:
  ```bash
  alias run-demo-fan-ev3='uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm_evt3.dat --use-evlib'
  ```
- Only include the flag if Step A requires it; otherwise omit.
- Update the banner text (under “Demo Aliases”) to list the new command and clarify its purpose (“evlib path on EVT3 data”).

### Step C – Document Usage

- Update `docs/setup.md` or `docs/evlib-integration.md` to add a “Demo” subsection:
  ```
  run-demo-fan        # Legacy loader on custom .dat
  run-demo-fan-ev3    # evlib loader on converted EVT3 .dat (requires convert-all-datasets)
  ```
- Mention dataset prerequisites (`unzip-datasets`, `convert-all-datasets`).

### Step D – Quick Validation

- Inside `nix develop`, run both commands:
  ```
  run-demo-fan
  run-demo-fan-ev3
  ```
- Verify both render windows appear and exit cleanly (Ctrl+C). If automated verification desired, capture short recordings (see visual smoke-test plan).

### Step E – Track in WIP Docs

- Update `docs/plans/wip-evlib-integration.md` (“Next Actions” or “Implementation Notes”) to check off that evlib demo parity exists.

---

## 3. Acceptance Criteria

1. `run-demo-fan-ev3` is available in the shellHook banner and alias list.
2. Command launches the evlib-backed playback on `fan_const_rpm_evt3.dat` without needing manual flags.
3. Documentation references the new command and explains when to use it.
4. Both demos (legacy + evlib) run from the repo root with `nix develop` active.

Once in place, this alias supports the upcoming visual smoke tests and gives contributors a quick way to sanity-check the new EVT3 pipeline.***
