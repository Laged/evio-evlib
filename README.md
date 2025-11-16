# evio-evlib – Event-Camera Detection Workbench

Reproducible, Nix-powered event-camera toolkit for fan RPM and drone propeller detection. Uses evlib + Polars for fast windowing and OpenCV UI; ships a menu-driven MVP (`run-mvp-demo`) plus evlib demos and tests via flake aliases.

## Quickstart (dev or demo)
```bash
git clone <repo-url> evio-evlib
cd evio-evlib
nix develop --command unzip-datasets        # or download-datasets if not present locally
nix develop --command run-mvp-demo          # Menu UI + detectors/datasets
nix develop --command run-fan-rpm-demo      # CLI fan RPM (evlib/Polars)
nix develop --command run-drone-detector-demo
```
Data: legacy `.dat` exports to `_legacy.h5` for evlib; see `docs/prod/07_runbook.md` for download/convert steps.

## What’s inside (runnable today)
- `run-mvp-demo`: menu-driven OpenCV UI (`evio/scripts/mvp_launcher.py`) wrapping detectors/datasets.
- `run-fan-rpm-demo`, `run-drone-detector-demo`: evlib/Polars demos (ellipse + DBSCAN pipelines) on HDF5 exports.
- Legacy players (`run-demo-fan`, `run-demo-fan-ev3`, MVP1/2) for reference.
- Tests: `run-evlib-tests` (evlib vs legacy loader parity sanity check).

## Roadmap (truth-in-progress)
- Planned: stream adapter (Metavision), plugin API + `detector-ui`, shared `evio-core`, storage sink (ClickHouse/TSDB) for bbox/RPM telemetry.
- Current gaps: no live camera path yet; MVP UI still tied to legacy script paths; storage not implemented.

## Docs (slide-friendly, numbered)
- Problem & scope: `docs/prod/01_problem_statement.md`
- Solution overview: `docs/prod/02_solution_approach.md`
- Architecture (with mermaid): `docs/prod/03_architecture.md`
- Tech stack & commands: `docs/prod/04_tech_stack.md`
- Repo layout: `docs/prod/05_repo_structure.md`
- Status (DONE/TODO): `docs/prod/06_status.md`
- Runbook (data, demos, tests): `docs/prod/07_runbook.md`
- Data conversion (formats, why HDF5, how to convert): `docs/prod/08_data_conversion.md`
- Detectors (fan/drone pipelines): `docs/prod/09_drone_detection.md`

Dev history/legacy plans remain in existing `docs/` files; production-facing docs live under `docs/prod/`.
