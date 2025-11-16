# 04 – Tech Stack & Commands

## Stack
- Nix flake + uv: reproducible env with Rust/evlib/OpenCV, Python deps.
- evlib + Polars: fast event loading and windowed filtering.
- OpenCV: visualization/UI overlays.

## flake.nix aliases (inside `nix develop`)
- `run-mvp-demo` – menu UI (`evio/scripts/mvp_launcher.py`) with detectors/datasets.
- `run-fan-rpm-demo` – evlib/Polars fan RPM demo on `_legacy.h5`.
- `run-drone-detector-demo` – evlib/Polars drone propeller demo on `_legacy.h5`.
- `run-demo-fan`, `run-demo-fan-ev3`, `run-mvp-1`, `run-mvp-2` – legacy reference players.
- `run-evlib-tests` – evlib vs legacy loader comparison tests.
- `download-datasets`, `generate-thumbnails`, `run-evlib-raw-demo`, `run-evlib-raw-player` – supporting tools/sandboxes.

## Data prep
- Legacy `.dat` → `_legacy.h5` export required for evlib demos.
- See `docs/prod/07_runbook.md` for download/convert steps and example commands.
