# 07 – Runbook (Data, Demos, Tests)

## Prereqs
```bash
git clone <repo-url> evio-evlib
cd evio-evlib
nix develop
```

## Data
- Input: legacy `.dat` files (custom format) and EVT3 `.raw` (different recordings).
- Convert legacy `.dat` → evlib-friendly `_legacy.h5` before running evlib demos.
  - Single: `convert-legacy-dat-to-hdf5`
  - Batch: `convert-all-legacy-to-hdf5`
- Use `unzip-datasets` or `download-datasets` first to get data locally.

## Demos / UI
- Menu UI: `run-mvp-demo`
- Fan RPM (evlib/Polars): `run-fan-rpm-demo evio/data/fan/fan_const_rpm_legacy.h5`
- Drone propellers (evlib/Polars): `run-drone-detector-demo evio/data/drone_idle/drone_idle_legacy.h5`
- Legacy players: `run-demo-fan`, `run-demo-fan-ev3`, `run-mvp-1`, `run-mvp-2`

## Tests
- Loader parity sanity check: `run-evlib-tests` (evlib vs legacy comparison)
- Additional tests per package (use `uv run pytest` inside the target package if needed)

## Troubleshooting
- Data missing: run `download-datasets` (if configured) or supply paths to your exports.
- HDF5 conversion issues: verify export format matches EVT3; re-export and ensure timestamps are intact.
- UI not showing detectors: ensure `nix develop` environment is active so deps and aliases resolve.
