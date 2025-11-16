# 08 – Data Conversion & Formats

## What we found
- Two different datasets in the bundle:
  - Legacy Sensofusion `.dat` (custom header, `% Height / % Version 2 / % end`). These are **not** valid EVT3; evlib rejects them.
  - IDS `.raw` EVT3 recordings (Nov 2025). EVT3 header present; evlib can load them, but they are different sessions (different resolution/event stats) from the legacy `.dat`.
- The so-called “EVT3-based” `.dat` files are not EVT3; they require conversion to be evlib-friendly.

## Why HDF5?
- Evlib accepts HDF5 with event columns; HDF5 is an open, widely supported container that preserves timestamps/coords cleanly.
- It also aligns with downstream ML tooling (e.g., RVT and other event-native models) that can read HDF5 or convert from it.

## Conversion path (legacy .dat → .h5)
- Use the legacy loader (`evio.core.recording.open_dat`) to unpack events, then write HDF5 with columns `t, x, y, polarity` and attrs `width, height`.
- Provided aliases in `flake.nix`:
  - `convert-legacy-dat-to-hdf5` – single file
  - `convert-all-legacy-to-hdf5` – batch
- After conversion, run evlib-based demos against the `_legacy.h5` outputs (e.g., `run-fan-rpm-demo evio/data/fan/fan_const_rpm_legacy.h5`).

## EVT3 `.raw` vs legacy `.dat`
- `.raw` EVT3 (IDS): standard ASCII EVT3 header, evlib loads directly; different recordings (longer, higher apparent resolution span, polarity quirks).
- `.dat` legacy: short custom header, not EVT2/EVT3, incompatible with evlib; must convert.
- Do **not** assume `.raw` and `.dat` are the same dataset; they are not comparable for parity tests.

## Scripts / tooling
- Conversion aliases in `flake.nix` call the converter to emit `_legacy.h5` files; they live under `evio/` scripts.
- Experimental converters for IDS `.raw` → `_evt3.dat` exist but are sandboxed and not needed for legacy parity.

## Recommended workflow
1) Obtain datasets (`unzip-datasets` or `download-datasets` in `nix develop`).
2) Convert legacy `.dat` to `_legacy.h5` (`convert-legacy-dat-to-hdf5` or `convert-all-legacy-to-hdf5`).
3) Run evlib demos/UI on the HDF5 outputs (`run-mvp-demo`, `run-fan-rpm-demo`, `run-drone-detector-demo`).
4) For EVT3 `.raw`, you can load directly with evlib, but treat them as separate experimental data.

## Limitations / caveats
- Legacy `.dat` remain unsupported by evlib without conversion; upstreaming a custom decoder to evlib would require Rust changes and full spec of the custom header/packing.
- Polarity in IDS `.raw` appears all OFF in provided samples; treat these as experimental only.
- Data files are large; keep them out of Git and fetch/unzip per the runbook.
