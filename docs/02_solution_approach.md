# 02 – Solution Approach

- Use event-native processing with evlib + Polars for fast windowing and representations.
- Normalize data into open HDF5 (`.dat` → `_legacy.h5`) to avoid proprietary blobs and enable evlib.
- Detection strategies:
  - Fan RPM: two-pass pipeline (accum frames → ellipse fit; fine windows → DBSCAN blade clustering → angle regression to RPM).
  - Drone propellers: multi-ellipse detection, per-propeller clustering, per-propeller RPM with overlays.
- Noise rejection: geometry gating (ellipses) + density clustering to suppress trees/birds/planes.
- Reproducibility: Nix + uv so agents and humans share the same toolchain and demos.
- Data interoperability: convert legacy `.dat` to open HDF5 for evlib and downstream ML (e.g., RVT and event-native models); treat EVT3 `.raw` as separate experimental recordings (see `08_data_conversion.md`).
