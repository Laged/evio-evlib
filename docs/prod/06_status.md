# 06 – Status (DONE / TODO)

## DONE
- evlib + Polars pipelines for fan RPM and drone propellers (ellipse + DBSCAN + RPM).
- Menu-driven MVP UI (`run-mvp-demo`) wrapping detectors/datasets with OpenCV overlays.
- Reproducible Nix/uv environment; flake aliases for demos/tests.
- Data normalization to open `_legacy.h5` for evlib ingestion.

## TODO / Gaps
- Stream adapter (Metavision SDK) to support live cameras.
- Plugin API + `evio-core` package; consolidate detector logic out of demos/legacy scripts.
- Unified `detector-ui` app using plugin discovery (file/stream parity).
- Storage/telemetry sink (ClickHouse/TSDB) for detector outputs and audit/training artifacts.
- Improve data conversion tooling (automate `.dat` → `.h5` workflow).
