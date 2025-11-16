# 03 – Architecture (Current → Target)

```mermaid
flowchart LR
    subgraph Data
        A[Legacy .dat (EVT3)] -->|export| B[Open .h5 (evlib-ready)]
    end
    subgraph Ingest & Windowing
        B --> C[evlib.load_events (lazy)]
        C --> D[Polars window filters<br/>+ accumulation]
    end
    subgraph Detectors (today)
        D --> E[Fan RPM demo<br/>(2-pass ellipse + DBSCAN + RPM fit)]
        D --> F[Drone demo<br/>(multi-ellipse + per-propeller RPM)]
    end
    subgraph UI
        E & F --> G[run-mvp-demo<br/>(menu-driven OpenCV UI)]
    end
    subgraph Planned Core
        D --> H[EventSource adapters<br/>(file + stream)]
        H --> I[Detector plugin API<br/>(evio-core)]
        I --> J[detector-ui (hot-swap)]
    end
    subgraph Outputs
        G --> K[(Storage: ClickHouse/TSDB)<br/>(recommended; not built)]
    end
```

## Current runnable pieces
- `run-mvp-demo`: menu UI (`evio/scripts/mvp_launcher.py`) wrapping detectors/datasets; uses evlib-backed detector utilities.
- `run-fan-rpm-demo`, `run-drone-detector-demo`: evlib/Polars CLI demos on `_legacy.h5` exports.
- Legacy players (`run-demo-fan`, `run-demo-fan-ev3`, MVP1/2) for reference.

## Planned build-out
- Adapters: file (evlib) + stream (Metavision SDK) under a common `EventSource` API.
- Plugin API: detector interface (`process(events, width, height)`), auto-discovered packages, shared overlays/utilities in `evio-core`.
- UI: `detector-ui` hot-swappable detectors/data, same code for file/stream.
- Storage: ClickHouse/TSDB sink for `{t, bbox/ellipse, rpm, detector_id, source}` + reduced event stats for audit/training (not implemented yet).
- Data conversion: legacy `.dat` must be converted to `_legacy.h5` before evlib-based demos/UI; see `docs/prod/08_data_conversion.md`.
- Detectors: fan/drone pipelines documented in `docs/prod/09_drone_detection.md`.
