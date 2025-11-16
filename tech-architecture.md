# Technical Architecture (Current & Planned)

## End-to-End Data Path (Today → Target)
```mermaid
flowchart LR
    subgraph Data
        A[Legacy .dat (EVT3)] -->|export| B[Open .h5 (evlib-ready)]
    end
    subgraph Ingest & Windowing
        B --> C[evlib.load_events (lazy)]
        C --> D[Polars window filters<br/>+ accumulation]
    end
    subgraph Detectors (CLI demos today)
        D --> E[Fan RPM demo<br/>(2-pass ellipse + DBSCAN + RPM fit)]
        D --> F[Drone demo<br/>(multi-ellipse + per-propeller RPM)]
    end
    subgraph Planned Core
        D --> G[EventSource adapters<br/>(File + Stream)]
        G --> H[Detector interface<br/>(plugin API)]
    end
    subgraph UI & Outputs
        E & F --> I[Overlays/BBoxes/RPM text<br/>(OpenCV windows today)]
        H --> J[Detector UI (hot-swap)<br/>(planned app)]
        I --> K[(Storage: ClickHouse/TSDB)<br/>(recommended; not built)]
    end
```

## What Exists Now
- Data conversion: manual export `.dat` → `_legacy.h5` to avoid proprietary blobs.
- Ingest/windowing: evlib + Polars (`detector-commons/loaders.py`) for fast slicing, timestamp normalization (Duration vs int64), accumulation, polarity visualization.
- Fan RPM demo: two-pass pipeline (coarse ellipse fit on accumulated frames, fine DBSCAN blade clustering, angle regression to RPM).
- Drone demo: multi-ellipse detection, per-propeller clustering and RPM; overlays via OpenCV.
- MVP GUI: `run-mvp-demo` (Nix shell alias to `evio/scripts/mvp_launcher.py`) wraps detectors/datasets into a unified menu + playback UI; still uses legacy evio paths and bundled detector utilities, not the planned plugin API.
- Reproducibility: Nix flake + uv; `nix develop` sets Rust/evlib/OpenCV toolchain.

## Planned Build-Out
- `EventSource` adapters:
  - File adapter (evlib-backed, loops playback).
  - Stream adapter (Metavision SDK) with background buffering; not implemented yet.
- Detector interface:
  - Simple plugin protocol (name/key/description + `process(events, width, height)`).
  - Auto-discovery in workspace packages.
- `evio-core` library:
  - Promote detector-commons into `libs/evio-core` with adapters, window API, representations, overlays, benchmarks, tests.
- `detector-ui` app:
  - Hot-swap detectors/data (keys), common visualization, same codepath for file/stream.
- Storage/telemetry (not built):
  - Recommended: ClickHouse/Time-Series DB for on-device buffered ingestion of `{timestamp, bbox/ellipse, rpm, detector_id, source}` plus downsampled event stats.
  - Benefits: fast time-range queries, rollups for alerts, and exporting labeled spans for ML training.

## Operational Flow (target)
1) Acquire: `.dat` or live camera → evlib loader (file/stream).  
2) Window: Polars filters to fixed Δt; normalize timestamps.  
3) Detect: plugins consume window → results (RPM, bbox/ellipses, debug overlays).  
4) Visualize: UI overlays results; hot-swap detectors/data.  
5) Persist (future): stream detector outputs to ClickHouse/TSDB; optionally persist reduced event stats (histograms/voxels) for later audit/training.  

## Gaps / Risks
- No stream adapter yet; live cameras remain a plan.
- No packaged `evio-core` or plugin API; current detectors live in demos and the MVP launcher, not in a shared plugin system.
- UI gap: `run-mvp-demo` exists but is tied to legacy evio script paths and detector_utils; needs consolidation into the planned plugin-based `detector-ui`.
- Storage/Audit path (ClickHouse/TSDB) unimplemented; current runs are ephemeral.

## Recommendations
- Finish file + stream adapters and promote them into `evio-core` with tests/benchmarks.
- Ship a minimal plugin protocol and migrate current demos as plugins.
- Build the detector UI to exercise file/stream parity and hot-swap scenarios.
- Add a slim telemetry sink (ClickHouse/TSDB) for detector outputs; start with append-only tables and rollups.
