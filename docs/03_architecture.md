# 03 – Architecture (Current → Target)

```mermaid
flowchart TB
    subgraph "Data Sources"
        A["<b>Legacy .dat<br/>Sensofusion</b>"]
        B["EVT3 .raw<br/>IDS camera"]
    end

    subgraph "Data Conversion"
        A --> C["<b>convert-legacy-dat-to-hdf5</b>"]
        C --> D["<b>HDF5 _legacy.h5</b>"]
    end

    subgraph "Ingest & Windowing"
        D --> E["<b>evlib + Polars<br/>lazy loading</b>"]
        B --> E
    end

    subgraph "Detection Pipelines"
        E --> F["<b>Fan RPM<br/>2-pass ellipse + DBSCAN</b>"]
        E --> G["<b>Drone propellers<br/>multi-ellipse bbox</b>"]
        G --> G2["Drone zoom-in<br/>bbox crop"]
    end

    subgraph "Visualization (DONE)"
        F --> H["<b>MVP Launcher<br/>run-mvp-demo</b>"]
        G --> H
        G2 --> H
        F --> I["CLI Demos<br/>run-fan-rpm-demo"]
        G --> J["CLI Demos<br/>run-drone-detector-demo"]
        G2 --> J
        E --> K["Legacy players<br/>run-demo-fan-ev3"]
    end

    subgraph "Testing"
        D --> L["run-evlib-tests<br/>loader parity"]
    end

    subgraph "Storage (TODO)"
        H -.-> M["ClickHouse/TSDB<br/>telemetry sink"]
    end

    subgraph "Event-Based ML (PARTIAL)"
        E --> N["RVT<br/>event-to-frame"]
    end

    %% Styling
    classDef done fill:#90EE90,stroke:#228B22,stroke-width:2px,color:#000
    classDef partial fill:#FFFFE0,stroke:#FFD700,stroke-width:2px,color:#000
    classDef future fill:#FFB6C1,stroke:#DC143C,stroke-width:2px,color:#000,stroke-dasharray: 5 5

    class A,C,D,E,F,G,H,I,J,K,L done
    class B,G2,N partial
    class M future
```

## Components by Status

### 🟢 Complete (Green)
- **Data Sources**: Legacy `.dat` files from Sensofusion
- **Data Conversion**: `convert-legacy-dat-to-hdf5` → `_legacy.h5`
- **Ingest**: evlib + Polars lazy loading and windowing
- **Detection**:
  - Fan RPM (2-pass ellipse + DBSCAN + RPM fit)
  - Drone propellers multi-ellipse bounding box detection
- **Visualization**:
  - MVP Launcher (`run-mvp-demo`) - fullscreen UI with all detectors
  - CLI demos (`run-fan-rpm-demo`, `run-drone-detector-demo`)
  - Legacy players (`run-demo-fan-ev3`)
- **Testing**: `run-evlib-tests` (evlib vs legacy parity)

### 🟡 Partial (Yellow)
- **Data Sources**: EVT3 `.raw` (IDS camera - experimental, different from legacy)
- **Detection**: Drone zoom-in feature (bbox crop - incomplete)
- **Event-Based ML**: RVT event-to-frame (runs successfully, retraining failed)

### 🔴 Not Started (Red)
- **Storage**: ClickHouse/TSDB telemetry sink (planned)

See `docs/06_status.md` and `docs/07_runbook.md` for commands
