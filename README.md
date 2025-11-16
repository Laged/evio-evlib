# evio-evlib – Event Camera Detection Workbench

**Nix-powered toolkit for fan RPM and drone detection using event cameras.**

Fast windowing with evlib + Polars, fullscreen MVP UI with pink Y2K branding, reproducible environment.

## Quick Start
```bash
nix develop
run-mvp-demo  # Fullscreen UI with fan/drone detectors
```

## What Works
- 🟢 **Detectors**: Fan RPM (ellipse + DBSCAN), Drone propellers (multi-ellipse)
- 🟢 **MVP UI**: Fullscreen launcher, 0.25x-100x playback, 10μs-100ms windows
- 🟢 **Performance**: Lazy loading, HDF5 chunking, frame skipping for 6000+ FPS
- 🟢 **Visual**: 870x435 thumbnails, Sensofusion gray + Y2K pink palette

## Commands
```bash
run-mvp-demo               # Menu UI with all detectors/datasets
run-fan-rpm-demo <file>    # CLI fan RPM (evlib/Polars)
run-drone-detector-demo <file>
convert-all-legacy-to-hdf5 # Convert .dat → .h5 for evlib
```

## Roadmap
- 🔴 Stream adapter (Metavision SDK) for live cameras
- 🔴 Storage sink (ClickHouse/TSDB) for telemetry
- 🟡 RVT integration (runs, retraining incomplete)

## Docs (`docs/prod/`)
1. Problem statement
2. Solution approach
3. **Architecture** (mermaid diagram)
4. Tech stack & commands
5. Repo structure
6. **Status** (green/yellow/red)
7. **Runbook** (data prep, demos)
8. Data conversion
9. Detector pipelines
