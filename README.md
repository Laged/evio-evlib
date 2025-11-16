# evio-evlib – Event Camera Detection Workbench

**Nix-powered toolkit for fan RPM and drone detection using event cameras.**

Fast windowing with evlib + Polars, fullscreen MVP UI with pink Y2K branding, reproducible environment.

_Crafted for the Junction 2025 Sensofusion challenge by team **weUseNixBtw** — [Matti Parkkila (Team Lead)](https://github.com/laged), [Irene Bandera Moreno](https://github.com/irenebm), and [Jesse Karjalainen](https://github.com/tupakkatapa)._

## Highlights
- A unified GUI for exploring event-based camera data
- A preprocessing pipeline to transform proprietary binary data from .dat to industry-standard .h5
- A blazingly-fast event-data loader (yes, python bindings with rust backend)
- A robust, multi-layered computer vision algorithm for real-time RPM counting & drone detection
- All packaged in an elegant flake.nix - just "nix develop" and "run-mvp-demo"

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

## Docs (`docs/`)
1. Problem statement
2. Solution approach
3. **Architecture** (mermaid diagram)
4. Tech stack & commands
5. Repo structure
6. **Status** (green/yellow/red)
7. **Runbook** (data prep, demos)
8. Data conversion
9. Detector pipelines
10. RVT proof of concept
