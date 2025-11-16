# 06 – Status (DONE / IN PROGRESS / TODO)

## ✅ DONE (Green)
- 🟢 **evlib + Polars pipelines** for fan RPM and drone propellers (ellipse + DBSCAN + RPM)
- 🟢 **Menu-driven MVP UI** (`run-mvp-demo`) with fullscreen support, pink Y2K branding, thumbnails
- 🟢 **Reproducible Nix/uv environment** with flake aliases for demos/tests
- 🟢 **Data normalization** to open `_legacy.h5` for evlib ingestion
- 🟢 **Performance optimization** (schema caching, HDF5 chunking, frame skipping)
- 🟢 **Playback controls** (speed: 0.25x-100x, window: 10μs-100ms, arrow keys)
- 🟢 **Visual polish** (870x435 thumbnails, Sensofusion gray + Y2K pink palette)

- 🟢 **Unified detector app** (`run-mvp-demo`) with menu-driven detector/dataset selection

## 🟡 IN PROGRESS / PARTIAL (Yellow)
- 🟡 **RVT integration** (event-to-frame representation)
  - 🟢 Successfully ran RVT on exported HDF5 data
  - 🔴 Retraining failed (insufficient data/time for fine-tuning)
  - Status: Can process events, cannot retrain models
- 🟡 **Plugin API** + `evio-core` package
  - Detector utilities extracted and working
  - Full plugin discovery system not yet implemented

## 🔴 TODO / Not Started (Red)
- 🔴 **Stream adapter** (Metavision SDK) to support live event cameras
- 🔴 **Storage/telemetry sink** (ClickHouse/TSDB) for detector outputs and training artifacts
- 🔴 **Automated data conversion** workflow (`.dat` → `.h5` pipeline)
