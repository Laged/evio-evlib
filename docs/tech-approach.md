# Technical Approach – weUseNixBtw

## Problem Context
- Sensofusion already detects/locates drones via RF/acoustic; optical-fiber drones are silent and far.
- 360° rotating coverage demands sub-millisecond sensing; frame cameras miss fast movers.
- Event cameras fit the gap: microsecond latency, low blur, high dynamic range.
- Challenge ask: start with fan RPM → varying RPM → idle/moving drone detection, then scale to tracking.

## What We Built (Current Demos)
- Event data handling: convert legacy `.dat` → open `.h5` (evlib-compatible) to escape proprietary blobs.
- Two evlib-accelerated detectors (CLI demos today):
  - Fan RPM: two-pass pipeline (coarse ellipse fit on accumulated frames, DBSCAN blade clustering on fine windows, RPM regression).
  - Drone propellers: multi-ellipse detection, per-propeller blade clustering, per-propeller RPM with overlays.
- Shared utilities (`detector-commons`): evlib loaders + Polars windowing, accumulation, polarity visuals, temporal geometry lookup, DBSCAN helpers.
- Unified UI entrypoint: `run-mvp-demo` (Nix shell alias) launches menu-driven OpenCV UI wrapping detectors/datasets; still tied to legacy script paths but good for demos.
- Reproducible dev env: Nix flake + uv; `nix develop` gives Rust/evlib/OpenCV; consistent across dev/prod.

## How This Meets the Challenge
- Starts simple (fan RPM), works on varying RPM, extends to drone propellers (idle/moving).
- Uses event-native ops (Polars filters, evlib representations) to keep latency and throughput high.
- Shows how to suppress irrelevant motion via geometry gating (ellipses) + density clustering, a path to filtering trees/birds/planes.

## Architecture Vision (Scaling Plan)
- Data path: `.dat` → `.h5` (open) → evlib loaders → windowed processing → detector outputs (RPM, bbox/ellipses).
- Core library (planned): promote `detector-commons` into `libs/evio-core` with:
  - `EventSource` adapters (file, live stream/Metavision) with uniform window API.
  - Detector interface for plug-and-play algorithms (fan RPM, drone tracker, future ML).
  - Visualization/overlay helpers and benchmarks.
- App layer (planned): `detector-ui` hot-swappable detectors/data, same app for files and live streams; auto-discovery of plugins.
- Data products: automatic bbox/ellipse overlays to generate labeled event datasets; foundation for SOTA models (e.g., RVT) on event streams.

## ML + Data Strategy
- Goal: build labeled event datasets (event stream + bbox/prop labels) to unlock event-native models (RVT and successors).
- Pipeline validated end-to-end (`.dat` → `.h5` → RVT ingest), though current RVT results are limited due to lack of drone-labeled training data.
- Next step: automated zoom/bbox generation on drones to create real data for training + threat assessment by drone type.

## Next Steps to Production
- Implement live `StreamEventAdapter` (Metavision SDK) and unify with file adapter.
- Promote current utilities into `libs/evio-core` package; publish detector protocol; add tests/benchmarks.
- Build the interactive UI: hot-swap detectors/data, overlays for RPM/track/bboxes, one command to run.
- Automate dataset labeling loop: run detectors, auto-zoom, emit labeled event sequences for training.
- Harden nix packaging (`nix run .#run-mvp-demo`) for on-site deployments and CI parity.

## Why Nix/uv Matter
- Reproducibility for mission-critical workloads: identical toolchains for agents and humans.
- Fast spin-up (`nix develop`, `uv sync`) and portable demos; fewer “works on my machine” failures.
- Basis for shipping the same environment to production systems performing heavy event processing.
