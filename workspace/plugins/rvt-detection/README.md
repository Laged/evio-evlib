# RVT Detector Plugin

RVT-powered detector implementation that plugs into the evio-evlib workbench. It
wraps the official RVT Lightning module, feeds it stacked histograms generated
with evlib, and exposes the `DetectorPlugin` API described in
`docs/architecture.md`. Shared plumbing (Hydra config loading, tensor helpers,
asset discovery) now lives in `workspace/libs/rvt-adapter/` so other packages
can import the exact same routines.

Before running inference, confirm that the vendored RVT repo is initialized and
that at least one checkpoint is available:

```bash
# From repo root
uv run --package rvt-detector python -m rvt_detector.env_check
```

This command reports missing assets plus CUDA availability so you can fix
issues before launching the heavier OpenCV demos.

## Usage

```python
from pathlib import Path
from rvt_detector import RVTDetectorPlugin
from evio.evlib_loader import load_events_with_evlib

events = load_events_with_evlib("../../evio/data/fan/fan_const_rpm.dat").collect()
plugin = RVTDetectorPlugin(
    dataset_name="gen1",
    experiment="small",
)
detections = plugin.process(events)
```

Install dependencies with `uv sync` (workspace root) or `pip install -e .` from
this directory. Provide the RVT repository and checkpoint paths via parameters
or helper config when integrating with the detector UI. The
`rvt_adapter.events_to_hist_tensor` helper mirrors the preprocessing pipeline
documented in `docs/rvt-integration-plan.md`.
