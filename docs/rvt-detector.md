# RVT Detector Integration

## Overview

The RVT (Recurrent Vision Transformer) detector is a state-of-the-art neural network for event-based object detection. This integration provides a plugin-based implementation that can be used with the evio-evlib workbench.

## Architecture

The RVT integration consists of three main components:

### 1. rvt-adapter (`workspace/libs/rvt-adapter`)

A shared library providing:
- Hydra configuration composition for RVT
- Event histogram generation using evlib
- Asset discovery and validation
- Environment reporting

### 2. rvt-detector (`workspace/plugins/rvt-detection`)

The main detector plugin implementing:
- `RVTDetectorPlugin` class with the detector interface
- Model loading and checkpoint management
- Recurrent state management for temporal coherence
- Detection post-processing

### 3. detector-ui (`workspace/apps/detector-ui`)

An interactive OpenCV-based application for:
- Loading event data from files (.dat, .h5, .aedat)
- Running RVT inference on event windows
- Visualizing detections in real-time

## Setup

### Prerequisites

1. **Initialize RVT submodule**:
   ```bash
   git submodule update --init --recursive
   ```

2. **Download RVT checkpoints**:
   - Obtain `rvt-models.zip` and place in `workspace/plugins/rvt-detection/models/`
   - Extract: `unzip-checkpoints`

3. **CUDA support** (recommended):
   - RVT runs significantly faster on GPU
   - CPU inference is supported but slow

### Installation

From the nix development shell:

```bash
# Sync all workspace dependencies
uv sync
```

The workspace manager will automatically install:
- evio-core (event processing library with evlib)
- rvt-adapter (RVT helpers)
- rvt-detector (detector plugin)
- detector-ui (visualization application)

## Usage

### Run Demos

### Fan Dataset with RVT

```bash
run-detector evio/data/fan/fan_const_rpm_legacy.h5
```

**What you'll see:**
- Event visualization showing rotating fan motion
- Green bounding boxes around detected objects
- Confidence scores and class IDs
- Real-time inference on event windows

**Controls:** Press 'q' or ESC to quit

### Drone Dataset with RVT

```bash
# First convert drone data to HDF5
convert-legacy-dat-to-hdf5 evio/data/drone_idle/drone_idle.dat

# Run detector
run-detector evio/data/drone_idle/drone_idle_legacy.h5 --checkpoint workspace/plugins/rvt-detection/models/rvt-s-1mpx.ckpt --dataset gen4
```

**What you'll see:**
- Event-based propeller detection
- Bounding boxes tracking fast-moving propellers
- Higher resolution Gen4 sensor data (1 Mpx)

**Controls:** Press 'q' or ESC to quit

### Manual Invocation

```bash
cd workspace
uv run --package detector-ui python -m detector_ui.main \
    <event_file.h5> \
    --rvt-repo plugins/rvt-detection/RVT \
    --checkpoint plugins/rvt-detection/models/rvt-s-gen1.ckpt \
    --device cuda \
    --window-ms 50 \
    --histogram-bins 10
```

#### Arguments

- `dat`: Path to event file (.dat, .h5, .aedat) or use `--mock` for synthetic data
- `--window-ms`: Temporal window duration in milliseconds (default: 50.0)
- `--width`, `--height`: Visualization dimensions (default: 1280x720)
- `--dataset-name`: RVT dataset config - `gen1` or `gen4` (default: gen1)
- `--experiment`: RVT experiment config - `small`, `base`, or `tiny` (default: small)
- `--rvt-repo`: Override path to RVT repository
- `--checkpoint`: Override path to RVT checkpoint (.ckpt)
- `--device`: Torch device (`cuda`, `cpu`, or auto-detect)
- `--histogram-bins`: Number of temporal bins for event histogram (default: 10)
- `--max-windows`: Limit number of windows processed
- `--no-loop`: Disable playback looping
- `--delay-ms`: OpenCV waitKey delay for playback speed control (default: 1)

### Environment Validation

Check RVT environment setup:

```bash
cd workspace
uv run --package rvt-detector python -m rvt_detector.env_check \
    --rvt-repo plugins/rvt-detection/RVT \
    --checkpoint plugins/rvt-detection/models/rvt-s-gen1.ckpt
```

This validates:
- RVT repository exists and contains Hydra configs
- Checkpoint file exists
- CUDA availability
- Python environment

## Configuration

### Dataset Configs

RVT supports different sensor configurations via Hydra:

- **gen1**: Prophesee Gen1 sensors (240x180 resolution)
- **gen4**: Prophesee Gen4 sensors (higher resolution)

### Experiment Configs

Pre-trained model variants:

- **small** (rvt-s): Lightweight model, faster inference
- **base** (rvt-b): Standard model, balanced performance
- **tiny** (rvt-t): Minimal model (if available)

### Checkpoints

Available checkpoints in `workspace/plugins/rvt-detection/models/`:

- `rvt-s-gen1.ckpt`: RVT-small trained on Gen1 datasets
- `rvt-b-gen1.ckpt`: RVT-base trained on Gen1 datasets
- `rvt-s-1mpx.ckpt`: RVT-small for 1 megapixel sensors
- `rvt-b-1mpx.ckpt`: RVT-base for 1 megapixel sensors

## Programmatic Usage

### Using the Plugin Directly

```python
import polars as pl
from rvt_detector import RVTDetectorPlugin
from evio.evlib_loader import load_events_with_evlib

# Load events
events = load_events_with_evlib("data/events.h5").collect()

# Initialize plugin
plugin = RVTDetectorPlugin(
    dataset_name="gen1",
    experiment="small",
    histogram_bins=10,
    window_duration_ms=50.0,
    device="cuda",
)

# Reset recurrent states (important for new sequences)
plugin.reset_states()

# Run inference
detections = plugin.process(events)

# Process results
for det in detections:
    print(f"Class {det.class_id}: {det.confidence:.2f} at {det.bbox}")
```

### Detection Format

Each detection is an `RVTDetection` dataclass:

```python
@dataclass
class RVTDetection:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float                         # Combined objectness * class confidence
    class_id: int                             # Class label index
```

## Implementation Details

### Event Processing Pipeline

1. **Event Loading**: Events loaded via evlib as Polars DataFrames
2. **Windowing**: Events partitioned into temporal windows
3. **Histogram**: Each window converted to stacked histogram tensor (bins × 2 × H × W)
4. **Inference**: RVT forward pass with recurrent state management
5. **Post-processing**: YOLOX-style NMS and confidence thresholding
6. **Visualization**: Detections overlaid on event frames

### Recurrent States

RVT maintains ConvLSTM hidden states across frames for temporal coherence:

- Call `plugin.reset_states()` at sequence boundaries
- States automatically updated during `plugin.process()`
- Improves detection quality on continuous streams

### Performance Considerations

- **CUDA highly recommended**: CPU inference is 10-100× slower
- **Batch size**: Currently fixed at 1 for real-time processing
- **Memory**: Recurrent states require ~200-500 MB VRAM depending on model size

## Troubleshooting

### Missing RVT Repository

**Error**: `RVT repository not found at workspace/plugins/rvt-detection/RVT`

**Solution**:
```bash
git submodule update --init --recursive
```

### Missing Checkpoint

**Error**: `RVT checkpoint not found at ...models/rvt-s-gen1.ckpt`

**Solution**:
1. Obtain `rvt-models.zip` and place in `workspace/plugins/rvt-detection/models/`
2. Run: `unzip-checkpoints`

### CUDA Not Available

**Warning**: `CUDA not available – RVT will run significantly slower on CPU`

**Solutions**:
- Install CUDA toolkit and PyTorch with CUDA support
- Or accept CPU-only inference (significantly slower)

### Hydra Config Errors

**Error**: `Missing Hydra config directory`

**Solution**:
- Ensure RVT submodule is properly initialized
- Verify `workspace/plugins/rvt-detection/RVT/config/` exists

### Import Errors

**Error**: `ImportError: cannot import name 'create_stacked_histogram'`

**Solution**:
- Ensure evio-core is installed: `uv sync`
- Verify evlib Python bindings are built correctly

## References

- RVT Paper: [Recurrent Vision Transformers for Object Detection with Event Cameras](https://arxiv.org/abs/2212.05598)
- RVT Repository: https://github.com/uzh-rpg/RVT
- evlib Documentation: See `docs/evlib-integration.md`
