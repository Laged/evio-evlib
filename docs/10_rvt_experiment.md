# RVT Object Detection - Experimental Integration

**Status**: Proof of concept, not production ready

## Summary

We successfully integrated RVT (Recurrent Vision Transformer) for event-based object detection. The tech works, but pre-trained models don't generalize to our fan/drone data - custom training would be needed for production use.

**What works:**
- ✅ RVT integration with evlib pipeline
- ✅ Inference processes event streams correctly
- ✅ Checkpoint loading and detection generation

**What doesn't:**
- ❌ Pre-trained models (trained on cars/pedestrians) don't detect fans/drones
- 🔧 Would need domain-specific training data

The code is preserved for future work with custom-trained models.

## Architecture

Three workspace components:

1. **rvt-adapter** (`workspace/libs/rvt-adapter`) - Shared utilities for configs, tensors, asset discovery
2. **rvt-detector** (`workspace/plugins/rvt-detection`) - Plugin with checkpoint management and inference
3. **detector-ui** (`workspace/apps/detector-ui`) - OpenCV visualization for real-time playback

## Setup

```bash
# 1. Clone RVT submodule
git submodule update --init --recursive

# 2. Extract pre-trained checkpoints
# Place rvt-models.zip in workspace/plugins/rvt-detection/, then:
unzip-checkpoints

# 3. Install dependencies
uv sync
```

**Requirements:**
- NVIDIA GPU with CUDA (~2-4GB VRAM)
- ~5GB disk space for PyTorch + dependencies
- Pre-converted HDF5 data (see below)

## Running Demos

### RVT Object Detector

⚠️ **Expectations**: Pre-trained models won't detect much on our data (out-of-distribution).

```bash
# Quick demos (won't detect much)
run-rvt-fan      # Fan with varying RPM
run-rvt-drone    # Moving drone

# Custom usage
run-detector <file.h5> [options]

# Examples:
run-detector evio/data/fan/fan_varying_rpm_turning_legacy.h5 \
  --checkpoint workspace/plugins/rvt-detection/models/rvt-s-gen1.ckpt \
  --experiment small

# CPU mode (very slow)
run-detector evio/data/fan/fan_varying_rpm_turning_legacy.h5 \
  --device cpu --max-windows 10
```

**Options:**
- `--checkpoint <path>` - Model checkpoint
- `--dataset gen1|gen4` - Sensor config
- `--experiment small|base|tiny` - Model size
- `--device cuda|cpu` - Inference device
- `--window-ms 50` - Event window duration
- `--width/--height` - Visualization size

**Controls**: Press 'q' or ESC to quit

### Data Preparation

RVT requires evlib HDF5 format:

```bash
convert-legacy-dat-to-hdf5 evio/data/fan/fan_varying_rpm_turning.dat
convert-legacy-dat-to-hdf5 evio/data/drone_moving/drone_moving.dat
```

### Available Checkpoints

**Gen1 Models** (Prophesee Gen1 sensors - 240×180):
- `rvt-t-gen1.ckpt` - Tiny (fastest, 51MB)
- `rvt-s-gen1.ckpt` - Small (good balance, 114MB)
- `rvt-b-gen1.ckpt` - Base (slowest, 213MB)

**1 Mpx Models** (Gen4 sensors - higher resolution):
- `rvt-t-1mpx.ckpt` - Tiny (fastest, 68MB)
- `rvt-s-1mpx.ckpt` - Small (good balance, 114MB)
- `rvt-b-1mpx.ckpt` - Base (slowest, 284MB)

All perform similarly (poorly) on our data since they weren't trained for fans/drones.

## Python API

```python
from rvt_detector import RVTDetectorPlugin
from evio.evlib_loader import load_events_with_evlib

# Load events
events = load_events_with_evlib("data/events.h5").collect()

# Initialize detector
plugin = RVTDetectorPlugin(
    dataset_name="gen1",
    experiment="small",
    device="cuda",
)

# Process
plugin.reset_states()  # Important for new sequences
detections = plugin.process(events)

for det in detections:
    print(f"Class {det.class_id}: {det.confidence:.2f} at {det.bbox}")
```

Each detection has: `bbox` (x1, y1, x2, y2), `confidence`, and `class_id`.

**How it works:**
1. Events → temporal windows → stacked histograms
2. RVT inference with recurrent states (ConvLSTM)
3. NMS post-processing
4. Visualization overlay

RVT maintains hidden states across frames for temporal coherence. Always call `reset_states()` at sequence boundaries.

---

## Other Detectors

### Classical CV Detectors

These use custom computer vision algorithms (no neural networks, no GPU required):

**Fan RPM Detector** - Ellipse fitting on event patterns
```bash
run-fan-detector  # Works on constant RPM fan dataset
```

**Drone Propeller Detector** - Event clustering and density analysis
```bash
run-drone-detector  # Works on stationary drone dataset
```

### Playback Demos

Simple event data visualization (no detection):

```bash
run-demo-fan        # Legacy .dat loader
run-demo-fan-ev3    # evlib HDF5 loader
run-mvp-1           # Event density visualization
run-mvp-2           # Voxel grid + FFT analysis
```

---

## Troubleshooting

**Missing RVT repo**: `git submodule update --init --recursive`

**Missing checkpoint**: Get `rvt-models.zip`, place in `workspace/plugins/rvt-detection/models/`, run `unzip-checkpoints`

**CUDA not available**:
- Install NVIDIA CUDA Toolkit OR
- Use CPU-only PyTorch (very slow): `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

**Import errors**: Run `uv sync` to install dependencies

**Missing HDF5 files**: Convert with `convert-legacy-dat-to-hdf5 <file.dat>`

## Performance

**Recommended**: CUDA GPU with 4+ GB VRAM

**Expected FPS:**
- GPU: 30-60 FPS (model dependent)
- CPU: 1-5 FPS (not usable for real-time)

**Model size trade-offs:**
- Tiny: Fastest, lower accuracy
- Small: Good balance ✅ Default
- Base: Slower, potentially higher accuracy

## References

- [RVT Paper](https://arxiv.org/abs/2212.05598) - Recurrent Vision Transformers for Object Detection
- [RVT Repository](https://github.com/uzh-rpg/RVT) - Original implementation
- [Git LFS required](https://git-lfs.github.com/) - For cloning checkpoint files (842MB total)
