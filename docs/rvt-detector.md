# RVT Object Detection - Exploratory Work

**Status**: Proof of concept, not production ready

## What we found

We successfully integrated RVT (Recurrent Vision Transformer) to test event-based object detection. The tech works, but the pre-trained models don't generalize to our fan/drone data. You'd need custom training to get useful results.

**What works:**
- ✅ RVT integration with evlib
- ✅ Inference pipeline processes event streams correctly
- ✅ Can load checkpoints and generate detections

**What doesn't:**
- ❌ Pre-trained models (Gen1/Gen4 benchmarks) don't work on our specific scenarios
- 🔧 Would need domain-specific training data

The code is here for reference if we want to revisit this with custom-trained models later.

## How it's organized

Three workspace components:

1. **rvt-adapter** (`workspace/libs/rvt-adapter`) - Shared helpers for Hydra configs, event histograms, asset discovery
2. **rvt-detector** (`workspace/plugins/rvt-detection`) - The detector plugin with checkpoint management and inference
3. **detector-ui** (`workspace/apps/detector-ui`) - OpenCV visualization app for real-time detection playback

## Quick setup

```bash
# 1. Get the RVT submodule
git submodule update --init --recursive

# 2. Extract checkpoints (if you have rvt-models.zip)
unzip-checkpoints

# 3. Sync dependencies
uv sync
```

Note: GPU highly recommended, CPU inference is painfully slow.

## Try it out

Basic demos (won't detect much due to pre-trained model limitations):

```bash
# Fan dataset
run-rvt-fan

# Drone dataset
run-rvt-drone
```

Press 'q' or ESC to quit.

### Custom usage

```bash
run-detector <your_file.h5> --checkpoint <path_to.ckpt> --dataset gen1
```

Common options:
- `--dataset gen1|gen4` - Sensor config
- `--checkpoint <path>` - Model checkpoint
- `--device cuda|cpu` - Inference device
- `--window-ms 50` - Event window size

## Available models

Pre-trained checkpoints in `workspace/plugins/rvt-detection/models/`:

- `rvt-s-gen1.ckpt` / `rvt-b-gen1.ckpt` - Gen1 sensors (240x180)
- `rvt-s-1mpx.ckpt` / `rvt-b-1mpx.ckpt` - Gen4 sensors (1 Mpx)

The `-s` models are smaller/faster, `-b` are larger/slower.

## Using in code

```python
from rvt_detector import RVTDetectorPlugin
from evio.evlib_loader import load_events_with_evlib

# Load events
events = load_events_with_evlib("data/events.h5").collect()

# Initialize plugin
plugin = RVTDetectorPlugin(
    dataset_name="gen1",
    experiment="small",
    device="cuda",
)

plugin.reset_states()  # Important for new sequences
detections = plugin.process(events)

for det in detections:
    print(f"Class {det.class_id}: {det.confidence:.2f} at {det.bbox}")
```

Each detection has: `bbox` (x1, y1, x2, y2), `confidence`, and `class_id`.

## How it works

1. Events → temporal windows → stacked histograms
2. RVT inference with recurrent states (ConvLSTM)
3. NMS post-processing
4. Overlay on visualization

RVT maintains hidden states across frames for temporal coherence. Call `reset_states()` at sequence boundaries.

## Common issues

**Missing RVT repo**: `git submodule update --init --recursive`

**Missing checkpoint**: Get `rvt-models.zip`, place in `workspace/plugins/rvt-detection/models/`, run `unzip-checkpoints`

**No CUDA**: You can use CPU but it'll be slow. Install PyTorch with CUDA support if you have a GPU.

**Import errors**: Run `uv sync` to ensure evio-core and evlib are installed.

## References

- [RVT Paper](https://arxiv.org/abs/2212.05598) - Recurrent Vision Transformers for Object Detection with Event Cameras
- [RVT Repository](https://github.com/uzh-rpg/RVT) - Original implementation
