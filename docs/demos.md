# Event Camera Demos

This guide shows how to run different event-based detection demos.

## Prerequisites

⚠️ **CUDA Required**: The RVT detector requires CUDA. PyTorch attempts to load CUDA libraries on import, even when using `--device cpu`. To run without CUDA, you must install CPU-only PyTorch builds (see Troubleshooting section).

1. **Datasets extracted**: Run `unzip-datasets` if you haven't already
2. **Convert to HDF5**: For RVT demos, convert legacy .dat files to HDF5:
   ```bash
   convert-legacy-dat-to-hdf5 evio/data/fan/fan_varying_rpm_turning.dat
   convert-legacy-dat-to-hdf5 evio/data/drone_moving/drone_moving.dat
   ```
3. **RVT checkpoints**: Run `unzip-checkpoints` to extract model checkpoints
4. **CUDA toolkit**: NVIDIA GPU with CUDA support (or CPU-only PyTorch, see below)

## Run Demos

### Fan Detector

```bash
run-fan-detector
```

**What you'll see:**
- Event visualization showing rotating fan motion with varying RPM and turning movement
- Green bounding boxes around detected objects
- Confidence scores and class IDs overlaid
- Real-time inference on 50ms event windows

**Dataset**: Fan with varying RPM and turning motion
**Model**: RVT-Base trained on Gen1 (Prophesee Gen1 sensors)
**Resolution**: 1280x720 visualization
**Controls**: Press 'q' or ESC to quit

---

### Drone Detector

```bash
run-drone-detector
```

**What you'll see:**
- High-resolution event data (1 Mpx sensor)
- Bounding boxes tracking fast-moving drone and propellers
- Confidence scores for detections
- Gen4 sensor processing with moving camera

**Dataset**: Moving drone with spinning propellers
**Model**: RVT-Base trained on 1 Mpx (Gen4 sensors)
**Resolution**: 1280x720 visualization
**Controls**: Press 'q' or ESC to quit

---

## Advanced Usage

### Custom Detector Configuration

Use the general `run-detector` command for more control:

```bash
# Use small model on fan dataset
run-detector evio/data/fan/fan_varying_rpm_turning_legacy.h5 \
  --checkpoint workspace/plugins/rvt-detection/models/rvt-s-gen1.ckpt \
  --experiment small

# CPU mode for testing (slow)
run-detector evio/data/fan/fan_varying_rpm_turning_legacy.h5 \
  --device cpu \
  --max-windows 10

# Adjust visualization parameters
run-detector evio/data/drone_moving/drone_moving_legacy.h5 \
  --checkpoint workspace/plugins/rvt-detection/models/rvt-b-1mpx.ckpt \
  --dataset gen4 \
  --width 1920 \
  --height 1080 \
  --window-ms 100.0
```

### Available Models

**Gen1 Models** (Prophesee Gen1 sensors - 240x180):
- `rvt-s-gen1.ckpt` - Small (114 MB) ✅ Default
- `rvt-b-gen1.ckpt` - Base (213 MB)
- `rvt-t-gen1.ckpt` - Tiny (download separately)

**1 Mpx Models** (Gen4 sensors - higher resolution):
- `rvt-s-1mpx.ckpt` - Small (114 MB)
- `rvt-b-1mpx.ckpt` - Base (284 MB)
- `rvt-t-1mpx.ckpt` - Tiny (download separately)

## Legacy Demos

These demos use classical event processing (no neural networks):

### Fan Player (Legacy)

```bash
run-demo-fan
```

Plays fan dataset with legacy .dat loader.

### Fan Player (evlib HDF5)

```bash
run-demo-fan-ev3
```

Plays fan dataset using evlib's HDF5 loader.

### MVP Demos

```bash
# Event density visualization
run-mvp-1

# Voxel grid + FFT analysis
run-mvp-2
```

## Troubleshooting

### CUDA Not Available

**Error**: `ValueError: libcublas.so.*[0-9] not found` or `OSError: libcurand.so.10: cannot open shared object file`

**Cause**: PyTorch loads CUDA libraries on import, even when using `--device cpu`.

**Solutions**:

**Option 1: Install CUDA** (Recommended)
- Install NVIDIA CUDA Toolkit for your system
- Restart the nix shell after installation

**Option 2: CPU-Only PyTorch** (Slow, for testing only)
```bash
# Exit nix shell first
exit

# Install CPU-only PyTorch in workspace
cd workspace
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Re-enter nix shell and test
nix develop
run-detector <file> --device cpu --max-windows 10
```

⚠️ **Warning**: CPU inference is 10-100× slower than GPU. Not recommended for real-time use.

### Missing HDF5 Files

**Error**: `Event file not found: ...legacy.h5`

**Solution**: Convert .dat files to HDF5:
```bash
convert-legacy-dat-to-hdf5 evio/data/fan/fan_varying_rpm_turning.dat
convert-legacy-dat-to-hdf5 evio/data/drone_moving/drone_moving.dat
```

### Missing Checkpoints

**Error**: `Checkpoint not found`

**Solution**:
```bash
unzip-checkpoints
```

Or download missing models:
```bash
cd workspace/plugins/rvt-detection/models/
curl -L -o rvt-t-gen1.ckpt https://download.ifi.uzh.ch/rpg/RVT/checkpoints/gen1/rvt-t.ckpt
```

## Performance

**Recommended**: CUDA GPU with 4+ GB VRAM
**Expected FPS**:
- GPU (CUDA): 30-60 FPS depending on model size
- CPU: 1-5 FPS (not recommended for real-time)

**Model Size vs Speed**:
- Tiny: Fastest, lower accuracy
- Small: Good balance ✅ Default
- Base: Slower, higher accuracy
