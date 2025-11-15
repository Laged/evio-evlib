#!/usr/bin/env bash
set -euo pipefail

# Run event-based object detector
# Usage: run-detector <event-file> [options...]

show_help() {
  cat << EOF
Usage: run-detector <event-file> [options...]

Run event-based object detector on event data.

Arguments:
  event-file              : Path to event data (.dat/.h5/.aedat)

Options:
  --checkpoint PATH       : Path to RVT checkpoint (default: rvt-s-gen1.ckpt)
  --dataset gen1|gen4     : Sensor type (default: gen1)
  --experiment small|base|tiny : Model size (default: small)
  --device cuda|cpu       : Compute device (default: auto -> prefers CUDA)
  --window-ms MS          : Time window duration in ms (default: 50.0)
  --max-windows N         : Limit playback to N windows
  --width W               : Frame width for visualization (default: 1280)
  --height H              : Frame height for visualization (default: 720)
  --histogram-bins N      : Temporal bins for histogram (default: 10)
  --delay-ms MS           : OpenCV waitKey delay (default: 1)
  --no-loop               : Disable playback looping
  --help, -h              : Show this help message

Examples:
  # Basic usage with defaults
  run-detector evio/data/fan/fan_const_rpm_legacy.h5

  # Use different checkpoint
  run-detector evio/data/fan/fan_const_rpm_legacy.h5 --checkpoint workspace/plugins/rvt-detection/models/rvt-b-gen1.ckpt

  # Run on drone data with 1 Mpx checkpoint
  run-detector evio/data/drone_idle/drone_idle_legacy.h5 --checkpoint workspace/plugins/rvt-detection/models/rvt-s-1mpx.ckpt --dataset gen4

  # CPU mode with limited windows for testing
  run-detector evio/data/fan/fan_const_rpm_legacy.h5 --device cpu --max-windows 10
EOF
}

# Parse arguments
if [ $# -lt 1 ]; then
  show_help
  exit 1
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  show_help
  exit 0
fi

EVENT_FILE="$1"
shift

# Default values
DEFAULT_CHECKPOINT="workspace/plugins/rvt-detection/models/rvt-s-gen1.ckpt"
CHECKPOINT="$DEFAULT_CHECKPOINT"
DATASET_NAME="gen1"
EXPERIMENT="small"
DEVICE=""
WINDOW_MS="50.0"
WIDTH="1280"
HEIGHT="720"
HISTOGRAM_BINS="10"
DELAY_MS="1"
MAX_WINDOWS=""
NO_LOOP=""

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --dataset)
      DATASET_NAME="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --window-ms)
      WINDOW_MS="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --histogram-bins)
      HISTOGRAM_BINS="$2"
      shift 2
      ;;
    --delay-ms)
      DELAY_MS="$2"
      shift 2
      ;;
    --max-windows)
      MAX_WINDOWS="--max-windows $2"
      shift 2
      ;;
    --no-loop)
      NO_LOOP="--no-loop"
      shift
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Run 'run-detector --help' for usage information"
      exit 1
      ;;
  esac
done

# Validate inputs
if [ ! -f "$EVENT_FILE" ]; then
  echo "❌ Event file not found: $EVENT_FILE"
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "❌ Checkpoint not found: $CHECKPOINT"
  echo ""
  echo "Available checkpoints:"
  ls -1 workspace/plugins/rvt-detection/models/*.ckpt 2>/dev/null || echo "  (none found)"
  exit 1
fi

RVT_REPO="workspace/plugins/rvt-detection/RVT"
if [ ! -d "$RVT_REPO/config" ]; then
  echo "❌ RVT repository not initialized"
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

# Display configuration
echo "🎯 Running Event Camera Detector"
echo "  Event file: $EVENT_FILE"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: $DATASET_NAME"
echo "  Experiment: $EXPERIMENT"
if [ -n "$DEVICE" ]; then
  display_device="$DEVICE"
else
  display_device="auto"
fi
echo "  Device: $display_device"
echo "  Window: ${WINDOW_MS}ms"
echo ""

# Run detector
DEVICE_ARGS=()
if [ -n "$DEVICE" ]; then
  DEVICE_ARGS+=(--device "$DEVICE")
fi

cd workspace
uv run --package detector-ui python -m detector_ui.main \
  "../$EVENT_FILE" \
  --rvt-repo plugins/rvt-detection/RVT \
  --checkpoint "../$CHECKPOINT" \
  --dataset-name "$DATASET_NAME" \
  --experiment "$EXPERIMENT" \
  "${DEVICE_ARGS[@]}" \
  --window-ms "$WINDOW_MS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --histogram-bins "$HISTOGRAM_BINS" \
  --delay-ms "$DELAY_MS" \
  $MAX_WINDOWS \
  $NO_LOOP
