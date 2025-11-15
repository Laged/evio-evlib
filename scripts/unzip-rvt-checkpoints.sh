#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="workspace/plugins/rvt-detection/models"
ZIP_PATH="$MODEL_DIR/rvt-models.zip"

if [ ! -f "$ZIP_PATH" ]; then
  echo "❌ Checkpoint archive not found: $ZIP_PATH"
  echo ""
  echo "Please obtain rvt-models.zip and place it in $MODEL_DIR"
  exit 1
fi

cd "$MODEL_DIR"
unzip -o rvt-models.zip >/dev/null
echo "✅ Extracted checkpoints into $MODEL_DIR"
ls -lh *.ckpt
