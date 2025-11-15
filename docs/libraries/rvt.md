# RVT – Recurrent Vision Transformer for Event Cameras

This document summarizes the upstream RVT project (https://github.com/uzh-rpg/RVT) and how we integrate it into the `evio-evlib` architecture. Use it as the canonical reference when planning detector plugins or Claude skills around RVT.

---

## 1. Upstream Snapshot

- **Project:** RVT (Recurrent Vision Transformer for real-time event-based object detection)
- **Authors:** Robotics and Perception Group, University of Zurich
- **Language:** PyTorch (Python) with CUDA kernels
- **Purpose:** Detect/track objects from event tensors with transformer attention + temporal recurrence
- **Input Tensor:** `(batch, time_bins, polarity, height, width)` (e.g., `(1, 10, 2, 480, 640)`)
- **Outputs:** Bounding boxes + classes; trained on Prophesee GEN1/GEN4 datasets

---

## 2. Architectural Role

| Layer (`docs/architecture.md`) | RVT Contribution | Repo Touchpoints |
| --- | --- | --- |
| Layer 3 – Processing Pipelines | Deep-learning branch consuming evlib tensors | `workspace/plugins/*` (dedicated RVT detector plugin) |
| Layer 4 – Task Modules | Provides robust detection + classification (fan bbox, drone tracking) | `workspace/plugins/fan-bbox` now, future detectors later |
| Layer 5 – Fusion | RVT outputs feed ensembles with classical RPM/rotation estimates | `workspace/libs/evio-core/src/evio_core/fusion.py` (planned) |
| Layer 6 – Apps | Detector UI loads RVT weights, toggles between classical + RVT detectors | `workspace/apps/detector-ui` |

RVT is the "heavy" inference path that complements our classical algorithms. evlib supplies the input tensors (see `docs/libraries/evlib.md`), ensuring both paths share the same data source.

---

## 3. Dependencies & Environment

Minimum stack (per upstream RVT repo):
- Python 3.8+
- PyTorch 1.13+ (CUDA 11.7 recommended) plus `torchvision`, `torchaudio`
- Auxiliary libs: `tqdm`, `numpy`, `opencv-python`, `matplotlib`, `einops`
- GPU with >8 GB VRAM for inference (10-bin 640×480 tensors)

Inside our UV workspace:

```bash
# Example: add RVT deps to a plugin (e.g., fan-bbox or rvt-detector)
uv add --package fan-bbox torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv add --package fan-bbox einops opencv-python tqdm
uv sync
```

Always run this from `nix develop` so CUDA, pkg-config, etc., are visible. CPU-only inference is possible for debugging but far slower.

---

## 4. Data & Preprocessing (evlib ↔ RVT)

RVT expects temporally binned event tensors. `evio/docs/refactor-to-evlib.md` describes how `evlib.representations.create_stacked_histogram(...)` produces exactly what RVT requires, including:
- `bins = 10`
- `window_duration_ms = 50`
- Polarity split channels (ON/OFF)

Recommended pipeline:

```python
import torch
import evlib.representations as evr

def events_to_rvt_tensor(events, height, width, bins=10):
    hist = evr.create_stacked_histogram(events, height=height, width=width, bins=bins, window_duration_ms=50.0)
    tensor = torch.zeros((bins, 2, height, width), dtype=torch.float32)
    for row in hist.iter_rows(named=True):
        tensor[row["time_bin"], row["polarity"], row["y"], row["x"]] = row["count"]
    return tensor
```

This is the same helper captured in `evio/docs/evlib-rvt-architecture.md` and should live in `workspace/libs/evio-core`.

---

## 5. Planned Plugin Layout

1. **`workspace/plugins/rvt-detector/` (new)** – Houses model loading, preprocessing, inference loop.
2. **`workspace/libs/evio-core/plugins.py`** – Defines `DetectorPlugin` protocol. RVT implementation will subclass/implement it.
3. **Runtime flow:**
   - `evio-core` provides `EventSource` windows (file or live).
   - Plugin converts window to RVT tensor via helpers above.
   - RVT model produces detections; plugin returns bounding boxes, confidences, classes.
   - UI displays boxes and fuses with classical outputs for RPM/sanity checking.

---

## 6. Data Challenges & Mitigations

- **Legacy `.dat` format:** RVT only sees evlib outputs. Until fan `.dat` recordings are converted to EVT2/AEDAT/H5 (`evio/docs/dat-format-compatibility.md`), we must either (a) convert them in preprocessing or (b) run RVT on public GEN1/GEN4 datasets for benchmarking.
- **Resolution consistency:** RVT weights are trained on specific sensor resolutions (e.g., 346×260, 640×480). Document resolution per dataset and resize or retrain accordingly.
- **Temporal alignment:** Keep window duration/bins identical to training (50 ms, 10 bins) to avoid domain shift.
- **Compute budget:** Provide clear instructions in `.claude/skills` on which GPU is required and whether to fall back to CPU mocks if hardware is missing.

---

## 7. Inference Workflow (for Claude/Codex)

```bash
# 1. Enter dev shell
nix develop

# 2. Sync deps (once)
uv sync

# 3. Run RVT detector (placeholder command)
uv run --package rvt-detector python -m rvt_detector.run \
  --weights artifacts/rvt_gen4.pth \
  --source data/fan_const_rpm.h5 \
  --visualize
```

Implementation tasks before this works:
- Port RVT inference scripts into `workspace/plugins/rvt-detector`.
- Wire CLI args to our EventSource abstraction.
- Save weights into `artifacts/` or fetch on demand (ensure instructions for restricted networks).

---

## 8. Testing & Validation

- **Offline validation:** Use Prophesee public datasets (converted to `.h5`) to compare detections with upstream RVT repository results.
- **Cross-check with classical detectors:** In `apps/detector-ui`, render both outputs simultaneously to ensure coordinate systems match (origin at bottom-left, same scaling).
- **Performance:** Benchmark end-to-end latency (window creation + RVT inference) aiming for <30 ms on an RTX 3080 for 640×480 windows.

---

## 9. LLM/Skill Notes

When creating `.claude/skills` entries:
- Reference this file plus `evio/docs/evlib-rvt-architecture.md`.
- Include reminders to check GPU availability (`nvidia-smi`) before launching RVT.
- Provide fallbacks for CPU dry-runs (reduced resolution/bins, or record-only mode).
- Highlight data-format requirements so models do not assume the legacy `.dat` files work in evlib.

With this reference, agents can reason about RVT’s expectations, know where it plugs into the repo, and keep instructions consistent with our planned architecture.

