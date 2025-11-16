# Development Environment Setup

**Date:** 2025-11-15
**Status:** Ready for use

---

## Quick Start

### Prerequisites

- Nix with flakes enabled
- Git

### Setup (One Time)

```bash
# Clone repo
git clone <repo-url> evio-evlib
cd evio-evlib

# Enter Nix environment
nix develop

# Initialize workspace (first time only)
uv sync

# Workspace is ready:
# - All packages installed
# - Shell aliases available
# - Ready to run demos
```

You're ready to develop!

### Data Files

Event camera dataset files are located in `evio/data/`:
- `evio/data/fan/*.dat` - Fan rotation datasets (202-489 MB each)
- Large binary files, currently tracked in git for hackathon

**Note:** These files may be gitignored in future. See `evio/data/README.md` for details on obtaining datasets.

---

## Daily Workflow

### Enter Environment

```bash
cd evio-evlib
nix develop
```

**First time setup:**
```bash
# After entering nix shell for the first time:
uv sync  # Install all workspace packages
```

**After git pull with lockfile changes:**
```bash
git pull
nix develop
uv sync  # Update packages if uv.lock changed
```

### Run Commands

All commands run from repo root using `uv run --package <member>`:

```bash
# Run evio legacy demos (note: scripts and data are in evio/ directory)
uv run --package evio python evio/scripts/play_dat.py evio/data/fan/fan_const_rpm.dat

# Use shell aliases (already use correct paths)
run-demo-fan
run-mvp-1
run-mvp-2

# Run workspace member code
uv run --package evio-core python -m evio_core
```

### Add Dependencies

```bash
# Add to workspace member
uv add --package evio-core scikit-learn

# Update lockfile
uv sync

# Verify lockfile updated
git status  # Should show uv.lock modified
```

### Never Use pip

❌ `pip install <package>`
✅ `uv add --package <member> <package> && uv sync`

---

## Workspace Structure

```
evio-evlib/               # UV workspace root
├── flake.nix            # Nix environment (system deps)
├── pyproject.toml       # UV workspace config
├── uv.lock              # Shared Python lockfile (committed)
├── .venv/               # Shared virtual env (auto-generated)
│
├── evio/                # Workspace member: legacy evio
│   ├── pyproject.toml
│   ├── scripts/         # MVP demos
│   └── src/evio/
│
└── workspace/           # Workspace members
    ├── libs/evio-core/      # Core library
    ├── plugins/fan-bbox/    # Fan detector
    └── apps/detector-ui/    # Interactive UI
```

---

## Package Management

### System Packages (Nix)

Provided by `flake.nix`:
- Python 3.11
- UV package manager
- Rust toolchain (for evlib)
- OpenCV
- pkg-config, zlib

**Never add Python packages to flake.nix**

### Python Packages (UV)

Managed via `pyproject.toml` in workspace members:
- evlib, polars, numpy (evio-core)
- opencv-python (detector-ui)
- scikit-learn (plugins, as needed)

**Always use UV, never pip**

---

## Verification

### Test 1: Workspace Members Import

```bash
uv run python -c "import evio_core; print('evio-core:', evio_core.__version__)"
uv run python -c "import fan_bbox; print('fan-bbox:', fan_bbox.__version__)"
uv run python -c "import detector_ui; print('detector-ui:', detector_ui.__version__)"
```

Expected: All print version 0.1.0

### Test 2: evio Legacy Works

```bash
uv run --package evio python -c "import evio; print('evio works')"
```

Expected: `evio works`

### Test 3: evlib Available

```bash
uv run python -c "import evlib; print('evlib available')"
```

Expected: `evlib available`

---

## Running Detector Demos

### Prerequisites

```bash
nix develop
unzip-datasets  # Extract datasets (one-time)

# Convert legacy .dat to HDF5 for evlib demos
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
convert-legacy-dat-to-hdf5 evio/data/drone_idle/drone_idle.dat
```

### Fan Detector Demo

**Legacy loader:**
```bash
run-fan-detector
```

Shows:
- Pass 1: Ellipse fitting on rotating fan (30ms windows)
- Pass 2: DBSCAN blade tracking (0.5ms windows)
- Matplotlib plots: Angle tracking and RPM estimation

**Controls:** Press 'q' or ESC to quit

### Drone Detector Demo

**Legacy loader:**
```bash
run-drone-detector
```

Shows:
- Dual propeller detection (up to 2 ellipses)
- Per-propeller RPM estimation
- "WARNING: DRONE DETECTED" overlay

**Controls:** Press 'q' or ESC to quit

### Expected Outputs

**Fan Detector:**
- RPM: ~250-400 (depending on dataset)
- Single ellipse tracking fan rotation

**Drone Detector:**
- RPM: ~3000-10000 per propeller
- 1-2 ellipses (horizontal orientation)
- Warning overlay

---

## Next Steps

See `docs/demo-workflow.md` for evlib migration path.

---

## Troubleshooting

### "uv: command not found"

You're not in Nix shell. Run: `nix develop`

### "Package not found"

Run `uv sync` to ensure all packages installed.

### "Import error"

Ensure you're using `uv run` to execute Python:
```bash
uv run python script.py
```

### Need to add dependency

Never pip! Use UV:
```bash
uv add --package <member> <package>
uv sync
```

---

## Next Steps

See `.claude/skills/dev-environment.md` for mandatory workflow rules.

See `docs/architecture.md` for system design.

See `docs/plans/` for implementation plans.

---

## Work Streams

Three parallel development tracks:

1. **nix-infra** (this branch): Infrastructure ✅ Complete
2. **hackathon-poc**: Implement evio-core, detectors, UI
3. **realtime-integration**: Design StreamEventAdapter

Each uses git worktrees for isolated development.
