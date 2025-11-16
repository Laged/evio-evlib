# Detector Demos Quick Start

**Want to see fan/drone detection in action?** This guide gets you running in 2 minutes.

---

## Prerequisites

```bash
# Install Nix with flakes enabled
# See: https://nixos.org/download.html

# Clone repo
git clone <repo-url> evio-evlib
cd evio-evlib
```

---

## Setup (One Time)

```bash
# Enter Nix environment
nix develop

# Initialize workspace
uv sync

# Extract datasets
unzip-datasets

# Convert to HDF5 for evlib demos
convert-legacy-dat-to-hdf5 evio/data/fan/fan_const_rpm.dat
convert-legacy-dat-to-hdf5 evio/data/drone_idle/drone_idle.dat
```

---

## Run Demos

### Fan RPM Detector

```bash
run-fan-detector
```

**What you'll see:**
- Pass 1: Green ellipse tracking rotating fan
- Pass 2: Blue circles on blade tips (DBSCAN clusters)
- Matplotlib: Angle tracking and RPM estimate (~300 RPM)

**Controls:** Press 'q' to quit

### Drone Propeller Detector

```bash
run-drone-detector
```

**What you'll see:**
- Green ellipses on 1-2 propellers
- Red "WARNING: DRONE DETECTED" overlay
- RPM estimates per propeller (~3000-10000 RPM)

**Controls:** Press 'q' to quit

---

## Troubleshooting

**"No module named evio"**
→ Run `uv sync` from repo root

**"File not found: evio/data/..."**
→ Run `unzip-datasets` to extract datasets

**"uv: command not found"**
→ Run `nix develop` to enter environment

**Window doesn't show**
→ Check X11/XQuartz on macOS, or DISPLAY on Linux

---

## Next Steps

- **See implementation:** `evio/scripts/fan_detector_demo.py`, `evio/scripts/drone_detector_demo.py`
- **evlib migration:** `docs/plans/2025-11-16-detector-commons-evlib-integration.md`
- **Architecture:** `docs/architecture.md`

---

**Questions?** Open an issue or check `docs/setup.md`
