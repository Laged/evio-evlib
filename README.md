# Event Camera Detection Workbench - Monorepo

**Sensofusion Junction Hackathon Challenge**: Build production-ready event camera detection system with evlib integration and real-time streaming capabilities.

---

## Repository Structure

```
evio-evlib/                        # Monorepo root
├── README.md                      # This file
├── docs/                          # Parent-level documentation
│   └── architecture.md           # Architecture design (moved from evio/)
│
├── evio/                          # Original evio implementation (reference/legacy)
│   ├── src/evio/
│   ├── scripts/                   # MVP demos (mvp_1, mvp_2, etc.)
│   ├── docs/                      # Original documentation
│   └── flake.nix                 # Original Nix environment
│
└── workspace/                     # NEW: UV workspace (created in worktrees)
    ├── pyproject.toml            # UV workspace config
    ├── libs/evio-core/           # Core library with evlib
    ├── plugins/                  # Detection plugins
    │   ├── fan-bbox/
    │   ├── fan-rpm/
    │   └── drone-tracker/
    └── apps/detector-ui/         # Interactive workbench
```

---

## Three Parallel Work Streams

This monorepo supports **three independent work streams** using **git worktrees** for isolated development:

### Work Stream 1: Nix Infrastructure 🔧

**Branch:** `nix-infra`
**Worktree:** `../evio-nix-infra/`
**Owner:** Infrastructure team
**Timeline:** Day 1-2

**Goals:**
- Set up UV workspace structure
- Enhance `flake.nix` for UV + evlib
- Create reproducible dev environment
- Document setup process

**Deliverables:**
- ✅ `workspace/pyproject.toml` (UV workspace config)
- ✅ Enhanced `flake.nix` with UV, Rust toolchain, OpenCV
- ✅ `workspace/libs/evio-core/` skeleton
- ✅ Setup documentation in `docs/setup.md`

**Commands:**
```bash
# Create worktree
git worktree add -b nix-infra ../evio-nix-infra

# Work in isolation
cd ../evio-nix-infra
nix develop
# Set up workspace structure
# Test: nix develop && cd workspace && uv sync
```

---

### Work Stream 2: Hackathon PoC 🎯

**Branch:** `hackathon-poc`
**Worktree:** `../evio-hackathon-poc/`
**Owner:** Algorithm team
**Timeline:** Day 2-5

**Goals:**
- Implement evlib integration
- Build first detector (fan bounding box)
- Create interactive UI with hot-swapping
- Prove end-to-end pipeline works

**Deliverables:**
- ✅ `libs/evio-core/` with FileEventAdapter using evlib
- ✅ `plugins/fan-bbox/` minimal detector (time surface → bbox)
- ✅ `apps/detector-ui/` interactive workbench
- ✅ Working demo: load .dat → detect fan → visualize
- ✅ Performance benchmarks (evlib vs manual parsing)

**Commands:**
```bash
# Create worktree
git worktree add -b hackathon-poc ../evio-hackathon-poc

# Work in isolation
cd ../evio-hackathon-poc
nix develop
cd workspace
uv sync

# Develop
cd plugins/fan-bbox
# Edit detector.py
uv run pytest

# Run PoC
cd ../..
uv run detector-ui ../data/fan/fan_const_rpm.dat
```

---

### Work Stream 3: Real-Time Integration 📡

**Branch:** `realtime-integration`
**Worktree:** `../evio-realtime/`
**Owner:** Integration team
**Timeline:** Async (implement at hackathon venue)

**Goals:**
- Design StreamEventAdapter interface
- Document Metavision SDK integration points
- Prepare for live camera deployment
- Create venue deployment guide

**Deliverables:**
- ✅ `StreamEventAdapter` stub implementation
- ✅ Metavision SDK integration plan
- ✅ Venue deployment checklist
- ✅ Quick-start guide for live cameras

**Commands:**
```bash
# Create worktree
git worktree add -b realtime-integration ../evio-realtime

# Work in isolation
cd ../evio-realtime
nix develop
cd workspace/libs/evio-core

# Design StreamEventAdapter
# Document integration points

# At venue: Install Metavision SDK
uv add metavision-sdk-core metavision-sdk-driver
# Implement adapter
uv run detector-ui camera:0
```

---

## Quick Start

### Initial Setup (Everyone)

```bash
# Clone monorepo
git clone <repo-url> evio-evlib
cd evio-evlib

# Enter Nix environment
nix develop

# Check status
git worktree list  # See all active worktrees
```

### Join a Work Stream

```bash
# Option 1: Create new worktree (if not exists)
git worktree add -b <branch-name> ../<worktree-dir>

# Option 2: Use existing worktree
cd ../<worktree-dir>

# Start working
nix develop
cd workspace  # (if workspace exists)
uv sync
```

### Sync Your Work

```bash
# In your worktree
git add .
git commit -m "feat: implement X"
git push origin <branch-name>

# Back in main repo
cd /path/to/evio-evlib
git fetch origin
git merge origin/<branch-name>  # Or create PR
```

---

## Development Workflow

### Day-to-Day (Per Work Stream)

**Nix Infra Team:**
```bash
cd ../evio-nix-infra
# Edit flake.nix
nix flake check
git commit -m "build: enhance flake with UV support"
```

**Hackathon PoC Team:**
```bash
cd ../evio-hackathon-poc/workspace
# Work on detector
cd plugins/fan-bbox
uv add scikit-learn
# Edit detector.py
uv run pytest
git commit -m "feat: add DBSCAN clustering to fan detector"
```

**Real-Time Team:**
```bash
cd ../evio-realtime/workspace
# Design stream adapter
cd libs/evio-core
# Edit adapters.py (stub)
git commit -m "design: add StreamEventAdapter interface"
```

### Integration (Periodic)

```bash
# In main repo
cd /path/to/evio-evlib

# Merge nix-infra changes
git merge origin/nix-infra

# Merge hackathon-poc changes
git merge origin/hackathon-poc

# Merge realtime-integration changes
git merge origin/realtime-integration

# Resolve conflicts if any
git push origin main
```

---

## Architecture Overview

### Plugin-Based Detection System

**Key Abstractions:**

1. **EventSource Protocol** - Unified interface for files and streams
   - `FileEventAdapter` (evlib-based, 10x faster)
   - `StreamEventAdapter` (Metavision SDK, future)

2. **DetectorPlugin Protocol** - Extensible detection algorithms
   - `FanBBoxDetector` (Challenge 1)
   - `FanRPMDetector` (Challenge 2)
   - `DroneTracker` (Challenge 3)

3. **Interactive UI** - Hot-swapping for rapid experimentation
   - Press `1, 2, 3` → switch detector
   - Press `d` → load different .dat file
   - Press `l` → toggle looping

**Data Flow:**
```
.dat file → evlib.load_events() → FileEventAdapter.get_window()
  → DetectorPlugin.process() → Visualizer.draw()
  → OpenCV window with results
```

**Performance Gains (evlib):**
- File loading: **10x faster**
- Voxel grids: **55x faster**
- Histograms: **200x faster**
- ROI filtering: **53x faster**

See `docs/architecture.md` for complete design.

---

## Dependencies

### System Dependencies (via Nix)

```nix
# flake.nix provides:
- Python 3.11
- UV package manager
- Rust toolchain (for evlib compilation)
- OpenCV system libraries
- pkg-config
```

### Python Dependencies (via UV)

**Core Library (`libs/evio-core`):**
- `evlib>=0.8.0` - Rust-powered event processing
- `polars>=0.20.0` - Fast DataFrames
- `numpy>=1.24.0`

**Plugins (examples):**
- `scikit-learn>=1.3.0` - DBSCAN clustering
- `scipy>=1.11.0` - Frequency analysis

**App (`apps/detector-ui`):**
- `opencv-python>=4.8.0` - Visualization

### Installing Dependencies

```bash
# Enter Nix shell (provides system deps)
nix develop

# Install Python packages (per workspace)
cd workspace
uv sync  # Installs everything from workspace lockfile
```

---

## Testing

### Run Tests (Per Work Stream)

```bash
cd workspace

# Test core library
uv run pytest libs/evio-core/tests -v

# Test specific plugin
uv run pytest plugins/fan-bbox/tests -v

# Test everything
uv run pytest -v
```

### Benchmarks

```bash
# Compare evlib vs manual parsing
cd workspace
uv run python tools/benchmark.py ../data/fan/fan_const_rpm.dat

# Expected output:
#   Custom loader: 1200ms
#   evlib loader:  120ms
#   Speedup: 10x
```

---

## Deployment

### Development (Offline)

```bash
# Work with .dat files
nix develop
cd workspace
uv run detector-ui ../data/fan/fan_const_rpm.dat
```

### Production (Hackathon Venue)

```bash
# Install Metavision SDK (provided by Sensofusion)
# On Linux machines at venue

# Add SDK dependency
cd workspace/libs/evio-core
uv add metavision-sdk-core metavision-sdk-driver

# Implement StreamEventAdapter (30 minutes)
# Edit src/evio_core/adapters.py

# Test with live camera
cd ../..
uv run detector-ui camera:0

# Same app, same plugins, now with live data!
```

---

## Troubleshooting

### Nested Git Repositories

**Problem:** `evio/` is a git repo inside monorepo.

**Solution:**
```bash
# Option 1: Treat as regular directory
echo "evio/.git" >> .gitignore
git add evio/  # Adds files, ignores .git

# Option 2: Make it a submodule
git rm --cached evio
git submodule add <evio-url> evio
```

### UV Workspace Issues

**Problem:** `uv sync` fails with dependency conflicts.

**Solution:**
```bash
# Clear cache
uv cache clean

# Update lockfile
cd workspace
rm uv.lock
uv lock

# Sync again
uv sync
```

### evlib Compilation Errors

**Problem:** `evlib` fails to compile (missing Rust toolchain).

**Solution:**
```bash
# Ensure in Nix shell (provides Rust)
exit  # Exit current shell
nix develop  # Re-enter

# Verify Rust available
rustc --version
cargo --version

# Retry evlib install
cd workspace
uv sync --reinstall evlib
```

---

## Documentation

- **Architecture Design:** `docs/architecture.md`
- **Setup Guide:** `docs/setup.md` (TODO: created by nix-infra team)
- **API Reference:** `workspace/libs/evio-core/README.md` (TODO)
- **Plugin Development:** `workspace/tools/plugin-template/README.md` (TODO)
- **Venue Deployment:** `docs/venue-deployment.md` (TODO: created by realtime team)

---

## Contributing

### Adding a New Plugin

```bash
# In hackathon-poc worktree
cd ../evio-hackathon-poc/workspace/plugins

# Copy template
cp -r fan-bbox my-detector

# Edit metadata
cd my-detector
# Edit pyproject.toml (change name to "my-detector")
# Edit src/my_detector/detector.py

# Register in workspace
cd ../..
# Add "plugins/my-detector" to workspace/pyproject.toml members

# Install
uv sync

# Test
uv run pytest plugins/my-detector/tests
```

### Merging Changes from Worktree

```bash
# In main repo
cd /path/to/evio-evlib

# Fetch changes
git fetch origin <branch-name>

# Option 1: Merge directly
git merge origin/<branch-name>

# Option 2: Create PR for review
# (Use GitHub/GitLab web UI)

# Push to main
git push origin main
```

---

## Success Criteria

### Week 1 (Nix Infra + PoC)
- ✅ Three worktrees active and working
- ✅ UV workspace with evio-core, fan-bbox, detector-ui
- ✅ Nix environment provides all system deps
- ✅ `uv sync` installs all Python deps
- ✅ Demo: Fan bounding box detection working

### Week 2 (Enhancement)
- ✅ Multiple detector plugins (bbox, rpm, tracking)
- ✅ Interactive hot-swapping (plugins and data)
- ✅ Performance benchmarks showing 10-200x speedup

### At Hackathon (Real-Time)
- ✅ StreamEventAdapter implemented
- ✅ Live camera demonstration
- ✅ Same app running on files and live streams

---

## Resources

- **evlib Documentation:** https://github.com/ac-freeman/evlib
- **UV Documentation:** https://docs.astral.sh/uv/
- **Nix Flakes:** https://nixos.wiki/wiki/Flakes
- **Polars:** https://pola-rs.github.io/polars/
- **Metavision SDK:** https://docs.prophesee.ai/

---

## Team Communication

**Work Stream Leads:**
- Nix Infrastructure: TBD
- Hackathon PoC: TBD
- Real-Time Integration: TBD

**Sync Meetings:**
- Daily standups (async): Share progress in each worktree
- Integration sync (bi-weekly): Merge branches, resolve conflicts
- Venue deployment (at hackathon): Real-time team leads integration

---

## License

MIT (inherited from original evio)

---

**Last Updated:** 2025-11-15
**Repository:** https://github.com/<your-org>/evio-evlib
**Contact:** <your-email>
