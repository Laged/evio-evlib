# Next Steps - Getting Started with evio-evlib

**Repository:** https://github.com/Laged/evio-evlib

This document guides you through setting up your local environment and starting work on any of the three parallel work streams.

---

## Initial Setup (For New Contributors)

### 1. Clone the Repository

```bash
# Clone the monorepo
git clone git@github.com:Laged/evio-evlib.git
cd evio-evlib

# Verify you're on main branch
git branch
# Should show: * main
```

### 2. Set Up All Three Worktrees

Create isolated working directories for each work stream:

```bash
# Create worktree for Nix Infrastructure (Work Stream 1)
git worktree add -b nix-infra ../evio-nix-infra origin/nix-infra

# Create worktree for Hackathon PoC (Work Stream 2)
git worktree add -b hackathon-poc ../evio-hackathon-poc origin/hackathon-poc

# Create worktree for Real-Time Integration (Work Stream 3)
git worktree add -b realtime-integration ../evio-realtime origin/realtime-integration
```

### 3. Verify Worktree Setup

```bash
git worktree list
```

Expected output:
```
/home/you/evio-evlib              <commit> [main]
/home/you/evio-nix-infra          <commit> [nix-infra]
/home/you/evio-hackathon-poc      <commit> [hackathon-poc]
/home/you/evio-realtime           <commit> [realtime-integration]
```

### 4. Enter Nix Development Environment

```bash
# In the main repo or any worktree
nix develop

# This provides:
# - Python 3.11
# - UV package manager
# - Rust toolchain (for evlib)
# - OpenCV system libraries
# - All system dependencies
```

---

## Choose Your Work Stream

### Work Stream 1: Nix Infrastructure 🔧

**Who:** Infrastructure team
**Timeline:** Day 1-2
**Location:** `../evio-nix-infra/`

**Goals:**
- Create UV workspace structure
- Enhance `flake.nix` for UV + evlib
- Set up reproducible dev environment
- Document setup process

**Get Started:**
```bash
cd ../evio-nix-infra
nix develop

# Your tasks:
# 1. Create workspace/ directory structure
# 2. Write workspace/pyproject.toml (UV workspace config)
# 3. Create libs/evio-core/ skeleton
# 4. Update flake.nix with UV support
# 5. Test: nix develop && cd workspace && uv sync

# See README.md "Work Stream 1" section for details
```

---

### Work Stream 2: Hackathon PoC 🎯

**Who:** Algorithm/ML team
**Timeline:** Day 2-5
**Location:** `../evio-hackathon-poc/`

**Goals:**
- Implement evlib integration (FileEventAdapter)
- Build first detector (fan bounding box)
- Create interactive UI with hot-swapping
- Demonstrate end-to-end pipeline

**Get Started:**
```bash
cd ../evio-hackathon-poc
nix develop

# Your tasks:
# 1. Implement libs/evio-core/src/evio_core/loaders.py (evlib integration)
# 2. Implement libs/evio-core/src/evio_core/adapters.py (FileEventAdapter)
# 3. Create plugins/fan-bbox/src/fan_bbox/detector.py
# 4. Build apps/detector-ui/src/detector_ui/main.py
# 5. Test with: uv run detector-ui ../evio/data/fan/fan_const_rpm.dat

# See README.md "Work Stream 2" section for details
```

---

### Work Stream 3: Real-Time Integration 📡

**Who:** Integration team
**Timeline:** Async (implement at hackathon venue)
**Location:** `../evio-realtime/`

**Goals:**
- Design StreamEventAdapter interface
- Document Metavision SDK integration points
- Prepare venue deployment checklist
- Create quick-start guide

**Get Started:**
```bash
cd ../evio-realtime
nix develop

# Your tasks:
# 1. Design StreamEventAdapter protocol (stub implementation)
# 2. Document Metavision SDK requirements
# 3. Create docs/venue-deployment.md checklist
# 4. Write integration guide for live cameras
# 5. Plan 30-minute deployment at venue

# See README.md "Work Stream 3" section for details
```

---

## Daily Development Workflow

### Working in Your Stream

```bash
# Navigate to your worktree
cd ../evio-<your-stream>/

# Pull latest changes
git pull origin <your-branch>

# Make changes
# ... edit files ...

# Stage and commit
git add .
git commit -m "feat: implement X"

# Push to your branch
git push origin <your-branch>
```

### Example: Working in Nix Infrastructure

```bash
cd ../evio-nix-infra

# Pull latest
git pull origin nix-infra

# Create workspace structure
mkdir -p workspace/libs/evio-core
cd workspace
# ... create pyproject.toml ...

# Commit changes
git add .
git commit -m "build: create UV workspace structure with evio-core skeleton"

# Push
git push origin nix-infra
```

---

## Integration (Periodic Merges)

**When:** After completing a milestone or at sync meetings

**Who:** Lead/Integration team

**Process:**

```bash
# Navigate to main repo
cd /path/to/evio-evlib

# Ensure main is up to date
git checkout main
git pull origin main

# Merge work stream branches
git merge origin/nix-infra
git merge origin/hackathon-poc
git merge origin/realtime-integration

# Resolve any conflicts
# ... fix conflicts if any ...

# Push integrated changes
git push origin main

# Update all worktrees with latest main
git worktree list | while read -r path commit branch; do
  if [ -d "$path" ]; then
    cd "$path" && git pull origin main
  fi
done
```

---

## Useful Commands

### Git Worktree Management

```bash
# List all worktrees
git worktree list

# Remove a worktree (when done)
git worktree remove ../evio-nix-infra

# Prune stale worktrees
git worktree prune
```

### UV Workspace Commands

```bash
# Sync all dependencies (in workspace/)
uv sync

# Add a dependency to a package
cd libs/evio-core
uv add evlib polars numpy

# Run app from workspace root
cd workspace
uv run detector-ui ../evio/data/fan/fan_const_rpm.dat

# Run tests
uv run pytest libs/evio-core/tests -v
```

### Nix Development

```bash
# Enter development shell
nix develop

# Check flake
nix flake check

# Update flake inputs
nix flake update

# Build a package
nix build .#evio-core
```

---

## Quick Reference

### Directory Structure

```
evio-evlib/                        [main repo]
├── README.md                      # Main documentation
├── NEXT_STEPS.md                  # This file
├── docs/architecture.md           # Architecture design
└── evio/                          # Original implementation (reference)

../evio-nix-infra/                 [nix-infra branch]
└── workspace/                     # Create this (Work Stream 1)
    ├── pyproject.toml
    ├── libs/evio-core/
    └── ...

../evio-hackathon-poc/             [hackathon-poc branch]
└── workspace/                     # Use after Stream 1 completes
    ├── libs/evio-core/           # Implement core
    ├── plugins/fan-bbox/          # First detector
    └── apps/detector-ui/          # Interactive UI

../evio-realtime/                  [realtime-integration branch]
└── workspace/                     # Design StreamEventAdapter
    └── libs/evio-core/
        └── src/evio_core/adapters.py
```

### Branch Purpose

| Branch | Purpose | Active Development |
|--------|---------|-------------------|
| `main` | Integration point, stable releases | Merge target |
| `nix-infra` | Infrastructure setup, build system | Days 1-2 |
| `hackathon-poc` | Core implementation, first PoC | Days 2-5 |
| `realtime-integration` | Live streaming preparation | Async/Venue |

---

## Troubleshooting

### Worktree Already Exists

**Problem:**
```
fatal: '/path/to/evio-nix-infra' already exists
```

**Solution:**
```bash
# Remove the directory first
rm -rf ../evio-nix-infra

# Or use existing worktree
cd ../evio-nix-infra
git pull origin nix-infra
```

### Worktree Out of Sync

**Problem:** Worktree branch behind main

**Solution:**
```bash
cd ../evio-<your-stream>
git pull origin main          # Merge latest main
git push origin <your-branch> # Push updated branch
```

### UV Not Found

**Problem:** `uv: command not found`

**Solution:**
```bash
# Ensure you're in Nix shell
nix develop

# Verify UV available
which uv
uv --version
```

---

## Resources

- **Architecture Design:** `docs/architecture.md`
- **Main README:** `README.md`
- **evlib Documentation:** https://github.com/ac-freeman/evlib
- **UV Documentation:** https://docs.astral.sh/uv/
- **Nix Flakes:** https://nixos.wiki/wiki/Flakes

---

## Communication

**Sync Points:**
- Daily async updates (post progress in each worktree's commits)
- Bi-weekly integration syncs (merge branches to main)
- Pre-hackathon final integration (ensure all streams work together)

**Questions?**
- Open GitHub issues for architecture questions
- Use PR discussions for code reviews
- Tag work stream leads in comments

---

**Good luck with the hackathon! 🚀**

Last updated: 2025-11-15
