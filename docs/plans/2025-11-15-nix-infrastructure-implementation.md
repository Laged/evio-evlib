# Nix Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up minimal Nix + UV workspace infrastructure for evio-evlib monorepo with three workspace member skeletons.

**Architecture:** Root flake.nix provides system deps (Python, UV, Rust, OpenCV). Root pyproject.toml defines UV workspace with evio and workspace/* members. All Python deps managed by UV via shared lockfile.

**Tech Stack:** Nix flakes, UV (Python package manager), Python 3.11, evlib, Polars

---

## Prerequisites

**Working Directory:** `/Users/laged/Codings/laged/evio-evlib` (main branch)

**Expected State:**
- Current branch: `main`
- Clean working tree
- evio/ directory exists with pyproject.toml

**Verify before starting:**
```bash
git status  # Should show clean
ls evio/pyproject.toml  # Should exist
```

---

## Task 1: Create Root UV Workspace Configuration

**Goal:** Define UV workspace that includes evio and workspace members

**Files:**
- Create: `pyproject.toml`

**Step 1: Create root pyproject.toml**

Create file with workspace configuration:

```toml
[tool.uv.workspace]
members = [
    "evio",
    "workspace/libs/*",
    "workspace/plugins/*",
    "workspace/apps/*",
]

[tool.uv]
# Require Python 3.11+ across all workspace members
# This is the intersection of all member requirements
requires-python = ">=3.11"
```

**Step 2: Verify file created**

Run: `cat pyproject.toml`

Expected: File contents match above

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add root UV workspace configuration"
```

Expected: Commit succeeds

---

## Task 2: Create Minimal Root Flake

**Goal:** Set up Nix flake with system dependencies only (Python, UV, Rust, OpenCV)

**Files:**
- Create: `flake.nix`

**Step 1: Create flake.nix**

Create file with minimal system dependencies:

```nix
{
  description = "Event Camera Detection Workbench - Nix Infrastructure";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Core tools
            python
            pkgs.uv                 # UV package manager

            # Rust toolchain (for evlib compilation)
            pkgs.rustc
            pkgs.cargo
            pkgs.pkg-config

            # System libraries
            pkgs.opencv4            # OpenCV for visualization
            pkgs.zlib               # Required by some Rust packages
          ];

          # Set LD_LIBRARY_PATH for Rust-backed libraries (evlib)
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ]}";

          shellHook = ''
            echo "=========================================="
            echo "  Event Camera Detection Workbench"
            echo "=========================================="
            echo ""
            echo "Python: $(python --version)"
            echo "UV: $(uv --version)"
            echo ""

            # Create workspace structure if missing
            if [ ! -d workspace ]; then
              echo "Creating workspace structure..."
              mkdir -p workspace/libs workspace/plugins workspace/apps
            fi

            # Initialize UV workspace (creates .venv, runs uv sync)
            echo "Initializing UV workspace..."
            uv sync --quiet 2>/dev/null || echo "Workspace initialized"

            echo ""
            echo "📦 Package Management:"
            echo "  NEVER use pip - UV only!"
            echo "  Add dependency: uv add --package <member> <package>"
            echo "  Sync workspace: uv sync"
            echo ""
            echo "🚀 Running Commands (from repo root):"
            echo "  uv run --package <member> <command>"
            echo ""
            echo "Demo Aliases:"
            echo "  run-demo-fan         : Play fan dataset"
            echo "  run-mvp-1            : MVP 1 - Event density"
            echo "  run-mvp-2            : MVP 2 - Voxel FFT"
            echo ""

            # Shell aliases for convenience
            alias run-demo-fan='uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat'
            alias run-mvp-1='uv run --package evio python scripts/mvp_1_density.py data/fan/fan_const_rpm.dat'
            alias run-mvp-2='uv run --package evio python scripts/mvp_2_voxel.py data/fan/fan_varying_rpm.dat'

            echo "Read .claude/skills/dev-environment.md for workflow guidelines"
            echo "=========================================="
            echo ""
          '';
        };
      }
    );
}
```

**Step 2: Verify flake syntax**

Run: `nix flake check`

Expected: No errors (will download inputs first time)

**Step 3: Commit**

```bash
git add flake.nix
git commit -m "build: add minimal root flake.nix with UV + Rust support"
```

Expected: Commit succeeds

---

## Task 3: Create Development Workflow Skill

**Goal:** Document mandatory workflow rules for all subagents

**Files:**
- Create: `.claude/skills/dev-environment.md`

**Step 1: Create .claude/skills directory**

```bash
mkdir -p .claude/skills
```

**Step 2: Create dev-environment.md**

Create file with workflow guidelines:

```markdown
---
name: dev-environment
description: Development workflow guidelines for evio-evlib monorepo - UV workspace + Nix
---

# Development Environment Workflow

## MANDATORY Rules for All Subagents

### Package Management

**System Packages (Nix):**
- ✅ Provided by `flake.nix`: Python, UV, Rust, OpenCV, pkg-config
- ❌ **NEVER** add Python packages to `flake.nix`
- ❌ **NEVER** modify `buildInputs` for Python deps

**Python Packages (UV):**
- ✅ Managed via `pyproject.toml` in workspace members
- ✅ Use `uv add --package <member> <dependency>`
- ✅ Run `uv sync` after adding dependencies
- ❌ **NEVER** run `pip install`
- ❌ **NEVER** use poetry, conda, or other tools

### Running Commands

**From Repo Root (ALWAYS):**
```bash
uv run --package <member> <command>
```

**Examples:**
```bash
# Run evio demo
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat

# Run workspace member script
uv run --package evio-core python -m evio_core.loaders

# Use shell aliases
run-demo-fan
run-mvp-1
```

**Never:**
- ❌ `cd workspace/libs/evio-core && python ...`
- ❌ `cd evio && uv run ...`

### Adding Dependencies

**To Workspace Member:**
```bash
uv add --package evio-core polars numpy
uv sync
```

**To evio (legacy):**
```bash
uv add --package evio scipy
uv sync
```

**Verify lockfile updated:**
```bash
git status  # Should show uv.lock modified
```

### Development Flow

```bash
# 1. Enter Nix environment
nix develop

# 2. Workspace auto-initialized by shellHook
# .venv created, uv sync runs

# 3. Run commands from root
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat

# 4. Add dependencies
uv add --package evio-core scikit-learn

# 5. Update lockfile
uv sync
```

### When Stuck

1. Read this file
2. Check `flake.nix` shellHook for aliases
3. Verify you're in `nix develop`
4. Use `uv run --package <member>` for all commands
5. Never pip, never cd

### Common Mistakes

❌ `pip install requests`
✅ `uv add --package <member> requests && uv sync`

❌ `cd workspace/libs/evio-core && python -m evio_core`
✅ `uv run --package evio-core python -m evio_core`

❌ Adding numpy to `flake.nix` buildInputs
✅ Adding numpy to workspace member's `pyproject.toml`

## Workspace Structure

```
evio-evlib/               # UV workspace root
├── pyproject.toml       # Workspace config
├── uv.lock              # Shared lockfile
├── evio/                # Member: legacy evio
├── workspace/
│   ├── libs/evio-core/  # Member: core library
│   ├── plugins/*/       # Members: detector plugins
│   └── apps/*/          # Members: applications
```

## Key Concepts

**UV Workspace:**
- One lockfile (`uv.lock`) for all members
- Consistent dependencies across packages
- Editable installations for workspace members

**Nix Shell:**
- Provides system dependencies only
- Reproducible environment
- No Python package management in Nix

**The Rule:**
- Nix = system packages (Python, UV, Rust, OpenCV)
- UV = Python packages (evlib, polars, numpy, etc.)
- Never mix the two
```

**Step 3: Verify file created**

Run: `cat .claude/skills/dev-environment.md | head -20`

Expected: File shows frontmatter and beginning of content

**Step 4: Commit**

```bash
git add .claude/skills/dev-environment.md
git commit -m "docs: add dev-environment workflow skill for subagents"
```

Expected: Commit succeeds

---

## Task 4: Create evio-core Library Skeleton

**Goal:** Create workspace member for core event processing library

**Files:**
- Create: `workspace/libs/evio-core/pyproject.toml`
- Create: `workspace/libs/evio-core/src/evio_core/__init__.py`
- Create: `workspace/libs/evio-core/README.md`

**Step 1: Create directory structure**

```bash
mkdir -p workspace/libs/evio-core/src/evio_core
```

**Step 2: Create pyproject.toml**

Create `workspace/libs/evio-core/pyproject.toml`:

```toml
[project]
name = "evio-core"
version = "0.1.0"
description = "Core event camera processing library with evlib integration"
requires-python = ">=3.11"
dependencies = [
    "evlib>=0.8.0",
    "polars>=0.20.0",
    "numpy>=1.24.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 3: Create __init__.py**

Create `workspace/libs/evio-core/src/evio_core/__init__.py`:

```python
"""Core event camera processing library with evlib integration."""

__version__ = "0.1.0"
```

**Step 4: Create README.md**

Create `workspace/libs/evio-core/README.md`:

```markdown
# evio-core

Core event camera processing library with evlib integration.

## Purpose

Provides:
- FileEventAdapter using evlib (10x faster file loading)
- EventSource protocol for file/stream abstraction
- DetectorPlugin protocol for extensible algorithms
- Representation wrappers (time surface, voxel grids)

## Status

🚧 Skeleton - awaiting Work Stream 2 (hackathon-poc) implementation
```

**Step 5: Verify structure**

Run: `ls -R workspace/libs/evio-core/`

Expected:
```
workspace/libs/evio-core/:
README.md  pyproject.toml  src

workspace/libs/evio-core/src:
evio_core

workspace/libs/evio-core/src/evio_core:
__init__.py
```

**Step 6: Commit**

```bash
git add workspace/libs/evio-core/
git commit -m "feat: add evio-core library skeleton"
```

Expected: Commit succeeds

---

## Task 5: Create fan-bbox Plugin Skeleton

**Goal:** Create workspace member for fan bounding box detector

**Files:**
- Create: `workspace/plugins/fan-bbox/pyproject.toml`
- Create: `workspace/plugins/fan-bbox/src/fan_bbox/__init__.py`
- Create: `workspace/plugins/fan-bbox/README.md`

**Step 1: Create directory structure**

```bash
mkdir -p workspace/plugins/fan-bbox/src/fan_bbox
```

**Step 2: Create pyproject.toml**

Create `workspace/plugins/fan-bbox/pyproject.toml`:

```toml
[project]
name = "fan-bbox"
version = "0.1.0"
description = "Fan bounding box detector plugin"
requires-python = ">=3.11"
dependencies = [
    "evio-core",
]

[tool.uv.sources]
evio-core = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 3: Create __init__.py**

Create `workspace/plugins/fan-bbox/src/fan_bbox/__init__.py`:

```python
"""Fan bounding box detector plugin."""

__version__ = "0.1.0"
```

**Step 4: Create README.md**

Create `workspace/plugins/fan-bbox/README.md`:

```markdown
# fan-bbox

Fan bounding box detector plugin.

## Purpose

Challenge 1: Detect and draw bounding box around rotating fan using time surface.

## Algorithm

1. Create time surface from event window (evlib)
2. Find active pixels above threshold
3. Compute spatial bounds (min/max x/y)
4. Return bounding box coordinates

## Status

🚧 Skeleton - awaiting Work Stream 2 (hackathon-poc) implementation
```

**Step 5: Verify structure**

Run: `ls -R workspace/plugins/fan-bbox/`

Expected:
```
workspace/plugins/fan-bbox/:
README.md  pyproject.toml  src

workspace/plugins/fan-bbox/src:
fan_bbox

workspace/plugins/fan-bbox/src/fan_bbox:
__init__.py
```

**Step 6: Commit**

```bash
git add workspace/plugins/fan-bbox/
git commit -m "feat: add fan-bbox plugin skeleton"
```

Expected: Commit succeeds

---

## Task 6: Create detector-ui App Skeleton

**Goal:** Create workspace member for interactive detection workbench

**Files:**
- Create: `workspace/apps/detector-ui/pyproject.toml`
- Create: `workspace/apps/detector-ui/src/detector_ui/__init__.py`
- Create: `workspace/apps/detector-ui/README.md`

**Step 1: Create directory structure**

```bash
mkdir -p workspace/apps/detector-ui/src/detector_ui
```

**Step 2: Create pyproject.toml**

Create `workspace/apps/detector-ui/pyproject.toml`:

```toml
[project]
name = "detector-ui"
version = "0.1.0"
description = "Interactive event camera detection workbench"
requires-python = ">=3.11"
dependencies = [
    "evio-core",
    "opencv-python>=4.8.0",
]

[tool.uv.sources]
evio-core = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 3: Create __init__.py**

Create `workspace/apps/detector-ui/src/detector_ui/__init__.py`:

```python
"""Interactive event camera detection workbench."""

__version__ = "0.1.0"
```

**Step 4: Create README.md**

Create `workspace/apps/detector-ui/README.md`:

```markdown
# detector-ui

Interactive event camera detection workbench.

## Purpose

Provides:
- Event playback from files or live streams
- Hot-swappable detector plugins (press 1, 2, 3...)
- Hot-swappable data sources (press 'd' to load new file)
- Real-time visualization with OpenCV

## Controls

- `1`, `2`, `3`: Switch detector plugin
- `d`: Load different data file
- `l`: Toggle looping
- `q`: Quit

## Status

🚧 Skeleton - awaiting Work Stream 2 (hackathon-poc) implementation
```

**Step 5: Verify structure**

Run: `ls -R workspace/apps/detector-ui/`

Expected:
```
workspace/apps/detector-ui/:
README.md  pyproject.toml  src

workspace/apps/detector-ui/src:
detector_ui

workspace/apps/detector-ui/src/detector_ui:
__init__.py
```

**Step 6: Commit**

```bash
git add workspace/apps/detector-ui/
git commit -m "feat: add detector-ui app skeleton"
```

Expected: Commit succeeds

---

## Task 7: Update .gitignore for UV Workspace

**Goal:** Ignore UV-generated files and Python artifacts

**Files:**
- Modify: `.gitignore`

**Step 1: Check existing .gitignore**

Run: `cat .gitignore`

Expected: Shows existing rules (UV section already present)

**Step 2: Verify UV section exists**

Run: `grep -A 5 "# UV" .gitignore`

Expected:
```
# UV
.venv/
venv/
uv.lock
.python-version
```

**Step 3: Update if needed**

If `uv.lock` is in .gitignore, remove it (we want to commit lockfile):

Read current .gitignore, then write updated version without `uv.lock` line.

Run: `cat .gitignore | grep -v "^uv.lock$" > .gitignore.tmp && mv .gitignore.tmp .gitignore`

**Step 4: Verify uv.lock not ignored**

Run: `grep "uv.lock" .gitignore`

Expected: No output (uv.lock should be committed)

**Step 5: Commit if changed**

```bash
git add .gitignore
git diff --cached .gitignore
# Only commit if there are changes
git commit -m "build: update .gitignore to track uv.lock" || echo "No changes to .gitignore"
```

Expected: Commit if modified, or message "No changes"

---

## Task 8: Initialize UV Workspace (Test)

**Goal:** Verify UV workspace setup by running uv sync

**Files:**
- Will create: `uv.lock` (auto-generated)
- Will create: `.venv/` (auto-generated)

**Step 1: Check prerequisites**

Run: `which uv`

Expected: Shows path to uv binary (should be in Nix shell)

If not in Nix shell:
```bash
nix develop
```

**Step 2: Run uv sync**

Run: `uv sync`

Expected:
```
Resolved X packages in Y.Zs
Installed X packages in Y.Zs
  + evio
  + evio-core
  + fan-bbox
  + detector-ui
  + evlib
  + polars
  + numpy
  + opencv-python
  ...
```

**Step 3: Verify lockfile created**

Run: `ls -lh uv.lock`

Expected: File exists, size > 0

**Step 4: Verify venv created**

Run: `ls .venv/`

Expected: Shows bin/, lib/, etc.

**Step 5: Commit lockfile**

```bash
git add uv.lock
git commit -m "build: add UV workspace lockfile"
```

Expected: Commit succeeds

---

## Task 9: Test Workspace Member Import

**Goal:** Verify workspace members are importable

**Files:**
- None (testing only)

**Step 1: Test evio-core import**

Run: `uv run python -c "import evio_core; print(f'evio-core v{evio_core.__version__}')"`

Expected: `evio-core v0.1.0`

**Step 2: Test fan-bbox import**

Run: `uv run python -c "import fan_bbox; print(f'fan-bbox v{fan_bbox.__version__}')"`

Expected: `fan-bbox v0.1.0`

**Step 3: Test detector-ui import**

Run: `uv run python -c "import detector_ui; print(f'detector-ui v{detector_ui.__version__}')"`

Expected: `detector-ui v0.1.0`

**Step 4: Test evio import**

Run: `uv run python -c "import evio; print('evio imported successfully')"`

Expected: `evio imported successfully`

**Step 5: Test evlib import (via evio-core deps)**

Run: `uv run python -c "import evlib; print('evlib imported successfully')"`

Expected: `evlib imported successfully`

---

## Task 10: Test Nix Shell Initialization

**Goal:** Verify nix develop creates workspace and runs uv sync

**Files:**
- None (testing only)

**Step 1: Exit Nix shell**

Run: `exit`

Expected: Returns to system shell

**Step 2: Remove workspace directory (test auto-creation)**

Run: `rm -rf workspace/`

Expected: Directory removed

**Step 3: Enter Nix shell**

Run: `nix develop`

Expected:
```
==========================================
  Event Camera Detection Workbench
==========================================

Python: Python 3.11.X
UV: uv X.Y.Z

Creating workspace structure...
Initializing UV workspace...

📦 Package Management:
  NEVER use pip - UV only!
  ...
```

**Step 4: Verify workspace recreated**

Run: `ls workspace/`

Expected: Shows libs/, plugins/, apps/

**Step 5: Verify uv sync ran**

Run: `ls .venv/`

Expected: Shows bin/, lib/, etc.

---

## Task 11: Test Demo Aliases

**Goal:** Verify shell aliases work (note: requires data files)

**Files:**
- None (testing only)

**Step 1: Check if evio data exists**

Run: `ls evio/data/fan/ 2>/dev/null || echo "Data files not present"`

Expected: Either lists files OR "Data files not present"

**Step 2: Test alias definition**

Run: `alias run-demo-fan`

Expected: Shows alias command

**Step 3: Document data requirement**

Note: Demo aliases require event camera data files in `evio/data/`.
These are not committed to git (large files, in .gitignore).

For testing at hackathon venue: Download data from Sensofusion.

---

## Task 12: Create Setup Documentation

**Goal:** Document how to use the infrastructure

**Files:**
- Create: `docs/setup.md`

**Step 1: Create docs/setup.md**

Create file:

```markdown
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

# Workspace auto-initialized by shellHook:
# - Creates workspace/ directory structure
# - Runs uv sync to install all packages
# - Sets up shell aliases
```

That's it! You're ready to develop.

---

## Daily Workflow

### Enter Environment

```bash
cd evio-evlib
nix develop
```

### Run Commands

All commands run from repo root using `uv run --package <member>`:

```bash
# Run evio legacy demos
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat

# Use shell aliases
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
```

**Step 2: Verify file created**

Run: `cat docs/setup.md | head -30`

Expected: Shows beginning of setup guide

**Step 3: Commit**

```bash
git add docs/setup.md
git commit -m "docs: add development environment setup guide"
```

Expected: Commit succeeds

---

## Task 13: Final Verification

**Goal:** Run all verification tests to confirm infrastructure is complete

**Files:**
- None (testing only)

**Step 1: Check git status**

Run: `git status`

Expected: Clean working tree, all changes committed

**Step 2: List commits**

Run: `git log --oneline -15`

Expected: Shows all commits from this implementation

**Step 3: Verify file structure**

Run: `ls -la`

Expected: Shows flake.nix, pyproject.toml, workspace/, .claude/, docs/

**Step 4: Verify workspace structure**

Run: `tree workspace/ -L 3` or `find workspace/ -type f`

Expected: Shows all created workspace members

**Step 5: Run full test suite**

```bash
# Test 1: Nix environment
nix develop --command bash -c "echo 'Nix shell works'"

# Test 2: UV workspace
uv sync
echo "UV sync: $?"  # Should be 0

# Test 3: Import all members
uv run python -c "import evio_core, fan_bbox, detector_ui, evio; print('All imports successful')"

# Test 4: evlib available
uv run python -c "import evlib; print('evlib available')"
```

Expected: All tests pass

**Step 6: Create summary**

Print summary of what was built:

```bash
echo "Infrastructure Implementation Complete!"
echo ""
echo "Created:"
echo "  - Root flake.nix (system deps: Python, UV, Rust, OpenCV)"
echo "  - Root pyproject.toml (UV workspace config)"
echo "  - .claude/skills/dev-environment.md (workflow guide)"
echo "  - workspace/libs/evio-core/ (core library skeleton)"
echo "  - workspace/plugins/fan-bbox/ (detector plugin skeleton)"
echo "  - workspace/apps/detector-ui/ (UI app skeleton)"
echo "  - docs/setup.md (setup documentation)"
echo "  - uv.lock (committed lockfile)"
echo ""
echo "Commits: $(git log --oneline main..HEAD | wc -l)"
echo ""
echo "Next Steps:"
echo "  1. Review commits: git log --oneline -15"
echo "  2. Test environment: nix develop"
echo "  3. Create nix-infra worktree (if needed)"
echo "  4. Merge to main when validated"
```

---

## Task 14: Document Completion

**Goal:** Mark implementation as complete

**Files:**
- Modify: `docs/plans/2025-11-15-nix-infrastructure-design.md`

**Step 1: Read current design doc**

Run: `cat docs/plans/2025-11-15-nix-infrastructure-design.md | grep "Status:"`

Expected: Shows `**Status:** Approved Design`

**Step 2: Update status in design doc**

Update line 3:
```markdown
**Status:** ✅ Implemented (2025-11-15)
```

**Step 3: Commit**

```bash
git add docs/plans/2025-11-15-nix-infrastructure-design.md
git commit -m "docs: mark nix infrastructure design as implemented"
```

Expected: Commit succeeds

---

## Success Criteria

### Must Have ✅
- Root `flake.nix` with minimal system dependencies
- Root `pyproject.toml` with workspace members
- `.claude/skills/dev-environment.md` enforcing workflow
- Workspace skeletons created and importable
- `nix develop` auto-initializes environment
- `uv sync` installs all members
- `uv run --package evio` imports successfully
- `docs/setup.md` guide created

### Verification Checklist

- [ ] `nix flake check` passes
- [ ] `uv sync` completes without errors
- [ ] `uv run python -c "import evio_core"` succeeds
- [ ] `uv run python -c "import fan_bbox"` succeeds
- [ ] `uv run python -c "import detector_ui"` succeeds
- [ ] `uv run python -c "import evio"` succeeds
- [ ] `uv run python -c "import evlib"` succeeds
- [ ] `git status` shows clean working tree
- [ ] All commits have descriptive messages
- [ ] `uv.lock` is committed

---

## Next Steps After Implementation

1. **Review all commits:** `git log --oneline -20`
2. **Test in fresh clone:** Clone repo, run `nix develop`, verify workspace
3. **Create nix-infra worktree** (optional): `git worktree add -b nix-infra ../evio-nix-infra`
4. **Merge to main:** When validated
5. **Handoff to Work Stream 2:** hackathon-poc can begin implementing evio-core

---

## Rollback Procedure

If something goes wrong:

```bash
# See commits
git log --oneline -20

# Soft reset to before implementation
git reset --soft <commit-before-implementation>

# Or hard reset (loses changes)
git reset --hard <commit-before-implementation>

# Clean workspace
git clean -fdx workspace/ .venv/
```

---

**Implementation Time Estimate:** 20-30 minutes

**Tasks:** 14 total
**Commits:** ~10-12 commits
**Files Created:** ~15 files
**Files Modified:** ~2 files

---

**Ready to execute with superpowers:executing-plans or superpowers:subagent-driven-development**
