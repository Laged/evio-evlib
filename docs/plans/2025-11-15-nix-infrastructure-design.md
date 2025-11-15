# Nix Infrastructure Design - UV Workspace Setup

**Date:** 2025-11-15
**Status:** Approved Design
**Work Stream:** 1 - Nix Infrastructure
**Owner:** Infrastructure Lead

---

## Executive Summary

Set up minimal, production-ready Nix + UV workspace infrastructure for the evio-evlib monorepo. Enable three parallel work streams to develop independently using git worktrees while sharing a unified development environment.

**Key Decisions:**
- Root `flake.nix` provides system dependencies only (Python, UV, Rust, OpenCV)
- UV manages all Python dependencies via workspace lockfile
- evio included as workspace member for `--package` integration
- Never pip, only UV for Python package management
- All commands run from repo root using `uv run --package <member>`

---

## Design Principles

1. **Minimal Nix Flake**: System packages only, no Python package building
2. **UV Workspace**: Single lockfile, consistent dependencies across members
3. **No cd Required**: Run any member's code from root using `--package` flag
4. **Never pip**: UV is the sole Python package manager
5. **Enforce via Documentation**: `.claude/skills/dev-environment.md` guides all subagents

---

## Repository Structure

```
evio-evlib/                          # Root repo (UV workspace root)
├── flake.nix                        # NEW: Minimal Nix environment
├── flake.lock                       # NEW: Nix lockfile
├── pyproject.toml                   # NEW: UV workspace config
├── uv.lock                          # Auto-generated: Shared Python lockfile
├── .venv/                           # Auto-generated: Shared virtual env
│
├── .claude/
│   └── skills/
│       └── dev-environment.md       # NEW: Development workflow guide
│
├── docs/
│   ├── architecture.md              # Existing
│   └── plans/
│       └── 2025-11-15-nix-infrastructure-design.md  # This file
│
├── evio/                            # Workspace member (legacy/reference)
│   ├── flake.nix                   # Keep for reference, not used
│   ├── pyproject.toml              # Existing evio project config
│   ├── scripts/                    # MVP demos
│   │   ├── play_dat.py
│   │   ├── mvp_1_density.py
│   │   └── ...
│   ├── src/evio/                   # Existing evio library
│   └── data/                       # Event camera datasets
│
└── workspace/                       # NEW: Workspace member root
    ├── libs/
    │   └── evio-core/              # NEW: Workspace member
    │       ├── pyproject.toml
    │       ├── src/evio_core/
    │       │   └── __init__.py
    │       └── README.md
    ├── plugins/
    │   └── fan-bbox/               # NEW: Workspace member
    │       ├── pyproject.toml
    │       ├── src/fan_bbox/
    │       │   └── __init__.py
    │       └── README.md
    └── apps/
        └── detector-ui/            # NEW: Workspace member
            ├── pyproject.toml
            ├── src/detector_ui/
            │   └── __init__.py
            └── README.md
```

---

## Root `pyproject.toml` - UV Workspace Configuration

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

**Key Points:**
- Glob patterns discover workspace members automatically
- Single lockfile (`uv.lock`) ensures dependency consistency
- evio included as member for `--package evio` support
- No dependencies at root level (members declare their own)

---

## Root `flake.nix` - Minimal System Dependencies

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

**Philosophy:**
- Nix provides **system packages only** (Python, UV, Rust, OpenCV)
- UV manages **all Python dependencies** via workspace lockfile
- shellHook auto-initializes workspace on first `nix develop`
- Aliases for convenience, but users can run any command via `uv run --package`

---

## Workspace Member Skeletons

### `workspace/libs/evio-core/pyproject.toml`

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

### `workspace/plugins/fan-bbox/pyproject.toml`

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

### `workspace/apps/detector-ui/pyproject.toml`

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

**Pattern:** Members declare workspace dependencies using `[tool.uv.sources]` with `workspace = true` for editable installations.

---

## `.claude/skills/dev-environment.md` - Workflow Guidelines

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

---

## Implementation Checklist

### Phase 1: Root Infrastructure (Today)

- [ ] Create root `pyproject.toml` with workspace config
- [ ] Create root `flake.nix` (minimal, system deps only)
- [ ] Create `.claude/skills/dev-environment.md`
- [ ] Test: `nix develop` creates workspace structure
- [ ] Test: `uv sync` runs successfully

### Phase 2: Workspace Skeletons (Today)

- [ ] Create `workspace/libs/evio-core/` skeleton
- [ ] Create `workspace/plugins/fan-bbox/` skeleton
- [ ] Create `workspace/apps/detector-ui/` skeleton
- [ ] Test: `uv sync` installs all members
- [ ] Test: `uv run --package evio python scripts/play_dat.py` works

### Phase 3: Git Worktree Setup (After Phase 2)

- [ ] Create `nix-infra` worktree
- [ ] Commit infrastructure changes to `nix-infra` branch
- [ ] Document setup in `docs/setup.md`
- [ ] Merge to `main` when validated

### Phase 4: Handoff to Work Streams (Next)

- [ ] Work Stream 2 (hackathon-poc): Implement evio-core with evlib
- [ ] Work Stream 3 (realtime): Design StreamEventAdapter stubs
- [ ] All teams use shared infrastructure

---

## Verification Steps

### Test 1: Nix Environment
```bash
nix develop
# Should see welcome message
# Should create workspace/ dirs
# Should run uv sync
```

### Test 2: UV Workspace
```bash
uv sync
# Should install evio, evio-core, fan-bbox, detector-ui
cat uv.lock  # Should show all dependencies
```

### Test 3: Run evio Demo
```bash
run-demo-fan
# OR
uv run --package evio python scripts/play_dat.py data/fan/fan_const_rpm.dat
# Should play fan dataset (requires data file)
```

### Test 4: Workspace Member Import
```bash
uv run python -c "import evio_core; print('Success')"
# Should import successfully (even if __init__.py is empty)
```

---

## Success Criteria

### Must Have
- ✅ Root `flake.nix` with minimal system dependencies
- ✅ Root `pyproject.toml` with workspace members
- ✅ `.claude/skills/dev-environment.md` enforcing workflow
- ✅ Workspace skeletons created and importable
- ✅ `nix develop` auto-initializes environment
- ✅ `uv sync` installs all members
- ✅ `uv run --package evio` runs legacy demos

### Should Have
- ✅ Shell aliases for common commands
- ✅ Comprehensive documentation in design doc
- ✅ Verification tests pass

### Nice to Have
- ✅ Committed to `nix-infra` branch
- ✅ Documented in `docs/setup.md`
- ✅ Merged to `main` for other work streams

---

## Open Questions / Decisions

1. **evio data files**: Assume they exist in `evio/data/`? Or document separately?
2. **Git worktree now or later**: Set up `nix-infra` worktree in this session?
3. **Workspace member source dirs**: Use `src/` layout or flat layout?

---

## Next Steps

1. **Review this design** with infrastructure lead
2. **Implement Phase 1**: Root infrastructure files
3. **Implement Phase 2**: Workspace skeletons
4. **Verify**: Run all tests
5. **Document**: Create `docs/setup.md` guide
6. **Commit**: Push to `nix-infra` branch (or main)

---

**Ready to implement?**
