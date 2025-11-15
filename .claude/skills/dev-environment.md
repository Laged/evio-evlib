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
