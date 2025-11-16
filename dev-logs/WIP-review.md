# WIP Review Log

## Work Stream 1 – Nix Infrastructure

### Design doc (`docs/plans/2025-11-15-nix-infrastructure-design.md`)
- Reviewer: Codex
- Status: ✅ **RESOLVED** - Infrastructure fixes implemented
- Original Blockers:
  1. ~~Shell hook auto-runs `uv sync` on every `nix develop`~~ → **FIXED**: Removed auto-sync, added conditional warnings (commit 13aa9de)
  2. ~~Documentation references outdated file paths~~ → **FIXED**: All paths updated to use evio/ prefix (commits 6511caf, a949780, 11e9870)
  3. ~~Workspace bootstrapping fragile~~ → **FIXED**: No auto-sync on empty workspace, clear guidance messages (commit 13aa9de)

### Implementation plan (`docs/plans/2025-11-15-nix-infrastructure-implementation.md`)
- Reviewer: Codex
- Status: ✅ **IMPLEMENTED** on main branch
- Implementation Summary:
  - Root `pyproject.toml` with UV workspace configuration
  - Root `flake.nix` with minimal system dependencies (Python, UV, Rust, OpenCV, HDF5)
  - Workspace member skeletons: evio-core, fan-bbox, detector-ui
  - `.claude/skills/dev-environment.md` workflow guide for subagents
  - `docs/setup.md` developer onboarding guide
  - Critical fixes: HDF5 support for evlib, demo alias paths, manual sync workflow

### Infrastructure Fixes (`docs/plans/2025-11-15-infrastructure-fixes-implementation.md`)
- Status: ✅ **COMPLETED** (5 commits: 13aa9de, 6511caf, a949780, 8e8e29e, 11e9870)
- All three blockers resolved:
  - Shell hook no longer hides sync failures (manual `uv sync` only)
  - Documentation uses correct evio/ prefixed paths throughout
  - Workspace initialization robust with clear guidance messages

## Upcoming Reviews
- N/A - All infrastructure work complete and ready for use by other work streams

