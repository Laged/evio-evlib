# WIP Review Log

## Work Stream 1 – Nix Infrastructure

### Design doc (`docs/plans/2025-11-15-nix-infrastructure-design.md`)
- Reviewer: Codex
- Status: Feedback sent
- Notes:
  - Proposed root-level `pyproject.toml` contradicts the workspace layout documented in `docs/architecture.md` and `README.md`; awaiting clarification before implementation proceeds.
  - `flake.nix` shell hook currently auto-runs `uv sync` on every `nix develop`, which could thrash network state and fail in offline reviews; suggested to make sync explicit or cache-aware.

### Implementation plan (`docs/plans/2025-11-15-nix-infrastructure-implementation.md`)
- Reviewer: Codex
- Status: Feedback sent
- Notes:
  - Plan assumes all work happens on `main`, but `README.md` mandates the `nix-infra` worktree/branch for this stream; needs revision so contributors do not clobber the shared root.
  - Will review repo changes (`flake.nix`, workspace skeletons, `.claude/skills/dev-environment.md`, `docs/setup.md`) once Claude lands them.

## Upcoming Reviews
- Validate the implemented Nix + UV infrastructure once the `nix-infra` branch/worktree is ready for review.

