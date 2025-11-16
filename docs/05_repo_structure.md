# 05 – Repo Structure (active vs legacy)

- `evio/` – legacy reference code and scripts (MVPs, launcher, legacy loaders).
- `workspace/tools/` – evlib-based demos and shared utilities:
  - `detector-commons` (evlib loaders, Polars windowing, DBSCAN helpers, visuals).
  - `fan-rpm-demo`, `drone-detector-demo` (CLI evlib/Polars detectors).
  - `evlib-examples`, `downloader`, `evio-verifier` (supporting tools).
- `workspace/libs/evio-core/` – placeholder package (planned home for adapters/plugin API).
- `docs/` – production-facing docs (numbered 01-09, architecture diagrams, status).
- `dev-logs/` – AI development logs and working notes (migration history, debugging summaries).
- `scripts/`, `flake.nix`, `pyproject.toml`, `uv.lock` – tooling and env.

Status: Active code paths live in `evio/` (MVP UI) and `workspace/tools/` (evlib demos). Planned convergence: move shared logic into `libs/evio-core/` and a plugin-based `detector-ui`.
