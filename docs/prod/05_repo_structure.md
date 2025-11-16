# 05 – Repo Structure (active vs legacy)

- `evio/` – legacy reference code and scripts (MVPs, launcher, legacy loaders).
- `workspace/tools/` – evlib-based demos and shared utilities:
  - `detector-commons` (evlib loaders, Polars windowing, DBSCAN helpers, visuals).
  - `fan-rpm-demo`, `drone-detector-demo` (CLI evlib/Polars detectors).
  - `evlib-examples`, `downloader`, `evio-verifier` (supporting tools).
- `workspace/libs/evio-core/` – placeholder package (planned home for adapters/plugin API).
- `docs/` – existing design/plan docs (to remain as legacy references).
- `docs/prod/` – production-facing docs (this numbered set).
- `scripts/`, `flake.nix`, `pyproject.toml`, `uv.lock` – tooling and env.

Status: Active code paths live in `evio/` (MVP UI) and `workspace/tools/` (evlib demos). Planned convergence: move shared logic into `libs/evio-core/` and a plugin-based `detector-ui`.
