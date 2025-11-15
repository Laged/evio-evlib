# rvt-adapter

Helper library that bundles all RVT-specific plumbing used across the
evio-evlib workspace. It centralizes:

- Resolving the vendored RVT repository and checkpoint assets
- Building Hydra configs for inference-only runs
- Selecting a compute device (CUDA/CPU)
- Converting evlib event windows into RVT histogram tensors

Use this package instead of open-coding Hydra/torch glue inside every plugin
or app. Future detectors can import from `rvt_adapter` to stay aligned with the
official integration plan in `docs/rvt-integration-plan.md`.
