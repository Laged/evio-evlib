"""Utility helpers shared across RVT integrations."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import polars as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from evio.representations import create_stacked_histogram as _evlib_histogram


def _discover_repo_root() -> Path:
    """Walk upward from this file until we find the workspace directory."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        plugin_dir = parent / "workspace" / "plugins" / "rvt-detection"
        if plugin_dir.is_dir():
            return parent
    raise RuntimeError(
        f"Unable to locate repo root for {__file__}. "
        "Ensure the RVT plugin lives under workspace/plugins/rvt-detection."
    )


def _resolve_plugin_root(repo_root: Path) -> Path:
    """Respect RVT_PLUGIN_ROOT override or fall back to the bundled plugin."""

    override = os.environ.get("RVT_PLUGIN_ROOT")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"RVT_PLUGIN_ROOT={candidate} does not exist or is not a directory."
            )
        return candidate
    plugin_root = repo_root / "workspace" / "plugins" / "rvt-detection"
    if not plugin_root.is_dir():
        raise FileNotFoundError(
            f"Expected RVT plugin at {plugin_root}. "
            "Verify the repository structure matches docs/architecture.md."
        )
    return plugin_root


_REPO_ROOT = _discover_repo_root()
_PLUGIN_ROOT = _resolve_plugin_root(_REPO_ROOT)

DEFAULT_RVT_REPO = _PLUGIN_ROOT / "RVT"
DEFAULT_MODELS_DIR = _PLUGIN_ROOT / "models"
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODELS_DIR / "rvt-s-gen1.ckpt"


def _stacked_histogram(
    events: pl.DataFrame | pl.LazyFrame,
    height: int,
    width: int,
    bins: int,
    window_duration_ms: float,
) -> pl.DataFrame:
    """Create stacked histogram using evlib's native implementation."""
    lazy = events.lazy() if isinstance(events, pl.DataFrame) else events
    return _evlib_histogram(
        lazy,
        height=height,
        width=width,
        bins=bins,
        window_duration_ms=window_duration_ms,
    )


@dataclass
class EnvReport:
    """Environment report surface for quick diagnostics."""

    repo_path: Path
    checkpoint_path: Path
    errors: list[str]
    warnings: list[str]
    device: str
    cuda_available: bool


def resolve_repo_path(repo: Path | str | None = None) -> Path:
    """Resolve and verify the RVT repository path."""

    candidate = Path(repo).expanduser() if repo else DEFAULT_RVT_REPO
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"RVT repository not found at {candidate}. Did you init the submodule?"
        )
    return candidate


def resolve_checkpoint_path(checkpoint: Path | str | None = None) -> Path:
    """Resolve and verify the checkpoint path."""

    candidate = Path(checkpoint).expanduser() if checkpoint else DEFAULT_CHECKPOINT_PATH
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"RVT checkpoint not found at {candidate}. Download or point to a valid .ckpt file."
        )
    return candidate


def ensure_repo_on_path(repo: Path) -> None:
    """Append the given repo to sys.path for Hydra imports."""

    import sys

    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.append(repo_str)


def determine_device(device: Optional[str] = None) -> torch.device:
    """Pick a torch device, defaulting to CUDA when available."""

    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but no compatible GPU is available; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_hydra_config(
    repo: Path,
    dataset_name: str,
    experiment: str,
    checkpoint_path: Path,
    extra_overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Compose the Hydra validation config used for inference."""

    config_dir = (repo / "config").resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Missing Hydra config directory: {config_dir}")

    dataset_dir = repo / ".rvt_adapter_dummy_dataset"
    dataset_dir.mkdir(exist_ok=True)

    overrides = [
        f"dataset={dataset_name}",
        f"+experiment/{dataset_name}={experiment}",
        f"dataset.path={dataset_dir}",
        "+wandb.group_name=rvt-plugin",
        "+wandb.project_name=RVT",
        "+logging.train.metrics.compute=false",
        "+logging.train.high_dim.enable=false",
        "+logging.validation.high_dim.enable=false",
        "batch_size.eval=1",
        "hardware.gpus=0",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base="1.2",
        job_name="rvt_adapter",
    ):
        cfg = compose(config_name="val", overrides=overrides)
    cfg.checkpoint = str(checkpoint_path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def events_to_hist_tensor(
    events: pl.DataFrame | pl.LazyFrame,
    sensor_hw: tuple[int, int],
    bins: int,
    window_duration_ms: float,
) -> torch.Tensor:
    """Convert evlib events to the tensor RVT expects."""

    hist = _stacked_histogram(
        events,
        height=sensor_hw[0],
        width=sensor_hw[1],
        bins=bins,
        window_duration_ms=window_duration_ms,
    )
    tensor = torch.zeros(
        (1, bins * 2, sensor_hw[0], sensor_hw[1]),
        dtype=torch.float32,
    )
    if hist.height == 0:
        return tensor
    num_channels = bins * 2
    for row in hist.iter_rows(named=True):
        time_bin = int(row["time_bin"])
        polarity = int(row["polarity"])
        channel = time_bin * 2 + polarity
        y = int(row["y"])
        x = int(row["x"])
        if not (0 <= channel < num_channels):
            continue
        if not (0 <= y < sensor_hw[0] and 0 <= x < sensor_hw[1]):
            continue
        tensor[0, channel, y, x] = float(row["count"])
    return tensor


def gather_environment_report(
    repo: Path | str | None = None,
    checkpoint: Path | str | None = None,
) -> EnvReport:
    """Collect quick status info about the RVT runtime environment."""

    errors: list[str] = []
    warnings: list[str] = []

    try:
        repo_path = resolve_repo_path(repo)
    except FileNotFoundError as exc:  # pragma: no cover - trivial
        errors.append(str(exc))
        repo_path = Path(repo) if repo else DEFAULT_RVT_REPO

    try:
        checkpoint_path = resolve_checkpoint_path(checkpoint)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        checkpoint_path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT_PATH

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    if not cuda_available:
        warnings.append("CUDA not available – RVT will run significantly slower on CPU.")

    return EnvReport(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        errors=errors,
        warnings=warnings,
        device=device,
        cuda_available=cuda_available,
    )
