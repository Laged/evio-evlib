import os
from pathlib import Path

import polars as pl
import torch

from rvt_adapter import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_RVT_REPO,
    build_hydra_config,
    determine_device,
    events_to_hist_tensor,
    gather_environment_report,
    resolve_checkpoint_path,
    resolve_repo_path,
)


def test_resolve_repo_and_checkpoint_defaults_exist():
    repo = resolve_repo_path()
    checkpoint = resolve_checkpoint_path()
    assert repo == DEFAULT_RVT_REPO
    assert checkpoint == DEFAULT_CHECKPOINT_PATH
    assert repo.is_dir()
    assert checkpoint.is_file()


def test_build_hydra_config_sets_dataset_and_checkpoint(tmp_path):
    repo = DEFAULT_RVT_REPO
    checkpoint = DEFAULT_CHECKPOINT_PATH
    cfg = build_hydra_config(
        repo=repo,
        dataset_name="gen1",
        experiment="default",
        checkpoint_path=checkpoint,
        extra_overrides=[f"logging.dir={tmp_path}"],
    )
    dataset_dir = Path(cfg.dataset.path)
    assert dataset_dir.exists()
    assert Path(cfg.checkpoint) == checkpoint
    assert cfg.batch_size.eval == 1


def test_build_hydra_config_accepts_relative_repo_path(tmp_path):
    repo = Path(os.path.relpath(DEFAULT_RVT_REPO, start=Path.cwd()))
    checkpoint = Path(os.path.relpath(DEFAULT_CHECKPOINT_PATH, start=Path.cwd()))
    cfg = build_hydra_config(
        repo=resolve_repo_path(repo),
        dataset_name="gen1",
        experiment="default",
        checkpoint_path=resolve_checkpoint_path(checkpoint),
        extra_overrides=[f"logging.dir={tmp_path}"],
    )
    assert Path(cfg.checkpoint) == resolve_checkpoint_path(checkpoint)


def test_determine_device_falls_back_when_cuda_missing(monkeypatch):
    from rvt_adapter import helpers

    monkeypatch.setattr(helpers.torch.cuda, "is_available", lambda: False)
    device = determine_device("cuda")
    assert device.type == "cpu"


def test_events_to_hist_tensor_handles_empty_frames():
    events = pl.DataFrame(
        {
            "t": pl.Series([], dtype=pl.Int64),
            "x": pl.Series([], dtype=pl.Int64),
            "y": pl.Series([], dtype=pl.Int64),
            "polarity": pl.Series([], dtype=pl.Int8),
        }
    )
    tensor = events_to_hist_tensor(events, sensor_hw=(4, 4), bins=2, window_duration_ms=10.0)
    assert tensor.shape == (1, 4, 4, 4)
    assert torch.all(tensor == 0)


def test_events_to_hist_tensor_populates_bins(monkeypatch):
    events = pl.DataFrame(
        {
            "t": [0, 40_000, 80_000],
            "x": [0, 1, 1],
            "y": [0, 0, 1],
            "polarity": [0, 1, 0],
        }
    )
    tensor = events_to_hist_tensor(events, sensor_hw=(2, 2), bins=2, window_duration_ms=50.0)
    assert tensor.shape == (1, 4, 2, 2)
    # First event lands in channel 0
    assert tensor[0, 0, 0, 0] == 1.0
    # Second event: bin=1, polarity=1 → channel 3
    assert tensor[0, 3, 0, 1] == 1.0
    # Third event: bin clipped to last channel and position (1,1)
    assert tensor[0, 2, 1, 1] == 1.0


def test_events_to_hist_tensor_fallback_without_evlib(monkeypatch):
    from rvt_adapter import helpers

    monkeypatch.setattr(helpers, "_evlib_histogram", None)
    events = pl.DataFrame(
        {
            "t": [0, 25_000],
            "x": [0, 0],
            "y": [0, 0],
            "polarity": [0, 1],
        }
    )
    tensor = events_to_hist_tensor(events, sensor_hw=(1, 1), bins=2, window_duration_ms=50.0)
    assert tensor[0, 0, 0, 0] == 1.0
    assert tensor[0, 1, 0, 0] == 1.0


def test_gather_environment_report_success():
    report = gather_environment_report()
    assert report.repo_path == DEFAULT_RVT_REPO
    assert report.checkpoint_path == DEFAULT_CHECKPOINT_PATH
    assert report.errors == []
