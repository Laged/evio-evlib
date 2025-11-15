from pathlib import Path

from rvt_detector import DEFAULT_CHECKPOINT_PATH, DEFAULT_RVT_REPO
from rvt_detector.env_check import validate_assets


def test_validate_assets_flags_missing_repo(tmp_path):
    repo = tmp_path / "repo"
    checkpoint = tmp_path / "models" / "ckpt.ckpt"
    report = validate_assets(repo, checkpoint)
    assert any("repository" in err.lower() for err in report.errors)
    assert report.repo_path == repo


def test_validate_assets_flags_missing_checkpoint(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    checkpoint = tmp_path / "missing.ckpt"
    report = validate_assets(repo, checkpoint)
    assert any("checkpoint" in err.lower() for err in report.errors)
    assert report.checkpoint_path == checkpoint


def test_validate_assets_detects_defaults():
    report = validate_assets(DEFAULT_RVT_REPO, DEFAULT_CHECKPOINT_PATH)
    assert report.errors == []
    assert report.repo_path == DEFAULT_RVT_REPO
    assert report.checkpoint_path == DEFAULT_CHECKPOINT_PATH
