"""Utility CLI for validating RVT detector prerequisites."""

from __future__ import annotations

import argparse
from pathlib import Path

from rvt_adapter import gather_environment_report


def validate_assets(repo: Path | str | None, checkpoint: Path | str | None):
    """Return an EnvReport describing any missing requirements."""

    return gather_environment_report(repo=repo, checkpoint=checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RVT detector environment (repo, checkpoints, CUDA)."
    )
    parser.add_argument(
        "--rvt-repo",
        type=Path,
        default=None,
        help="Override path to the RVT repository (defaults to vendored submodule).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override path to an RVT checkpoint (.ckpt).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_assets(args.rvt_repo, args.checkpoint)
    if report.errors:
        print("❌ RVT environment check failed:")
        for err in report.errors:
            print(f"  - {err}")
    else:
        print("✅ RVT assets detected.")
    for warn in report.warnings:
        print(f"⚠️  {warn}")
    print(f"Selected device: {report.device}")
    return 1 if report.errors else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
