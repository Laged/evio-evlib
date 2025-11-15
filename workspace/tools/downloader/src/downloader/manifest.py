"""Manifest loading and validation."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    Load and validate manifest file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Validated manifest dictionary

    Raises:
        SystemExit: On validation errors (exit code 2)
    """
    if not manifest_path.exists():
        print(f"❌ Error: Manifest not found: {manifest_path}")
        print(f"   Expected location: {manifest_path.absolute()}")
        sys.exit(2)

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in manifest: {e}")
        sys.exit(2)

    # Validate required fields
    if 'datasets' not in manifest:
        print("❌ Error: Manifest missing 'datasets' field")
        sys.exit(2)

    if not isinstance(manifest['datasets'], list):
        print("❌ Error: 'datasets' must be an array")
        sys.exit(2)

    # Validate each dataset entry
    required_fields = ['id', 'name', 'path', 'size']
    for i, dataset in enumerate(manifest['datasets']):
        for field in required_fields:
            if field not in dataset:
                print(f"❌ Error: Dataset {i} missing required field '{field}'")
                sys.exit(2)

    return manifest


def filter_datasets(
    datasets: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate datasets into download queue and skip list based on file existence.

    Args:
        datasets: List of dataset metadata dicts

    Returns:
        (to_download, to_skip) tuple
    """
    to_download = []
    to_skip = []

    for dataset in datasets:
        path = Path(dataset['path'])

        # File doesn't exist - download it
        if not path.exists():
            dataset['download_reason'] = "not present"
            to_download.append(dataset)
            continue

        # File exists - check size
        actual_size = path.stat().st_size
        expected_size = dataset['size']

        if actual_size == expected_size:
            # Exact match - skip
            dataset['skip_reason'] = "already present (correct size)"
            to_skip.append(dataset)
        else:
            # Size mismatch - will be removed and re-downloaded
            dataset['download_reason'] = f"size mismatch ({actual_size} != {expected_size})"
            to_download.append(dataset)

    return to_download, to_skip
