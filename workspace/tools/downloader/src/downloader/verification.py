"""File verification (size and checksum)."""

import hashlib
from pathlib import Path
from typing import Dict, Tuple, Any


def verify_size(path: Path, expected: int) -> Tuple[bool, str]:
    """
    Verify file size matches expected.

    Args:
        path: File path
        expected: Expected size in bytes

    Returns:
        (valid, error_message)
    """
    actual = path.stat().st_size

    if actual != expected:
        return False, f"Size mismatch: {actual} != {expected}"

    return True, ""


def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of file (streaming for large files).

    Args:
        path: File path
        chunk_size: Chunk size for streaming (default 8KB)

    Returns:
        SHA256 hex digest
    """
    sha256 = hashlib.sha256()

    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_checksum(dataset: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify SHA256 checksum if present in manifest.

    Args:
        dataset: Dataset metadata dict

    Returns:
        (valid, error_message)
    """
    if not dataset.get('sha256'):
        return True, "No checksum in manifest"

    path = Path(dataset['path'])
    actual = compute_sha256(path)
    expected = dataset['sha256']

    if actual != expected:
        return False, f"Checksum mismatch"

    return True, ""


def check_inventory() -> Dict[str, Dict[str, int]]:
    """
    Check what datasets are actually present on disk.

    Returns:
        Inventory dict: {category: {ext: count}}
    """
    inventory = {}

    # Fan datasets
    fan_path = Path('evio/data/fan')
    if fan_path.exists():
        dat_files = list(fan_path.glob('*.dat'))
        raw_files = list(fan_path.glob('*.raw'))
        inventory['fan'] = {
            'dat': len(dat_files),
            'raw': len(raw_files)
        }
    else:
        inventory['fan'] = {'dat': 0, 'raw': 0}

    # Drone datasets (idle + moving combined)
    drone_idle_path = Path('evio/data/drone_idle')
    drone_moving_path = Path('evio/data/drone_moving')

    drone_dat_files = []
    drone_raw_files = []

    if drone_idle_path.exists():
        drone_dat_files.extend(list(drone_idle_path.glob('*.dat')))
        drone_raw_files.extend(list(drone_idle_path.glob('*.raw')))

    if drone_moving_path.exists():
        drone_dat_files.extend(list(drone_moving_path.glob('*.dat')))
        drone_raw_files.extend(list(drone_moving_path.glob('*.raw')))

    inventory['drone'] = {
        'dat': len(drone_dat_files),
        'raw': len(drone_raw_files)
    }

    # Fred-0 datasets
    fred_path = Path('evio/data/fred-0/Event')
    if fred_path.exists():
        dat_files = list(fred_path.glob('*.dat'))
        raw_files = list(fred_path.glob('*.raw'))
        inventory['fred-0'] = {
            'dat': len(dat_files),
            'raw': len(raw_files)
        }
    else:
        inventory['fred-0'] = {'dat': 0, 'raw': 0}

    return inventory


def print_inventory(inventory: Dict[str, Dict[str, int]]):
    """
    Print dataset inventory in human-readable format.

    Args:
        inventory: Inventory dict from check_inventory()
    """
    print("Current dataset inventory:")

    fan = inventory.get('fan', {})
    print(f"  Fan datasets:   {fan.get('dat', 0)} .dat files ({fan.get('raw', 0)} .raw files)")

    drone = inventory.get('drone', {})
    print(f"  Drone datasets: {drone.get('dat', 0)} .dat files ({drone.get('raw', 0)} .raw files)")

    fred = inventory.get('fred-0', {})
    print(f"  Fred-0 events:  {fred.get('dat', 0)} .dat file ({fred.get('raw', 0)} .raw file)")

    total_dat = sum(v.get('dat', 0) for v in inventory.values())
    total_raw = sum(v.get('raw', 0) for v in inventory.values())
    print()
    print(f"Total: {total_dat + total_raw} files")
