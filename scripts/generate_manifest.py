#!/usr/bin/env python3
"""
Generate dataset manifest from Google Drive folder.

Uses gdown's internal metadata fetching to get file IDs, names, and sizes
without downloading the actual files.
"""

import json
import sys
from pathlib import Path

try:
    import gdown
    import requests
except ImportError:
    print("Error: gdown or requests not installed")
    print("Run: uv sync")
    sys.exit(1)


FOLDER_URL = "https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE"

# Map Drive filenames to repo paths
PATH_MAPPING = {
    # Fan datasets
    "fan_const_rpm.dat": "evio/data/fan/fan_const_rpm.dat",
    "fan_const_rpm.raw": "evio/data/fan/fan_const_rpm.raw",
    "fan_varying_rpm.dat": "evio/data/fan/fan_varying_rpm.dat",
    "fan_varying_rpm.raw": "evio/data/fan/fan_varying_rpm.raw",
    "fan_varying_rpm_turning.dat": "evio/data/fan/fan_varying_rpm_turning.dat",
    "fan_varying_rpm_turning.raw": "evio/data/fan/fan_varying_rpm_turning.raw",

    # Drone datasets
    "drone_idle.dat": "evio/data/drone/drone_idle.dat",
    "drone_idle.raw": "evio/data/drone/drone_idle.raw",
    "drone_moving.dat": "evio/data/drone/drone_moving.dat",
    "drone_moving.raw": "evio/data/drone/drone_moving.raw",

    # Fred-0 datasets
    "events.dat": "evio/data/fred-0/events/events.dat",
    "events.raw": "evio/data/fred-0/events/events.raw",
}


def main():
    """Generate manifest from Drive folder."""
    print(f"Fetching metadata from: {FOLDER_URL}")
    print()

    # Use gdown's download_folder with skip_download=True to get metadata only
    print("Fetching file list (metadata only, no downloads)...")
    print()

    try:
        # Get file metadata without downloading
        file_list = gdown.download_folder(
            url=FOLDER_URL,
            skip_download=True,
            quiet=False,
            remaining_ok=True
        )
    except Exception as e:
        print(f"Error fetching folder contents: {e}")
        print("gdown API may have changed. Please check gdown documentation.")
        sys.exit(1)

    if not file_list:
        print("Error: No files found in folder")
        sys.exit(1)

    print(f"Found {len(file_list)} files")
    print()

    # Build manifest
    datasets = []
    unmapped = []

    for file_obj in file_list:
        # GoogleDriveFileToDownload object attributes
        name = Path(file_obj.path).name  # Extract filename from Drive path
        file_id = file_obj.id

        # Skip if no name or ID
        if not name or not file_id:
            continue

        # Only process files we want in the manifest
        if name not in PATH_MAPPING:
            unmapped.append(name)
            continue

        # Get file size from Google Drive using HEAD request
        drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            response = requests.head(drive_url, allow_redirects=True, timeout=10)
            size = int(response.headers.get('Content-Length', 0))
            if size == 0:
                print(f"⚠ {name} - Could not determine size, using 0")
            size_mb = size / 1024 / 1024
            print(f"✓ {name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"⚠ {name} - Error getting size: {e}, using 0")
            size = 0

        datasets.append({
            "id": file_id,
            "name": name,
            "path": PATH_MAPPING[name],
            "size": size,
            "sha256": ""  # To be computed later
        })

    if unmapped:
        print()
        print("Unmapped files (not in manifest):")
        for name in unmapped:
            print(f"  - {name}")

    # Create manifest
    manifest = {
        "version": "1.0",
        "source": FOLDER_URL,
        "generated": "2025-11-15",
        "datasets": datasets
    }

    # Write to file
    output_path = Path("docs/data/datasets.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"✅ Manifest written to: {output_path}")
    print(f"   {len(datasets)} datasets included")
    print()
    print("Next steps:")
    print("  1. Review docs/data/datasets.json")
    print("  2. Optionally compute SHA256 checksums")
    print("  3. Commit the manifest")


if __name__ == '__main__':
    main()
