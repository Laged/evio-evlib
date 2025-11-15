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
    from gdown import google_drive
except ImportError:
    print("Error: gdown not installed")
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

    # Parse folder ID from URL
    parsed = google_drive.parse_url(FOLDER_URL)
    folder_id = parsed.get("id")

    if not folder_id:
        print("Error: Could not parse folder ID from URL")
        sys.exit(1)

    print(f"Folder ID: {folder_id}")
    print("Fetching file list...")
    print()

    # Get file metadata (gdown's internal helper)
    try:
        files = google_drive._get_folder_contents(
            folder_id,
            remaining_ok=True
        )
    except AttributeError:
        # Try alternative method name (gdown API may vary)
        try:
            files = []
            # Walk folder recursively
            def walk_folder(fid):
                items = google_drive._get_directory_structure(fid, None)
                for item in items:
                    if item[1] == "file":
                        files.append({
                            "id": item[0],
                            "title": item[2],
                            "fileSize": item[3] if len(item) > 3 else 0
                        })
                    elif item[1] == "folder":
                        walk_folder(item[0])

            walk_folder(folder_id)
        except Exception as e:
            print(f"Error fetching folder contents: {e}")
            print("gdown API may have changed. Please check gdown documentation.")
            sys.exit(1)

    print(f"Found {len(files)} files")
    print()

    # Build manifest
    datasets = []
    unmapped = []

    for file_info in files:
        name = file_info.get("title", "")
        file_id = file_info.get("id", "")
        size = int(file_info.get("fileSize", 0))

        # Skip directories and non-data files
        if not name or not file_id:
            continue

        # Map to repo path
        if name in PATH_MAPPING:
            datasets.append({
                "id": file_id,
                "name": name,
                "path": PATH_MAPPING[name],
                "size": size,
                "sha256": ""  # To be computed later
            })
            print(f"✓ {name} ({size / 1024 / 1024:.1f} MB)")
        else:
            unmapped.append(name)

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
