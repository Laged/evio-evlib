"""CLI entry point for dataset downloader."""

import argparse
import asyncio
import sys
from pathlib import Path

from .manifest import load_manifest, filter_datasets
from .progress import download_with_progress
from .verification import verify_checksum, check_inventory, print_inventory


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download event camera datasets from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive download with defaults
  %(prog)s --yes              # Skip confirmation prompt
  %(prog)s --concurrency 5    # Download 5 files simultaneously
  %(prog)s --verify           # Verify checksums after download
  %(prog)s --dry-run          # Show what would be downloaded
        """
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        default=3,
        metavar='N',
        help='Number of parallel downloads (default: 3, warn if >5)'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify SHA256 checksums after download'
    )

    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('docs/data/datasets.json'),
        metavar='PATH',
        help='Path to manifest file (default: docs/data/datasets.json)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without downloading'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output, no progress bars'
    )

    return parser.parse_args()


def confirm_download(to_download: list, skip_yes: bool) -> bool:
    """
    Ask user to confirm download.

    Args:
        to_download: List of datasets to download
        skip_yes: Skip prompt if True

    Returns:
        True if user confirms, False otherwise
    """
    if skip_yes:
        return True

    # Calculate total size
    total_size = sum(ds['size'] for ds in to_download)
    total_mb = total_size / 1024 / 1024
    total_gb = total_size / 1024 / 1024 / 1024

    print("⚠️  WARNING: This will download:")
    print(f"   {len(to_download)} files")
    if total_gb >= 1:
        print(f"   {total_gb:.2f} GB")
    else:
        print(f"   {total_mb:.1f} MB")
    print()
    print("Dataset includes:")
    print("  - Fan datasets (constant/varying RPM)")
    print("  - Drone datasets (idle/moving)")
    print("  - Fred-0 reference data")
    print()

    response = input("Continue with download? (y/N): ").strip().lower()
    return response in ('y', 'yes')


def main():
    """Main entry point."""
    args = parse_args()

    # Warn about high concurrency
    if args.concurrency > 5:
        print("⚠️  High concurrency (>5) may trigger rate limits")
        print("    Consider using default (3) for reliable downloads")
        print()

    # Load manifest
    manifest = load_manifest(args.manifest)

    print("=" * 50)
    print("  Event Camera Dataset Downloader")
    print("=" * 50)
    print()
    print(f"Loaded {len(manifest['datasets'])} datasets from manifest")
    print(f"Concurrency: {args.concurrency}")
    print(f"Verify checksums: {args.verify}")
    print()

    # Filter datasets
    to_download, to_skip = filter_datasets(manifest['datasets'])

    print(f"To download: {len(to_download)} files")
    print(f"To skip: {len(to_skip)} files (already present)")
    print()

    # Dry run mode
    if args.dry_run:
        if to_download:
            print("Would download:")
            for ds in to_download:
                size_mb = ds['size'] / 1024 / 1024
                print(f"  - {ds['name']} ({size_mb:.1f} MB) - {ds['download_reason']}")

        if to_skip:
            print()
            print("Would skip:")
            for ds in to_skip:
                print(f"  - {ds['name']} - {ds['skip_reason']}")

        return 0

    # Check if nothing to download
    if not to_download:
        print("✅ All files already present!")
        print()
        inventory = check_inventory()
        print_inventory(inventory)
        return 0

    # Confirm download
    if not confirm_download(to_download, args.yes):
        print("Download cancelled.")
        return 0

    # Download files
    print()
    print("Downloading datasets...")
    print(f"(Parallel downloads: {args.concurrency})")
    print()

    if args.quiet:
        # Simple progress without rich
        print("Downloading in quiet mode...")
        # TODO: Implement simple progress
        successes = []
        failures = []
    else:
        # Rich progress display
        successes, failures = asyncio.run(
            download_with_progress(to_download, args.concurrency)
        )

    # Verify checksums if requested
    verify_failures = []
    if args.verify and successes:
        print()
        print("Verifying checksums...")
        print()

        for dataset in successes:
            valid, error = verify_checksum(dataset)
            if not valid:
                verify_failures.append((dataset, error))
                print(f"  ✗ {dataset['name']}: {error}")
            else:
                if dataset.get('sha256'):
                    print(f"  ✓ {dataset['name']}")
                else:
                    print(f"  ⊘ {dataset['name']} (no checksum in manifest)")

        # Add verify failures to failures list
        if verify_failures:
            failures.extend(verify_failures)
            successes = [s for s in successes if s not in [d for d, _ in verify_failures]]

    # Summary
    print()
    print("=" * 50)
    print("  Download Summary")
    print("=" * 50)
    print()
    print(f"✅ Successfully downloaded: {len(successes)} files")
    print(f"⏭️  Skipped (already present): {len(to_skip)} files")

    if failures:
        print(f"❌ Failed: {len(failures)} files")
        print()

        # Check if all/most failures are quota exceeded
        quota_failures = [(d, e) for d, e in failures if e == "QUOTA_EXCEEDED"]
        other_failures = [(d, e) for d, e in failures if e != "QUOTA_EXCEEDED"]

        if quota_failures:
            print("⚠️  Google Drive Quota Exceeded:")
            print()
            print("   Too many users have downloaded these files recently.")
            print("   Google Drive has temporarily blocked access.")
            print()
            print("   Solutions:")
            print("   1. Wait 24 hours and try again")
            print("   2. Download manually via browser:")
            print("      https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE")
            print("   3. Contact dataset owner for alternative access")
            print()
            print(f"   Affected files ({len(quota_failures)}):")
            for dataset, _ in quota_failures:
                print(f"     - {dataset['name']}")

        if other_failures:
            if quota_failures:
                print()
            print("Other failures:")
            for dataset, error in other_failures:
                print(f"  - {dataset['name']}: {error}")
                print(f"    Manual: https://drive.google.com/uc?id={dataset['id']}")

    print()
    inventory = check_inventory()
    print_inventory(inventory)

    # Show demo commands if fan datasets present
    if inventory.get('fan', {}).get('dat', 0) > 0:
        print()
        print("Ready to run:")
        print("  run-demo-fan")
        print("  run-mvp-1")
        print("  run-mvp-2")
    else:
        print()
        print("⚠️  No fan datasets found - demos may not work")

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
