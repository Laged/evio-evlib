# Dataset Downloader

Parallel HTTP downloader for event camera datasets from Google Drive.

## Features

- **Parallel downloads** - 3 concurrent by default (configurable)
- **Rich progress bars** - Per-chunk + per-file + overall progress
- **Smart resume** - HTTP Range requests for partial files
- **Drive token handling** - Automatic confirmation for large files (>100MB)
- **Verification** - Size check always, SHA256 optional

## Usage

From nix develop shell in repo root:

```bash
download-datasets               # Interactive with defaults
download-datasets --yes         # Skip confirmation
download-datasets --concurrency 5  # 5 parallel downloads
download-datasets --verify      # Verify SHA256 checksums
download-datasets --dry-run     # Preview without downloading
```

## Architecture

- `cli.py` - CLI entry point, argument parsing
- `download.py` - aiohttp download manager with semaphore
- `drive.py` - Google Drive confirmation token handling
- `manifest.py` - JSON manifest loading and validation
- `progress.py` - Rich progress bar management
- `verification.py` - Size and SHA256 verification
