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

## Troubleshooting

### Google Drive Quota Exceeded

Google Drive limits how many times a file can be downloaded in a 24-hour period. If you see:

```
⚠️  Google Drive Quota Exceeded:
   Too many users have downloaded these files recently.
```

**Solutions:**

1. **Wait 24 hours** - Google's quota resets after ~24 hours
2. **Manual browser download**:
   - Visit: https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE
   - Download files directly through Google Drive web interface
   - Place in correct directories:
     - `evio/data/fan/` - Fan datasets
     - `evio/data/drone/` - Drone datasets
     - `evio/data/fred-0/events/` - Fred-0 events
3. **Alternative access** - Contact dataset owner for different sharing method

### Resume Interrupted Downloads

The downloader automatically resumes partial downloads using HTTP Range requests:

```bash
download-datasets --yes  # Will resume any partial files
```

## Architecture

- `cli.py` - CLI entry point, argument parsing
- `download.py` - aiohttp download manager with semaphore
- `drive.py` - Google Drive confirmation token handling
- `manifest.py` - JSON manifest loading and validation
- `progress.py` - Rich progress bar management
- `verification.py` - Size and SHA256 verification
