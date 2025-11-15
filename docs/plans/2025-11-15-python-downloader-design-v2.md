# Python Dataset Downloader - Design Document (v2 - Revised)

**Date:** 2025-11-15
**Status:** Approved Design (Revised)
**Context:** Replace shell-based gdown with aiohttp-based Python downloader providing parallel HTTP streaming, progress tracking, and robust error handling

**Requirements Source:** `docs/data/download-feedback.md`
**Supersedes:** `2025-11-15-python-downloader-design.md` (v1)

---

## Executive Summary

Replace the current sequential shell-based dataset downloader with an aiohttp-based Python CLI that provides:
- **True parallel HTTP streaming** (3 concurrent by default, configurable)
- **Rich progress bars** (per-chunk + per-file + overall progress)
- **Smart resume** (HTTP Range requests for partial files)
- **Robust error handling** (continue on individual failures, detailed reporting)
- **Manifest-driven** (centralized metadata in JSON, generated from Drive)
- **Isolated dependencies** (workspace/tools/downloader package)

**Key improvement:** ~3x faster downloads through parallel HTTP streaming, better UX with chunk-level progress, resumable downloads.

**Architecture:** aiohttp async HTTP client with semaphore-based concurrency, rich progress bars, manifest-driven metadata, isolated workspace package.

---

## Design Principles

1. **True Parallelism**: Concurrent HTTP streams (not wrapped sync calls)
2. **Safe Defaults**: Works without configuration (`download-datasets` just works)
3. **Fail Gracefully**: Individual file failures don't abort entire download
4. **Verify Integrity**: Size check always, SHA256 optional
5. **Clear Feedback**: Rich progress bars show exactly what's happening
6. **Isolated Dependencies**: Tool deps don't pollute evio library

---

## Requirements Traceability

All requirements trace to `docs/data/download-feedback.md`:

| Requirement ID | Requirement | Implementation | Section |
|----------------|-------------|----------------|---------|
| REQ-1 | Parallelism (3 concurrent) | asyncio.Semaphore(3) with aiohttp | §4.1 Download Manager |
| REQ-2 | Progress visualization | rich.progress with per-chunk updates | §4.4 Progress Tracker |
| REQ-3 | Resumable downloads | HTTP Range requests | §4.3 Resume Logic |
| REQ-4 | Drive API resilience | Manual confirmation token handling | §4.2 Drive Handler |
| REQ-5 | Manifest-based | JSON metadata with real IDs | §3 Manifest Structure |

---

## Architecture

### 1. Package Structure

**New Workspace Member:** `workspace/tools/downloader`

```
workspace/tools/downloader/
├── pyproject.toml              # Package config with deps
├── README.md                   # Tool documentation
└── src/
    └── downloader/
        ├── __init__.py         # Package init
        ├── cli.py              # CLI entry point, arg parsing
        ├── download.py         # aiohttp download logic
        ├── manifest.py         # Manifest loading/validation
        ├── progress.py         # Rich progress bar management
        ├── verification.py     # Size/SHA256 verification
        └── drive.py            # Drive confirmation token handling
```

**Dependencies (workspace/tools/downloader/pyproject.toml):**
```toml
[project]
name = "downloader"
version = "0.1.0"
description = "Event camera dataset downloader with parallel streaming"
requires-python = ">=3.11"
dependencies = [
    "aiohttp>=3.9.0",      # Async HTTP client
    "rich>=13.0.0",        # Progress bars
    "aiofiles>=23.0.0",    # Async file I/O
]

[project.scripts]
download-datasets = "downloader.cli:main"
```

**UV Workspace Integration (root pyproject.toml):**
```toml
[tool.uv.workspace]
members = [
    "evio",
    "workspace/libs/*",
    "workspace/plugins/*",
    "workspace/apps/*",
    "workspace/tools/*",    # Add tools directory
]
```

---

### 2. Component Architecture

**Component Diagram:**
```
┌─────────────────────────────────────────────────┐
│ CLI (cli.py)                                    │
│ - Parse arguments                               │
│ - Validate options                              │
│ - Orchestrate download flow                     │
└─────────┬───────────────────────────────────────┘
          │
          ├──> Manifest Loader (manifest.py)
          │    - Load JSON
          │    - Validate schema
          │    - Filter datasets
          │
          ├──> Download Manager (download.py)
          │    - Create aiohttp session
          │    - Semaphore-based concurrency
          │    - Coordinate parallel downloads
          │    │
          │    ├──> Drive Handler (drive.py)
          │    │    - Check for confirmation page
          │    │    - Extract confirmation token
          │    │    - Retry with token
          │    │
          │    └──> File Downloader
          │         - HTTP Range support
          │         - Chunked streaming
          │         - Progress callbacks
          │
          ├──> Progress Tracker (progress.py)
          │    - Rich progress bars
          │    - Per-file tasks
          │    - Overall task
          │    - Speed/ETA calculation
          │
          └──> Verification (verification.py)
               - Size verification
               - SHA256 computation
               - Inventory check
```

**Data Flow:**
```
Manifest (JSON)
    ↓
Manifest Loader → Filter datasets (check existing)
    ↓
Download Manager
    ↓
┌───────────────────────────────┐
│ Semaphore (max 3 concurrent)  │
│   ├─> File 1 ─> aiohttp ─> chunks ─> progress
│   ├─> File 2 ─> aiohttp ─> chunks ─> progress
│   └─> File 3 ─> aiohttp ─> chunks ─> progress
└───────────────────────────────┘
    ↓
Verification (size + optional SHA256)
    ↓
Summary Report + Inventory
```

---

## 3. Manifest Structure

### 3.1 Manifest Format

**File:** `docs/data/datasets.json`

```json
{
  "version": "1.0",
  "source": "https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE",
  "generated": "2025-11-15",
  "datasets": [
    {
      "id": "1abc123def456...",
      "name": "fan_const_rpm.dat",
      "path": "evio/data/fan/fan_const_rpm.dat",
      "size": 212336640,
      "sha256": "a3f5e8c9..."
    }
  ]
}
```

**Fields:**
- `version`: Manifest format version (semantic versioning)
- `source`: Original Google Drive folder URL (for reference)
- `generated`: ISO date when manifest was generated
- `datasets`: Array of dataset objects
  - `id`: Google Drive file ID
  - `name`: Display name (from Drive)
  - `path`: Target path in repo (relative to root)
  - `size`: File size in bytes (for verification and progress)
  - `sha256`: Optional checksum (empty string if not computed)

### 3.2 Manifest Generation

**Tool:** `scripts/generate_manifest.py` (already created)

**Workflow:**
```bash
# From repo root in nix develop
python scripts/generate_manifest.py

# Fetches metadata from Drive folder (no download)
# Outputs: docs/data/datasets.json
```

**How it works:**
```python
from gdown import google_drive

folder_id = google_drive.parse_url(FOLDER_URL)["id"]
files = google_drive._get_folder_contents(folder_id, remaining_ok=True)

manifest = {
    "datasets": [
        {
            "id": f["id"],
            "name": f["title"],
            "path": PATH_MAPPING[f["title"]],  # Manual mapping
            "size": int(f["fileSize"]),
            "sha256": ""  # Compute later if needed
        }
        for f in files
        if f["title"] in PATH_MAPPING
    ]
}
```

**Benefits:**
- No full Drive API / OAuth needed
- Real file IDs and sizes (not placeholders)
- Regenerable when folder changes
- Version-controlled metadata

---

## 4. Core Components

### 4.1 Download Manager

**File:** `src/downloader/download.py`

**Responsibilities:**
- Create aiohttp.ClientSession
- Manage concurrency with asyncio.Semaphore
- Coordinate parallel downloads
- Handle exceptions and retries

**Key Implementation:**
```python
async def download_all(
    datasets: List[Dict],
    concurrency: int = 3,
    progress: Progress = None
) -> Tuple[List[Dict], List[Tuple[Dict, str]]]:
    """
    Download all datasets in parallel.

    Args:
        datasets: List of dataset metadata dicts
        concurrency: Max concurrent downloads
        progress: Rich progress tracker

    Returns:
        (successes, failures)
    """
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_file(session, ds, semaphore, progress)
            for ds in datasets
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successes and failures
    successes = []
    failures = []

    for dataset, result in zip(datasets, results):
        if isinstance(result, Exception):
            failures.append((dataset, str(result)))
        elif result[0]:  # (success, error) tuple
            successes.append(dataset)
        else:
            failures.append((dataset, result[1]))

    return successes, failures
```

### 4.2 Drive Confirmation Handler

**File:** `src/downloader/drive.py`

**Problem:** Google Drive files >100MB require confirmation token to prevent abuse.

**Detection:**
```python
async def needs_confirmation(session: aiohttp.ClientSession, file_id: str) -> bool:
    """Check if file requires confirmation token."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    async with session.head(url, allow_redirects=True) as resp:
        # Check for confirmation redirect
        if "confirm=" in str(resp.url):
            return True

        # Check for download warning in headers
        if "download_warning" in resp.headers.get("content-disposition", ""):
            return True

    return False
```

**Token Extraction:**
```python
async def get_confirmation_token(session: aiohttp.ClientSession, file_id: str) -> Optional[str]:
    """Extract confirmation token from Drive page."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    async with session.get(url) as resp:
        text = await resp.text()

        # Find confirm token in HTML/redirect
        # Pattern: confirm=<token>
        match = re.search(r'confirm=([^&"\']+)', text)
        if match:
            return match.group(1)

        # Alternative: UUID in download link
        match = re.search(r'download\?id=.*&confirm=([^&"\']+)', text)
        if match:
            return match.group(1)

    return None
```

**Download with Token and Cookie Preservation:**
```python
async def download_with_confirmation(
    session: aiohttp.ClientSession,
    file_id: str,
    path: Path,
    progress_callback
) -> Tuple[bool, str]:
    """
    Download file, handling Drive confirmation tokens and cookies.

    Google Drive large files (>100MB) require:
    1. Confirmation token (extracted from HTML page)
    2. download_warning cookie (set by that HTML response)

    Both must be present in the final download request.
    The aiohttp.ClientSession automatically preserves cookies.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Step 1: Check if confirmation needed
    async with session.head(url, allow_redirects=True) as resp:
        needs_confirm = "confirm=" in str(resp.url) or \
                       "download_warning" in resp.headers.get("content-disposition", "")

    if not needs_confirm:
        # Small file, direct download
        async with session.get(url) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}: {resp.reason}"

            async with aiofiles.open(path, 'wb') as f:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    await f.write(chunk)
                    if progress_callback:
                        progress_callback(len(chunk))
        return True, ""

    # Step 2: Fetch confirmation page to get token AND cookie
    async with session.get(url) as resp:
        html = await resp.text()

        # Extract confirmation token from HTML
        token = None
        for pattern in [
            r'confirm=([^&"\']+)',
            r'download\?id=.*&confirm=([^&"\']+)',
            r'id="download-form".*?action=".*?confirm=([^&"\']+)',
        ]:
            match = re.search(pattern, html)
            if match:
                token = match.group(1)
                break

        if not token:
            return False, "Could not extract confirmation token from Drive page"

        # CRITICAL: Session now has download_warning cookie from this response
        # aiohttp.ClientSession automatically preserves cookies for the domain

    # Step 3: Download with token (cookie automatically included by session)
    download_url = f"{url}&confirm={token}"

    async with session.get(download_url) as resp:
        if resp.status != 200:
            return False, f"HTTP {resp.status}: {resp.reason}"

        # Verify we got file, not HTML error page
        content_type = resp.headers.get('content-type', '')
        if 'text/html' in content_type:
            # Still got HTML - token/cookie didn't work
            snippet = (await resp.text())[:200]
            return False, f"Got HTML instead of file. Response: {snippet}..."

        # Stream download
        async with aiofiles.open(path, 'wb') as f:
            async for chunk in resp.content.iter_chunked(1 << 20):  # 1MB chunks
                await f.write(chunk)
                if progress_callback:
                    progress_callback(len(chunk))

    return True, ""
```

**Key Points:**
- **aiohttp.ClientSession maintains cookie jar automatically** - Cookies set by Drive HTML page persist for domain
- **Fetch HTML page first** - Sets `download_warning` cookie in session
- **Extract token** - Parse from HTML using multiple patterns
- **Download with both** - Token in URL, cookie in session headers (automatic)
- **Verify content-type** - Detect if we still got HTML instead of file bytes (failure case)

### 4.3 Resume Logic

**File:** `src/downloader/download.py`

**Strategy:** HTTP Range requests with automatic fallback

```python
async def download_file_resumable(
    session: aiohttp.ClientSession,
    dataset: Dict,
    semaphore: asyncio.Semaphore,
    progress: Progress,
    task_id: int
) -> Tuple[bool, str]:
    """
    Download file with resume support and fallback.

    Strategy:
    1. If file complete → skip
    2. If partial file exists → try Range request
    3. If Range fails (206 not supported) → delete partial, full download
    4. If full download fails → error
    """
    path = Path(dataset['path'])
    file_id = dataset['id']
    expected_size = dataset['size']

    # Check existing file
    resume_from = 0
    if path.exists():
        actual_size = path.stat().st_size

        if actual_size == expected_size:
            # Complete - skip
            progress.update(task_id, description=f"[green]✓ {dataset['name']} (already present)")
            progress.update(task_id, completed=expected_size)
            return True, ""

        elif actual_size < expected_size:
            # Partial - try resume
            resume_from = actual_size
            progress.update(task_id, completed=actual_size)
            progress.update(task_id, description=f"[yellow]↻ {dataset['name']} (resuming from {actual_size / 1024 / 1024:.1f} MB)")
        else:
            # Larger than expected - corrupt
            path.unlink()
            resume_from = 0

    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build URL with Drive confirmation handling
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if await needs_confirmation(session, file_id):
        token = await get_confirmation_token(session, file_id)
        if token:
            url = f"{url}&confirm={token}"
        else:
            return False, "Could not get confirmation token"

    async with semaphore:
        try:
            # Attempt 1: Resume with Range header (if partial file)
            if resume_from > 0:
                headers = {'Range': f'bytes={resume_from}-'}
                mode = 'ab'  # Append mode

                async with session.get(url, headers=headers) as resp:
                    if resp.status == 206:
                        # Server supports Range - resume from offset
                        progress.update(task_id, description=f"[yellow]↓ {dataset['name']} (resuming)")

                        async with aiofiles.open(path, mode) as f:
                            async for chunk in resp.content.iter_chunked(1 << 20):
                                await f.write(chunk)
                                progress.update(task_id, advance=len(chunk))

                        # Verify size
                        actual_size = path.stat().st_size
                        if actual_size != expected_size:
                            return False, f"Size mismatch after resume: {actual_size} != {expected_size}"

                        progress.update(task_id, description=f"[green]✓ {dataset['name']}")
                        return True, ""

                    elif resp.status == 200:
                        # Server doesn't support Range, sent full file
                        # FALLBACK: Delete partial, accept full download
                        progress.update(task_id, description=f"[yellow]↓ {dataset['name']} (Range not supported, restarting)")
                        path.unlink()  # Delete partial
                        resume_from = 0
                        progress.update(task_id, completed=0)  # Reset progress

                        # Download full file from this response
                        async with aiofiles.open(path, 'wb') as f:
                            async for chunk in resp.content.iter_chunked(1 << 20):
                                await f.write(chunk)
                                progress.update(task_id, advance=len(chunk))

                        # Verify size
                        actual_size = path.stat().st_size
                        if actual_size != expected_size:
                            return False, f"Size mismatch: {actual_size} != {expected_size}"

                        progress.update(task_id, description=f"[green]✓ {dataset['name']}")
                        return True, ""

                    else:
                        # Unexpected status - fall through to full download
                        progress.update(task_id, description=f"[yellow]↓ {dataset['name']} (Range failed, restarting)")
                        path.unlink()
                        resume_from = 0
                        progress.update(task_id, completed=0)

            # Attempt 2: Full download (no Range header)
            if resume_from == 0:
                progress.update(task_id, description=f"[yellow]↓ {dataset['name']}")

                async with session.get(url) as resp:
                    if resp.status != 200:
                        return False, f"HTTP {resp.status}: {resp.reason}"

                    # Verify content-type (detect HTML error pages)
                    content_type = resp.headers.get('content-type', '')
                    if 'text/html' in content_type:
                        snippet = (await resp.text())[:200]
                        return False, f"Got HTML instead of file: {snippet}..."

                    async with aiofiles.open(path, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(1 << 20):
                            await f.write(chunk)
                            progress.update(task_id, advance=len(chunk))

                # Verify size
                actual_size = path.stat().st_size
                if actual_size != expected_size:
                    return False, f"Size mismatch: {actual_size} != {expected_size}"

                progress.update(task_id, description=f"[green]✓ {dataset['name']}")
                return True, ""

        except Exception as e:
            # Clean up partial file on exception
            if path.exists():
                path.unlink()
            return False, f"Download failed: {str(e)}"

    return False, "Unexpected code path"
```

**Resume Flow:**
```
Partial file exists (500 MB of 1 GB)
  ↓
Try Range request with bytes=500000000-
  ↓
├─ 206 Partial Content → Resume from 500 MB ✓
├─ 200 OK → Delete partial, download full file from this response ✓
└─ 416/Other → Delete partial, retry full download ✓
```

**Benefits:**
- Resume from byte offset when server supports it (save bandwidth)
- Automatic fallback when Range not supported (avoid stuck state)
- Clean up partial files on errors (prevent corruption)
- Multiple fallback paths (robust recovery)
- Skip already-complete files

### 4.4 Progress Tracker

**File:** `src/downloader/progress.py`

**Requirements:** Per-chunk, per-file, and overall progress

**Implementation:**
```python
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TextColumn,
    TimeRemainingColumn,
)

def create_progress() -> Progress:
    """Create rich progress tracker."""
    return Progress(
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

async def download_with_progress(
    datasets: List[Dict],
    concurrency: int = 3
) -> Tuple[List[Dict], List[Tuple]]:
    """Download with rich progress display."""

    with create_progress() as progress:
        # Overall task
        overall_task = progress.add_task(
            "[cyan]Overall Progress",
            total=len(datasets)
        )

        # Create task for each file
        tasks_map = {}
        for dataset in datasets:
            task_id = progress.add_task(
                f"[dim]{dataset['name']}",
                total=dataset['size']
            )
            tasks_map[dataset['name']] = task_id

        # Download all
        successes, failures = await download_all(
            datasets,
            concurrency,
            progress=progress,
            tasks_map=tasks_map
        )

        # Update overall
        progress.update(overall_task, completed=len(datasets))

    return successes, failures
```

**Progress Display:**
```
fan_const_rpm.dat          [████████▌·] 85.2% • 181/212 MB • 4.2 MB/s • 0:00:07
fan_varying_rpm.dat        [███▌······] 35.1% • 171/489 MB • 3.8 MB/s • 0:01:23
drone_idle.dat             [··········]  0.0% • 0/150 MB • waiting...

Overall Progress           [████▌·····] 45.0% • 3/12 files • 0:02:15 remaining
```

---

## 5. CLI Interface

### 5.1 Command Structure

```bash
download-datasets [OPTIONS]
```

### 5.2 Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--yes`, `-y` | flag | False | Skip confirmation prompt |
| `--concurrency N` | int | 3 | Number of parallel downloads (warn if >5) |
| `--verify` | flag | False | Verify SHA256 checksums after download |
| `--manifest PATH` | path | `docs/data/datasets.json` | Path to manifest file |
| `--dry-run` | flag | False | Show what would be downloaded |
| `--quiet`, `-q` | flag | False | Minimal output (no progress bars) |
| `--resume` | flag | True | Enable resume (HTTP Range requests) |

### 5.3 Usage Examples

```bash
# Basic usage (interactive, safe defaults)
download-datasets

# Unattended/CI mode
download-datasets --yes --quiet

# Tune performance
download-datasets --concurrency 5

# Paranoid verification
download-datasets --verify

# Preview without downloading
download-datasets --dry-run

# Disable resume (force full redownload)
download-datasets --no-resume
```

### 5.4 Exit Codes

- **0**: All files downloaded successfully (or all already present)
- **1**: Some files failed (partial success)
- **2**: Complete failure (manifest error, no files downloaded)

---

## 6. Error Handling

### 6.1 Error Categories

| Error Type | Handling | User Impact |
|------------|----------|-------------|
| **Manifest parse error** | Abort immediately | Clear error, exit code 2 |
| **Network timeout** | Retry once, then fail | Continue with other files |
| **Permission denied** | Skip file, warn | Show manual download URL |
| **Disk space full** | Abort immediately | Critical error, can't continue |
| **Drive rate limit** | Exponential backoff | Slow down, retry |
| **Size mismatch** | Mark as failed | Re-download on next run |
| **Checksum mismatch** | Mark as failed (--verify) | Re-download on next run |

### 6.2 Error Reporting

**Per-File Errors:**
```python
{
    'dataset': {...},
    'error': 'HTTP 403: Forbidden',
    'url': 'https://drive.google.com/uc?id=...'
}
```

**Summary Report:**
```
❌ Failed: 2 files

Failed downloads:
  - hong_kong_city.glb (HTTP 403: Forbidden)
    Manual download: https://drive.google.com/uc?id=18gpWinw...

  - drone_moving.dat (Connection timeout)
    Will retry on next run
```

---

## 7. Verification

### 7.1 Size Verification

**When:** Always (after every download)

```python
def verify_size(path: Path, expected: int) -> Tuple[bool, str]:
    """Verify file size matches expected."""
    actual = path.stat().st_size

    if actual != expected:
        return False, f"Size mismatch: {actual} != {expected}"

    return True, ""
```

### 7.2 Checksum Verification

**When:** Optional (with `--verify` flag)

```python
import hashlib

async def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash (streaming for large files)."""
    sha256 = hashlib.sha256()

    async with aiofiles.open(path, 'rb') as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()

async def verify_checksum(dataset: Dict) -> Tuple[bool, str]:
    """Verify SHA256 checksum."""
    if not dataset.get('sha256'):
        return True, "No checksum in manifest"

    path = Path(dataset['path'])
    actual = await compute_sha256(path)
    expected = dataset['sha256']

    if actual != expected:
        return False, f"Checksum mismatch"

    return True, ""
```

---

## 8. Integration

### 8.1 Workspace and Flake Integration

**Step 1: Update root pyproject.toml to include tools directory**

**File:** `pyproject.toml` (repo root)

**Add to workspace members:**
```toml
[tool.uv.workspace]
members = [
    "evio",
    "workspace/libs/*",
    "workspace/plugins/*",
    "workspace/apps/*",
    "workspace/tools/*",    # ADD THIS LINE
]
```

**Step 2: Regenerate lockfile**

```bash
uv lock
```

**Expected:** `uv.lock` updated with downloader package dependencies (aiohttp, rich, aiofiles).

**Step 3: Verify package discovery**

```bash
uv run --package downloader download-datasets --help
```

**Expected:** Shows help text (package discovered successfully).

**Step 4: Update flake.nix shell alias**

**File:** `flake.nix`

**Replace old shell script with new alias:**
```nix
# Replace old download-datasets shell script (lines 15-197) with:
alias download-datasets='uv run --package downloader download-datasets'
```

**Remove:** Old shell-based download script (lines 15-197)

### 8.2 Documentation Updates

**1. evio/data/README.md:**

```markdown
## Downloading Datasets

From the `nix develop` shell:

```bash
download-datasets
```

**Features:**
- **Parallel HTTP streaming** - 3 files simultaneously (3x faster)
- **Rich progress bars** - Per-chunk progress with speed/ETA
- **Smart resume** - Continue partial downloads (HTTP Range requests)
- **Optional verification** - SHA256 checksums with `--verify`
- **Configurable** - Tune concurrency with `--concurrency N`

**Options:**
```bash
download-datasets               # Interactive with progress bars
download-datasets --yes         # Skip confirmation
download-datasets --concurrency 5  # 5 parallel downloads
download-datasets --verify      # Verify checksums
download-datasets --dry-run     # Preview without downloading
```

Downloads ~1.4 GB of event camera datasets from Google Drive.
```

**2. docs/setup.md:**

Add to workflow section:
```markdown
## Downloading Event Camera Datasets

Datasets are not included in the repository (large binary files).

Download using the automated tool:

```bash
nix develop
download-datasets
```

Features:
- Parallel downloads (3x faster than before)
- Progress tracking with visual bars
- Automatic resume on interruption
- Integrity verification

For CI/unattended:
```bash
download-datasets --yes --quiet
```
```

---

## 9. Testing Strategy

### 9.1 Unit Tests (Optional)

**Test manifest loading:**
```python
def test_load_manifest_valid():
    manifest = load_manifest(Path("tests/fixtures/valid.json"))
    assert "datasets" in manifest
    assert len(manifest["datasets"]) > 0

def test_load_manifest_invalid():
    with pytest.raises(SystemExit):
        load_manifest(Path("tests/fixtures/invalid.json"))
```

**Test file checking:**
```python
def test_should_download_missing_file():
    should_dl, reason = should_download_file({"path": "/nonexistent", "size": 100})
    assert should_dl == True
    assert "not present" in reason

def test_should_skip_existing_file(tmp_path):
    file = tmp_path / "test.dat"
    file.write_bytes(b"x" * 100)

    should_dl, reason = should_download_file({"path": str(file), "size": 100})
    assert should_dl == False
    assert "already present" in reason
```

### 9.2 Manual Testing Checklist

See `docs/plans/python-downloader-testing.md`:

- [ ] `--help` shows all options
- [ ] `--dry-run` previews downloads
- [ ] `--yes` skips confirmation
- [ ] `--concurrency 1` downloads sequentially
- [ ] `--concurrency 10` shows warning
- [ ] `--verify` verifies checksums (when present)
- [ ] Progress bars update correctly
- [ ] Resume works (interrupt and restart)
- [ ] Size verification catches mismatches
- [ ] Failed files continue with others
- [ ] Summary shows successes/failures/skips

---

## Success Criteria

### Must Have ✅
- Parallel HTTP streaming (aiohttp-based)
- Rich progress bars (per-chunk + per-file + overall)
- Smart resume (HTTP Range requests)
- Robust error handling (continue on failures)
- Manifest-driven (real IDs from generate_manifest.py)
- Isolated package (workspace/tools/downloader)
- Works with zero config (`download-datasets`)

### Should Have ✅
- Configurable concurrency with warnings
- Optional SHA256 verification
- Detailed summary report
- Inventory check
- Dry-run mode
- Quiet mode for CI
- Drive confirmation token handling

### Nice to Have
- Unit tests for core functions
- Progress state persistence (resume after restart)
- Bandwidth limiting (--max-speed flag)

---

## Open Questions

**None** - Design is complete and addresses all blockers.

---

## Next Steps

1. ✅ Manifest generator created (`scripts/generate_manifest.py`)
2. **Run manifest generator** - Get real file IDs
3. **Create workspace/tools/downloader** - Package structure
4. **Implement core components** - Following detailed implementation plan
5. **Test thoroughly** - Manual testing checklist
6. **Merge to main** - After all tests pass

---

## Appendices

### A. Requirements Traceability Matrix

| Feedback Requirement | Design Component | Implementation File |
|---------------------|------------------|---------------------|
| Parallelism | asyncio.Semaphore + aiohttp | download.py |
| Progress bars | rich.progress | progress.py |
| Resumable | HTTP Range requests | download.py |
| Drive resilience | Confirmation token handler | drive.py |
| Manifest-based | JSON loader + validator | manifest.py |
| SHA256 verification | hashlib streaming | verification.py |
| Concurrency config | CLI --concurrency flag | cli.py |

### B. Comparison to Original Design

| Aspect | Original (v1) | Revised (v2) |
|--------|--------------|--------------|
| HTTP Client | gdown (sync) | aiohttp (async) |
| Parallelism | asyncio.to_thread wrapper | Native async HTTP |
| Progress | Per-file only | Per-chunk + per-file |
| Resume | Size check, re-download | HTTP Range requests |
| Package | evio/scripts | workspace/tools/downloader |
| Dependencies | In evio | Isolated package |
| Manifest IDs | Placeholders | Real (from generator) |

---

**Ready for implementation plan.**
