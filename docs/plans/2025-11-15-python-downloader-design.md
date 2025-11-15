# Python Dataset Downloader - Design Document

**Date:** 2025-11-15
**Status:** Approved Design
**Context:** Replace shell-based gdown script with Python downloader for parallel downloads, progress tracking, and better UX

---

## Executive Summary

Replace the current sequential shell-based dataset downloader with a Python CLI that provides:
- **Parallel downloads** (3 concurrent by default, configurable)
- **Rich progress bars** (per-file + overall progress)
- **Smart resume** (skip already-downloaded files based on size)
- **Robust error handling** (continue on individual failures, detailed reporting)
- **Manifest-based** (centralized metadata in JSON)

**Key improvement:** ~3x faster downloads through parallelism, better UX with progress visibility.

---

## Design Principles

1. **Safe Defaults**: Works without configuration (`download-datasets` just works)
2. **Parallel by Default**: 3 concurrent downloads (safe for Drive rate limits)
3. **Fail Gracefully**: Individual file failures don't abort entire download
4. **Verify Integrity**: Size check always, SHA256 optional
5. **Clear Feedback**: Rich progress bars show exactly what's happening

---

## Architecture

### Component Structure

**Location:** `evio/scripts/download_datasets.py`
**Invocation:** `uv run --package evio python evio/scripts/download_datasets.py`
**Alias:** `download-datasets` (defined in flake.nix)

### Key Components

1. **Manifest Parser** - Reads `docs/data/datasets.json` for file metadata
2. **Download Manager** - Orchestrates parallel downloads with asyncio + semaphore
3. **File Downloader** - Wraps gdown in async tasks, handles Drive tokens
4. **Progress Tracker** - Uses `rich.progress` for visual feedback
5. **Verification Engine** - Size check (always), SHA256 (optional)

### Data Flow

```
Manifest → Parse metadata → Check existing files (size) →
Queue missing/incomplete files → Download in parallel (3 at a time) →
Show progress → Verify sizes → Optional checksum → Report summary
```

### Dependencies

- **gdown** (>=5.0.0) - Google Drive download with token handling
- **rich** (>=13.0.0) - Progress bars and terminal formatting
- **Standard library:** asyncio, hashlib, pathlib, argparse, json

---

## Manifest Structure

**File:** `docs/data/datasets.json`

```json
{
  "version": "1.0",
  "datasets": [
    {
      "id": "1abc123...",
      "name": "fan_const_rpm.dat",
      "path": "evio/data/fan/fan_const_rpm.dat",
      "size": 212336640,
      "sha256": "abc123def456..."
    },
    {
      "id": "1def456...",
      "name": "fan_const_rpm.raw",
      "path": "evio/data/fan/fan_const_rpm.raw",
      "size": 124567890,
      "sha256": "789ghi012..."
    }
  ]
}
```

**Fields:**
- `version`: Manifest format version (future-proofing)
- `datasets`: Array of file metadata objects
  - `id`: Google Drive file ID
  - `name`: Display name for progress/logging
  - `path`: Target path relative to repo root
  - `size`: Expected file size in bytes (for verification)
  - `sha256`: Optional checksum for `--verify` mode

---

## CLI Interface

### Usage Examples

```bash
# Basic usage (safe defaults: confirm prompt, 3 concurrent, no checksum)
download-datasets

# Quiet mode (skip confirmation, for CI/scripts)
download-datasets --yes

# Custom concurrency (warn if >5 to avoid rate limits)
download-datasets --concurrency 5

# Verify checksums after download
download-datasets --verify

# Custom manifest location (for testing/alternative datasets)
download-datasets --manifest path/to/custom.json

# Dry run (show what would be downloaded without downloading)
download-datasets --dry-run
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--yes` / `-y` | False | Skip confirmation prompt |
| `--concurrency N` | 3 | Number of parallel downloads (warn if >5) |
| `--verify` | False | Verify SHA256 checksums after download |
| `--manifest PATH` | `docs/data/datasets.json` | Path to manifest file |
| `--dry-run` | False | Show what would be downloaded, don't download |
| `--quiet` / `-q` | False | Minimal output (no progress bars) |

### Exit Codes

- **0**: All files downloaded/verified successfully
- **1**: Some files failed (partial success, see summary)
- **2**: Complete failure (manifest error, no files downloaded)

---

## Download Logic

### Download Flow

```python
async def download_file(file_info, progress, semaphore):
    async with semaphore:  # Limit to N concurrent downloads
        # 1. Check if file exists with correct size
        if file_exists_and_size_matches(file_info):
            progress.update(task, description=f"[skip] {file_info['name']}")
            return SUCCESS

        # 2. Ensure target directory exists
        Path(file_info['path']).parent.mkdir(parents=True, exist_ok=True)

        # 3. Download using gdown (handles Drive confirmation tokens)
        try:
            await asyncio.to_thread(
                gdown.download,
                id=file_info['id'],
                output=file_info['path'],
                quiet=False
            )
        except Exception as e:
            log_error(file_info['name'], str(e))
            return FAILURE

        # 4. Verify file size
        if not verify_size(file_info):
            return FAILURE

        return SUCCESS

async def main(manifest, concurrency=3):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [download_file(f, progress, semaphore) for f in manifest['datasets']]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return summarize_results(results)
```

### Concurrency Strategy

**Default:** 3 simultaneous downloads
**Rationale:** Safe for Google Drive rate limits, good performance balance
**Warning threshold:** >5 concurrent (risk of throttling)

```python
if args.concurrency > 5:
    print("⚠️  High concurrency (>5) may trigger rate limits")
    print("    Consider using default (3) for reliable downloads")
```

### Resume Strategy (Option D: Hybrid)

```python
def should_skip_file(file_info):
    path = Path(file_info['path'])

    # File doesn't exist - download it
    if not path.exists():
        return False

    # File exists - check size (fast)
    actual_size = path.stat().st_size
    expected_size = file_info['size']

    if actual_size == expected_size:
        return True  # Assume correct (skip)
    else:
        # Size mismatch - remove and redownload
        path.unlink()
        return False
```

**Checksum verification (optional with `--verify`):**
```python
def verify_checksum(file_info):
    if 'sha256' not in file_info:
        return True  # No checksum in manifest, assume OK

    actual = compute_sha256(file_info['path'])
    expected = file_info['sha256']

    if actual != expected:
        print(f"❌ Checksum mismatch: {file_info['name']}")
        return False

    return True
```

---

## Error Handling

### Error Categories

| Error Type | Handling Strategy |
|------------|-------------------|
| Individual file failure | Log error, continue with other files, report in summary |
| Network timeout | Retry once with exponential backoff, then mark as failed |
| Permission error (Drive) | Skip file, warn in summary with manual download URL |
| Disk space error | Abort immediately (can't continue safely) |
| Manifest parse error | Abort immediately (can't proceed without metadata) |

### Error Reporting

**Per-file errors:**
```python
try:
    await download_file(file_info)
except PermissionError as e:
    errors.append({
        'file': file_info['name'],
        'error': 'Permission denied',
        'url': f"https://drive.google.com/uc?id={file_info['id']}"
    })
```

**Summary errors:**
```
❌ Failed: 1 file

Failed downloads:
  - hong_kong_city.glb (permission denied)
    Manual download: https://drive.google.com/uc?id=18gpWinw...
```

---

## Progress Display

### Progress Bar Layout (using `rich`)

```
fan_const_rpm.dat      [████████░░] 45% (96 MB/212 MB) 4.2 MB/s
fan_varying_rpm.dat    [██░░░░░░░░] 15% (73 MB/489 MB) 3.8 MB/s
drone_idle.dat         [waiting...]

Overall: [████░░░░░░] 35% (3/12 files completed) - ETA: 3m 42s
```

### Implementation

```python
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn

with Progress(
    *Progress.get_default_columns(),
    DownloadColumn(),
    TransferSpeedColumn(),
) as progress:
    overall_task = progress.add_task("[cyan]Overall", total=total_files)

    for file_info in manifest['datasets']:
        task = progress.add_task(file_info['name'], total=file_info['size'])
        # Download and update progress.update(task, advance=chunk_size)
```

### Quiet Mode (`--quiet`)

Simple text output:
```
[1/12] Downloading fan_const_rpm.dat... done (212 MB)
[2/12] Downloading fan_varying_rpm.dat... done (489 MB)
[3/12] Skipping drone_idle.dat (already exists)
```

---

## Verification & Reporting

### Verification Strategy

```python
def verify_file(file_info, verify_checksum=False):
    path = Path(file_info['path'])

    # Always: Size check (fast, catches most issues)
    actual_size = path.stat().st_size
    expected_size = file_info['size']
    if actual_size != expected_size:
        return FAILED, f"Size mismatch: {actual_size} != {expected_size}"

    # Optional: Checksum verification (--verify flag)
    if verify_checksum and 'sha256' in file_info:
        actual_hash = compute_sha256(path)
        expected_hash = file_info['sha256']
        if actual_hash != expected_hash:
            return FAILED, f"Checksum mismatch"

    return SUCCESS, None

def compute_sha256(path, chunk_size=8192):
    """Stream-based hash to handle large files efficiently"""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()
```

### Summary Report

```
==========================================
  Download Summary
==========================================

✅ Successfully downloaded: 10 files (1.2 GB)
⏭️  Skipped (already present): 2 files (200 MB)
❌ Failed: 1 file

Failed downloads:
  - hong_kong_city.glb (permission denied)
    Manual download: https://drive.google.com/uc?id=18gpWinw...

Current dataset inventory:
  Fan datasets:   3 .dat files (3 .raw files)
  Drone datasets: 2 .dat files (2 .raw files)
  Fred-0 events:  1 .dat file (1 .raw file)

Total: 12 files, 1.4 GB

Ready to run:
  run-demo-fan
  run-mvp-1
  run-mvp-2
```

### Inventory Check

After download, scan actual directories:
```python
def check_inventory():
    fan_count = len(list(Path('evio/data/fan').glob('*.dat')))
    drone_count = len(list(Path('evio/data/drone').glob('*.dat')))
    fred_count = len(list(Path('evio/data/fred-0/events').glob('*.dat')))

    return {
        'fan': fan_count,
        'drone': drone_count,
        'fred-0': fred_count
    }
```

---

## Integration

### flake.nix Changes

**Add Python dependencies:**
```nix
devShells.default = pkgs.mkShell {
  buildInputs = [
    # ... existing dependencies ...
    python
    pkgs.uv
    pkgs.gdown  # Keep CLI tool available
  ];

  # ... existing shellHook ...

  # Update alias to use Python script
  alias download-datasets='uv run --package evio python evio/scripts/download_datasets.py'
};
```

**Remove:** Old shell-based download script (lines 15-104 in current flake.nix)

### evio/pyproject.toml Updates

Add dependencies:
```toml
[project]
dependencies = [
  # ... existing ...
  "gdown>=5.0.0",
  "rich>=13.0.0",
]
```

### Documentation Updates

**1. evio/data/README.md:**
- Update "Automated Download" section to mention new features:
  - Parallel downloads (3x faster)
  - Progress bars showing per-file status
  - Automatic resume (skip already-downloaded files)
  - Optional checksum verification

**2. docs/setup.md:**
- Update workflow to show `download-datasets` command
- Add note about `--verify` flag for paranoid users
- Mention `--concurrency` for tuning

**3. Create docs/data/datasets.json:**
- Initial manifest with all known datasets
- Include file IDs, sizes, paths
- Add SHA256 checksums (compute from current files)

### Backward Compatibility

- **Breaking:** None - same command name `download-datasets`
- **Behavior change:** Parallel downloads (faster but different output)
- **Migration:** Automatic - users just run `download-datasets` as before

---

## Testing Strategy

### Manual Testing Checklist

- [ ] Fresh download (no existing files) - all files download
- [ ] Resume scenario (some files exist with correct size) - skipped properly
- [ ] Resume scenario (partial file exists) - removed and re-downloaded
- [ ] Failed file handling (simulate permission error) - continues with others
- [ ] `--verify` flag - checksums verified
- [ ] `--concurrency 5` - 5 files download simultaneously
- [ ] `--concurrency 10` - warning displayed
- [ ] `--dry-run` - shows what would download, doesn't download
- [ ] `--quiet` - minimal output, no progress bars
- [ ] Large file (>100 MB) - gdown handles confirmation token
- [ ] Network interruption - graceful failure, clear error message
- [ ] Disk full - aborts with clear error
- [ ] Invalid manifest JSON - aborts with parse error

### Edge Cases

- Empty manifest (no datasets)
- Manifest with missing required fields
- File ID doesn't exist on Drive
- File size mismatch after download
- Checksum mismatch (corrupted download)
- All files already present (skip all)

---

## Implementation Plan

### Phase 1: Core Downloader
1. Create `evio/scripts/download_datasets.py`
2. Implement manifest parser
3. Implement basic download logic with gdown
4. Add asyncio parallelism with semaphore
5. Add size-based resume logic

### Phase 2: Progress & UX
6. Integrate `rich` progress bars
7. Add CLI argument parsing
8. Add confirmation prompt
9. Add dry-run mode
10. Add quiet mode

### Phase 3: Verification & Reporting
11. Add size verification
12. Add SHA256 verification (optional)
13. Implement summary report
14. Add inventory check

### Phase 4: Integration
15. Create `docs/data/datasets.json` manifest
16. Update `evio/pyproject.toml` dependencies
17. Update `flake.nix` (remove old script, add alias)
18. Update `evio/data/README.md`
19. Update `docs/setup.md`

### Phase 5: Testing & Polish
20. Manual testing (all checklist items)
21. Error message polish
22. Documentation review
23. Final verification

---

## Success Criteria

### Must Have
- ✅ Parallel downloads (default 3 concurrent)
- ✅ Rich progress bars (per-file + overall)
- ✅ Resume capability (skip existing files)
- ✅ Robust error handling (continue on individual failures)
- ✅ Manifest-based metadata
- ✅ Works with `download-datasets` command (no config needed)

### Should Have
- ✅ Configurable concurrency with warnings
- ✅ Optional checksum verification
- ✅ Detailed summary report
- ✅ Inventory check after download
- ✅ Dry-run mode
- ✅ Quiet mode for CI

### Nice to Have
- Custom manifest path support
- Per-file download retry logic
- ETA calculation for overall progress

---

## Open Questions

**None** - Design is complete and ready for implementation.

---

## Next Steps

1. Use `superpowers:writing-plans` to create detailed implementation plan
2. Execute implementation in `python-downloader` worktree
3. Test thoroughly with manual checklist
4. Merge to main when complete

---

**Ready for implementation plan?**
