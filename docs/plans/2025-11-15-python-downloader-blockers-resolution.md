# Python Downloader - Blocker Resolution

**Date:** 2025-11-15
**Reviewer:** Codex
**Status:** Blockers Addressed, Ready for Revised Implementation

---

## Blockers Identified

### Blocker 1: Architecture Conflict ✅ RESOLVED

**Problem:** Original plan used gdown wrapped in asyncio, which is still synchronous HTTP per-file. Doesn't provide:
- True parallel HTTP streaming
- Per-chunk progress updates
- HTTP Range request support for resume

**Resolution:** Switch to **aiohttp**-based architecture per feedback document recommendations.

**New Architecture:**
```python
# aiohttp for parallel HTTP streaming
async with aiohttp.ClientSession() as session:
    async with session.get(url) as resp:
        async for chunk in resp.content.iter_chunked(1 << 20):
            # Stream chunk-by-chunk with progress updates
```

**Benefits:**
- True parallel downloads (concurrent HTTP streams)
- Per-chunk progress tracking (accurate ETAs)
- HTTP Range support (resume from byte offset)
- Handles Drive confirmation tokens manually (more control)

---

### Blocker 2: Placeholder Manifest ✅ RESOLVED

**Problem:** Manifest had PLACEHOLDER_* IDs with fake sizes - couldn't test or deploy.

**Resolution:** Created `scripts/generate_manifest.py` using gdown's internal metadata helpers.

**How It Works:**
```python
from gdown import google_drive

folder_id = google_drive.parse_url(FOLDER_URL)["id"]
files = google_drive._get_folder_contents(folder_id)  # Metadata only

manifest = [{
    "id": f["id"],
    "name": f["title"],
    "size": int(f["fileSize"]),
    "path": PATH_MAPPING[f["title"]]
} for f in files]
```

**Benefits:**
- Real Drive file IDs from actual folder
- Actual file sizes (not estimates)
- No full Drive API / OAuth needed
- Can regenerate manifest when folder changes

---

### Blocker 3: Dependency Scope ✅ RESOLVED

**Problem:** Adding downloader deps (aiohttp, rich) to evio/pyproject.toml pollutes the published library.

**Resolution:** Create new workspace member: `workspace/tools/downloader`

**Structure:**
```
workspace/tools/downloader/
├── pyproject.toml       # Own deps (aiohttp, rich)
├── src/
│   └── downloader/
│       ├── __init__.py
│       ├── cli.py       # Main CLI
│       ├── download.py  # aiohttp download logic
│       ├── manifest.py  # Manifest loading
│       └── progress.py  # Rich progress bars
└── README.md
```

**Benefits:**
- Clean separation (tool deps don't pollute evio)
- Proper workspace member (can be versioned independently)
- UV handles dependencies correctly
- flake.nix alias: `uv run --package downloader python -m downloader.cli`

---

### Blocker 4: Documentation Traceability ✅ RESOLVED

**Problem:** Plan didn't reference requirements source (docs/data/download-feedback.md).

**Resolution:** Add explicit requirement traceability in all documents.

**Requirements Source:** `docs/data/download-feedback.md`

**Key Requirements:**
1. **Parallelism** - 3 concurrent downloads (configurable)
2. **Progress Visualization** - rich progress bars (per-file + overall)
3. **Resumable Downloads** - HTTP Range requests for partial resume
4. **Drive API Resilience** - Handle >100MB files with confirmation tokens
5. **Manifest-Based** - JSON metadata (IDs, paths, SHA256)

**Traceability:**
- Design doc references feedback doc in "Context" section
- Implementation plan links to requirements
- Each feature traces back to specific requirement

---

## Revised Architecture

### Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| HTTP Client | **aiohttp** | Async parallel streaming, Range requests |
| Progress UI | **rich** | Beautiful progress bars, terminal formatting |
| Concurrency | **asyncio.Semaphore** | Limit concurrent connections (avoid rate limits) |
| Manifest | **JSON** | Simple, version-controllable metadata |
| Drive Tokens | **Manual handling** | Parse confirmation page, retry with token |

### Package Structure

**New Workspace Member:** `workspace/tools/downloader`

```toml
# workspace/tools/downloader/pyproject.toml
[project]
name = "downloader"
version = "0.1.0"
description = "Event camera dataset downloader"
requires-python = ">=3.11"
dependencies = [
    "aiohttp>=3.9.0",
    "rich>=13.0.0",
    "aiofiles>=23.0.0",  # Async file I/O
]

[project.scripts]
download-datasets = "downloader.cli:main"
```

**UV Root Workspace:** Add to root pyproject.toml:
```toml
[tool.uv.workspace]
members = [
    "evio",
    "workspace/libs/*",
    "workspace/plugins/*",
    "workspace/apps/*",
    "workspace/tools/*",  # Add tools
]
```

**flake.nix Alias:**
```nix
alias download-datasets='uv run --package downloader download-datasets'
```

### Download Flow

```
1. Load manifest (docs/data/datasets.json)
2. Check existing files (size-based skip)
3. Confirm with user (--yes to skip)
4. Create aiohttp session with semaphore (concurrency=3)
5. For each file:
   a. Check if Drive confirmation needed (file >100MB)
   b. If yes: Fetch confirmation token, retry with token
   c. Stream download with chunked progress updates
   d. Verify size on completion
6. Optional: Verify SHA256 checksums (--verify)
7. Report summary (successes/failures/skips)
8. Show inventory of actual files
```

### Drive Confirmation Token Handling

```python
async def download_with_confirmation(session, file_id, path, progress):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    async with session.get(url, allow_redirects=False) as resp:
        # Check if confirmation page
        if "accounts.google.com/ServiceLogin" in str(resp.url):
            # File requires login (permission denied)
            return False, "Permission denied"

        if "confirm=" in str(resp.url) or "download_warning" in await resp.text():
            # Extract confirmation token
            text = await resp.text()
            token = extract_confirm_token(text)
            url = f"{url}&confirm={token}"

    # Download with token
    async with session.get(url) as resp:
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0

        async with aiofiles.open(path, 'wb') as f:
            async for chunk in resp.content.iter_chunked(1 << 20):  # 1MB chunks
                await f.write(chunk)
                downloaded += len(chunk)
                progress.update(task, advance=len(chunk))

    return True, ""
```

---

## Updated Implementation Plan Summary

**Phase 1: Workspace Setup**
1. Create workspace/tools/downloader package
2. Add dependencies (aiohttp, rich, aiofiles)
3. Generate real manifest with scripts/generate_manifest.py

**Phase 2: Core Downloader**
4. Manifest loader with validation
5. File existence check (skip logic)
6. aiohttp download with Drive token handling
7. Async parallel execution with semaphore

**Phase 3: Progress & UX**
8. Rich progress bars (per-file + overall)
9. CLI argument parsing (--yes, --concurrency, --verify)
10. Confirmation prompt
11. Summary reporting

**Phase 4: Verification & Docs**
12. SHA256 checksum verification (optional)
13. Inventory check (scan actual files)
14. Update documentation (evio/data/README.md, docs/setup.md)
15. Update flake.nix alias

**Total Tasks:** ~15 tasks (revised from original plan)

---

## Requirements Traceability

| Requirement (from feedback doc) | Implementation |
|--------------------------------|----------------|
| **Parallelism** (3 concurrent) | asyncio.Semaphore(3) limiting aiohttp sessions |
| **Progress Bars** (per-file + overall) | rich.progress with DownloadColumn, TransferSpeedColumn |
| **Resumable** (HTTP Range) | aiohttp session.get with Range header for partial files |
| **Drive Resilience** (>100MB tokens) | Parse confirmation page, extract token, retry with &confirm= |
| **Manifest-Based** (JSON metadata) | docs/data/datasets.json with real IDs from generate_manifest.py |
| **SHA256 Verification** | hashlib.sha256() streaming with --verify flag |
| **Configurable Concurrency** | --concurrency N flag (default 3, warn >5) |
| **Dry Run** | --dry-run shows what would download |
| **Quiet Mode** | --quiet for CI (minimal output) |

---

## Next Steps

1. ✅ **Manifest Generator** - Created scripts/generate_manifest.py
2. **Run Generator** - Execute to get real file IDs (needs uv sync first)
3. **Create Workspace Member** - Set up workspace/tools/downloader
4. **Update Design Doc** - Reflect aiohttp architecture
5. **Update Implementation Plan** - Detailed tasks with aiohttp approach
6. **Execute Implementation** - Use superpowers:subagent-driven-development

---

## Migration from Original Plan

**What Changes:**
- ❌ Remove gdown from download logic (keep for manifest generation only)
- ✅ Add aiohttp for HTTP streaming
- ✅ Add aiofiles for async file I/O
- ✅ Create workspace/tools/downloader (not evio/scripts)
- ✅ Manual Drive token handling (not gdown's automatic handling)

**What Stays:**
- ✅ Rich progress bars
- ✅ Manifest-based metadata
- ✅ CLI flags (--yes, --concurrency, --verify, --dry-run)
- ✅ Smart resume (size-based skip)
- ✅ Summary reporting and inventory check

---

**Status:** All blockers resolved. Ready for revised implementation.

**See:** Revised design and implementation plan (to be created next)
