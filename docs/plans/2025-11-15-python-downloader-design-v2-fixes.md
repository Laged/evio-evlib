# Python Downloader Design v2 - Critical Fixes

**Date:** 2025-11-15
**Status:** Fixes for 3 remaining blockers
**Applies to:** `2025-11-15-python-downloader-design-v2.md`

---

## Blocker 1: Workspace Membership Update Missing

**Location:** Lines 91-100 vs §8.1 (lines 700-705)

**Problem:** Design claims "No changes needed" for root pyproject.toml in §8.1, but earlier shows adding `workspace/tools/*`. Current main branch doesn't include tools directory, so `uv run --package downloader` will fail.

**Fix:**

### Section 8.1 - Replace lines 702-705

**OLD (WRONG):**
```nix
**Add workspace/tools to UV workspace (already in root pyproject.toml):**
```nix
# No changes needed - UV discovers workspace/tools/* automatically
```
```

**NEW (CORRECT):**
```markdown
**Update root pyproject.toml to include tools:**

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

**Then regenerate lockfile:**
```bash
uv lock
```

**Expected:** `uv.lock` updated with downloader package dependencies (aiohttp, rich, aiofiles).

**Verify:**
```bash
uv run --package downloader download-datasets --help
```

Expected: Shows help text (package discovered successfully).
```

---

## Blocker 2: Drive Confirmation Flow Doesn't Preserve Cookies

**Location:** Lines 308-367 (Drive confirmation handler)

**Problem:** Design fetches confirmation token but doesn't preserve `download_warning` cookie from HTML response. Google Drive requires this cookie for the actual download, otherwise you get HTML warning page instead of file bytes.

**Fix:**

### Section 4.2 - Replace Drive Handler Implementation

**Add cookie preservation to download flow:**

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
    1. Confirmation token (from HTML page)
    2. download_warning cookie (set by that page)

    Both must be present in the final download request.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Step 1: Check if confirmation needed
    async with session.head(url, allow_redirects=True) as resp:
        # If redirect to confirm page, we need token
        needs_confirm = "confirm=" in str(resp.url) or \
                       "download_warning" in resp.headers.get("content-disposition", "")

    if not needs_confirm:
        # Small file, direct download
        return await download_direct(session, url, path, progress_callback)

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
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0

        async with aiofiles.open(path, 'wb') as f:
            async for chunk in resp.content.iter_chunked(1 << 20):  # 1MB chunks
                await f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(len(chunk))

    return True, ""
```

**Key Changes:**
1. **Session preserves cookies automatically** - aiohttp.ClientSession maintains cookie jar per domain
2. **Fetch HTML page** - Sets download_warning cookie in session
3. **Extract token** - Parse from HTML
4. **Download with both** - Token in URL, cookie in session headers (automatic)
5. **Verify content-type** - Detect if we still got HTML (failure case)

**Why this works:**
- aiohttp.ClientSession uses aiohttp.CookieJar automatically
- Cookies set by Drive HTML page persist for domain
- Subsequent requests to drive.google.com include cookie
- No manual cookie handling needed (session does it)

---

## Blocker 3: Resume Logic Lacks Fallback When Range Fails

**Location:** Lines 436-464 (Resume logic), Lines 473-475 (Limitations note)

**Problem:** Design notes "Range support is unreliable" but returns False immediately on Range failure, leaving partial file in place. User stuck in failure loop.

**Fix:**

### Section 4.3 - Replace Resume Logic Implementation

**Add fallback to full download when Range fails:**

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

**Key Changes:**
1. **Try Range first** - If partial file exists
2. **Detect Range failure** - Check for 206 vs 200 vs other
3. **Fallback on 200** - Delete partial, accept full file from current response
4. **Fallback on other errors** - Delete partial, retry full download
5. **Clean partial on exception** - Prevent stuck state
6. **Progress updates** - Show resume/restart status

**Flow:**
```
Partial file exists (500 MB of 1 GB)
  ↓
Try Range request with bytes=500000000-
  ↓
├─ 206 Partial Content → Resume from 500 MB ✓
├─ 200 OK → Delete partial, download full file from this response ✓
└─ 416/Other → Delete partial, retry full download ✓
```

**Why this works:**
- Never leaves user stuck with partial file
- Auto-recovers from Range failures
- Utilizes Range when possible (bandwidth savings)
- Falls back gracefully when not supported

---

## Summary of Fixes

### Blocker 1: Workspace Membership
**Was:** "No changes needed"
**Now:** Explicit `pyproject.toml` edit + `uv lock` step

### Blocker 2: Drive Cookies
**Was:** Only token handling
**Now:** Session cookie jar preserves `download_warning` automatically

### Blocker 3: Range Fallback
**Was:** Fail immediately on Range error
**Now:** Delete partial + retry full download (3 fallback paths)

---

## Integration Instructions

These fixes should be merged into `2025-11-15-python-downloader-design-v2.md`:

1. **Replace Section 8.1** (lines 700-715) with Blocker 1 fix
2. **Replace Section 4.2** (Drive handler) with Blocker 2 fix
3. **Replace Section 4.3** (Resume logic) with Blocker 3 fix

After fixes, all three blockers will be resolved and the design will be implementable.

---

**Status:** Fixes ready for review and merge into design v2.
