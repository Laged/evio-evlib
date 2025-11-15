"""Google Drive confirmation token handling."""

import re
from typing import Optional, Tuple
from pathlib import Path

import aiohttp
import aiofiles


async def download_with_confirmation(
    session: aiohttp.ClientSession,
    file_id: str,
    path: Path,
    progress_callback=None
) -> Tuple[bool, str]:
    """
    Download file from Google Drive, handling confirmation tokens for large files.

    Google Drive files >100MB require:
    1. Confirmation token (extracted from HTML page)
    2. download_warning cookie (set by HTML response)

    Both must be present in the final download request.
    aiohttp.ClientSession automatically preserves cookies.

    Args:
        session: aiohttp ClientSession (maintains cookies)
        file_id: Google Drive file ID
        path: Output file path
        progress_callback: Optional callback(bytes_downloaded) for progress

    Returns:
        (success, error_message)
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Step 1: Try to get the file - check if we get HTML (confirmation needed)
    # Google Drive doesn't reliably indicate confirmation via HEAD, so we just try GET
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}: {resp.reason}"

            # Check if we got HTML (needs confirmation) or binary data (direct download)
            content_type = resp.headers.get('content-type', '')

            if 'text/html' not in content_type:
                # Got the file directly - download it
                async with aiofiles.open(path, 'wb') as f:
                    # Read response body
                    content = await resp.read()
                    await f.write(content)
                    if progress_callback:
                        progress_callback(len(content))
                return True, ""

            # Got HTML - check for quota exceeded error first
            html = await resp.text()

            # Check for quota exceeded (Google Drive rate limit)
            if 'Quota exceeded' in html or 'Too many users have viewed or downloaded' in html:
                return False, "QUOTA_EXCEEDED"

            # Extract confirmation value from hidden input
            # <input type="hidden" name="confirm" value="t">
            confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', html)
            if not confirm_match:
                return False, "Could not extract confirmation value from Drive page"
            confirm = confirm_match.group(1)

            # Extract UUID from hidden input (optional, but recommended)
            # <input type="hidden" name="uuid" value="...">
            uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html)
            uuid = uuid_match.group(1) if uuid_match else None

            # CRITICAL: Session now has cookies from this response
            # aiohttp.ClientSession automatically preserves cookies for the domain
    except Exception as e:
        return False, f"Failed to fetch initial response: {str(e)}"

    # Step 2: Download with confirmation parameters
    # Build URL for drive.usercontent.google.com (where the form action points)
    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm}"
    if uuid:
        download_url += f"&uuid={uuid}"

    try:
        async with session.get(download_url) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}: {resp.reason}"

            # Verify we got file, not HTML error page
            content_type = resp.headers.get('content-type', '')
            if 'text/html' in content_type:
                # Still got HTML - check for quota exceeded
                html = await resp.text()
                if 'Quota exceeded' in html or 'Too many users have viewed or downloaded' in html:
                    return False, "QUOTA_EXCEEDED"
                # Other HTML error
                snippet = html[:200]
                return False, f"Got HTML instead of file. Response: {snippet}..."

            # Stream download
            async with aiofiles.open(path, 'wb') as f:
                async for chunk in resp.content.iter_chunked(1 << 20):  # 1MB chunks
                    await f.write(chunk)
                    if progress_callback:
                        progress_callback(len(chunk))

        return True, ""
    except Exception as e:
        return False, f"Download with token failed: {str(e)}"
