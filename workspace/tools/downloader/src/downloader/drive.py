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

            # Got HTML - need to extract confirmation token
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
    except Exception as e:
        return False, f"Failed to fetch initial response: {str(e)}"

    # Step 2: Download with token (cookie automatically included by session)
    download_url = f"{url}&confirm={token}"

    try:
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
    except Exception as e:
        return False, f"Download with token failed: {str(e)}"
