"""Async download manager with HTTP Range resume support."""

import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import aiohttp
import aiofiles
from rich.progress import Progress

from .drive import download_with_confirmation


async def download_file(
    session: aiohttp.ClientSession,
    dataset: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Download single file with smart resume and fallback.

    Strategy:
    1. If file complete → skip
    2. If partial file exists → try Range request
    3. If Range fails (206 not supported) → delete partial, full download
    4. If full download fails → error

    Args:
        session: aiohttp ClientSession
        dataset: Dataset metadata dict (id, name, path, size)
        semaphore: Concurrency limiter
        progress: Optional Rich progress tracker
        task_id: Optional progress task ID

    Returns:
        (success, error_message)
    """
    async with semaphore:
        path = Path(dataset['path'])
        file_id = dataset['id']
        expected_size = dataset['size']

        # Update progress
        if progress and task_id is not None:
            progress.update(task_id, description=f"[yellow]⏳ {dataset['name']}")

        # Check existing file
        resume_from = 0
        if path.exists():
            actual_size = path.stat().st_size

            if actual_size == expected_size:
                # Complete - skip
                if progress and task_id is not None:
                    progress.update(
                        task_id,
                        description=f"[green]✓ {dataset['name']} (already present)",
                        completed=expected_size
                    )
                return True, ""

            elif actual_size < expected_size:
                # Partial - try resume
                resume_from = actual_size
                if progress and task_id is not None:
                    progress.update(task_id, completed=actual_size)
                    progress.update(
                        task_id,
                        description=f"[yellow]↻ {dataset['name']} (resuming from {actual_size / 1024 / 1024:.1f} MB)"
                    )
            else:
                # Larger than expected - corrupt
                path.unlink()
                resume_from = 0

        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build Drive URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"

        try:
            # Attempt 1: Resume with Range header (if partial file)
            if resume_from > 0:
                headers = {'Range': f'bytes={resume_from}-'}

                async with session.get(url, headers=headers) as resp:
                    if resp.status == 206:
                        # Server supports Range - resume from offset
                        if progress and task_id is not None:
                            progress.update(task_id, description=f"[yellow]↓ {dataset['name']} (resuming)")

                        mode = 'ab'  # Append mode
                        async with aiofiles.open(path, mode) as f:
                            async for chunk in resp.content.iter_chunked(1 << 20):
                                await f.write(chunk)
                                if progress and task_id is not None:
                                    progress.update(task_id, advance=len(chunk))

                        # Verify size
                        actual_size = path.stat().st_size
                        if actual_size != expected_size:
                            return False, f"Size mismatch after resume: {actual_size} != {expected_size}"

                        if progress and task_id is not None:
                            progress.update(task_id, description=f"[green]✓ {dataset['name']}")
                        return True, ""

                    elif resp.status == 200:
                        # Server doesn't support Range, sent full file
                        # FALLBACK: Delete partial, accept full download
                        if progress and task_id is not None:
                            progress.update(
                                task_id,
                                description=f"[yellow]↓ {dataset['name']} (Range not supported, restarting)"
                            )
                        path.unlink()  # Delete partial
                        resume_from = 0
                        if progress and task_id is not None:
                            progress.update(task_id, completed=0)  # Reset progress

                        # Download full file from this response
                        async with aiofiles.open(path, 'wb') as f:
                            async for chunk in resp.content.iter_chunked(1 << 20):
                                await f.write(chunk)
                                if progress and task_id is not None:
                                    progress.update(task_id, advance=len(chunk))

                        # Verify size
                        actual_size = path.stat().st_size
                        if actual_size != expected_size:
                            return False, f"Size mismatch: {actual_size} != {expected_size}"

                        if progress and task_id is not None:
                            progress.update(task_id, description=f"[green]✓ {dataset['name']}")
                        return True, ""

                    else:
                        # Unexpected status - fall through to full download
                        if progress and task_id is not None:
                            progress.update(
                                task_id,
                                description=f"[yellow]↓ {dataset['name']} (Range failed, restarting)"
                            )
                        path.unlink()
                        resume_from = 0
                        if progress and task_id is not None:
                            progress.update(task_id, completed=0)

            # Attempt 2: Full download with Drive confirmation handling
            if resume_from == 0:
                if progress and task_id is not None:
                    progress.update(task_id, description=f"[yellow]↓ {dataset['name']}")

                # Use Drive confirmation handler
                def progress_callback(bytes_downloaded):
                    if progress and task_id is not None:
                        progress.update(task_id, advance=bytes_downloaded)

                success, error = await download_with_confirmation(
                    session, file_id, path, progress_callback
                )

                if not success:
                    if progress and task_id is not None:
                        progress.update(task_id, description=f"[red]✗ {dataset['name']}")
                    return False, error

                # Verify size
                actual_size = path.stat().st_size
                if actual_size != expected_size:
                    if progress and task_id is not None:
                        progress.update(task_id, description=f"[red]✗ {dataset['name']}")
                    return False, f"Size mismatch: {actual_size} != {expected_size}"

                if progress and task_id is not None:
                    size_mb = actual_size / 1024 / 1024
                    progress.update(
                        task_id,
                        description=f"[green]✓ {dataset['name']} ({size_mb:.1f} MB)",
                        completed=expected_size
                    )
                return True, ""

        except Exception as e:
            # Clean up partial file on exception
            if path.exists():
                path.unlink()
            if progress and task_id is not None:
                progress.update(task_id, description=f"[red]✗ {dataset['name']}")
            return False, f"Download failed: {str(e)}"

    return False, "Unexpected code path"


async def download_all(
    datasets: List[Dict[str, Any]],
    concurrency: int = 3,
    progress: Optional[Progress] = None,
    tasks_map: Optional[Dict[str, int]] = None
) -> Tuple[List[Dict], List[Tuple[Dict, str]]]:
    """
    Download all datasets in parallel with semaphore-based concurrency control.

    Args:
        datasets: List of dataset metadata dicts
        concurrency: Max concurrent downloads
        progress: Optional Rich progress tracker
        tasks_map: Optional map of dataset names to progress task IDs

    Returns:
        (successes, failures) tuple
    """
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        download_tasks = []

        for dataset in datasets:
            task_id = None
            if tasks_map and dataset['name'] in tasks_map:
                task_id = tasks_map[dataset['name']]

            task = download_file(
                session,
                dataset,
                semaphore,
                progress,
                task_id
            )
            download_tasks.append((dataset, task))

        # Execute downloads in parallel
        results = await asyncio.gather(
            *[t for _, t in download_tasks],
            return_exceptions=True
        )

        # Separate successes and failures
        successes = []
        failures = []

        for (dataset, _), result in zip(download_tasks, results):
            if isinstance(result, Exception):
                failures.append((dataset, str(result)))
            else:
                success, error = result
                if success:
                    successes.append(dataset)
                else:
                    failures.append((dataset, error))

    return successes, failures
