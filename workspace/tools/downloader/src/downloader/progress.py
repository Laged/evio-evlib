"""Rich progress bar management."""

from typing import Dict, List, Tuple, Any

from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .download import download_all


def create_progress() -> Progress:
    """Create rich progress tracker with download-optimized columns."""
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
    datasets: List[Dict[str, Any]],
    concurrency: int = 3
) -> Tuple[List[Dict], List[Tuple]]:
    """
    Download datasets with rich progress display.

    Args:
        datasets: List of dataset metadata dicts
        concurrency: Max concurrent downloads

    Returns:
        (successes, failures) tuple
    """
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

        # Download all with progress tracking
        successes, failures = await download_all(
            datasets,
            concurrency,
            progress=progress,
            tasks_map=tasks_map
        )

        # Update overall progress
        progress.update(overall_task, completed=len(datasets))

    return successes, failures
