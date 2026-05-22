"""
File watcher for real-time dashboard updates — Enhancement 3.

Monitors the master_dataset.csv for file-system changes and broadcasts
a WebSocket update notification to all connected frontend clients.
Run this as a background process alongside the FastAPI server.
"""

import asyncio
import logging
import time
from pathlib import Path
from threading import Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WATCH_PATH = Path("data/processed/master_dataset.csv")
POLL_INTERVAL_SECONDS = 30


def watch_and_notify(
    ws_clients: list,
    loop: asyncio.AbstractEventLoop,
    watch_path: Path | None = None,
    stop_event: Event | None = None,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> None:
    """Poll the master dataset file and notify WebSocket clients on change.

    Compares the file's last-modified timestamp on each poll interval.
    When a change is detected, broadcasts a JSON update message to all
    currently connected WebSocket clients.

    Args:
        ws_clients: List of active FastAPI WebSocket connection objects.
        loop: The asyncio event loop running the FastAPI application.
        watch_path: Dataset path to monitor. Defaults to WATCH_PATH.
        stop_event: Optional event used to stop the loop during shutdown.
        poll_interval: Seconds between file checks.
    """
    target_path = watch_path or WATCH_PATH
    last_mtime = target_path.stat().st_mtime if target_path.exists() else 0.0

    while not (stop_event and stop_event.is_set()):
        time.sleep(poll_interval)
        try:
            if not target_path.exists():
                continue
            current_mtime = target_path.stat().st_mtime
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                logger.info("Dataset updated — broadcasting to %d clients.", len(ws_clients))
                for ws in list(ws_clients):
                    asyncio.run_coroutine_threadsafe(
                        ws.send_json({"type": "update", "timestamp": current_mtime}),
                        loop,
                    )
        except Exception as exc:
            logger.warning("Watcher error: %s", exc)
