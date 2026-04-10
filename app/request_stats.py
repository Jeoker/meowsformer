"""In-process HTTP/WebSocket request counters persisted to stats.json."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger

STATS_FILE = Path("stats.json")
_start_time = time.time()

TRANSLATE_PATHS = frozenset(
    {"/api/translate", "/api/v1/translate", "/ws/translate"}
)


def _load_counts() -> defaultdict[str, int]:
    if STATS_FILE.exists():
        try:
            return defaultdict(int, json.loads(STATS_FILE.read_text()))
        except Exception:
            pass
    return defaultdict(int)


def _save_counts() -> None:
    try:
        STATS_FILE.write_text(json.dumps(dict(_request_counts)))
    except Exception as e:
        logger.warning(f"Could not save stats: {e}")


async def periodic_save() -> None:
    while True:
        await asyncio.sleep(60)
        _save_counts()


_request_counts: defaultdict[str, int] = _load_counts()


def increment(path: str) -> None:
    _request_counts[path] += 1


def save() -> None:
    _save_counts()


def stats_payload() -> dict:
    total_translate = sum(
        v for k, v in _request_counts.items() if k in TRANSLATE_PATHS
    )
    return {
        "uptime_seconds": int(time.time() - _start_time),
        "total_translate_calls": total_translate,
        "requests_by_path": dict(
            sorted(_request_counts.items(), key=lambda x: -x[1])
        ),
    }


def snapshot_for_log() -> dict[str, int]:
    return dict(_request_counts)
