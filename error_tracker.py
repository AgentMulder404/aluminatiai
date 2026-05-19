"""
Agent-side error tracking with ring buffer and hourly stats.

Collects errors from the main loop (collection, upload, scheduler, etc.)
and provides summaries for heartbeat payloads + periodic error uploads.
"""

from __future__ import annotations

import traceback
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ErrorEntry:
    timestamp: float
    error_type: str
    message: str
    gpu_index: Optional[int] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "message": self.message,
        }
        if self.gpu_index is not None:
            d["gpu_index"] = self.gpu_index
        if self.stack_trace:
            d["stack_trace"] = self.stack_trace
        return d


class ErrorTracker:
    """Ring-buffer error tracker with hourly window stats."""

    def __init__(self, max_entries: int = 1000):
        self._buffer: deque[ErrorEntry] = deque(maxlen=max_entries)
        self._total_count: int = 0

    def record(
        self,
        error_type: str,
        message: str,
        gpu_index: Optional[int] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        stack = None
        if exc is not None:
            stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if len(stack) > 4000:
                stack = stack[:4000] + "\n... truncated"

        entry = ErrorEntry(
            timestamp=time.time(),
            error_type=error_type,
            message=message[:2000],
            gpu_index=gpu_index,
            stack_trace=stack,
        )
        self._buffer.append(entry)
        self._total_count += 1

    def get_recent(self, n: int = 50) -> list[ErrorEntry]:
        entries = list(self._buffer)
        return entries[-n:]

    def get_unsent(self, since: float) -> list[ErrorEntry]:
        return [e for e in self._buffer if e.timestamp > since]

    def get_stats(self) -> dict:
        now = time.time()
        hour_ago = now - 3600
        hour_count = sum(1 for e in self._buffer if e.timestamp > hour_ago)

        last_error = self._buffer[-1] if self._buffer else None

        return {
            "error_count_total": self._total_count,
            "error_count_last_hour": hour_count,
            "last_error_message": last_error.message if last_error else None,
            "last_error_at": last_error.timestamp if last_error else None,
        }

    @property
    def total_count(self) -> int:
        return self._total_count
