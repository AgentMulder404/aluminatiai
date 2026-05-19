# Copyright 2026 Kevin (AluminatiAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AluminatiAI — https://github.com/AgentMulder404/AluminatAI
"""
TagClient: polls GET /api/v1/tag and caches active job registrations.

Usage:
    client = TagClient(api_endpoint, api_key, poll_interval=30)
    client.start()           # starts background polling thread

    match = client.match(gpu_index=0, pid=12345, ts=datetime.now(UTC))
    if match:
        # match.job_id, match.team_id, match.model_tag
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass
class TagRecord:
    id: str
    job_id: str
    team_id: Optional[str]
    model_tag: Optional[str]
    gpu_indices: Optional[list[int]]   # None = all GPUs
    start_time: datetime
    end_time: Optional[datetime]       # None = open-ended
    pid: Optional[int]


# ── Client ─────────────────────────────────────────────────────────────────────


class TagClient:
    """
    Background-polling client for the /api/v1/tag REST endpoint.

    Thread-safe: `match()` can be called from any thread.
    """

    def __init__(
        self,
        api_endpoint: str,          # e.g. "https://aluminatiai.com/v1/metrics/ingest"
        api_key: str,
        poll_interval: int = 30,    # seconds between polls
    ):
        # Derive base URL from the ingest endpoint (strip trailing path)
        # Expected:  https://host/v1/metrics/ingest → https://host
        parts = api_endpoint.split("/v1/")
        self._base_url = parts[0].rstrip("/")
        self._tag_url = f"{self._base_url}/api/v1/tag"

        self._api_key = api_key
        self._poll_interval = poll_interval

        self._lock = threading.Lock()
        self._tags: list[TagRecord] = []
        self._last_poll: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="tag-client")
        self._thread.start()
        logger.info("TagClient started — polling %s every %ds", self._tag_url, self._poll_interval)

    def stop(self) -> None:
        """Signal the polling thread to stop."""
        self._stop_event.set()

    # ── Matching ───────────────────────────────────────────────────────────────

    def match(
        self,
        gpu_index: int,
        pid: Optional[int],
        ts: datetime,
    ) -> Optional[TagRecord]:
        """
        Return the highest-priority tag that covers this (gpu_index, pid, timestamp).

        Priority: most-recently-started tag wins when multiple match.
        Returns None if no tag matches.
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        with self._lock:
            candidates: list[TagRecord] = []
            for tag in self._tags:
                # Time window: start_time ≤ ts ≤ end_time (or open-ended)
                if ts < tag.start_time:
                    continue
                if tag.end_time is not None and ts > tag.end_time:
                    continue
                # GPU index filter (None = all GPUs)
                if tag.gpu_indices is not None and gpu_index not in tag.gpu_indices:
                    continue
                # PID filter (None = any PID)
                if tag.pid is not None and pid != tag.pid:
                    continue
                candidates.append(tag)

            if not candidates:
                return None

            # Most recently started tag wins
            return max(candidates, key=lambda t: t.start_time)

    # ── Internal polling ───────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        _consecutive_failures = 0
        _current_interval = self._poll_interval
        while not self._stop_event.is_set():
            try:
                self._fetch()
                _consecutive_failures = 0
                _current_interval = self._poll_interval
            except Exception as exc:
                _consecutive_failures += 1
                logger.warning("TagClient poll error (%d consecutive): %s",
                               _consecutive_failures, exc)
                if _consecutive_failures >= 3:
                    _current_interval = min(_current_interval * 2, 300)
            _jitter = random.uniform(-0.2 * _current_interval, 0.2 * _current_interval)
            self._stop_event.wait(max(1.0, _current_interval + _jitter))

    def _fetch(self) -> None:
        # Ask for tags from the last 25 hours (slightly more than a day to catch
        # long-running jobs that started before the last poll window).
        since = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        headers = {"X-API-Key": self._api_key}

        resp = requests.get(
            self._tag_url,
            params={"since": since},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()

        raw_tags = resp.json().get("tags", [])
        parsed: list[TagRecord] = []
        for t in raw_tags:
            try:
                parsed.append(TagRecord(
                    id=t["id"],
                    job_id=t["job_id"],
                    team_id=t.get("team_id"),
                    model_tag=t.get("model_tag"),
                    gpu_indices=t.get("gpu_indices"),
                    start_time=_parse_dt(t["start_time"]),
                    end_time=_parse_dt(t["end_time"]) if t.get("end_time") else None,
                    pid=t.get("pid"),
                ))
            except Exception as exc:
                logger.debug("TagClient: skipping malformed tag %s: %s", t.get("id"), exc)

        with self._lock:
            self._tags = parsed
            self._last_poll = datetime.now(timezone.utc)

        logger.debug("TagClient: fetched %d tag(s)", len(parsed))

    @property
    def tag_count(self) -> int:
        with self._lock:
            return len(self._tags)


def _parse_dt(value: str) -> datetime:
    """Parse ISO 8601 timestamp to an aware datetime (UTC)."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
