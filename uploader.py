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
Metrics uploader for AluminatAI API — v0.2.0

Features:
  - Exponential backoff with jitter (1s → 2s → 4s → 8s → 16s, capped 60s)
  - Respects Retry-After on 429
  - Permanent failure on 401/403 → writes to WAL immediately
  - WAL-based local buffer (append-only newline-delimited JSON)
  - TLS / mTLS / proxy support from config
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Dict, List

try:
    import fcntl  # POSIX only — not available on Windows
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

import requests

from config import (
    API_ENDPOINT, API_KEY, UPLOAD_BATCH_SIZE,
    WAL_DIR, WAL_MAX_AGE_HOURS, WAL_MAX_MB,
    UPLOAD_MAX_RETRIES, UPLOAD_MAX_RETRY_DELAY, UPLOAD_TIMEOUT,
    HTTPS_PROXY, CA_BUNDLE, CLIENT_CERT, CLIENT_KEY,
    OFFLINE_MODE, DRY_RUN,
)

MAX_BUFFER_SIZE = 10_000
QUARANTINE_FILE = WAL_DIR / "metrics.quarantine"

logger = logging.getLogger(__name__)

# ── WAL encryption (optional — requires cryptography package) ─────────────────


def _init_fernet():
    """Return a Fernet instance derived from the API key, or None.

    Key derivation: SHA-256(API_KEY) → 32 raw bytes → URL-safe base64 → Fernet key.
    Activated automatically when ALUMINATAI_API_KEY is set AND cryptography is installed.
    Falls back gracefully to plaintext WAL with a one-time WARNING if the package is absent.
    """
    if not API_KEY:
        return None
    try:
        from cryptography.fernet import Fernet
        import hashlib
        import base64
        raw = hashlib.sha256(API_KEY.encode()).digest()
        return Fernet(base64.urlsafe_b64encode(raw))
    except ImportError:
        logger.warning(
            "WAL encryption unavailable — install aluminatai[secure] for encrypted WAL"
        )
        return None


_FERNET = _init_fernet()


def _quarantine_lines(lines: list[str]) -> None:
    """Append unreadable WAL lines to the quarantine file for post-mortem analysis."""
    if not lines:
        return
    try:
        QUARANTINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(QUARANTINE_FILE, "a") as f:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for raw in lines:
                    f.write(raw + "\n")
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
        logger.error(
            "WAL: quarantined %d unreadable lines → %s",
            len(lines), QUARANTINE_FILE,
        )
    except OSError as exc:
        logger.error("WAL: failed to write quarantine file: %s", exc)


# ── WAL helpers ───────────────────────────────────────────────────────────────

WAL_FILE = WAL_DIR / "metrics.wal"

# Approximate count of pending WAL rows, maintained by append/clear/rewrite.
# This is an in-process estimate — it resets to 0 on restart.
_WAL_PENDING: int = 0


def _wal_append(batch: List[Dict]) -> None:
    """Append a batch to the WAL as newline-delimited JSON entries.

    Uses an exclusive flock on POSIX to prevent concurrent writers (e.g. two
    agent processes during a K8s rolling update) from interleaving partial
    writes and corrupting the WAL.
    """
    global _WAL_PENDING
    WAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(WAL_FILE, "a") as f:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for row in batch:
                    entry = {"ts": time.time(), "row": row}
                    line_bytes = json.dumps(entry).encode()
                    if _FERNET:
                        f.write(_FERNET.encrypt(line_bytes).decode() + "\n")
                    else:
                        f.write(line_bytes.decode() + "\n")
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
        _WAL_PENDING += len(batch)
        logger.info("WAL: appended %d rows → %s", len(batch), WAL_FILE)
    except OSError as exc:
        logger.error("WAL write failed: %s", exc,
                     extra={"event": "wal_write_failed", "path": str(WAL_FILE), "error": str(exc)})


def _wal_read_valid() -> List[Dict]:
    """Read WAL, filter by TTL, enforce size cap, return metric dicts."""
    if not WAL_FILE.exists():
        return []

    cutoff = time.time() - WAL_MAX_AGE_HOURS * 3600
    rows: list[dict] = []
    raw_lines: list[str] = []

    try:
        with open(WAL_FILE) as f:
            raw_lines = f.readlines()
    except OSError:
        return []

    quarantine_batch: list[str] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        entry = None
        if _FERNET:
            try:
                entry = json.loads(_FERNET.decrypt(line.encode()))
            except Exception:
                # Wrong key or plaintext line — try raw JSON as fallback
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, KeyError):
                    quarantine_batch.append(line)
                    continue
        else:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, KeyError):
                quarantine_batch.append(line)
                continue
        if entry and entry.get("ts", 0) >= cutoff:
            try:
                rows.append(entry["row"])
            except KeyError:
                pass
    if quarantine_batch:
        _quarantine_lines(quarantine_batch)

    # Size cap: drop oldest if WAL is too large
    wal_mb = WAL_FILE.stat().st_size / (1024 * 1024) if WAL_FILE.exists() else 0
    if wal_mb > WAL_MAX_MB:
        drop = max(0, len(rows) - len(rows) // 2)
        rows = rows[drop:]
        logger.warning("WAL exceeded %dMB — dropped %d oldest rows", WAL_MAX_MB, drop)

    return rows


def _wal_clear() -> None:
    """Delete the WAL after a successful full replay."""
    global _WAL_PENDING
    try:
        WAL_FILE.unlink(missing_ok=True)
        _WAL_PENDING = 0
    except OSError:
        pass


def _wal_rewrite(rows: List[Dict]) -> None:
    """Atomically replace the WAL with only the given rows.

    Writes to a .tmp sibling file first, then renames over the WAL so that a
    crash between _wal_clear() and _wal_append() can never lose data — the
    old WAL remains intact until the rename succeeds.
    """
    WAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = WAL_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for row in rows:
                    entry = {"ts": time.time(), "row": row}
                    line_bytes = json.dumps(entry).encode()
                    if _FERNET:
                        f.write(_FERNET.encrypt(line_bytes).decode() + "\n")
                    else:
                        f.write(line_bytes.decode() + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
        tmp.replace(WAL_FILE)  # atomic on POSIX
        global _WAL_PENDING
        _WAL_PENDING = len(rows)
        logger.info("WAL: rewrote %d rows → %s", len(rows), WAL_FILE)
    except OSError as exc:
        logger.error("WAL rewrite failed: %s", exc,
                     extra={"event": "wal_rewrite_failed", "path": str(WAL_FILE), "error": str(exc)})
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ── Session factory ───────────────────────────────────────────────────────────


def _build_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    })
    if HTTPS_PROXY:
        session.proxies = {"https": HTTPS_PROXY, "http": HTTPS_PROXY}
    # TLS verification: always enabled. Never set verify=False — self-signed certs
    # are rejected by default. CA_BUNDLE overrides the system CA store (e.g. for
    # corporate proxy MITM certs); omit it to use the bundled Mozilla CA store.
    session.verify = CA_BUNDLE if CA_BUNDLE else True
    if CLIENT_CERT and CLIENT_KEY:
        session.cert = (CLIENT_CERT, CLIENT_KEY)
    return session


# ── Uploader ──────────────────────────────────────────────────────────────────


class MetricsUploader:
    """Upload GPU metrics to the AluminatAI API with backoff + WAL durability."""

    CIRCUIT_OPEN_THRESHOLD = 5
    CIRCUIT_COOLDOWN_SECONDS = 300

    def __init__(self, api_endpoint: str = API_ENDPOINT, api_key: str = API_KEY):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.session = _build_session(api_key)
        self.buffer: List[Dict] = []
        self._upload_success = 0
        self._upload_failure = 0
        self._wal_replay_uploaded = 0
        self._wal_replay_failed = 0
        self._consecutive_failures = 0
        self._circuit_open_since: float = 0.0
        if OFFLINE_MODE:
            logger.info("OFFLINE_MODE=1 — all metrics written to WAL; no HTTP uploads")
        else:
            logger.info("Uploader initialised → %s", api_endpoint)

    def add_metrics(self, metrics: List[Dict]) -> None:
        if len(self.buffer) + len(metrics) > MAX_BUFFER_SIZE:
            overflow = self.buffer + list(metrics)
            flush_rows = overflow[:MAX_BUFFER_SIZE]
            logger.warning(
                "Buffer exceeded %d rows — flushing %d to WAL",
                MAX_BUFFER_SIZE, len(flush_rows),
            )
            _wal_append(flush_rows)
            self.buffer = overflow[MAX_BUFFER_SIZE:]
        else:
            self.buffer.extend(metrics)

    def upload_batch(self, metrics: List[Dict]) -> bool:
        """
        Upload one batch with exponential backoff.

        Returns True on success.  Writes to WAL on permanent failure.
        In OFFLINE_MODE, writes to WAL immediately and returns False (no HTTP).
        """
        if DRY_RUN:
            logger.info(
                "DRY RUN — would upload %d metrics to %s (skipped)",
                len(metrics), self.api_endpoint,
            )
            return True
        if OFFLINE_MODE:
            _wal_append(metrics)
            return False

        # Circuit breaker: skip HTTP when API has been failing persistently
        if self._circuit_open_since > 0:
            elapsed = time.time() - self._circuit_open_since
            if elapsed < self.CIRCUIT_COOLDOWN_SECONDS:
                _wal_append(metrics)
                return False
            # Half-open: try one probe request
            logger.info("Circuit breaker half-open — probing API")
            self._circuit_open_since = 0.0

        delay = 1.0
        for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
            try:
                resp = self.session.post(self.api_endpoint, json=metrics, timeout=UPLOAD_TIMEOUT)
            except requests.Timeout:
                logger.warning(
                    "Upload timeout (attempt %d/%d)", attempt, UPLOAD_MAX_RETRIES,
                    extra={"event": "upload_timeout", "attempt": attempt,
                           "max_retries": UPLOAD_MAX_RETRIES, "next_delay_s": round(delay, 1)},
                )
                self._sleep_with_jitter(delay)
                delay = min(delay * 2, UPLOAD_MAX_RETRY_DELAY)
                continue
            except requests.ConnectionError as exc:
                logger.warning(
                    "Connection error (attempt %d/%d): %s", attempt, UPLOAD_MAX_RETRIES, exc,
                    extra={"event": "upload_connection_error", "attempt": attempt,
                           "max_retries": UPLOAD_MAX_RETRIES, "error": str(exc),
                           "next_delay_s": round(delay, 1)},
                )
                self._sleep_with_jitter(delay)
                delay = min(delay * 2, UPLOAD_MAX_RETRY_DELAY)
                continue
            except requests.RequestException as exc:
                logger.error(
                    "Unrecoverable request error: %s", exc,
                    extra={"event": "upload_unrecoverable", "error": str(exc),
                           "batch_size": len(metrics)},
                )
                _wal_append(metrics)
                self._upload_failure += 1
                return False

            if resp.status_code == 200:
                self._upload_success += len(metrics)
                self._consecutive_failures = 0
                return True

            if resp.status_code in (401, 403):
                logger.error(
                    "Permanent auth failure (%d) — check API key", resp.status_code,
                    extra={"event": "upload_auth_failed", "status_code": resp.status_code,
                           "batch_size": len(metrics)},
                )
                _wal_append(metrics)
                self._upload_failure += 1
                return False

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", delay))
                logger.warning(
                    "Rate-limited — waiting %.0fs", retry_after,
                    extra={"event": "upload_rate_limited", "retry_after_s": round(retry_after, 1),
                           "attempt": attempt},
                )
                time.sleep(retry_after + random.uniform(0, 1))
                delay = min(delay * 2, UPLOAD_MAX_RETRY_DELAY)
                continue

            # 5xx or other transient error
            logger.warning(
                "HTTP %d (attempt %d/%d)", resp.status_code, attempt, UPLOAD_MAX_RETRIES,
                extra={"event": "upload_http_error", "status_code": resp.status_code,
                       "attempt": attempt, "max_retries": UPLOAD_MAX_RETRIES,
                       "next_delay_s": round(delay, 1)},
            )
            self._sleep_with_jitter(delay)
            delay = min(delay * 2, UPLOAD_MAX_RETRY_DELAY)

        # Exhausted retries
        logger.error(
            "Upload failed after %d attempts — writing to WAL", UPLOAD_MAX_RETRIES,
            extra={"event": "upload_exhausted", "max_retries": UPLOAD_MAX_RETRIES,
                   "batch_size": len(metrics)},
        )
        _wal_append(metrics)
        self._upload_failure += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.CIRCUIT_OPEN_THRESHOLD:
            self._circuit_open_since = time.time()
            logger.error(
                "Circuit breaker OPEN — %d consecutive failures, skipping HTTP for %ds",
                self._consecutive_failures, self.CIRCUIT_COOLDOWN_SECONDS,
            )
        return False

    @staticmethod
    def _sleep_with_jitter(base_delay: float) -> None:
        jitter = random.uniform(-0.2 * base_delay, 0.2 * base_delay)
        time.sleep(max(0, base_delay + jitter))

    def flush(self) -> int:
        """Upload all buffered metrics. Returns number successfully uploaded."""
        if not self.buffer:
            return 0

        uploaded = 0
        remaining: list[dict] = []

        for i in range(0, len(self.buffer), UPLOAD_BATCH_SIZE):
            batch = self.buffer[i:i + UPLOAD_BATCH_SIZE]
            if self.upload_batch(batch):
                uploaded += len(batch)
            else:
                remaining.extend(batch)

        self.buffer = remaining
        if uploaded:
            logger.info("Flushed %d metrics", uploaded)
        return uploaded

    def retry_failed_uploads(self) -> int:
        """
        Replay the WAL at startup.  Returns number of metrics successfully re-uploaded.
        Clears the WAL if all entries are replayed.
        In OFFLINE_MODE, skips replay (WAL is the intended storage).
        """
        if DRY_RUN or OFFLINE_MODE:
            return 0
        rows = _wal_read_valid()
        if not rows:
            return 0

        logger.info(
            "WAL replay: %d rows pending", len(rows),
            extra={"event": "wal_replay_start", "pending_rows": len(rows)},
        )
        uploaded = 0
        failed: list[dict] = []

        for i in range(0, len(rows), UPLOAD_BATCH_SIZE):
            batch = rows[i:i + UPLOAD_BATCH_SIZE]
            if self.upload_batch(batch):
                uploaded += len(batch)
            else:
                failed.extend(batch)

        self._wal_replay_uploaded += uploaded
        self._wal_replay_failed += len(failed)

        if not failed:
            _wal_clear()
            logger.info(
                "WAL replay complete — all %d rows uploaded", uploaded,
                extra={"event": "wal_replay_complete", "uploaded": uploaded, "failed": 0},
            )
        else:
            # Atomically rewrite WAL with only the rows that still need upload.
            # _wal_rewrite() writes to a .tmp file then renames, so a crash
            # between these two steps cannot silently drop metrics.
            _wal_rewrite(failed)
            logger.warning(
                "WAL replay partial: %d uploaded, %d remain", uploaded, len(failed),
                extra={"event": "wal_replay_partial", "uploaded": uploaded,
                       "failed": len(failed), "reason": "upload_errors"},
            )

        return uploaded

    def get_status(self) -> Dict:
        wal_size = WAL_FILE.stat().st_size if WAL_FILE.exists() else 0
        return {
            "buffer_size": len(self.buffer),
            "wal_bytes": wal_size,
            "wal_entries_pending": _WAL_PENDING,
            "upload_success_total": self._upload_success,
            "upload_failure_total": self._upload_failure,
            "wal_replay_uploaded_total": self._wal_replay_uploaded,
            "wal_replay_failed_total": self._wal_replay_failed,
            "api_endpoint": self.api_endpoint,
            "has_api_key": bool(self.api_key),
            "circuit_breaker": "open" if self._circuit_open_since > 0 else "closed",
            "consecutive_failures": self._consecutive_failures,
        }
