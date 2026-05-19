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
PidSmoother — 30-second sliding-window PID stability filter.

Problem: DDP / multi-process GPU jobs spawn several workers in rapid
succession during initialisation.  Each worker may take 1–3 samples to
allocate GPU memory, so the NVML compute-process list is noisy for the
first ~15 s of a job.  Without smoothing, the attribution engine sees
flickery PID sets and may emit multiple partial-attribution rows — or
temporarily mis-attribute power to a *different* job entirely.

Solution: keep a per-GPU ring-buffer of (timestamp, frozen_pid_set)
observations.  `stable_pids()` returns only PIDs that appear in at
least `stable_threshold` (default 60 %) of the buffered samples.

Behaviour by phase
──────────────────
  cold start (< 3 samples in window)
    → return union of all seen PIDs so nothing is silently dropped

  warm window (≥ 3 samples)
    → return PIDs whose appearance frequency ≥ stable_threshold

  caller falls back to raw NVML list when `stable_pids()` returns ∅
    → ensures brand-new jobs are never silently dropped

Typical window at SAMPLE_INTERVAL=5 s: 6 observations → threshold
count = ceil(0.60 × 6) = 4, so a PID must survive at least 20 s before
it is considered stable.  Transient CUDA memcpy helpers or DDP spawn
workers that disappear within one interval are filtered out.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Deque, FrozenSet, Tuple


class PidSmoother:
    """
    Per-GPU sliding-window PID stability filter.

    Parameters
    ----------
    window_s : float
        Width of the sliding window in seconds (default 30).
    stable_threshold : float
        Fraction [0, 1] of window samples a PID must appear in to be
        considered stable (default 0.60).
    """

    def __init__(
        self,
        window_s: float = 30.0,
        stable_threshold: float = 0.60,
    ) -> None:
        self._window_s = window_s
        self._threshold = max(0.0, min(1.0, stable_threshold))
        # gpu_index → deque of (monotonic_ts, frozenset[pid])
        self._history: Dict[int, Deque[Tuple[float, FrozenSet[int]]]] = defaultdict(deque)

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, gpu_index: int, ts: float, pids: FrozenSet[int]) -> None:
        """
        Record the PID set observed on `gpu_index` at time `ts`.

        Evicts observations older than `window_s` in the same call so
        memory usage is bounded by (window_s / sample_interval) entries
        per GPU.
        """
        dq = self._history[gpu_index]
        dq.append((ts, pids))
        cutoff = ts - self._window_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def stable_pids(self, gpu_index: int) -> FrozenSet[int]:
        """
        Return the set of PIDs considered stable on `gpu_index`.

        A PID is stable when it appears in at least `stable_threshold`
        fraction of buffered samples.

        Special cases
        -------------
        * No history at all  → frozenset() (caller falls back to raw list)
        * Fewer than 3 samples in window  → union of all seen PIDs
          (cold-start grace period so new jobs are never silently dropped)
        """
        dq = self._history.get(gpu_index)
        if not dq:
            return frozenset()

        # Cold-start: not enough samples yet for a meaningful frequency check
        if len(dq) < 3:
            return frozenset(pid for _, pids in dq for pid in pids)

        # Count per-PID appearances across the window
        counts: dict[int, int] = {}
        total = len(dq)
        for _, pids in dq:
            for pid in pids:
                counts[pid] = counts.get(pid, 0) + 1

        # Threshold: at least stable_threshold × total samples
        min_count = self._threshold * total
        return frozenset(pid for pid, n in counts.items() if n >= min_count)

    # ── Introspection (for tests / banner) ───────────────────────────────

    def window_size(self, gpu_index: int) -> int:
        """Number of buffered samples for a given GPU."""
        dq = self._history.get(gpu_index)
        return len(dq) if dq else 0

    def stats(self) -> dict:
        """Return a summary dict suitable for logging / tests."""
        return {
            gpu: {
                "samples": len(dq),
                "oldest_age_s": round(time.monotonic() - dq[0][0], 1) if dq else 0.0,
                "unique_pids": len({pid for _, pids in dq for pid in pids}),
            }
            for gpu, dq in self._history.items()
        }
