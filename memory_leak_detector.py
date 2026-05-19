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
GPU memory leak detector — flags monotonically increasing VRAM usage.

Tracks memory_used_mb per GPU in a sliding window. If >= 85% of
consecutive sample pairs are increasing over the full window, the GPU
is flagged as having a potential memory leak (OOM precursor).
"""
from __future__ import annotations

import collections
from typing import Optional


class MemoryLeakDetector:
    """Detect monotonically increasing GPU memory over a sliding window."""

    def __init__(self, window_size: int = 60, threshold_pct: float = 0.85):
        self._window_size = window_size
        self._threshold_pct = threshold_pct
        self._history: dict[int, collections.deque] = {}
        self._alerted: dict[int, bool] = {}

    def update(self, gpu_index: int, memory_used_mb: float) -> bool:
        """Record a sample. Returns True on first leak detection for this GPU."""
        if gpu_index not in self._history:
            self._history[gpu_index] = collections.deque(maxlen=self._window_size)
            self._alerted[gpu_index] = False

        buf = self._history[gpu_index]
        buf.append(memory_used_mb)

        if len(buf) < self._window_size:
            return False

        increases = sum(1 for i in range(1, len(buf)) if buf[i] > buf[i - 1])
        total_pairs = len(buf) - 1
        leak_detected = (increases / total_pairs) >= self._threshold_pct

        if leak_detected and not self._alerted[gpu_index]:
            self._alerted[gpu_index] = True
            return True
        elif not leak_detected:
            self._alerted[gpu_index] = False

        return False

    def get_leak_score(self, gpu_index: int) -> float:
        """Return 0.0-1.0 leak probability for Prometheus gauge."""
        buf = self._history.get(gpu_index)
        if not buf or len(buf) < 10:
            return 0.0
        increases = sum(1 for i in range(1, len(buf)) if buf[i] > buf[i - 1])
        return increases / (len(buf) - 1)
