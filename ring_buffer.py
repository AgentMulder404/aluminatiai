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
Thread-safe per-GPU ring buffer for high-frequency power samples.

Stores recent NVML readings at 100-500ms intervals. The attribution
loop reads statistical summaries (p50/p95/p99/mean/max) at 5s intervals
instead of instantaneous spot readings.

Writer: FastCollector thread (100-500ms)
Reader: Main attribution loop (5s)
"""
from __future__ import annotations

import collections
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GPUSample:
    """Single high-frequency GPU sample."""
    timestamp: float
    power_w: float
    utilization_pct: int
    temperature_c: int
    memory_used_mb: float


@dataclass(frozen=True)
class GPUSummary:
    """Statistical summary of samples in a time window."""
    gpu_index: int
    sample_count: int
    window_seconds: float

    power_mean_w: float
    power_p50_w: float
    power_p95_w: float
    power_p99_w: float
    power_max_w: float
    power_min_w: float

    util_mean_pct: float
    temp_max_c: int
    memory_max_mb: float


class GPURingBuffer:
    """
    Thread-safe ring buffer for one GPU.

    Uses collections.deque(maxlen=N) for automatic eviction.
    At 200ms sampling with maxlen=100, holds 20s of history.
    """

    def __init__(self, gpu_index: int, max_samples: int = 100):
        self.gpu_index = gpu_index
        self._buf: collections.deque[GPUSample] = collections.deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def append(self, sample: GPUSample) -> None:
        with self._lock:
            self._buf.append(sample)

    def summarize(self, window_seconds: Optional[float] = None) -> Optional[GPUSummary]:
        """
        Compute statistical summary of buffered samples.

        If window_seconds is set, only include samples from the last N seconds.
        Returns None if buffer is empty or has fewer than 3 samples.
        """
        with self._lock:
            if not self._buf:
                return None
            if window_seconds:
                cutoff = time.time() - window_seconds
                samples = [s for s in self._buf if s.timestamp >= cutoff]
            else:
                samples = list(self._buf)

        if len(samples) < 3:
            return None

        powers = [s.power_w for s in samples]
        powers_sorted = sorted(powers)
        n = len(powers_sorted)

        return GPUSummary(
            gpu_index=self.gpu_index,
            sample_count=n,
            window_seconds=samples[-1].timestamp - samples[0].timestamp if n > 1 else 0,
            power_mean_w=statistics.mean(powers),
            power_p50_w=powers_sorted[n // 2],
            power_p95_w=powers_sorted[int(n * 0.95)] if n >= 20 else powers_sorted[-1],
            power_p99_w=powers_sorted[int(n * 0.99)] if n >= 100 else powers_sorted[-1],
            power_max_w=max(powers),
            power_min_w=min(powers),
            util_mean_pct=statistics.mean(s.utilization_pct for s in samples),
            temp_max_c=max(s.temperature_c for s in samples),
            memory_max_mb=max(s.memory_used_mb for s in samples),
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
