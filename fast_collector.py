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
Fast-path NVML poller — fills ring buffers at 100-500ms intervals.

Runs in a dedicated daemon thread. Only collects power/util/temp/memory
(no PID resolution, no attribution — that's the main loop's job).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from ring_buffer import GPURingBuffer, GPUSample

logger = logging.getLogger(__name__)

try:
    import pynvml
    _NVML = True
except ImportError:
    _NVML = False


class FastCollector:
    """
    High-frequency GPU sampler that writes to per-GPU ring buffers.

    Lifecycle: start() -> ... -> stop()
    Thread-safe: ring buffers can be read from any thread.
    """

    def __init__(
        self,
        gpu_handles: list,
        ring_buffers: List[GPURingBuffer],
        sample_interval: float = 0.2,
    ):
        self._handles = gpu_handles
        self._buffers = ring_buffers
        self._interval = sample_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._consecutive_errors = 0
        self._total_samples = 0

    def start(self) -> None:
        if not _NVML:
            logger.warning("FastCollector: pynvml unavailable — disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="fast-collector",
        )
        self._thread.start()
        logger.info(
            "FastCollector started: %d GPUs at %.0fms intervals",
            len(self._handles), self._interval * 1000,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            logger.info(
                "FastCollector stopped after %d total samples", self._total_samples,
            )

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            now = time.time()

            for i, handle in enumerate(self._handles):
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU,
                    )
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    self._buffers[i].append(GPUSample(
                        timestamp=now,
                        power_w=power_w,
                        utilization_pct=util.gpu,
                        temperature_c=temp,
                        memory_used_mb=mem.used / (1024 * 1024),
                    ))
                    self._total_samples += 1
                    self._consecutive_errors = 0
                except Exception:
                    self._consecutive_errors += 1
                    if self._consecutive_errors == 50:
                        logger.error(
                            "FastCollector: %d consecutive NVML errors — throttling",
                            self._consecutive_errors,
                        )
                    if self._consecutive_errors >= 50:
                        self._stop_event.wait(1.0)
                        break

            elapsed = time.monotonic() - loop_start
            sleep_s = max(0.0, self._interval - elapsed)
            if sleep_s > 0:
                self._stop_event.wait(sleep_s)
