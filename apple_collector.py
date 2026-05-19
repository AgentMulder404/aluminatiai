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
Apple Silicon GPU Collector — energy monitoring for M-series Macs.

Two-tier backend:
  1. Primary: `sudo powermetrics` subprocess — accurate GPU/CPU/package power,
     die temperature, and GPU frequency from Apple's SMC.
  2. Fallback: `ioreg` — GPU utilization and memory from IOAccelerator, no
     power data (estimated from utilization × TDP).

Returns the same GPUMetrics dataclass so the agent loop is backend-agnostic.
Exposes exactly one "GPU" (Apple Silicon has a single integrated GPU).

Usage:
    collector = AppleSiliconCollector()
    metrics = collector.collect()   # List[GPUMetrics] with one entry
"""
from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from collector import GPUMetrics

logger = logging.getLogger(__name__)

# Known Apple Silicon TDP estimates (GPU portion, watts).
# Apple doesn't publish exact GPU TDP — these are empirically measured peaks.
_CHIP_GPU_TDP: dict[str, float] = {
    "Apple M1":          10.0,
    "Apple M1 Pro":      20.0,
    "Apple M1 Max":      40.0,
    "Apple M1 Ultra":    60.0,
    "Apple M2":          12.0,
    "Apple M2 Pro":      22.0,
    "Apple M2 Max":      45.0,
    "Apple M2 Ultra":    75.0,
    "Apple M3":          12.0,
    "Apple M3 Pro":      22.0,
    "Apple M3 Max":      45.0,
    "Apple M3 Ultra":    75.0,
    "Apple M4":          14.0,
    "Apple M4 Pro":      25.0,
    "Apple M4 Max":      50.0,
    "Apple M4 Ultra":    80.0,
    "Apple M5":          15.0,
    "Apple M5 Pro":      28.0,
    "Apple M5 Max":      55.0,
    "Apple M5 Ultra":    85.0,
}


@dataclass
class _SoCReading:
    """Parsed powermetrics sample."""
    gpu_power_w: float = 0.0
    cpu_power_w: float = 0.0
    package_power_w: float = 0.0
    die_temp_c: float = 0.0
    gpu_freq_mhz: int = 0
    gpu_util_pct: float = 0.0


def _detect_chip() -> str:
    """Return chip name like 'Apple M5' or '' if not Apple Silicon."""
    if platform.machine() != "arm64" or sys.platform != "darwin":
        return ""
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=5,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _get_total_memory_mb() -> float:
    """Read total system memory via sysctl (unified memory = GPU memory)."""
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5,
        ).strip()
        return int(raw) / (1024 * 1024)
    except (subprocess.SubprocessError, OSError, ValueError):
        return 0.0


def _get_gpu_core_count() -> int:
    """Read GPU core count from ioreg."""
    try:
        output = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            text=True, timeout=5,
        )
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', output)
        return int(m.group(1)) if m else 0
    except (subprocess.SubprocessError, OSError):
        return 0


def _gpu_tdp(chip_name: str) -> float:
    """Look up GPU TDP for a chip, with env override."""
    override = os.getenv("APPLE_CHIP_TDP_OVERRIDE", "")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    return _CHIP_GPU_TDP.get(chip_name, 15.0)


class _IOReg:
    """Fallback: read GPU utilization and memory from ioreg (no sudo)."""

    @staticmethod
    def read() -> dict:
        """Return dict with util_pct, memory_used_mb, or empty on failure."""
        try:
            output = subprocess.check_output(
                ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                text=True, timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            return {}

        result: dict = {}

        util_match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', output)
        if util_match:
            result["util_pct"] = int(util_match.group(1))

        mem_match = re.search(r'"In use system memory"\s*=\s*(\d+)', output)
        if mem_match:
            result["memory_used_mb"] = int(mem_match.group(1)) / (1024 * 1024)

        return result


class _PowerMetricsMonitor:
    """Background thread running `sudo powermetrics` and parsing output."""

    def __init__(self, interval_ms: int = 1000):
        self._interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest = _SoCReading()
        self._running = False
        self.available = False

    def start(self) -> bool:
        """Start the powermetrics subprocess. Returns True if successful."""
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        deadline = time.time() + (self._interval_ms / 1000.0) + 3.0
        while time.time() < deadline and not self.available:
            time.sleep(0.2)
        return self.available

    def stop(self) -> None:
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def latest(self) -> _SoCReading:
        with self._lock:
            return self._latest

    def _reader_loop(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [
                    "sudo", "-n", "powermetrics",
                    "--samplers", "gpu_power,smc",
                    "-i", str(self._interval_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (FileNotFoundError, OSError) as exc:
            logger.warning("powermetrics not available: %s", exc)
            return

        chunk: list[str] = []
        for line in self._proc.stdout:
            if not self._running:
                break
            if line.startswith("*****") or line.startswith("Machine model"):
                if chunk:
                    self._parse_chunk(chunk)
                chunk = [line]
            else:
                chunk.append(line)
        if chunk:
            self._parse_chunk(chunk)

    def _parse_chunk(self, lines: list[str]) -> None:
        text = "".join(lines)
        reading = _SoCReading()

        m = re.search(r"GPU Power:\s*([\d.]+)\s*mW", text)
        if m:
            reading.gpu_power_w = float(m.group(1)) / 1000.0

        m = re.search(r"CPU Power:\s*([\d.]+)\s*mW", text)
        if m:
            reading.cpu_power_w = float(m.group(1)) / 1000.0

        m = re.search(r"Combined Power.*?:\s*([\d.]+)\s*mW", text)
        if m:
            reading.package_power_w = float(m.group(1)) / 1000.0

        m = re.search(r"GPU (?:HW active|requested) frequency:\s*([\d.]+)\s*MHz", text)
        if m:
            reading.gpu_freq_mhz = int(float(m.group(1)))

        m = re.search(r"die temperature:\s*([\d.]+)\s*C", text, re.IGNORECASE)
        if m:
            reading.die_temp_c = float(m.group(1))

        m = re.search(r"GPU HW active residency:\s*([\d.]+)\s*%", text)
        if m:
            reading.gpu_util_pct = float(m.group(1))

        with self._lock:
            self._latest = reading

        if not self.available:
            self.available = True


class AppleSiliconCollector:
    """
    Collects GPU metrics from Apple Silicon via powermetrics or ioreg.

    Implements the same interface as GPUCollector:
      - collect() -> List[GPUMetrics]
      - get_gpu_count() -> int
      - get_gpu_info() -> List[Dict]
      - gpu_handles, gpu_uuids attributes
      - shutdown()
    """

    def __init__(self, collect_clocks: bool = False):
        self.collect_clocks = collect_clocks
        self.initialized = False
        self.gpu_count = 1
        self.gpu_handles: list = [0]
        self.gpu_info: list[dict] = []
        self.gpu_uuids: list[str] = []

        self.last_sample_time: dict[int, float] = {}
        self.last_power_draw: dict[int, float] = {}

        self._chip_name = _detect_chip()
        if not self._chip_name:
            raise RuntimeError(
                "Apple Silicon not detected — requires arm64 macOS"
            )

        self._gpu_tdp = _gpu_tdp(self._chip_name)
        self._total_memory_mb = _get_total_memory_mb()
        self._gpu_cores = _get_gpu_core_count()

        self._monitor: Optional[_PowerMetricsMonitor] = None
        self._powermetrics_available = False

        self._initialize()

    def _initialize(self):
        gpu_name = f"{self._chip_name} GPU"
        if self._gpu_cores:
            gpu_name += f" ({self._gpu_cores}-core)"

        gpu_uuid = f"apple-{self._chip_name.lower().replace(' ', '-')}-integrated-gpu"

        self.gpu_info = [{
            "index": 0,
            "uuid": gpu_uuid,
            "name": gpu_name,
        }]
        self.gpu_uuids = [gpu_uuid]

        # Try to start powermetrics (requires sudo -n, i.e. NOPASSWD)
        pm_enabled = os.getenv("APPLE_POWERMETRICS_ENABLED", "1").lower() not in ("0", "false", "no")
        interval_ms = int(os.getenv("APPLE_POWERMETRICS_INTERVAL_MS", "1000"))

        if pm_enabled:
            self._monitor = _PowerMetricsMonitor(interval_ms=interval_ms)
            self._powermetrics_available = self._monitor.start()
            if self._powermetrics_available:
                logger.info("Apple Silicon collector: powermetrics active, chip=%s, cores=%d",
                            self._chip_name, self._gpu_cores)
            else:
                logger.warning(
                    "powermetrics failed (sudo -n required) — falling back to ioreg "
                    "(no power data, utilization only)"
                )
                self._monitor.stop()
                self._monitor = None

        if not self._powermetrics_available:
            logger.info("Apple Silicon collector: ioreg fallback, chip=%s, cores=%d",
                        self._chip_name, self._gpu_cores)

        self.initialized = True

    @property
    def backend(self) -> str:
        return "powermetrics" if self._powermetrics_available else "ioreg"

    def get_gpu_count(self) -> int:
        return self.gpu_count

    def get_gpu_info(self) -> List[Dict]:
        return self.gpu_info

    def collect(self) -> List[GPUMetrics]:
        """Collect current GPU metrics."""
        if not self.initialized:
            raise RuntimeError("Apple Silicon collector not initialized")

        timestamp = datetime.now(timezone.utc).isoformat()
        now = time.monotonic()
        info = self.gpu_info[0]

        if self._powermetrics_available and self._monitor:
            reading = self._monitor.latest
            power_w = round(reading.gpu_power_w, 2)
            util_pct = int(reading.gpu_util_pct) if reading.gpu_util_pct > 0 else 0
            temp_c = int(reading.die_temp_c) if reading.die_temp_c > 0 else 0
            freq_mhz = reading.gpu_freq_mhz

            # Use ioreg for memory since powermetrics doesn't report it
            ioreg_data = _IOReg.read()
            mem_used_mb = round(ioreg_data.get("memory_used_mb", 0.0), 1)
            if util_pct == 0 and "util_pct" in ioreg_data:
                util_pct = ioreg_data["util_pct"]
        else:
            # ioreg fallback — no power data
            ioreg_data = _IOReg.read()
            util_pct = ioreg_data.get("util_pct", 0)
            mem_used_mb = round(ioreg_data.get("memory_used_mb", 0.0), 1)
            temp_c = 0
            freq_mhz = 0
            # Estimate power from utilization × TDP
            power_w = round(self._gpu_tdp * (util_pct / 100.0), 2)

        # Energy delta
        energy_delta_j = None
        last_time = self.last_sample_time.get(0)
        last_power = self.last_power_draw.get(0)
        if last_time is not None and last_power is not None:
            dt = now - last_time
            if dt > 0 and power_w > 0:
                avg_power = (last_power + power_w) / 2.0
                energy_delta_j = round(avg_power * dt, 4)

        self.last_sample_time[0] = now
        self.last_power_draw[0] = power_w

        return [GPUMetrics(
            timestamp=timestamp,
            gpu_index=0,
            gpu_uuid=info["uuid"],
            gpu_name=info["name"],
            power_draw_w=power_w,
            power_limit_w=self._gpu_tdp,
            energy_delta_j=energy_delta_j,
            utilization_gpu_pct=util_pct,
            utilization_memory_pct=0,
            temperature_c=temp_c,
            fan_speed_pct=0,
            sm_clock_mhz=freq_mhz if freq_mhz > 0 else None,
            memory_used_mb=mem_used_mb,
            memory_total_mb=round(self._total_memory_mb, 1),
        )]

    def shutdown(self):
        if self._monitor:
            self._monitor.stop()
        self.initialized = False
        logger.info("Apple Silicon collector shut down")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
