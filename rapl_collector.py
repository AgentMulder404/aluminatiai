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
CPU-only RAPL Collector — first-class collector for CPU/DRAM energy monitoring.

Drop-in replacement for GPUCollector on machines with no discrete GPU.
Returns the same GPUMetrics dataclass so the agent loop is backend-agnostic.

Each RAPL package (CPU socket) is exposed as one "GPU" entry. Power comes
from hardware energy counters in /sys/class/powercap; utilization comes
from /proc/stat; temperature from /sys/class/hwmon.

Usage:
    collector = RAPLCollector()
    metrics = collector.collect()   # List[GPUMetrics], one per CPU socket
"""
from __future__ import annotations

import hashlib
import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from collector import GPUMetrics
from efficiency.rapl import RaplReader

logger = logging.getLogger(__name__)


def _read_cpu_utilization() -> float:
    """Read aggregate CPU utilization from /proc/stat (Linux only)."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        if parts[0] != "cpu":
            return 0.0
        # user, nice, system, idle, iowait, irq, softirq, steal
        vals = [int(x) for x in parts[1:9]]
        idle = vals[3] + vals[4]  # idle + iowait
        total = sum(vals)
        return total, idle
    except (OSError, ValueError, IndexError):
        return None


def _read_temperatures() -> List[float]:
    """Read CPU temperatures from hwmon sysfs."""
    temps = []
    hwmon_base = "/sys/class/hwmon"
    try:
        for hwmon in sorted(os.listdir(hwmon_base)):
            hwmon_path = os.path.join(hwmon_base, hwmon)
            name_file = os.path.join(hwmon_path, "name")
            try:
                with open(name_file) as f:
                    name = f.read().strip()
            except (OSError, PermissionError):
                continue
            if name not in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
                continue
            for entry in sorted(os.listdir(hwmon_path)):
                if entry.startswith("temp") and entry.endswith("_input"):
                    try:
                        with open(os.path.join(hwmon_path, entry)) as f:
                            temps.append(int(f.read().strip()) / 1000.0)
                    except (OSError, ValueError, PermissionError):
                        continue
            if temps:
                break
    except OSError:
        pass
    return temps


def _read_memory_info() -> tuple[float, float]:
    """Read system memory from /proc/meminfo. Returns (used_mb, total_mb)."""
    total_kb = 0
    available_kb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    break
    except (OSError, ValueError):
        return 0.0, 0.0
    total_mb = total_kb / 1024.0
    used_mb = (total_kb - available_kb) / 1024.0
    return used_mb, total_mb


class RAPLCollector:
    """
    Collects CPU/DRAM energy metrics via RAPL sysfs counters.

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
        self.gpu_count = 0
        self.gpu_handles: list = []
        self.gpu_info: list[dict] = []
        self.gpu_uuids: list[str] = []

        self.last_sample_time: dict[int, float] = {}
        self.last_power_draw: dict[int, float] = {}

        # CPU utilization tracking (delta-based)
        self._last_cpu_total: Optional[int] = None
        self._last_cpu_idle: Optional[int] = None

        self._reader = RaplReader()
        if not self._reader.available:
            raise RuntimeError(
                "RAPL not available — requires Linux with readable "
                "/sys/class/powercap/intel-rapl:* or amd_rapl:* counters"
            )

        self._initialize()

    def _initialize(self):
        hostname = socket.gethostname()
        self.gpu_count = self._reader.package_count
        cpu_model = (
            os.getenv("RAPL_CPU_MODEL_OVERRIDE")
            or self._reader.cpu_model
            or "CPU (RAPL)"
        )

        for i in range(self.gpu_count):
            uuid_seed = f"RAPL-PKG-{i}-{hostname}"
            uuid = f"RAPL-{hashlib.sha256(uuid_seed.encode()).hexdigest()[:16]}"

            self.gpu_handles.append(i)
            self.gpu_info.append({
                "index": i,
                "uuid": uuid,
                "name": f"{cpu_model} [Socket {i}]" if self.gpu_count > 1 else cpu_model,
            })

        self.gpu_uuids = [g["uuid"] for g in self.gpu_info]
        self.initialized = True

        # Prime the RAPL reader with an initial reading
        self._reader.read_all()
        # Prime CPU utilization
        result = _read_cpu_utilization()
        if result is not None:
            self._last_cpu_total, self._last_cpu_idle = result

        logger.info("RAPL collector: %d package(s), cpu=%s", self.gpu_count, cpu_model)

    def get_gpu_count(self) -> int:
        return self.gpu_count

    def get_gpu_info(self) -> List[Dict]:
        return self.gpu_info

    def collect(self) -> List[GPUMetrics]:
        """Collect current metrics from all RAPL packages."""
        if not self.initialized:
            raise RuntimeError("RAPL collector not initialized")

        readings = self._reader.read_all()
        if not readings:
            return []

        # CPU utilization (shared across all sockets)
        cpu_util_pct = 0
        result = _read_cpu_utilization()
        if result is not None:
            total, idle = result
            if self._last_cpu_total is not None:
                dt = total - self._last_cpu_total
                di = idle - self._last_cpu_idle
                if dt > 0:
                    cpu_util_pct = int(100.0 * (1.0 - di / dt))
                    cpu_util_pct = max(0, min(100, cpu_util_pct))
            self._last_cpu_total = total
            self._last_cpu_idle = idle

        # Temperature (use max across all cores)
        temps = _read_temperatures()
        temp_c = int(max(temps)) if temps else 0

        # System memory
        mem_used_mb, mem_total_mb = _read_memory_info()

        timestamp = datetime.now(timezone.utc).isoformat()
        metrics: List[GPUMetrics] = []

        for reading in readings:
            info = self.gpu_info[reading.package_index] if reading.package_index < len(self.gpu_info) else self.gpu_info[0]
            now = time.monotonic()
            last_time = self.last_sample_time.get(reading.package_index)
            last_power = self.last_power_draw.get(reading.package_index)

            energy_delta_j = None
            if last_time is not None and last_power is not None:
                dt = now - last_time
                if dt > 0 and reading.package_watts > 0:
                    avg_power = (last_power + reading.package_watts) / 2.0
                    energy_delta_j = round(avg_power * dt, 4)

            self.last_sample_time[reading.package_index] = now
            self.last_power_draw[reading.package_index] = reading.package_watts

            metrics.append(GPUMetrics(
                timestamp=timestamp,
                gpu_index=reading.package_index,
                gpu_uuid=info["uuid"],
                gpu_name=info["name"],
                power_draw_w=round(reading.package_watts, 2),
                power_limit_w=reading.power_limit_w,
                energy_delta_j=energy_delta_j,
                utilization_gpu_pct=cpu_util_pct,
                utilization_memory_pct=0,
                temperature_c=temp_c,
                fan_speed_pct=0,
                memory_used_mb=round(mem_used_mb, 1),
                memory_total_mb=round(mem_total_mb, 1),
            ))

        return metrics

    def shutdown(self):
        self.initialized = False
        logger.info("RAPL collector shut down")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
