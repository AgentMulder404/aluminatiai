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
"""AMD GPU power monitoring for GreenTune.

Unified interface for reading power/energy/temperature from AMD GPUs.
Tries amdsmi Python bindings first, falls back to rocm-smi CLI parsing.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PowerSample:
    timestamp: float
    gpu_index: int
    power_w: float
    temperature_c: float
    memory_used_mb: float = 0.0
    utilization_pct: float = 0.0


@dataclass
class EnergyAccumulator:
    """Tracks cumulative energy via trapezoidal integration of power samples."""

    samples: list[PowerSample] = field(default_factory=list)
    total_joules: float = 0.0
    peak_power_w: float = 0.0
    _last_sample: Optional[PowerSample] = field(default=None, repr=False)

    def add(self, sample: PowerSample):
        if self._last_sample is not None:
            dt = sample.timestamp - self._last_sample.timestamp
            avg_power = (sample.power_w + self._last_sample.power_w) / 2
            self.total_joules += avg_power * dt
        self.peak_power_w = max(self.peak_power_w, sample.power_w)
        self._last_sample = sample
        self.samples.append(sample)

    @property
    def total_kwh(self) -> float:
        return self.total_joules / 3_600_000

    @property
    def avg_power_w(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        duration = self.samples[-1].timestamp - self.samples[0].timestamp
        return self.total_joules / duration if duration > 0 else 0.0

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp

    def to_dict(self) -> dict:
        return {
            "total_joules": round(self.total_joules, 2),
            "total_kwh": round(self.total_kwh, 8),
            "avg_power_w": round(self.avg_power_w, 1),
            "peak_power_w": round(self.peak_power_w, 1),
            "duration_s": round(self.duration_s, 1),
            "sample_count": len(self.samples),
        }


class AMDPowerMonitor:
    """Reads power/temp from a single AMD GPU."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._use_amdsmi = False
        self._amdsmi_handle = None
        self._init_backend()

    def _init_backend(self):
        try:
            from amdsmi import amdsmi_init, amdsmi_get_processor_handles

            amdsmi_init()
            handles = amdsmi_get_processor_handles()
            if self.gpu_index < len(handles):
                self._amdsmi_handle = handles[self.gpu_index]
                self._use_amdsmi = True
        except (ImportError, Exception):
            self._use_amdsmi = False

    def read(self) -> PowerSample:
        if self._use_amdsmi:
            return self._read_amdsmi()
        return self._read_cli()

    def _read_amdsmi(self) -> PowerSample:
        from amdsmi import (
            amdsmi_get_power_info,
            amdsmi_get_temp_metric,
            AmdSmiTemperatureType,
            AmdSmiTemperatureMetric,
        )

        pwr = amdsmi_get_power_info(self._amdsmi_handle)
        temp = amdsmi_get_temp_metric(
            self._amdsmi_handle,
            AmdSmiTemperatureType.HOTSPOT,
            AmdSmiTemperatureMetric.CURRENT,
        )
        return PowerSample(
            timestamp=time.time(),
            gpu_index=self.gpu_index,
            power_w=float(pwr.get("current_socket_power", 0)),
            temperature_c=float(temp),
        )

    def _read_cli(self) -> PowerSample:
        power_w = self._parse_rocm_smi("--showpower", r"([\d.]+)\s*W")
        temp_c = self._parse_rocm_smi(
            "--showtemp", r"([\d.]+)\s*c", keyword="edge"
        )
        return PowerSample(
            timestamp=time.time(),
            gpu_index=self.gpu_index,
            power_w=power_w,
            temperature_c=temp_c,
        )

    def _parse_rocm_smi(
        self, flag: str, pattern: str, keyword: str = ""
    ) -> float:
        try:
            out = subprocess.check_output(
                f"rocm-smi -d {self.gpu_index} {flag}",
                shell=True,
                text=True,
                timeout=5,
            )
            for line in out.splitlines():
                if keyword and keyword.lower() not in line.lower():
                    continue
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    return float(m.group(1))
        except Exception:
            pass
        return 0.0

    def close(self):
        if self._use_amdsmi:
            try:
                from amdsmi import amdsmi_shut_down

                amdsmi_shut_down()
            except Exception:
                pass


class PowerSamplerThread:
    """Background thread that samples GPU power at a fixed interval."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 1.0):
        self.monitor = AMDPowerMonitor(gpu_index)
        self.accumulator = EnergyAccumulator()
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> EnergyAccumulator:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.monitor.close()
        return self.accumulator

    def snapshot(self) -> dict:
        """Thread-safe snapshot of current energy stats."""
        with self._lock:
            return self.accumulator.to_dict()

    def _loop(self):
        while not self._stop.is_set():
            sample = self.monitor.read()
            with self._lock:
                self.accumulator.add(sample)
            self._stop.wait(self.interval_s)


if __name__ == "__main__":
    print("Sampling GPU 0 power for 10 seconds...")
    sampler = PowerSamplerThread(gpu_index=0, interval_s=1.0)
    sampler.start()
    time.sleep(10)
    acc = sampler.stop()
    print(f"Samples:    {len(acc.samples)}")
    print(f"Avg power:  {acc.avg_power_w:.1f} W")
    print(f"Peak power: {acc.peak_power_w:.1f} W")
    print(f"Energy:     {acc.total_joules:.1f} J ({acc.total_kwh:.6f} kWh)")
    print(f"Duration:   {acc.duration_s:.1f} s")
