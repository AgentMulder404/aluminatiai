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
Intel Arc / Data Center GPU Collector — energy monitoring for discrete Intel GPUs.

Two-tier backend:
  1. Primary: xpu-smi CLI — ``xpu-smi dump`` for metrics, ``xpu-smi discovery``
     for device enumeration.  Requires the ``xpu-smi`` tool shipped with Intel
     oneAPI Base Toolkit or the standalone GPU driver package.
  2. Fallback: hwmon sysfs (xe/i915 kernel driver) for power/energy/temperature
     plus ``intel_gpu_top -J`` for utilization.

Returns the same GPUMetrics dataclass so the agent loop is backend-agnostic.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from collector import GPUMetrics

logger = logging.getLogger(__name__)

# PCI device-ID → friendly name
_ARC_DEVICE_NAMES: dict[str, str] = {
    "0x56a0": "Intel Arc A770",
    "0x56a1": "Intel Arc A770",
    "0x56a5": "Intel Arc A750",
    "0x56a6": "Intel Arc A750",
    "0x56b0": "Intel Arc A580",
    "0xe20b": "Intel Arc B580",
    "0xe20c": "Intel Arc B580",
    "0x56c0": "Intel Data Center GPU Flex 170",
    "0x56c1": "Intel Data Center GPU Flex 140",
    "0x0bd5": "Intel Data Center GPU Max 1550",
    "0x0bd6": "Intel Data Center GPU Max 1100",
    "0x0bd9": "Intel Data Center GPU Max 1350",
}

# Default TDP when we can't read it from the driver
_DEFAULT_TDP: dict[str, float] = {
    "Intel Arc A770": 225.0,
    "Intel Arc A750": 225.0,
    "Intel Arc A580": 185.0,
    "Intel Arc B580": 190.0,
    "Intel Data Center GPU Flex 170": 150.0,
    "Intel Data Center GPU Flex 140": 75.0,
    "Intel Data Center GPU Max 1550": 600.0,
    "Intel Data Center GPU Max 1100": 300.0,
    "Intel Data Center GPU Max 1350": 450.0,
}


def _xpu_smi_path() -> str:
    return os.getenv("XPU_SMI_PATH", "xpu-smi")


def _xpu_smi_available() -> bool:
    path = _xpu_smi_path()
    if shutil.which(path):
        return True
    try:
        subprocess.run(
            [path, "--version"], capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _safe_float(s) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(s) -> int:
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


# ── xpu-smi backend ────────────────────────────────────────────────────────

class _XPUSMIBackend:
    """Collect Intel GPU metrics via xpu-smi CLI."""

    # xpu-smi dump metric IDs:
    #   0 = GPU Utilization (%)
    #   1 = GPU Power (W)
    #   2 = GPU Frequency (MHz)
    #   3 = GPU Core Temperature (C)
    #   4 = GPU Memory Used (bytes)
    #   5 = GPU Memory Utilization (%)
    #  18 = GPU Memory Bandwidth Utilization (%)
    _DUMP_METRICS = "0,1,2,3,4,5,18"

    def __init__(self):
        self.xpu_smi = _xpu_smi_path()
        self.device_count = 0
        self.device_info: list[dict] = []

        self._discover_devices()

    def _discover_devices(self):
        """Run xpu-smi discovery to enumerate devices."""
        try:
            result = subprocess.run(
                [self.xpu_smi, "discovery", "--json"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            raise RuntimeError(f"xpu-smi discovery failed: {exc}") from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"xpu-smi discovery returned code {result.returncode}: "
                f"{result.stderr.strip()}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"xpu-smi discovery JSON parse error: {exc}") from exc

        devices = data if isinstance(data, list) else data.get("device_list", [])

        for dev in devices:
            dev_id = str(dev.get("device_id", len(self.device_info)))
            pci_id = dev.get("pci_device_id", "")
            name = dev.get("device_name", "")

            if not name or name.startswith("0x"):
                name = _ARC_DEVICE_NAMES.get(pci_id, f"Intel GPU {dev_id}")

            uuid = dev.get("uuid", "") or dev.get("serial_number", "") or f"IGPU-{dev_id}"

            mem_bytes = dev.get("max_mem_alloc_size_byte", 0) or dev.get("memory_physical_size_byte", 0)
            mem_total_mb = _safe_float(mem_bytes) / (1024 * 1024) if mem_bytes else 0.0

            self.device_info.append({
                "index": int(dev_id),
                "uuid": uuid,
                "name": name,
                "pci_device_id": pci_id,
                "mem_total_mb": mem_total_mb,
            })

        self.device_count = len(self.device_info)
        if self.device_count == 0:
            raise RuntimeError("xpu-smi found no devices")

    def collect_device(self, device_id: int) -> dict:
        """Run xpu-smi dump for a single device, return raw metrics dict."""
        try:
            result = subprocess.run(
                [
                    self.xpu_smi, "dump",
                    "-d", str(device_id),
                    "-m", self._DUMP_METRICS,
                    "-n", "1",
                ],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("xpu-smi dump failed for device %d: %s", device_id, exc)
            return {}

        if result.returncode != 0:
            logger.warning(
                "xpu-smi dump device %d returned %d: %s",
                device_id, result.returncode, result.stderr.strip(),
            )
            return {}

        return self._parse_dump_output(result.stdout)

    @staticmethod
    def _parse_dump_output(output: str) -> dict:
        """
        Parse xpu-smi dump CSV output.

        Example line:
        Timestamp, DeviceId, GPU Utilization (%), GPU Power (W), GPU Frequency (MHz), GPU Core Temperature (C), GPU Memory Used (MiB), GPU Memory Utilization (%), GPU Memory Bandwidth Utilization (%)
        06:30:15.000,    0,  45.00,  120.50,  2100,  65,  4096.00,  32.00,  18.50
        """
        metrics: dict = {}

        for line in output.strip().splitlines():
            if line.startswith("Timestamp") or line.startswith("Device"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue

            # parts[0]=timestamp, parts[1]=device_id, then metric values
            metrics["utilization_pct"] = _safe_int(parts[2])
            metrics["power_w"] = _safe_float(parts[3])
            metrics["frequency_mhz"] = _safe_int(parts[4])
            metrics["temperature_c"] = _safe_int(parts[5])
            metrics["memory_used_mb"] = _safe_float(parts[6])
            metrics["memory_util_pct"] = _safe_int(parts[7])
            if len(parts) > 8:
                metrics["memory_bw_util_pct"] = _safe_int(parts[8])
            break

        return metrics

    def get_power_limit(self, device_id: int) -> float:
        """Query power limit from xpu-smi stats."""
        try:
            result = subprocess.run(
                [self.xpu_smi, "stats", "-d", str(device_id), "--json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for entry in (data if isinstance(data, list) else [data]):
                    limit = entry.get("power_limit", 0) or entry.get("default_power_limit", 0)
                    if limit and float(limit) > 0:
                        return float(limit)
        except Exception:
            pass
        return 0.0

    def shutdown(self):
        pass


# ── hwmon + intel_gpu_top fallback ──────────────────────────────────────────

class _HWMonBackend:
    """
    Collect Intel GPU metrics via sysfs hwmon (xe/i915 driver) and
    intel_gpu_top for utilization.
    """

    def __init__(self):
        self.device_count = 0
        self.device_info: list[dict] = []
        self._hwmon_paths: list[str] = []
        self._energy_counters: dict[int, tuple[float, float]] = {}

        self._discover_devices()

    def _discover_devices(self):
        """Find Intel GPU hwmon directories via sysfs."""
        # xe driver: /sys/class/drm/card*/device/hwmon/hwmon*/
        # i915 driver: same path structure
        for card_dir in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
            hwmon_parent = os.path.join(card_dir, "device", "hwmon")
            if not os.path.isdir(hwmon_parent):
                continue

            hwmon_dirs = sorted(glob.glob(os.path.join(hwmon_parent, "hwmon*")))
            if not hwmon_dirs:
                continue

            hwmon = hwmon_dirs[0]

            # Verify this is an Intel GPU by checking for energy or power files
            has_energy = os.path.exists(os.path.join(hwmon, "energy1_input"))
            has_power = os.path.exists(os.path.join(hwmon, "power1_input"))
            if not has_energy and not has_power:
                continue

            # Check driver name to ensure it's Intel (xe or i915)
            driver_link = os.path.join(card_dir, "device", "driver")
            driver_name = ""
            if os.path.islink(driver_link):
                driver_name = os.path.basename(os.readlink(driver_link))

            if driver_name not in ("xe", "i915", ""):
                continue

            idx = len(self._hwmon_paths)
            self._hwmon_paths.append(hwmon)

            # Read device name from sysfs
            pci_device_path = os.path.join(card_dir, "device", "device")
            pci_id = self._read_sysfs(pci_device_path).strip()
            name = _ARC_DEVICE_NAMES.get(pci_id, f"Intel GPU {idx}")

            uuid = f"IGPU-HWMON-{idx}"
            pci_addr_path = os.path.join(card_dir, "device", "uevent")
            pci_addr = self._grep_file(pci_addr_path, "PCI_SLOT_NAME=")
            if pci_addr:
                uuid = f"IGPU-{pci_addr}"

            mem_total_mb = 0.0
            mem_total_path = os.path.join(card_dir, "device", "mem_info_vram_total")
            mem_total_str = self._read_sysfs(mem_total_path)
            if mem_total_str:
                mem_total_mb = _safe_float(mem_total_str) / (1024 * 1024)

            self.device_info.append({
                "index": idx,
                "uuid": uuid,
                "name": name,
                "pci_device_id": pci_id,
                "mem_total_mb": mem_total_mb,
            })

        self.device_count = len(self.device_info)
        if self.device_count == 0:
            raise RuntimeError(
                "No Intel GPU hwmon interfaces found in sysfs"
            )

    def collect_device(self, index: int) -> dict:
        """Read power, temperature, memory from hwmon sysfs."""
        if index >= len(self._hwmon_paths):
            return {}

        hwmon = self._hwmon_paths[index]
        card_dir = os.path.dirname(os.path.dirname(hwmon))
        metrics: dict = {}

        # Power via energy counter delta (µJ)
        energy_file = os.path.join(hwmon, "energy1_input")
        energy_uj_str = self._read_sysfs(energy_file)
        if energy_uj_str:
            energy_uj = _safe_float(energy_uj_str)
            now = time.monotonic()
            if index in self._energy_counters:
                prev_uj, prev_time = self._energy_counters[index]
                dt = now - prev_time
                if dt > 0:
                    delta_uj = energy_uj - prev_uj
                    if delta_uj < 0:
                        delta_uj += 2**32
                    metrics["power_w"] = (delta_uj / 1e6) / dt
            self._energy_counters[index] = (energy_uj, now)
        else:
            power_file = os.path.join(hwmon, "power1_input")
            power_uw_str = self._read_sysfs(power_file)
            if power_uw_str:
                metrics["power_w"] = _safe_float(power_uw_str) / 1e6

        # Power limit (µW)
        limit_file = os.path.join(hwmon, "power1_max")
        limit_str = self._read_sysfs(limit_file)
        if limit_str:
            metrics["power_limit_w"] = _safe_float(limit_str) / 1e6

        # Temperature (milli-°C) — try temp2 (GPU) first, then temp1 (package)
        for temp_file in ("temp2_input", "temp3_input", "temp1_input"):
            temp_str = self._read_sysfs(os.path.join(hwmon, temp_file))
            if temp_str:
                metrics["temperature_c"] = _safe_int(temp_str) // 1000
                break

        # Fan (RPM → approximate %)
        fan_str = self._read_sysfs(os.path.join(hwmon, "fan1_input"))
        if fan_str:
            rpm = _safe_int(fan_str)
            metrics["fan_speed_pct"] = min(100, int(rpm / 30))

        # Memory from card sysfs
        mem_used_path = os.path.join(card_dir, "mem_info_vram_used")
        mem_used_str = self._read_sysfs(mem_used_path)
        if mem_used_str:
            metrics["memory_used_mb"] = _safe_float(mem_used_str) / (1024 * 1024)

        # Utilization via intel_gpu_top (if available)
        util = self._read_gpu_utilization(index)
        if util is not None:
            metrics["utilization_pct"] = util

        return metrics

    @staticmethod
    def _read_gpu_utilization(index: int) -> Optional[int]:
        """
        Get GPU utilization from intel_gpu_top.

        intel_gpu_top -J streams JSON continuously; we run it with timeout
        to get a single sample.
        """
        try:
            result = subprocess.run(
                ["intel_gpu_top", "-J", "-s", "500", "-o", "-"],
                capture_output=True, text=True, timeout=3,
            )
            # intel_gpu_top always exits non-zero when killed by timeout
            for line in result.stdout.strip().splitlines():
                line = line.strip().rstrip(",")
                if not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                    engines = data.get("engines", {})
                    # Sum render/compute engine busy percentages
                    total_busy = 0.0
                    count = 0
                    for eng_name, eng_data in engines.items():
                        if any(k in eng_name.lower() for k in ("render", "compute", "3d")):
                            total_busy += eng_data.get("busy", 0.0)
                            count += 1
                    if count > 0:
                        return min(100, int(total_busy / count))
                except json.JSONDecodeError:
                    continue
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return None

    @staticmethod
    def _read_sysfs(path: str) -> str:
        try:
            with open(path) as f:
                return f.read().strip()
        except (OSError, IOError):
            return ""

    @staticmethod
    def _grep_file(path: str, prefix: str) -> str:
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith(prefix):
                        return line[len(prefix):].strip()
        except (OSError, IOError):
            pass
        return ""

    def shutdown(self):
        pass


# ── Main collector ──────────────────────────────────────────────────────────

class IntelArcCollector:
    """
    Collects metrics from Intel Arc / Data Center GPUs.

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

        self._use_xpu_smi = False
        self._xpu_backend: Optional[_XPUSMIBackend] = None
        self._hwmon_backend: Optional[_HWMonBackend] = None
        self._power_limits: dict[int, float] = {}

        self._initialize()

    @property
    def backend(self) -> str:
        return "xpu-smi" if self._use_xpu_smi else "hwmon"

    def _initialize(self):
        # Try xpu-smi first
        if _xpu_smi_available():
            try:
                self._xpu_backend = _XPUSMIBackend()
                self._use_xpu_smi = True
                self.gpu_count = self._xpu_backend.device_count
                self.gpu_info = self._xpu_backend.device_info

                for i, info in enumerate(self.gpu_info):
                    limit = self._xpu_backend.get_power_limit(info["index"])
                    if limit > 0:
                        self._power_limits[i] = limit

                logger.info("Intel Arc collector: xpu-smi, %d device(s)", self.gpu_count)
            except Exception as exc:
                logger.warning("xpu-smi init failed: %s — trying hwmon", exc)
                self._xpu_backend = None

        # Fall back to hwmon sysfs
        if not self._use_xpu_smi:
            try:
                self._hwmon_backend = _HWMonBackend()
                self.gpu_count = self._hwmon_backend.device_count
                self.gpu_info = self._hwmon_backend.device_info
                logger.info("Intel Arc collector: hwmon sysfs, %d device(s)", self.gpu_count)
            except RuntimeError:
                raise RuntimeError(
                    "No Intel Arc/dGPU backend available — install xpu-smi "
                    "(oneAPI Base Toolkit) or ensure xe/i915 hwmon sysfs is accessible"
                )

        self.gpu_handles = list(range(self.gpu_count))
        self.gpu_uuids = [info["uuid"] for info in self.gpu_info]

        # Fill in default power limits
        for i, info in enumerate(self.gpu_info):
            if i not in self._power_limits:
                self._power_limits[i] = _DEFAULT_TDP.get(info["name"], 225.0)

        self.initialized = True

    def get_gpu_count(self) -> int:
        return self.gpu_count

    def get_gpu_info(self) -> List[Dict]:
        return self.gpu_info

    def collect(self) -> List[GPUMetrics]:
        if not self.initialized:
            raise RuntimeError("Intel Arc collector not initialized")

        timestamp = datetime.now(timezone.utc).isoformat()
        now = time.monotonic()
        metrics: List[GPUMetrics] = []

        for i in range(self.gpu_count):
            try:
                if self._use_xpu_smi and self._xpu_backend:
                    data = self._xpu_backend.collect_device(
                        self.gpu_info[i]["index"]
                    )
                elif self._hwmon_backend:
                    data = self._hwmon_backend.collect_device(i)
                else:
                    continue

                metrics.append(self._build_metrics(i, data, timestamp, now))
            except Exception as exc:
                logger.warning("Intel Arc collect failed for device %d: %s", i, exc)

        return metrics

    def _build_metrics(
        self, index: int, data: dict, timestamp: str, now: float
    ) -> GPUMetrics:
        info = self.gpu_info[index] if index < len(self.gpu_info) else {
            "uuid": f"IGPU-{index}", "name": f"Intel GPU {index}"
        }
        power_w = round(data.get("power_w", 0.0), 2)

        # Energy delta (trapezoidal integration)
        energy_delta_j = None
        last_time = self.last_sample_time.get(index)
        last_power = self.last_power_draw.get(index)
        if last_time is not None and last_power is not None:
            dt = now - last_time
            if dt > 0 and power_w > 0:
                avg_power = (last_power + power_w) / 2.0
                energy_delta_j = round(avg_power * dt, 4)

        self.last_sample_time[index] = now
        self.last_power_draw[index] = power_w

        power_limit = data.get("power_limit_w", 0.0)
        if power_limit <= 0:
            power_limit = self._power_limits.get(index, 225.0)

        mem_total = data.get("memory_total_mb", 0.0)
        if mem_total <= 0:
            mem_total = info.get("mem_total_mb", 0.0)

        return GPUMetrics(
            timestamp=timestamp,
            gpu_index=index,
            gpu_uuid=info["uuid"],
            gpu_name=info["name"],
            power_draw_w=power_w,
            power_limit_w=power_limit,
            energy_delta_j=energy_delta_j,
            utilization_gpu_pct=int(data.get("utilization_pct", 0)),
            utilization_memory_pct=int(data.get("memory_util_pct", 0)),
            temperature_c=int(data.get("temperature_c", 0)),
            fan_speed_pct=int(data.get("fan_speed_pct", 0)),
            memory_used_mb=round(data.get("memory_used_mb", 0.0), 1),
            memory_total_mb=round(mem_total, 1),
        )

    def shutdown(self):
        if self._xpu_backend:
            self._xpu_backend.shutdown()
        if self._hwmon_backend:
            self._hwmon_backend.shutdown()
        self.initialized = False
        logger.info("Intel Arc collector shut down")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
