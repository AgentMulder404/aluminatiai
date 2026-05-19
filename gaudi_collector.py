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
Intel Gaudi AI Accelerator Collector — energy monitoring for Gaudi2/3.

Two-tier backend:
  1. Primary: pyhlml Python SDK (ships with SynapseAI driver) — direct device
     queries for power, temperature, utilization, HBM memory, UUID.
  2. Fallback: hl-smi CLI in CSV mode — parses
     `hl-smi -Q index,name,uuid,power.draw,temperature.aip,utilization.aip,
              memory.used,memory.total -f csv,noheader,nounits`

Returns the same GPUMetrics dataclass so the agent loop is backend-agnostic.

Note: hl-smi `-Q power.draw` reports only the 54V rail on Gaudi2/3. The
pyhlml SDK `hlmlDeviceGetPowerUsage()` reports the same 54V-only value.
The combined 54V+12V power is only visible in the default table output.
"""
from __future__ import annotations

import csv
import io
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

# Device name → friendly name mapping
_GAUDI_NAMES: dict[str, str] = {
    "HL-205":  "Intel Gaudi",
    "HL-225":  "Intel Gaudi2",
    "HL-325":  "Intel Gaudi3",
    "HL-325L": "Intel Gaudi3",
}

# Try importing pyhlml (ships with SynapseAI)
# The newer habana-pyhlml uses snake_case, older pyhlml uses camelCase.
_PYHLML = False
_PYHLML_SNAKE = False  # True if using snake_case API

try:
    import pyhlml
    _PYHLML = True
    # Detect API style: snake_case (new) vs camelCase (old)
    if hasattr(pyhlml, "hlml_init"):
        _PYHLML_SNAKE = True
    elif hasattr(pyhlml, "hlmlInit"):
        _PYHLML_SNAKE = False
    else:
        _PYHLML = False
        logger.debug("pyhlml imported but missing init function")
except ImportError:
    pass


def _hl_smi_path() -> str:
    return os.getenv("HL_SMI_PATH", "hl-smi")


def _hl_smi_available() -> bool:
    path = _hl_smi_path()
    if shutil.which(path):
        return True
    try:
        subprocess.run(
            [path, "--version"], capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ── pyhlml wrapper (normalizes camelCase / snake_case) ───────────────────────

def _hlml_call(func_name: str, *args):
    """Call a pyhlml function, handling both API styles."""
    if _PYHLML_SNAKE:
        # camelCase → snake_case: hlmlDeviceGetCount → hlml_device_get_count
        snake = re.sub(r'([A-Z])', r'_\1', func_name).lower()
        if snake.startswith("_"):
            snake = snake[1:]
        fn = getattr(pyhlml, snake, None)
    else:
        fn = getattr(pyhlml, func_name, None)

    if fn is None:
        raise AttributeError(f"pyhlml has no function {func_name}")
    return fn(*args)


class _PyHLMLBackend:
    """Collect Gaudi metrics via pyhlml SDK."""

    def __init__(self):
        _hlml_call("hlmlInit")
        self.device_count = _hlml_call("hlmlDeviceGetCount")
        self.handles = []
        self.device_info: list[dict] = []

        for i in range(self.device_count):
            handle = _hlml_call("hlmlDeviceGetHandleByIndex", i)
            self.handles.append(handle)

            name = _hlml_call("hlmlDeviceGetName", handle)
            uuid = _hlml_call("hlmlDeviceGetUUID", handle)
            friendly = _GAUDI_NAMES.get(name, name)

            self.device_info.append({
                "index": i,
                "uuid": uuid,
                "name": friendly,
                "raw_name": name,
            })

    def collect_device(self, index: int) -> dict:
        """Collect metrics for one device. Returns dict of raw values."""
        handle = self.handles[index]

        power_mw = _hlml_call("hlmlDeviceGetPowerUsage", handle)
        try:
            power_limit_mw = _hlml_call("hlmlDeviceGetPowerManagementLimit", handle)
        except Exception:
            power_limit_mw = 0

        temp_c = _hlml_call("hlmlDeviceGetTemperature", handle, 0)
        util_pct = _hlml_call("hlmlDeviceGetUtilizationRates", handle)
        mem = _hlml_call("hlmlDeviceGetMemoryInfo", handle)

        return {
            "power_w": power_mw / 1000.0,
            "power_limit_w": power_limit_mw / 1000.0,
            "temperature_c": temp_c,
            "utilization_pct": util_pct,
            "memory_used_mb": mem.used / (1024 * 1024),
            "memory_total_mb": mem.total / (1024 * 1024),
        }

    def shutdown(self):
        try:
            _hlml_call("hlmlShutdown")
        except Exception:
            pass


class _HLSMIBackend:
    """Collect Gaudi metrics via hl-smi CLI in CSV mode."""

    _QUERY_FIELDS = "index,name,uuid,power.draw,temperature.aip,utilization.aip,memory.used,memory.total"

    def __init__(self):
        self.hl_smi = _hl_smi_path()
        self.device_count = 0
        self.device_info: list[dict] = []

        # Discovery: run once to get device list
        rows = self._query_csv()
        if not rows:
            raise RuntimeError("hl-smi returned no devices")

        self.device_count = len(rows)
        for row in rows:
            raw_name = row.get("name", "Unknown Gaudi")
            friendly = _GAUDI_NAMES.get(raw_name, raw_name)
            self.device_info.append({
                "index": int(row.get("index", 0)),
                "uuid": row.get("uuid", f"gaudi-{row.get('index', 0)}"),
                "name": friendly,
                "raw_name": raw_name,
            })

    def _query_csv(self) -> list[dict]:
        """Run hl-smi with CSV output, return list of dicts."""
        try:
            result = subprocess.run(
                [self.hl_smi, "-Q", self._QUERY_FIELDS, "-f", "csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("hl-smi query failed: %s", exc)
            return []

        if result.returncode != 0:
            logger.warning("hl-smi returned code %d: %s", result.returncode, result.stderr.strip())
            return []

        field_names = [f.strip() for f in self._QUERY_FIELDS.split(",")]
        rows = []
        reader = csv.reader(io.StringIO(result.stdout.strip()))
        for csv_row in reader:
            if len(csv_row) < len(field_names):
                continue
            row = {}
            for name, val in zip(field_names, csv_row):
                row[name] = val.strip()
            rows.append(row)

        return rows

    def collect_all(self) -> list[dict]:
        """Collect metrics for all devices. Returns list of dicts."""
        rows = self._query_csv()
        results = []
        for row in rows:
            results.append({
                "index": int(row.get("index", 0)),
                "power_w": _safe_float(row.get("power.draw", "0")),
                "power_limit_w": 0.0,  # not available via -Q
                "temperature_c": int(_safe_float(row.get("temperature.aip", "0"))),
                "utilization_pct": int(_safe_float(row.get("utilization.aip", "0"))),
                "memory_used_mb": _safe_float(row.get("memory.used", "0")) / (1024 * 1024),
                "memory_total_mb": _safe_float(row.get("memory.total", "0")) / (1024 * 1024),
            })
        return results

    def _parse_power_limit_from_table(self) -> dict[int, float]:
        """Parse default table output to get power limits (not in CSV mode)."""
        try:
            result = subprocess.run(
                [self.hl_smi], capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return {}

        limits: dict[int, float] = {}
        # Match lines like "| N/A   29C   N/A    93W / 600W |"
        for m in re.finditer(r"\|\s*N/A\s+\d+C\s+N/A\s+(\d+)W\s*/\s*(\d+)W\s*\|", result.stdout):
            idx = len(limits)
            limits[idx] = float(m.group(2))
        return limits

    def shutdown(self):
        pass


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


# Default power limits when hl-smi table parse fails
_DEFAULT_POWER_LIMITS: dict[str, float] = {
    "HL-205":  300.0,
    "HL-225":  600.0,
    "HL-325":  900.0,
    "HL-325L": 900.0,
}


class GaudiCollector:
    """
    Collects metrics from Intel Gaudi AI accelerators.

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

        self._use_pyhlml = False
        self._pyhlml_backend: Optional[_PyHLMLBackend] = None
        self._hlsmi_backend: Optional[_HLSMIBackend] = None
        self._power_limits: dict[int, float] = {}

        self._initialize()

    def _initialize(self):
        # Try pyhlml SDK first
        if _PYHLML:
            try:
                self._pyhlml_backend = _PyHLMLBackend()
                self._use_pyhlml = True
                self.gpu_count = self._pyhlml_backend.device_count
                self.gpu_info = self._pyhlml_backend.device_info
                logger.info("Gaudi collector: pyhlml SDK, %d device(s)", self.gpu_count)
            except Exception as exc:
                logger.warning("pyhlml init failed: %s — trying hl-smi", exc)
                self._pyhlml_backend = None

        # Fall back to hl-smi CLI
        if not self._use_pyhlml:
            if not _hl_smi_available():
                raise RuntimeError(
                    "No Gaudi backend available — install pyhlml (SynapseAI) "
                    "or ensure hl-smi is in PATH"
                )
            self._hlsmi_backend = _HLSMIBackend()
            self.gpu_count = self._hlsmi_backend.device_count
            self.gpu_info = self._hlsmi_backend.device_info

            # Get power limits from table output
            self._power_limits = self._hlsmi_backend._parse_power_limit_from_table()
            logger.info("Gaudi collector: hl-smi CLI, %d device(s)", self.gpu_count)

        # Populate handles and UUIDs
        self.gpu_handles = list(range(self.gpu_count))
        self.gpu_uuids = [info["uuid"] for info in self.gpu_info]

        # Set default power limits for devices missing them
        for i, info in enumerate(self.gpu_info):
            if i not in self._power_limits:
                raw = info.get("raw_name", "")
                self._power_limits[i] = _DEFAULT_POWER_LIMITS.get(raw, 600.0)

        self.initialized = True

    def get_gpu_count(self) -> int:
        return self.gpu_count

    def get_gpu_info(self) -> List[Dict]:
        return self.gpu_info

    def collect(self) -> List[GPUMetrics]:
        """Collect current metrics from all Gaudi devices."""
        if not self.initialized:
            raise RuntimeError("Gaudi collector not initialized")

        timestamp = datetime.now(timezone.utc).isoformat()
        now = time.monotonic()
        metrics: List[GPUMetrics] = []

        if self._use_pyhlml and self._pyhlml_backend:
            for i in range(self.gpu_count):
                try:
                    data = self._pyhlml_backend.collect_device(i)
                except Exception as exc:
                    logger.warning("pyhlml collect failed for device %d: %s", i, exc)
                    continue
                metrics.append(self._build_metrics(i, data, timestamp, now))
        elif self._hlsmi_backend:
            all_data = self._hlsmi_backend.collect_all()
            for data in all_data:
                idx = data.get("index", len(metrics))
                metrics.append(self._build_metrics(idx, data, timestamp, now))

        return metrics

    def _build_metrics(self, index: int, data: dict, timestamp: str, now: float) -> GPUMetrics:
        info = self.gpu_info[index] if index < len(self.gpu_info) else {"uuid": f"gaudi-{index}", "name": "Intel Gaudi"}
        power_w = round(data.get("power_w", 0.0), 2)

        # Energy delta
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
            power_limit = self._power_limits.get(index, 600.0)

        return GPUMetrics(
            timestamp=timestamp,
            gpu_index=index,
            gpu_uuid=info["uuid"],
            gpu_name=info["name"],
            power_draw_w=power_w,
            power_limit_w=power_limit,
            energy_delta_j=energy_delta_j,
            utilization_gpu_pct=int(data.get("utilization_pct", 0)),
            utilization_memory_pct=0,
            temperature_c=int(data.get("temperature_c", 0)),
            fan_speed_pct=0,
            memory_used_mb=round(data.get("memory_used_mb", 0.0), 1),
            memory_total_mb=round(data.get("memory_total_mb", 0.0), 1),
        )

    def shutdown(self):
        if self._pyhlml_backend:
            self._pyhlml_backend.shutdown()
        if self._hlsmi_backend:
            self._hlsmi_backend.shutdown()
        self.initialized = False
        logger.info("Gaudi collector shut down")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
