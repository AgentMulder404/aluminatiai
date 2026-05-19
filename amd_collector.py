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
AMD GPU Metrics Collector using amdsmi / rocm-smi CLI fallback.

Drop-in replacement for collector.GPUCollector on AMD ROCm systems.
Returns the same GPUMetrics dataclass so the agent loop is backend-agnostic.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from collector import GPUMetrics

logger = logging.getLogger(__name__)

try:
    import amdsmi
    AMDSMI_AVAILABLE = True
except ImportError:
    AMDSMI_AVAILABLE = False

_AMD_DEVICE_NAMES: dict[str, str] = {
    "0x74a0": "AMD Instinct MI300X",
    "0x74a1": "AMD Instinct MI300X",
    "0x74b5": "AMD Instinct MI300X",
    "0x7408": "AMD Instinct MI300A",
    "0x740c": "AMD Instinct MI300A",
    "0x74a5": "AMD Instinct MI325X",
    "0x7410": "AMD Instinct MI250X",
    "0x740f": "AMD Instinct MI250",
    "0x738c": "AMD Instinct MI200",
    "0x738e": "AMD Instinct MI210",
    "0x7388": "AMD Instinct MI100",
}


def _rocm_smi_available() -> bool:
    try:
        subprocess.run(
            ["rocm-smi", "--version"],
            capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


ROCM_CLI_AVAILABLE = _rocm_smi_available()


class AMDGPUCollector:
    """
    Collects metrics from AMD GPUs via amdsmi (preferred) or rocm-smi CLI.

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
        self._use_amdsmi = False

        self.last_sample_time: dict[int, float] = {}
        self.last_power_draw: dict[int, float] = {}

        self._initialize()

    def _initialize(self):
        if AMDSMI_AVAILABLE:
            try:
                self._init_amdsmi()
                return
            except Exception as exc:
                logger.warning("amdsmi init failed (%s) — trying rocm-smi CLI", exc)

        if ROCM_CLI_AVAILABLE:
            self._init_cli()
            return

        raise RuntimeError(
            "No AMD GPU backend available. "
            "Install amdsmi Python bindings or ensure rocm-smi is in PATH."
        )

    # ── amdsmi backend ──────────────────────────────────────────────────

    def _init_amdsmi(self):
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            raise RuntimeError("amdsmi found 0 GPU processors")

        self.gpu_count = len(handles)
        self.gpu_handles = handles
        self._use_amdsmi = True

        for i, h in enumerate(handles):
            uuid = self._amdsmi_uuid(h, i)
            name = self._amdsmi_name(h, i)
            mem_total_mb = self._init_vram_total(i)
            self.gpu_info.append({
                "index": i, "uuid": uuid, "name": name,
                "mem_total_mb": mem_total_mb,
            })

        self.gpu_uuids = [g["uuid"] for g in self.gpu_info]
        self.initialized = True
        logger.info("AMD collector: amdsmi — %d GPU(s)", self.gpu_count)

    @staticmethod
    def _amdsmi_uuid(handle, idx: int) -> str:
        try:
            return amdsmi.amdsmi_get_gpu_device_uuid(handle)
        except Exception:
            return f"AMD-GPU-{idx}"

    @staticmethod
    def _amdsmi_name(handle, idx: int) -> str:
        device_id = ""
        try:
            info = amdsmi.amdsmi_get_gpu_asic_info(handle)
            name = info.get("market_name", "") or ""
            if name and not name.startswith("0x"):
                return name
            device_id = info.get("device_id", "")
        except Exception:
            pass
        if device_id in _AMD_DEVICE_NAMES:
            return _AMD_DEVICE_NAMES[device_id]
        # Try rocm-smi for Card Model / GFX Version
        try:
            out = subprocess.check_output(
                ["rocm-smi", "-d", str(idx), "--showproductname"],
                text=True, timeout=5,
            )
            for line in out.splitlines():
                m = re.search(r"Card Model:\s*(0x\w+)", line)
                if m and m.group(1) in _AMD_DEVICE_NAMES:
                    return _AMD_DEVICE_NAMES[m.group(1)]
        except Exception:
            pass
        return f"AMD GPU {idx}"

    @staticmethod
    def _init_vram_total(idx: int) -> float:
        try:
            out = subprocess.check_output(
                ["rocm-smi", "-d", str(idx), "--showmeminfo", "vram"],
                text=True, timeout=10,
            )
            for line in out.splitlines():
                m = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", line)
                if m:
                    return float(m.group(1)) / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def _collect_amdsmi(
        self, gpu_index: int, timestamp: str, current_time: float
    ) -> GPUMetrics:
        h = self.gpu_handles[gpu_index]

        power_w, power_limit_w = 0.0, 0.0
        try:
            pwr = amdsmi.amdsmi_get_power_info(h)
            power_w = float(pwr.get("current_socket_power", 0))
            power_limit_w = float(
                pwr.get("power_limit", 0)
                or pwr.get("default_power_cap", 0)
            )
        except Exception:
            pass

        energy_delta = self._calc_energy_delta(gpu_index, current_time, power_w)

        temperature_c = 0
        try:
            temperature_c = int(
                amdsmi.amdsmi_get_temp_metric(
                    h,
                    amdsmi.AmdSmiTemperatureType.HOTSPOT,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
            )
        except Exception:
            pass

        util_gpu, util_mem = 0, 0
        try:
            activity = amdsmi.amdsmi_get_gpu_activity(h)
            util_gpu = int(activity.get("gfx_activity", 0))
            util_mem = int(activity.get("umc_activity", 0))
        except Exception:
            pass

        mem_total_mb = self.gpu_info[gpu_index].get("mem_total_mb", 0.0)
        mem_used_mb = 0.0
        try:
            vram = amdsmi.amdsmi_get_gpu_vram_usage(h)
            mem_used_mb = float(vram.get("vram_used", 0)) / (1024 * 1024)
            if mem_total_mb == 0.0:
                mem_total_mb = float(vram.get("vram_total", 0)) / (1024 * 1024)
        except Exception:
            pass
        if mem_used_mb == 0.0:
            cli_out = self._cli_run(["rocm-smi", "-d", str(gpu_index), "--showmeminfo", "vram"])
            vram_used_b = self._find_float(cli_out, r"VRAM Total Used Memory \(B\):\s*(\d+)")
            mem_used_mb = vram_used_b / (1024 * 1024) if vram_used_b else 0.0

        processes: list[dict] = []
        try:
            for p in amdsmi.amdsmi_get_gpu_process_list(h):
                if isinstance(p, dict):
                    processes.append({
                        "pid": p.get("pid", 0),
                        "used_gpu_memory": p.get("vram_usage", 0),
                    })
        except Exception:
            pass

        return GPUMetrics(
            timestamp=timestamp,
            gpu_index=gpu_index,
            gpu_uuid=self.gpu_info[gpu_index]["uuid"],
            gpu_name=self.gpu_info[gpu_index]["name"],
            power_draw_w=power_w,
            power_limit_w=power_limit_w,
            energy_delta_j=energy_delta,
            utilization_gpu_pct=util_gpu,
            utilization_memory_pct=util_mem,
            temperature_c=temperature_c,
            fan_speed_pct=0,
            memory_used_mb=mem_used_mb,
            memory_total_mb=mem_total_mb,
            processes=processes,
        )

    # ── rocm-smi CLI backend ────────────────────────────────────────────

    def _init_cli(self):
        out = self._cli_run(["rocm-smi", "--showid"])
        gpu_lines = re.findall(r"GPU\[(\d+)\]", out)
        self.gpu_count = max(len(set(gpu_lines)), 1)

        for i in range(self.gpu_count):
            init_out = self._cli_run(
                ["rocm-smi", "-d", str(i), "--showuniqueid", "--showproductname", "--showmaxpower"]
            )
            uuid = self._find_str(init_out, r"Unique ID:\s*(\S+)") or f"AMD-CLI-{i}"
            name = self._find_str(init_out, r"Card Series:\s*(.+)") or f"AMD GPU {i}"
            self.gpu_handles.append(i)
            self.gpu_info.append({
                "index": i,
                "uuid": uuid.strip(),
                "name": name.strip(),
            })

        self.gpu_uuids = [g["uuid"] for g in self.gpu_info]
        self.initialized = True
        logger.info("AMD collector: rocm-smi CLI — %d GPU(s)", self.gpu_count)

    def _collect_cli(
        self, gpu_index: int, timestamp: str, current_time: float
    ) -> GPUMetrics:
        out = self._cli_run([
            "rocm-smi", "-d", str(gpu_index),
            "--showpower", "--showmaxpower", "--showtemp", "--showuse", "--showmeminfo", "vram",
        ])

        power_w = self._find_float(out, r"Average Graphics Package Power \(W\):\s*([\d.]+)")
        if power_w == 0.0:
            power_w = self._find_float(out, r"([\d.]+)\s*W", keyword="current")
            if power_w == 0.0:
                power_w = self._find_float(out, r"([\d.]+)\s*W")

        power_limit_w = self._find_float(out, r"Max Graphics Package Power \(W\):\s*([\d.]+)")

        energy_delta = self._calc_energy_delta(gpu_index, current_time, power_w)

        temp_c = int(self._find_float(out, r"([\d.]+)\s*c", keyword="edge"))
        if temp_c == 0:
            temp_c = int(self._find_float(out, r"([\d.]+)\s*c", keyword="junction"))

        util_gpu = int(self._find_float(out, r"GPU use \(%\):\s*([\d.]+)"))

        vram_total_b = self._find_float(out, r"VRAM Total Memory \(B\):\s*([\d]+)")
        vram_used_b = self._find_float(out, r"VRAM Total Used Memory \(B\):\s*([\d]+)")
        mem_used_mb = vram_used_b / (1024 * 1024) if vram_used_b else 0.0
        mem_total_mb = vram_total_b / (1024 * 1024) if vram_total_b else 0.0

        return GPUMetrics(
            timestamp=timestamp,
            gpu_index=gpu_index,
            gpu_uuid=self.gpu_info[gpu_index]["uuid"],
            gpu_name=self.gpu_info[gpu_index]["name"],
            power_draw_w=power_w,
            power_limit_w=power_limit_w,
            energy_delta_j=energy_delta,
            utilization_gpu_pct=util_gpu,
            utilization_memory_pct=0,
            temperature_c=temp_c,
            fan_speed_pct=0,
            memory_used_mb=mem_used_mb,
            memory_total_mb=mem_total_mb,
            processes=[],
        )

    # ── Shared helpers ──────────────────────────────────────────────────

    def collect(self) -> List[GPUMetrics]:
        if not self.initialized:
            raise RuntimeError("AMD collector not initialized")

        metrics = []
        timestamp = datetime.now(timezone.utc).isoformat()
        current_time = time.time()

        for i in range(self.gpu_count):
            try:
                if self._use_amdsmi:
                    m = self._collect_amdsmi(i, timestamp, current_time)
                else:
                    m = self._collect_cli(i, timestamp, current_time)
                metrics.append(m)
            except Exception as exc:
                logger.warning("Failed to collect AMD GPU %d: %s", i, exc)

        return metrics

    def _calc_energy_delta(
        self, gpu_index: int, current_time: float, power_w: float
    ) -> Optional[float]:
        energy_delta = None
        if gpu_index in self.last_sample_time:
            dt = current_time - self.last_sample_time[gpu_index]
            avg_power = (power_w + self.last_power_draw[gpu_index]) / 2.0
            energy_delta = avg_power * dt
        self.last_sample_time[gpu_index] = current_time
        self.last_power_draw[gpu_index] = power_w
        return energy_delta

    @staticmethod
    def _cli_run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, text=True, timeout=10, stderr=subprocess.DEVNULL,
            )
        except Exception:
            return ""

    @staticmethod
    def _find_float(text: str, pattern: str, keyword: str = "") -> float:
        for line in text.splitlines():
            if keyword and keyword.lower() not in line.lower():
                continue
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return 0.0

    @staticmethod
    def _find_str(text: str, pattern: str) -> Optional[str]:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1) if m else None

    def get_gpu_count(self) -> int:
        return self.gpu_count

    def get_gpu_info(self) -> List[Dict]:
        return self.gpu_info

    def shutdown(self):
        if self.initialized and self._use_amdsmi:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass
        self.initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


if __name__ == "__main__":
    print("Testing AMD GPU Collector...")
    try:
        with AMDGPUCollector() as collector:
            print(f"Found {collector.get_gpu_count()} AMD GPUs:")
            for info in collector.get_gpu_info():
                print(f"  GPU {info['index']}: {info['name']} ({info['uuid'][:20]}...)")

            print("\nCollecting 3 samples (2s intervals)...")
            for i in range(3):
                metrics = collector.collect()
                for m in metrics:
                    ej = f"{m.energy_delta_j:.1f}J" if m.energy_delta_j else "N/A"
                    print(
                        f"  GPU {m.gpu_index}: {m.power_draw_w:.1f}W, "
                        f"{m.utilization_gpu_pct}% util, "
                        f"{m.temperature_c}°C, "
                        f"{m.memory_used_mb:.0f}/{m.memory_total_mb:.0f} MB, "
                        f"energy={ej}"
                    )
                if i < 2:
                    time.sleep(2)

            print("\n  AMD collector test passed!")
    except Exception as e:
        print(f"Error: {e}")
