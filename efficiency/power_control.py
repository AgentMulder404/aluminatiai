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
NVML power limit management utilities.

Provides safe get/set of GPU power management limits. Falls back to
nvidia-smi when NVML persistence mode isn't enabled (e.g. Colab).
"""
import subprocess

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore[assignment]


def set_power_limit(gpu_index: int, watts: int, quiet: bool = False) -> bool:
    """
    Set GPU power limit via NVML.

    Falls back to nvidia-smi if NVML persistence mode isn't enabled.
    Requires root/sudo on bare metal.

    Args:
        quiet: If True, suppress all print output (for pre-flight checks).
    """
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, watts * 1000)  # mW
            actual = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            if not quiet:
                print(f"  Power limit set to {actual:.0f}W via NVML")
            return True
        except pynvml.NVMLError:
            pass

    # Fallback: nvidia-smi (works on Colab without root)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "-pl", str(watts)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            if not quiet:
                print(f"  Power limit set to {watts}W via nvidia-smi")
            return True
        else:
            if not quiet:
                print(f"  WARNING: nvidia-smi -pl failed: {result.stderr.strip()}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        if not quiet:
            print(f"  WARNING: Could not set power limit: {e}")
        return False


def get_power_limit(gpu_index: int = 0) -> int:
    """Read current power management limit in watts."""
    if pynvml is None:
        return 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return int(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000)
    except pynvml.NVMLError:
        return 0


def get_default_power_limit(gpu_index: int = 0) -> int:
    """Read the factory default power limit in watts."""
    if pynvml is None:
        return 400
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return int(pynvml.nvmlDeviceGetDefaultPowerManagementLimit(handle) / 1000)
    except pynvml.NVMLError:
        return 400  # A100 SXM4 default
