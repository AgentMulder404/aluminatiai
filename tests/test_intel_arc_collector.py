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
Intel Arc / Data Center GPU collector unit tests.

Tests:
  1. xpu-smi discovery JSON parsing (single device)
  2. xpu-smi discovery JSON parsing (multi-device)
  3. xpu-smi dump CSV output parsing
  4. xpu-smi dump with missing fields
  5. Device name mapping (PCI ID → friendly name)
  6. IntelArcCollector returns valid GPUMetrics via xpu-smi
  7. Energy delta after two samples
  8. Context manager lifecycle
  9. Graceful failure when no backend available
  10. ArchSpec resolution for Intel Arc names
  11. Safe float / int parsing
  12. Default power limits from TDP table
  13. Backend property
  14. hwmon sysfs discovery (mocked)
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Sample xpu-smi outputs ─────────────────────────────────────────────────

XPU_SMI_DISCOVERY_SINGLE = """\
[
  {
    "device_id": "0",
    "device_name": "Intel Arc A770",
    "pci_device_id": "0x56a0",
    "uuid": "ARC-A770-0001-UUID",
    "max_mem_alloc_size_byte": 17179869184
  }
]
"""

XPU_SMI_DISCOVERY_MULTI = """\
[
  {
    "device_id": "0",
    "device_name": "Intel Arc A770",
    "pci_device_id": "0x56a0",
    "uuid": "ARC-A770-0001-UUID",
    "max_mem_alloc_size_byte": 17179869184
  },
  {
    "device_id": "1",
    "device_name": "Intel Arc A750",
    "pci_device_id": "0x56a5",
    "uuid": "ARC-A750-0002-UUID",
    "max_mem_alloc_size_byte": 8589934592
  }
]
"""

XPU_SMI_DISCOVERY_FLEX = """\
[
  {
    "device_id": "0",
    "device_name": "",
    "pci_device_id": "0x56c0",
    "uuid": "FLEX-170-UUID",
    "max_mem_alloc_size_byte": 17179869184
  }
]
"""

XPU_SMI_DISCOVERY_B580 = """\
[
  {
    "device_id": "0",
    "device_name": "Intel Arc B580",
    "pci_device_id": "0xe20b",
    "uuid": "ARC-B580-0001-UUID",
    "max_mem_alloc_size_byte": 12884901888
  }
]
"""

# xpu-smi dump output: header line + data line
# Metrics: 0=util%, 1=power_W, 2=freq_MHz, 3=temp_C, 4=mem_used_MiB, 5=mem_util%, 18=mem_bw_util%
XPU_SMI_DUMP_SINGLE = """\
Timestamp, DeviceId, GPU Utilization (%), GPU Power (W), GPU Frequency (MHz), GPU Core Temperature (C), GPU Memory Used (MiB), GPU Memory Utilization (%), GPU Memory Bandwidth Utilization (%)
06:30:15.000,    0,  45.00,  120.50,  2100,  65,  4096.00,  32.00,  18.50
"""

XPU_SMI_DUMP_IDLE = """\
Timestamp, DeviceId, GPU Utilization (%), GPU Power (W), GPU Frequency (MHz), GPU Core Temperature (C), GPU Memory Used (MiB), GPU Memory Utilization (%), GPU Memory Bandwidth Utilization (%)
06:30:15.000,    0,  0.00,  15.20,  300,  38,  128.00,  1.00,  0.50
"""

XPU_SMI_STATS_JSON = """\
{
  "power_limit": 225.0,
  "default_power_limit": 225.0
}
"""


def _mock_xpu_smi_run(discovery_json, dump_output="", stats_json=""):
    """Create a mock for subprocess.run that returns different outputs per command."""
    def side_effect(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "discovery" in cmd_str:
            result.stdout = discovery_json
        elif "dump" in cmd_str:
            result.stdout = dump_output
        elif "stats" in cmd_str:
            result.stdout = stats_json
        elif "--version" in cmd_str:
            result.stdout = "xpu-smi 1.2.0"
        else:
            result.stdout = ""
        return result
    return side_effect


class TestXPUSMIDiscovery(unittest.TestCase):
    """Test xpu-smi discovery JSON parsing."""

    def test_single_device(self):
        from intel_arc_collector import _XPUSMIBackend

        with patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(XPU_SMI_DISCOVERY_SINGLE)):
            backend = _XPUSMIBackend()

        self.assertEqual(backend.device_count, 1)
        self.assertEqual(backend.device_info[0]["name"], "Intel Arc A770")
        self.assertEqual(backend.device_info[0]["uuid"], "ARC-A770-0001-UUID")
        self.assertAlmostEqual(
            backend.device_info[0]["mem_total_mb"],
            17179869184 / (1024 * 1024),
            places=0,
        )

    def test_multi_device(self):
        from intel_arc_collector import _XPUSMIBackend

        with patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(XPU_SMI_DISCOVERY_MULTI)):
            backend = _XPUSMIBackend()

        self.assertEqual(backend.device_count, 2)
        self.assertEqual(backend.device_info[0]["name"], "Intel Arc A770")
        self.assertEqual(backend.device_info[1]["name"], "Intel Arc A750")

    def test_device_name_from_pci_id(self):
        from intel_arc_collector import _XPUSMIBackend

        with patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(XPU_SMI_DISCOVERY_FLEX)):
            backend = _XPUSMIBackend()

        self.assertEqual(backend.device_info[0]["name"], "Intel Data Center GPU Flex 170")

    def test_b580_device(self):
        from intel_arc_collector import _XPUSMIBackend

        with patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(XPU_SMI_DISCOVERY_B580)):
            backend = _XPUSMIBackend()

        self.assertEqual(backend.device_count, 1)
        self.assertEqual(backend.device_info[0]["name"], "Intel Arc B580")


class TestXPUSMIDumpParsing(unittest.TestCase):
    """Test xpu-smi dump CSV output parsing."""

    def test_parse_metrics(self):
        from intel_arc_collector import _XPUSMIBackend

        result = _XPUSMIBackend._parse_dump_output(XPU_SMI_DUMP_SINGLE)

        self.assertAlmostEqual(result["power_w"], 120.5)
        self.assertEqual(result["utilization_pct"], 45)
        self.assertEqual(result["temperature_c"], 65)
        self.assertAlmostEqual(result["memory_used_mb"], 4096.0)
        self.assertEqual(result["memory_util_pct"], 32)
        self.assertEqual(result["memory_bw_util_pct"], 18)
        self.assertEqual(result["frequency_mhz"], 2100)

    def test_parse_idle(self):
        from intel_arc_collector import _XPUSMIBackend

        result = _XPUSMIBackend._parse_dump_output(XPU_SMI_DUMP_IDLE)

        self.assertAlmostEqual(result["power_w"], 15.2)
        self.assertEqual(result["utilization_pct"], 0)

    def test_parse_empty(self):
        from intel_arc_collector import _XPUSMIBackend

        result = _XPUSMIBackend._parse_dump_output("")
        self.assertEqual(result, {})

    def test_parse_header_only(self):
        from intel_arc_collector import _XPUSMIBackend

        result = _XPUSMIBackend._parse_dump_output(
            "Timestamp, DeviceId, GPU Utilization (%)\n"
        )
        self.assertEqual(result, {})


class TestDeviceNameMapping(unittest.TestCase):

    def test_a770(self):
        from intel_arc_collector import _ARC_DEVICE_NAMES
        self.assertEqual(_ARC_DEVICE_NAMES["0x56a0"], "Intel Arc A770")

    def test_a750(self):
        from intel_arc_collector import _ARC_DEVICE_NAMES
        self.assertEqual(_ARC_DEVICE_NAMES["0x56a5"], "Intel Arc A750")

    def test_b580(self):
        from intel_arc_collector import _ARC_DEVICE_NAMES
        self.assertEqual(_ARC_DEVICE_NAMES["0xe20b"], "Intel Arc B580")

    def test_flex_170(self):
        from intel_arc_collector import _ARC_DEVICE_NAMES
        self.assertEqual(_ARC_DEVICE_NAMES["0x56c0"], "Intel Data Center GPU Flex 170")

    def test_max_1550(self):
        from intel_arc_collector import _ARC_DEVICE_NAMES
        self.assertEqual(_ARC_DEVICE_NAMES["0x0bd5"], "Intel Data Center GPU Max 1550")


class TestSafeHelpers(unittest.TestCase):

    def test_safe_float_valid(self):
        from intel_arc_collector import _safe_float
        self.assertEqual(_safe_float("120.5"), 120.5)

    def test_safe_float_empty(self):
        from intel_arc_collector import _safe_float
        self.assertEqual(_safe_float(""), 0.0)

    def test_safe_float_none(self):
        from intel_arc_collector import _safe_float
        self.assertEqual(_safe_float(None), 0.0)

    def test_safe_float_invalid(self):
        from intel_arc_collector import _safe_float
        self.assertEqual(_safe_float("N/A"), 0.0)

    def test_safe_int_valid(self):
        from intel_arc_collector import _safe_int
        self.assertEqual(_safe_int("65"), 65)

    def test_safe_int_float_str(self):
        from intel_arc_collector import _safe_int
        self.assertEqual(_safe_int("45.7"), 45)

    def test_safe_int_none(self):
        from intel_arc_collector import _safe_int
        self.assertEqual(_safe_int(None), 0)


class TestIntelArcCollectorXPUSMI(unittest.TestCase):
    """Test full IntelArcCollector via mocked xpu-smi."""

    def _make_collector(
        self,
        discovery_json=XPU_SMI_DISCOVERY_SINGLE,
        dump_output=XPU_SMI_DUMP_SINGLE,
        stats_json=XPU_SMI_STATS_JSON,
    ):
        from intel_arc_collector import IntelArcCollector

        with patch("intel_arc_collector._xpu_smi_available", return_value=True), \
             patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(discovery_json, dump_output, stats_json)):
            return IntelArcCollector()

    def test_returns_valid_gpu_metrics(self):
        collector = self._make_collector()

        with patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(
                       XPU_SMI_DISCOVERY_SINGLE, XPU_SMI_DUMP_SINGLE)):
            metrics = collector.collect()

        self.assertEqual(len(metrics), 1)
        m = metrics[0]
        self.assertEqual(m.gpu_index, 0)
        self.assertEqual(m.gpu_name, "Intel Arc A770")
        self.assertAlmostEqual(m.power_draw_w, 120.5)
        self.assertEqual(m.power_limit_w, 225.0)
        self.assertEqual(m.temperature_c, 65)
        self.assertEqual(m.utilization_gpu_pct, 45)
        self.assertEqual(m.utilization_memory_pct, 32)
        self.assertGreater(m.memory_used_mb, 0)
        self.assertGreater(m.memory_total_mb, 0)
        self.assertIsNotNone(m.timestamp)
        self.assertTrue(m.gpu_uuid.startswith("ARC-"))

        collector.shutdown()

    def test_multi_device(self):
        collector = self._make_collector(discovery_json=XPU_SMI_DISCOVERY_MULTI)

        self.assertEqual(collector.get_gpu_count(), 2)
        self.assertEqual(len(collector.gpu_uuids), 2)

        collector.shutdown()

    def test_energy_delta(self):
        collector = self._make_collector()

        with patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(
                       XPU_SMI_DISCOVERY_SINGLE, XPU_SMI_DUMP_SINGLE)):
            collector.collect()
            time.sleep(0.05)
            metrics = collector.collect()

        self.assertIsNotNone(metrics[0].energy_delta_j)
        self.assertGreater(metrics[0].energy_delta_j, 0)

        collector.shutdown()

    def test_context_manager(self):
        collector = self._make_collector()
        with collector:
            self.assertTrue(collector.initialized)
        self.assertFalse(collector.initialized)

    def test_backend_property(self):
        collector = self._make_collector()
        self.assertEqual(collector.backend, "xpu-smi")
        collector.shutdown()

    def test_default_power_limit_when_stats_fails(self):
        """When xpu-smi stats fails, use default TDP for A770 (225W)."""
        collector = self._make_collector(stats_json="")

        with patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(
                       XPU_SMI_DISCOVERY_SINGLE, XPU_SMI_DUMP_SINGLE)):
            metrics = collector.collect()

        self.assertEqual(metrics[0].power_limit_w, 225.0)
        collector.shutdown()

    def test_b580_default_tdp(self):
        collector = self._make_collector(
            discovery_json=XPU_SMI_DISCOVERY_B580, stats_json="",
        )

        with patch("intel_arc_collector.subprocess.run",
                   side_effect=_mock_xpu_smi_run(
                       XPU_SMI_DISCOVERY_B580, XPU_SMI_DUMP_IDLE)):
            metrics = collector.collect()

        self.assertEqual(metrics[0].power_limit_w, 190.0)
        collector.shutdown()


class TestIntelArcCollectorNoBackend(unittest.TestCase):

    def test_raises_when_nothing_available(self):
        from intel_arc_collector import IntelArcCollector

        with patch("intel_arc_collector._xpu_smi_available", return_value=False), \
             patch("intel_arc_collector.glob.glob", return_value=[]):
            with self.assertRaises(RuntimeError):
                IntelArcCollector()


class TestArchSpecIntelArc(unittest.TestCase):

    def test_resolves_a770(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Arc A770")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Arc")
        self.assertEqual(spec.tdp_w, 225.0)
        self.assertEqual(spec.memory_gb, 16)

    def test_resolves_a750(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Arc A750")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.tdp_w, 225.0)
        self.assertEqual(spec.memory_gb, 8)

    def test_resolves_b580(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Arc B580")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.tdp_w, 190.0)
        self.assertEqual(spec.memory_gb, 12)

    def test_resolves_a580(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Arc A580")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.tdp_w, 185.0)

    def test_resolves_flex_170(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Data Center GPU Flex 170")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Arc")
        self.assertEqual(spec.tdp_w, 150.0)

    def test_resolves_max_1550(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Data Center GPU Max 1550")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Ponte Vecchio")
        self.assertEqual(spec.tdp_w, 600.0)

    def test_nvidia_still_resolves(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("NVIDIA H100-SXM5-80GB")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Hopper")

    def test_gaudi_still_resolves(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Gaudi2")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Gaudi")

    def test_apple_still_resolves(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Apple M5 GPU")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Apple Silicon")


if __name__ == "__main__":
    unittest.main()
