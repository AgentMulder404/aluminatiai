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
Intel Gaudi collector unit tests.

Tests:
  1. hl-smi CSV output parsing (single device)
  2. hl-smi CSV output parsing (multi-device, 8x Gaudi2)
  3. hl-smi power limit from table output
  4. Device name mapping (HL-225 -> Intel Gaudi2)
  5. GaudiCollector returns valid GPUMetrics via hl-smi
  6. Energy delta after two samples
  7. Context manager lifecycle
  8. Graceful failure when no backend available
  9. ArchSpec resolution for Gaudi names
  10. Safe float parsing
  11. Gaudi3 CSV parsing
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Sample hl-smi CSV output for a single Gaudi2 device
# Fields: index, name, uuid, power.draw, temperature.aip, utilization.aip, memory.used, memory.total
HLSMI_CSV_SINGLE = """\
0, HL-225, HL2-0001-UUID, 285, 42, 78, 805306368, 103079215104
"""

# 8x Gaudi2 (typical DL2q instance)
HLSMI_CSV_8X_GAUDI2 = """\
0, HL-225, HL2-0001-UUID, 285, 42, 78, 805306368, 103079215104
1, HL-225, HL2-0002-UUID, 310, 45, 82, 1073741824, 103079215104
2, HL-225, HL2-0003-UUID, 275, 40, 65, 536870912, 103079215104
3, HL-225, HL2-0004-UUID, 295, 43, 75, 805306368, 103079215104
4, HL-225, HL2-0005-UUID, 302, 44, 80, 805306368, 103079215104
5, HL-225, HL2-0006-UUID, 288, 41, 70, 671088640, 103079215104
6, HL-225, HL2-0007-UUID, 315, 46, 85, 1073741824, 103079215104
7, HL-225, HL2-0008-UUID, 270, 39, 60, 402653184, 103079215104
"""

# Gaudi3 device
HLSMI_CSV_GAUDI3 = """\
0, HL-325L, HL3-0001-UUID, 450, 55, 90, 2147483648, 137438953472
"""

# Default table output for power limit parsing
HLSMI_TABLE_OUTPUT = """\
+-----------------------------------------------------------------------------+
| HL-SMI Version:                              hl-1.18.0-fw-52.0.0.0          |
| Driver Version:                                     1.18.0-abc123          |
|-------------------------------+----------------------+----------------------+
| AIP  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | AIP-Util  Compute M. |
|===============================+======================+======================|
|   0  HL-225              N/A  | 0000:19:00.0     N/A |                   0  |
| N/A   42C   N/A   285W / 600W |    768MiB / 98304MiB |    78%           N/A |
|-------------------------------+----------------------+----------------------+
|   1  HL-225              N/A  | 0000:1a:00.0     N/A |                   0  |
| N/A   45C   N/A   310W / 600W |   1024MiB / 98304MiB |    82%           N/A |
+-----------------------------------------------------------------------------+
"""


def _mock_subprocess_run(csv_output, table_output=""):
    """Create a mock for subprocess.run that returns different outputs."""
    def side_effect(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        if "-Q" in cmd:
            result.stdout = csv_output
        else:
            result.stdout = table_output
        return result
    return side_effect


class TestHLSMICSVParsing(unittest.TestCase):
    """Test hl-smi CSV output parsing."""

    def test_single_device(self):
        from gaudi_collector import _HLSMIBackend

        with patch("gaudi_collector.subprocess.run", side_effect=_mock_subprocess_run(HLSMI_CSV_SINGLE)), \
             patch("gaudi_collector._hl_smi_available", return_value=True):
            backend = _HLSMIBackend()

        self.assertEqual(backend.device_count, 1)
        self.assertEqual(backend.device_info[0]["name"], "Intel Gaudi2")
        self.assertEqual(backend.device_info[0]["raw_name"], "HL-225")

    def test_8x_gaudi2(self):
        from gaudi_collector import _HLSMIBackend

        with patch("gaudi_collector.subprocess.run", side_effect=_mock_subprocess_run(HLSMI_CSV_8X_GAUDI2)), \
             patch("gaudi_collector._hl_smi_available", return_value=True):
            backend = _HLSMIBackend()

        self.assertEqual(backend.device_count, 8)
        for info in backend.device_info:
            self.assertEqual(info["name"], "Intel Gaudi2")

    def test_gaudi3_device(self):
        from gaudi_collector import _HLSMIBackend

        with patch("gaudi_collector.subprocess.run", side_effect=_mock_subprocess_run(HLSMI_CSV_GAUDI3)), \
             patch("gaudi_collector._hl_smi_available", return_value=True):
            backend = _HLSMIBackend()

        self.assertEqual(backend.device_count, 1)
        self.assertEqual(backend.device_info[0]["name"], "Intel Gaudi3")

    def test_collect_all_metrics(self):
        from gaudi_collector import _HLSMIBackend

        with patch("gaudi_collector.subprocess.run", side_effect=_mock_subprocess_run(HLSMI_CSV_SINGLE)), \
             patch("gaudi_collector._hl_smi_available", return_value=True):
            backend = _HLSMIBackend()
            results = backend.collect_all()

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertAlmostEqual(r["power_w"], 285.0)
        self.assertEqual(r["temperature_c"], 42)
        self.assertEqual(r["utilization_pct"], 78)
        # Memory: 805306368 bytes = 768 MiB
        self.assertAlmostEqual(r["memory_used_mb"], 805306368 / (1024 * 1024), places=0)
        # Total: 103079215104 bytes = 96 GiB = 98304 MiB
        self.assertAlmostEqual(r["memory_total_mb"], 103079215104 / (1024 * 1024), places=0)


class TestHLSMIPowerLimitParse(unittest.TestCase):

    def test_parses_two_devices(self):
        from gaudi_collector import _HLSMIBackend

        with patch("gaudi_collector.subprocess.run",
                    side_effect=_mock_subprocess_run(HLSMI_CSV_SINGLE, HLSMI_TABLE_OUTPUT)), \
             patch("gaudi_collector._hl_smi_available", return_value=True):
            backend = _HLSMIBackend()
            limits = backend._parse_power_limit_from_table()

        self.assertEqual(limits[0], 600.0)
        self.assertEqual(limits[1], 600.0)


class TestDeviceNameMapping(unittest.TestCase):

    def test_hl225_to_gaudi2(self):
        from gaudi_collector import _GAUDI_NAMES
        self.assertEqual(_GAUDI_NAMES["HL-225"], "Intel Gaudi2")

    def test_hl325l_to_gaudi3(self):
        from gaudi_collector import _GAUDI_NAMES
        self.assertEqual(_GAUDI_NAMES["HL-325L"], "Intel Gaudi3")

    def test_hl205_to_gaudi(self):
        from gaudi_collector import _GAUDI_NAMES
        self.assertEqual(_GAUDI_NAMES["HL-205"], "Intel Gaudi")


class TestSafeFloat(unittest.TestCase):

    def test_valid(self):
        from gaudi_collector import _safe_float
        self.assertEqual(_safe_float("285.5"), 285.5)

    def test_empty(self):
        from gaudi_collector import _safe_float
        self.assertEqual(_safe_float(""), 0.0)

    def test_none(self):
        from gaudi_collector import _safe_float
        self.assertEqual(_safe_float(None), 0.0)

    def test_invalid(self):
        from gaudi_collector import _safe_float
        self.assertEqual(_safe_float("N/A"), 0.0)


class TestGaudiCollectorHLSMI(unittest.TestCase):
    """Test full GaudiCollector via mocked hl-smi."""

    def _make_collector(self, csv_output=HLSMI_CSV_SINGLE, table_output=HLSMI_TABLE_OUTPUT):
        from gaudi_collector import GaudiCollector

        with patch("gaudi_collector._PYHLML", False), \
             patch("gaudi_collector._hl_smi_available", return_value=True), \
             patch("gaudi_collector.subprocess.run",
                   side_effect=_mock_subprocess_run(csv_output, table_output)):
            return GaudiCollector()

    def test_returns_valid_gpu_metrics(self):
        collector = self._make_collector()

        with patch("gaudi_collector.subprocess.run",
                   side_effect=_mock_subprocess_run(HLSMI_CSV_SINGLE)):
            metrics = collector.collect()

        self.assertEqual(len(metrics), 1)
        m = metrics[0]
        self.assertEqual(m.gpu_index, 0)
        self.assertEqual(m.gpu_name, "Intel Gaudi2")
        self.assertAlmostEqual(m.power_draw_w, 285.0)
        self.assertEqual(m.power_limit_w, 600.0)
        self.assertEqual(m.temperature_c, 42)
        self.assertEqual(m.utilization_gpu_pct, 78)
        self.assertGreater(m.memory_used_mb, 0)
        self.assertGreater(m.memory_total_mb, 0)
        self.assertIsNotNone(m.timestamp)
        self.assertTrue(m.gpu_uuid.startswith("HL2-") or m.gpu_uuid.startswith("AIP"))

        collector.shutdown()

    def test_8x_devices(self):
        collector = self._make_collector(csv_output=HLSMI_CSV_8X_GAUDI2)

        self.assertEqual(collector.get_gpu_count(), 8)
        self.assertEqual(len(collector.gpu_uuids), 8)

        with patch("gaudi_collector.subprocess.run",
                   side_effect=_mock_subprocess_run(HLSMI_CSV_8X_GAUDI2)):
            metrics = collector.collect()

        self.assertEqual(len(metrics), 8)
        powers = [m.power_draw_w for m in metrics]
        self.assertIn(285.0, powers)
        self.assertIn(315.0, powers)

        collector.shutdown()

    def test_energy_delta(self):
        collector = self._make_collector()

        with patch("gaudi_collector.subprocess.run",
                   side_effect=_mock_subprocess_run(HLSMI_CSV_SINGLE)):
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

    def test_default_power_limit_when_table_fails(self):
        """When table parse fails, use default for HL-225 (600W)."""
        from gaudi_collector import GaudiCollector

        with patch("gaudi_collector._PYHLML", False), \
             patch("gaudi_collector._hl_smi_available", return_value=True), \
             patch("gaudi_collector.subprocess.run",
                   side_effect=_mock_subprocess_run(HLSMI_CSV_SINGLE, "")):
            collector = GaudiCollector()

        with patch("gaudi_collector.subprocess.run",
                   side_effect=_mock_subprocess_run(HLSMI_CSV_SINGLE)):
            metrics = collector.collect()

        self.assertEqual(metrics[0].power_limit_w, 600.0)
        collector.shutdown()


class TestGaudiCollectorNoBackend(unittest.TestCase):

    def test_raises_when_nothing_available(self):
        from gaudi_collector import GaudiCollector

        with patch("gaudi_collector._PYHLML", False), \
             patch("gaudi_collector._hl_smi_available", return_value=False):
            with self.assertRaises(RuntimeError):
                GaudiCollector()


class TestArchSpecGaudi(unittest.TestCase):

    def test_resolves_gaudi2(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Gaudi2")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Gaudi")
        self.assertEqual(spec.tdp_w, 600.0)
        self.assertEqual(spec.memory_gb, 96)

    def test_resolves_gaudi3(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Intel Gaudi3")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.tdp_w, 900.0)
        self.assertEqual(spec.memory_gb, 128)

    def test_nvidia_still_resolves(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("NVIDIA H100-SXM5-80GB")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Hopper")

    def test_apple_still_resolves(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Apple M5 GPU")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Apple Silicon")


if __name__ == "__main__":
    unittest.main()
