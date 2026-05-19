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
Apple Silicon collector unit tests.

Tests:
  1. Chip detection
  2. GPU TDP lookup (known + override)
  3. ioreg parsing (utilization + memory)
  4. powermetrics chunk parsing
  5. Collector returns valid GPUMetrics (ioreg fallback)
  6. Collector returns valid GPUMetrics (powermetrics)
  7. Context manager lifecycle
  8. Non-macOS raises RuntimeError
  9. ArchSpec resolution for Apple names
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


SAMPLE_IOREG_OUTPUT = """\
+-o AGXAcceleratorG17G  <class AGXAcceleratorG17G, id 0x100000534>
    {
      "model" = "Apple M5"
      "gpu-core-count" = 10
      "PerformanceStatistics" = {"In use system memory"=928219136,"Device Utilization %"=42,"Renderer Utilization %"=38,"Tiler Utilization %"=35}
    }
"""

SAMPLE_POWERMETRICS_CHUNK = """\
***** 2026-05-09 12:00:00 -0700 *****

*** GPU Power ***
GPU Power: 8500 mW
GPU HW active residency: 65.2 %
GPU HW active frequency: 1398 MHz

*** CPU Power ***
CPU Power: 4200 mW
Combined Power (CPU + GPU + ANE): 14500 mW

*** SMC ***
die temperature: 52.3 C
"""


class TestChipDetection(unittest.TestCase):

    def test_detects_apple_silicon(self):
        with patch("apple_collector.platform") as mock_platform, \
             patch("apple_collector.sys") as mock_sys, \
             patch("apple_collector.subprocess") as mock_sp:
            mock_platform.machine.return_value = "arm64"
            mock_sys.platform = "darwin"
            mock_sp.check_output.return_value = "Apple M4 Pro"

            from apple_collector import _detect_chip
            self.assertEqual(_detect_chip(), "Apple M4 Pro")

    def test_returns_empty_on_linux(self):
        with patch("apple_collector.platform") as mock_platform, \
             patch("apple_collector.sys") as mock_sys:
            mock_platform.machine.return_value = "x86_64"
            mock_sys.platform = "linux"

            from apple_collector import _detect_chip
            self.assertEqual(_detect_chip(), "")

    def test_returns_empty_on_intel_mac(self):
        with patch("apple_collector.platform") as mock_platform, \
             patch("apple_collector.sys") as mock_sys:
            mock_platform.machine.return_value = "x86_64"
            mock_sys.platform = "darwin"

            from apple_collector import _detect_chip
            self.assertEqual(_detect_chip(), "")


class TestGpuTdp(unittest.TestCase):

    def test_known_chip(self):
        from apple_collector import _gpu_tdp
        self.assertEqual(_gpu_tdp("Apple M4 Max"), 50.0)

    def test_unknown_chip_default(self):
        from apple_collector import _gpu_tdp
        self.assertEqual(_gpu_tdp("Apple M99"), 15.0)

    def test_env_override(self):
        with patch.dict(os.environ, {"APPLE_CHIP_TDP_OVERRIDE": "42.5"}):
            from apple_collector import _gpu_tdp
            self.assertEqual(_gpu_tdp("Apple M1"), 42.5)


class TestIORegParsing(unittest.TestCase):

    def test_parses_utilization_and_memory(self):
        with patch("apple_collector.subprocess") as mock_sp:
            mock_sp.check_output.return_value = SAMPLE_IOREG_OUTPUT
            mock_sp.SubprocessError = Exception

            from apple_collector import _IOReg
            data = _IOReg.read()

            self.assertEqual(data["util_pct"], 42)
            self.assertAlmostEqual(data["memory_used_mb"], 928219136 / (1024 * 1024), places=0)

    def test_handles_missing_data(self):
        with patch("apple_collector.subprocess") as mock_sp:
            mock_sp.check_output.return_value = "no gpu data here"
            mock_sp.SubprocessError = Exception

            from apple_collector import _IOReg
            data = _IOReg.read()
            self.assertEqual(data, {})

    def test_handles_subprocess_failure(self):
        with patch("apple_collector.subprocess") as mock_sp:
            mock_sp.check_output.side_effect = OSError("no ioreg")
            mock_sp.SubprocessError = OSError

            from apple_collector import _IOReg
            data = _IOReg.read()
            self.assertEqual(data, {})


class TestPowerMetricsParser(unittest.TestCase):

    def test_parses_all_fields(self):
        from apple_collector import _PowerMetricsMonitor
        monitor = _PowerMetricsMonitor.__new__(_PowerMetricsMonitor)
        monitor._lock = __import__("threading").Lock()
        monitor._latest = None
        monitor.available = False

        monitor._parse_chunk(SAMPLE_POWERMETRICS_CHUNK.splitlines(keepends=True))

        r = monitor._latest
        self.assertAlmostEqual(r.gpu_power_w, 8.5, places=1)
        self.assertAlmostEqual(r.cpu_power_w, 4.2, places=1)
        self.assertAlmostEqual(r.package_power_w, 14.5, places=1)
        self.assertAlmostEqual(r.gpu_util_pct, 65.2, places=1)
        self.assertEqual(r.gpu_freq_mhz, 1398)
        self.assertAlmostEqual(r.die_temp_c, 52.3, places=1)
        self.assertTrue(monitor.available)

    def test_handles_missing_fields(self):
        from apple_collector import _PowerMetricsMonitor
        monitor = _PowerMetricsMonitor.__new__(_PowerMetricsMonitor)
        monitor._lock = __import__("threading").Lock()
        monitor._latest = None
        monitor.available = False

        monitor._parse_chunk(["just some random text\n"])

        r = monitor._latest
        self.assertEqual(r.gpu_power_w, 0.0)
        self.assertEqual(r.die_temp_c, 0.0)


class TestAppleSiliconCollectorIoregFallback(unittest.TestCase):
    """Test collector in ioreg fallback mode (no sudo)."""

    def _make_collector(self):
        """Create collector with mocked externals, powermetrics disabled."""
        from apple_collector import AppleSiliconCollector

        with patch("apple_collector._detect_chip", return_value="Apple M5"), \
             patch("apple_collector._get_total_memory_mb", return_value=16384.0), \
             patch("apple_collector._get_gpu_core_count", return_value=10), \
             patch.dict(os.environ, {"APPLE_POWERMETRICS_ENABLED": "0"}):
            return AppleSiliconCollector()

    def test_returns_valid_gpu_metrics(self):
        collector = self._make_collector()

        with patch("apple_collector._IOReg.read", return_value={
            "util_pct": 55,
            "memory_used_mb": 2048.0,
        }):
            metrics = collector.collect()

        self.assertEqual(len(metrics), 1)
        m = metrics[0]
        self.assertEqual(m.gpu_index, 0)
        self.assertIn("Apple M5", m.gpu_name)
        self.assertIn("10-core", m.gpu_name)
        self.assertEqual(m.utilization_gpu_pct, 55)
        # Power estimated from util * TDP: 15.0 * 0.55 = 8.25
        self.assertAlmostEqual(m.power_draw_w, 8.25, places=2)
        self.assertEqual(m.power_limit_w, 15.0)
        self.assertAlmostEqual(m.memory_total_mb, 16384.0)
        self.assertAlmostEqual(m.memory_used_mb, 2048.0)
        self.assertEqual(m.temperature_c, 0)  # no temp in ioreg mode
        self.assertEqual(m.fan_speed_pct, 0)

        collector.shutdown()

    def test_energy_delta_after_two_samples(self):
        collector = self._make_collector()

        with patch("apple_collector._IOReg.read", return_value={"util_pct": 50}):
            collector.collect()
            time.sleep(0.05)
            metrics = collector.collect()

        m = metrics[0]
        self.assertIsNotNone(m.energy_delta_j)
        self.assertGreater(m.energy_delta_j, 0)

        collector.shutdown()

    def test_uuid_format(self):
        collector = self._make_collector()
        self.assertEqual(collector.gpu_uuids[0], "apple-apple-m5-integrated-gpu")
        self.assertEqual(collector.get_gpu_count(), 1)
        collector.shutdown()

    def test_context_manager(self):
        with self._make_collector() as collector:
            self.assertTrue(collector.initialized)
        self.assertFalse(collector.initialized)

    def test_backend_property(self):
        collector = self._make_collector()
        self.assertEqual(collector.backend, "ioreg")
        collector.shutdown()


class TestAppleSiliconCollectorPowermetrics(unittest.TestCase):
    """Test collector with mocked powermetrics monitor."""

    def test_returns_power_from_powermetrics(self):
        from apple_collector import AppleSiliconCollector, _SoCReading

        mock_monitor = MagicMock()
        mock_monitor.start.return_value = True
        mock_monitor.available = True
        mock_monitor.latest = _SoCReading(
            gpu_power_w=12.5,
            cpu_power_w=5.0,
            package_power_w=19.0,
            die_temp_c=58.0,
            gpu_freq_mhz=1398,
            gpu_util_pct=72.0,
        )

        with patch("apple_collector._detect_chip", return_value="Apple M4 Max"), \
             patch("apple_collector._get_total_memory_mb", return_value=65536.0), \
             patch("apple_collector._get_gpu_core_count", return_value=40), \
             patch("apple_collector._PowerMetricsMonitor", return_value=mock_monitor), \
             patch("apple_collector._IOReg.read", return_value={"memory_used_mb": 8192.0}):

            collector = AppleSiliconCollector()
            self.assertEqual(collector.backend, "powermetrics")

            metrics = collector.collect()
            m = metrics[0]

            self.assertAlmostEqual(m.power_draw_w, 12.5, places=1)
            self.assertEqual(m.utilization_gpu_pct, 72)
            self.assertEqual(m.temperature_c, 58)
            self.assertEqual(m.sm_clock_mhz, 1398)
            self.assertEqual(m.power_limit_w, 50.0)  # M4 Max TDP
            self.assertAlmostEqual(m.memory_used_mb, 8192.0)
            self.assertAlmostEqual(m.memory_total_mb, 65536.0)

            collector.shutdown()
            mock_monitor.stop.assert_called_once()


class TestNonMacOS(unittest.TestCase):

    def test_raises_on_linux(self):
        with patch("apple_collector._detect_chip", return_value=""):
            from apple_collector import AppleSiliconCollector
            with self.assertRaises(RuntimeError):
                AppleSiliconCollector()


class TestArchSpecApple(unittest.TestCase):

    def test_resolves_m5_gpu(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Apple M5 GPU")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Apple Silicon")
        self.assertEqual(spec.tdp_w, 15.0)

    def test_resolves_m4_max_with_core_suffix(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Apple M4 Max GPU (40-core)")
        self.assertIsNotNone(spec)
        self.assertIn("M4 Max", spec.name)

    def test_resolves_m2_ultra(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("Apple M2 Ultra GPU")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.memory_gb, 192)

    def test_nvidia_still_resolves(self):
        from efficiency.gpu_specs import resolve_arch
        spec = resolve_arch("NVIDIA A100-SXM4-80GB")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "Ampere")


if __name__ == "__main__":
    unittest.main()
