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
Unit tests for GPUCollector

Run with: python -m pytest tests/test_collector.py
Or: python tests/test_collector.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from collector import GPUCollector, GPUMetrics


def test_collector_initialization():
    """Test that collector initializes without errors"""
    try:
        collector = GPUCollector()
        assert collector.initialized
        assert collector.gpu_count >= 0
        collector.shutdown()
        print("✅ test_collector_initialization passed")
    except Exception as e:
        print(f"❌ test_collector_initialization failed: {e}")
        raise


def test_collect_metrics():
    """Test that metrics can be collected"""
    try:
        with GPUCollector() as collector:
            metrics = collector.collect()

            assert isinstance(metrics, list)
            assert len(metrics) == collector.gpu_count

            for m in metrics:
                assert isinstance(m, GPUMetrics)
                assert m.power_draw_w >= 0
                assert m.power_limit_w > 0
                assert 0 <= m.utilization_gpu_pct <= 100
                assert m.temperature_c >= 0

        print("✅ test_collect_metrics passed")
    except Exception as e:
        print(f"❌ test_collect_metrics failed: {e}")
        raise


def test_energy_calculation():
    """Test that energy delta is calculated correctly"""
    import time

    try:
        with GPUCollector() as collector:
            # First sample (no energy delta yet)
            metrics1 = collector.collect()

            # Wait and collect again
            time.sleep(2)
            metrics2 = collector.collect()

            # Second sample should have energy delta
            for m in metrics2:
                if m.energy_delta_j is not None:
                    assert m.energy_delta_j > 0, "Energy delta should be positive"

                    # Rough sanity check: E = P × t
                    # For 2s at typical power, should be < 2000J (1000W × 2s)
                    assert m.energy_delta_j < 2000, "Energy delta suspiciously high"

        print("✅ test_energy_calculation passed")
    except Exception as e:
        print(f"❌ test_energy_calculation failed: {e}")
        raise


def test_gpu_info():
    """Test GPU info retrieval"""
    try:
        with GPUCollector() as collector:
            info = collector.get_gpu_info()

            assert isinstance(info, list)
            assert len(info) == collector.gpu_count

            for gpu_info in info:
                assert 'index' in gpu_info
                assert 'uuid' in gpu_info
                assert 'name' in gpu_info
                assert isinstance(gpu_info['uuid'], str)

        print("✅ test_gpu_info passed")
    except Exception as e:
        print(f"❌ test_gpu_info failed: {e}")
        raise


def test_metrics_serialization():
    """Test that metrics can be serialized"""
    try:
        with GPUCollector() as collector:
            metrics = collector.collect()

            for m in metrics:
                # Test dict conversion
                d = m.to_dict()
                assert isinstance(d, dict)
                assert 'gpu_index' in d
                assert 'power_draw_w' in d

                # Test CSV row conversion
                row = m.to_csv_row()
                assert isinstance(row, list)
                assert len(row) > 0

        print("✅ test_metrics_serialization passed")
    except Exception as e:
        print(f"❌ test_metrics_serialization failed: {e}")
        raise


def run_all_tests():
    """Run all tests without pytest"""
    tests = [
        test_collector_initialization,
        test_collect_metrics,
        test_energy_calculation,
        test_gpu_info,
        test_metrics_serialization,
    ]

    print("\n🧪 Running Collector Unit Tests")
    print("="*70)

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception:
            failed += 1

    print("="*70)
    print(f"\n✅ Passed: {passed}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print()

    return failed == 0


if __name__ == '__main__':
    if PYTEST_AVAILABLE:
        # Run with pytest if available
        pytest.main([__file__, '-v'])
    else:
        # Run without pytest
        success = run_all_tests()
        sys.exit(0 if success else 1)
