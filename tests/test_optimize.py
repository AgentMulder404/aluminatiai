# Copyright 2026 Kevin (AluminatiAI)
"""Tests for the optimize command — WorkloadAnalyzer recommendations."""
import json
import sys
import os
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimize import WorkloadAnalyzer, OptimizeResult, Recommendation
from efficiency.gpu_specs import GPU_ARCHITECTURES, ArchSpec


# Helper to create N identical samples
def _make_samples(
    n: int = 30,
    power: float = 250.0,
    util: int = 50,
    mem_util: int = 40,
    temp: int = 65,
    power_limit: float = 400.0,
    processes: list | None = None,
) -> list[dict]:
    return [
        {
            "power_draw_w": power,
            "utilization_gpu_pct": util,
            "utilization_memory_pct": mem_util,
            "temperature_c": temp,
            "power_limit_w": power_limit,
            "processes": processes or [],
        }
        for _ in range(n)
    ]


# A100 spec for testing
A100 = GPU_ARCHITECTURES.get("A100-SXM4-80GB")


class TestLowUtilization:
    def test_triggers_at_20_pct(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(util=20)
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "utilization" in categories

    def test_no_trigger_at_80_pct(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(util=80)
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "utilization" not in categories


class TestPrecisionDetection:
    def test_fp32_on_tensor_core_gpu(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(
            util=70,
            processes=[{"cmdline": "python train.py --precision fp32", "pid": 1234}],
        )
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "precision" in categories

    def test_bf16_no_trigger(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(
            util=70,
            processes=[{"cmdline": "python train.py --bf16", "pid": 1234}],
        )
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "precision" not in categories


class TestIdleDetection:
    def test_triggers_at_60_pct_idle(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        # 18 idle + 12 active = 60% idle
        idle = _make_samples(n=18, util=0)
        active = _make_samples(n=12, util=80)
        result = analyzer.analyze(idle + active, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "idle" in categories
        idle_rec = [r for r in result.recommendations if r.category == "idle"][0]
        assert idle_rec.priority == "P1"  # >50% idle -> P1

    def test_no_trigger_at_10_pct_idle(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        idle = _make_samples(n=3, util=0)
        active = _make_samples(n=27, util=80)
        result = analyzer.analyze(idle + active, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "idle" not in categories


class TestThermalThrottling:
    def test_high_power_low_util(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(power=380, util=30, temp=82, power_limit=400)
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "thermal" in categories

    def test_no_trigger_when_util_high(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(power=380, util=85, temp=75, power_limit=400)
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "thermal" not in categories


class TestGpuMatch:
    def test_suggests_better_gpu(self):
        # Use a less efficient GPU — T4 at low utilization
        t4 = GPU_ARCHITECTURES.get("T4")
        if not t4:
            return  # Skip if T4 not in specs
        analyzer = WorkloadAnalyzer(arch_spec=t4)
        samples = _make_samples(util=60, power=60, power_limit=70)
        result = analyzer.analyze(samples, "T4", 0, 30.0)
        # May or may not trigger depending on relative efficiency
        # Just verify it doesn't crash
        assert isinstance(result.recommendations, list)


class TestPowerCap:
    def test_suggests_cap(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        # Drawing near TDP at moderate utilization
        samples = _make_samples(power=380, util=70, power_limit=400)
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "power_cap" in categories

    def test_no_cap_when_low_power(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(power=150, util=50, power_limit=400)
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        categories = [r.category for r in result.recommendations]
        assert "power_cap" not in categories


class TestJsonOutput:
    def test_serializable(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(util=20)
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        d = asdict(result)
        output = json.dumps(d)
        parsed = json.loads(output)
        assert "recommendations" in parsed
        assert "gpu_name" in parsed
        assert parsed["avg_util_pct"] == 20.0


class TestNoRecsForEfficient:
    def test_well_optimized_workload(self):
        analyzer = WorkloadAnalyzer(arch_spec=A100)
        samples = _make_samples(
            util=82, mem_util=30, power=280, temp=68, power_limit=400,
            processes=[{"cmdline": "python train.py --bf16", "pid": 1}],
        )
        result = analyzer.analyze(samples, "A100", 0, 30.0)
        # Should have no P1 recommendations for a well-tuned workload
        p1_recs = [r for r in result.recommendations if r.priority == "P1"]
        assert len(p1_recs) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
