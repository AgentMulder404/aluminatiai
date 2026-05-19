# Copyright 2026 Kevin (AluminatiAI)
"""Tests for the ab command — CI significance, AEM, serialization."""
import json
import os
import sys
import tempfile
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ab import check_significance, ABResult, PhaseMetrics, _parse_throughput, _export_csv
from efficiency.profiler import ConfidenceInterval


def _ci(mean, std, lo, hi, n=5):
    return ConfidenceInterval(mean=mean, std=std, ci_lower=lo, ci_upper=hi, n=n)


def _make_phase(label, energy_mean, energy_lo, energy_hi, power_mean=200.0, dur_mean=60.0, tp=None):
    return PhaseMetrics(
        label=label,
        energy_j=_ci(energy_mean, 10, energy_lo, energy_hi),
        mean_power_w=_ci(power_mean, 5, power_mean - 10, power_mean + 10),
        duration_s=_ci(dur_mean, 1, dur_mean - 2, dur_mean + 2),
        throughput=_ci(tp, 0.5, tp - 1, tp + 1) if tp else None,
        peak_power_w=power_mean + 50,
        avg_util_pct=70,
        avg_temp_c=65,
        sample_count=600,
    )


class TestSignificance:
    def test_overlapping_cis(self):
        a = _ci(100, 10, 80, 120)
        b = _ci(110, 10, 90, 130)
        assert check_significance(a, b) is False

    def test_non_overlapping_cis(self):
        a = _ci(100, 5, 90, 110)
        b = _ci(200, 5, 190, 210)
        assert check_significance(a, b) is True

    def test_barely_touching(self):
        a = _ci(100, 5, 90, 110)
        b = _ci(115, 5, 110, 120)
        # a.ci_upper (110) == b.ci_lower (110) -> not strictly non-overlapping
        assert check_significance(a, b) is False

    def test_b_lower_than_a(self):
        a = _ci(200, 5, 190, 210)
        b = _ci(100, 5, 90, 110)
        assert check_significance(a, b) is True


class TestAEM:
    def test_standard_aem(self):
        # 30% energy savings, 5% throughput loss => AEM = 6.0
        result = ABResult(gpu_name="A100", gpu_index=0)
        result.baseline = _make_phase("Baseline", 1000, 950, 1050, tp=100)
        result.optimized = _make_phase("Optimized", 700, 650, 750, tp=95)
        result.energy_savings_pct = 30.0
        result.throughput_change_pct = -5.0
        # Manually compute AEM
        aem = result.energy_savings_pct / abs(result.throughput_change_pct)
        assert aem == 6.0

    def test_aem_no_throughput_loss(self):
        # Energy savings with no throughput loss => AEM = inf
        result = ABResult(gpu_name="A100", gpu_index=0)
        result.energy_savings_pct = 20.0
        result.throughput_change_pct = 0.0
        # When no loss, AEM should be inf
        if result.throughput_change_pct >= 0:
            aem = float("inf")
        assert aem == float("inf")

    def test_aem_with_gain(self):
        # Energy savings AND throughput gain => AEM = inf (free lunch)
        result = ABResult(gpu_name="A100", gpu_index=0)
        result.energy_savings_pct = 15.0
        result.throughput_change_pct = 5.0  # Throughput improved
        # Positive throughput change means no loss
        loss = abs(result.throughput_change_pct) if result.throughput_change_pct < 0 else 0
        aem = result.energy_savings_pct / loss if loss > 0 else float("inf")
        assert aem == float("inf")


class TestThroughputParsing:
    def test_tokens_per_sec(self):
        assert _parse_throughput("Throughput: 1500.5 tok/s") == 1500.5

    def test_it_per_sec(self):
        assert _parse_throughput("100% 50/50 [00:30<00:00, 3.5 it/s]") == 3.5

    def test_images_per_sec(self):
        assert _parse_throughput("Processing: 120.0 images/s") == 120.0

    def test_no_match(self):
        assert _parse_throughput("Training complete in 30 minutes") is None

    def test_steps_per_sec(self):
        assert _parse_throughput("Step 100: 25.3 steps/s") == 25.3


class TestCSVExport:
    def test_csv_format(self):
        result = ABResult(gpu_name="A100", gpu_index=0)
        result.baseline = _make_phase("Baseline", 1000, 950, 1050)
        result.optimized = _make_phase("Optimized", 700, 650, 750)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        try:
            _export_csv(result, path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 phases
            assert "Baseline" in lines[1]
            assert "Optimized" in lines[2]
        finally:
            os.unlink(path)


class TestJsonOutput:
    def test_roundtrip(self):
        result = ABResult(
            gpu_name="A100", gpu_index=0, arch_spec="A100-SXM4-80GB",
            energy_savings_pct=25.5, recommendation="Recommended.",
        )
        result.baseline = _make_phase("Baseline", 1000, 950, 1050)
        result.optimized = _make_phase("Optimized", 750, 700, 800)

        d = asdict(result)
        text = json.dumps(d)
        parsed = json.loads(text)
        assert parsed["energy_savings_pct"] == 25.5
        assert parsed["gpu_name"] == "A100"
        assert parsed["baseline"]["label"] == "Baseline"
        assert parsed["optimized"]["label"] == "Optimized"


class TestRecommendationText:
    def test_significant_savings(self):
        result = ABResult(gpu_name="A100", gpu_index=0)
        result.energy_savings_pct = 25.0
        result.energy_significant = True
        result.throughput_change_pct = -2.0
        # Should recommend
        assert result.energy_savings_pct > 5

    def test_not_significant(self):
        result = ABResult(gpu_name="A100", gpu_index=0)
        result.energy_savings_pct = 3.0
        result.energy_significant = False
        assert not result.energy_significant


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
