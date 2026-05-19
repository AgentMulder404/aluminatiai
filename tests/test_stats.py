# Copyright 2026 Kevin (AluminatiAI)
"""Tests for efficiency.stats — CI computation and trapezoidal energy."""
import math
import sys
import os

# Ensure agent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from efficiency.stats import compute_ci, t_critical_95, trapezoidal_energy


class TestTCritical:
    def test_exact_lookup(self):
        assert t_critical_95(2) == 4.303
        assert t_critical_95(10) == 2.228

    def test_large_df_normal_approx(self):
        assert t_critical_95(500) == 1.960

    def test_interpolation(self):
        # df=35 is between 30 (2.042) and 40 (2.021)
        val = t_critical_95(35)
        assert 2.021 < val < 2.042


class TestComputeCI:
    def test_empty(self):
        ci = compute_ci([])
        assert ci.mean == 0.0
        assert ci.n == 0

    def test_single_value(self):
        ci = compute_ci([42.0])
        assert ci.mean == 42.0
        assert ci.ci_lower == 42.0
        assert ci.ci_upper == 42.0
        assert ci.n == 1

    def test_known_values(self):
        # 5 identical values => std=0, CI collapses to the mean
        ci = compute_ci([10.0, 10.0, 10.0, 10.0, 10.0])
        assert ci.mean == 10.0
        assert ci.std == 0.0
        assert ci.ci_lower == 10.0
        assert ci.ci_upper == 10.0

    def test_symmetric_spread(self):
        # [8, 10, 12] => mean=10, std=2, df=2, t=4.303
        ci = compute_ci([8.0, 10.0, 12.0])
        assert ci.mean == 10.0
        assert ci.n == 3
        # margin = 4.303 * (2 / sqrt(3)) ≈ 4.969
        assert ci.ci_lower < 10.0
        assert ci.ci_upper > 10.0
        # CI should be symmetric around mean
        margin = ci.ci_upper - ci.mean
        assert abs(margin - (ci.mean - ci.ci_lower)) < 0.001

    def test_ci_narrows_with_more_samples(self):
        few = compute_ci([10.0, 12.0, 14.0])
        many = compute_ci([10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        few_width = few.ci_upper - few.ci_lower
        many_width = many.ci_upper - many.ci_lower
        assert many_width < few_width


class TestTrapezoidalEnergy:
    def test_constant_power(self):
        # 100W for 10s = 1000J
        timestamps = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        powers = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        energy = trapezoidal_energy(timestamps, powers)
        assert abs(energy - 1000.0) < 0.001

    def test_linear_ramp(self):
        # Linear ramp from 0W to 200W over 10s
        # Area = 0.5 * 200 * 10 = 1000J
        timestamps = [float(i) for i in range(11)]
        powers = [20.0 * i for i in range(11)]
        energy = trapezoidal_energy(timestamps, powers)
        assert abs(energy - 1000.0) < 0.001

    def test_single_sample(self):
        assert trapezoidal_energy([0.0], [100.0]) == 0.0

    def test_empty(self):
        assert trapezoidal_energy([], []) == 0.0

    def test_mismatched_lengths(self):
        assert trapezoidal_energy([0.0, 1.0], [100.0]) == 0.0

    def test_two_samples(self):
        # 150W avg * 5s = 750J
        energy = trapezoidal_energy([0.0, 5.0], [100.0, 200.0])
        assert abs(energy - 750.0) < 0.001


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
