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
Standalone statistics utilities for energy profiling.

Extracted from ScientificEnergyProfiler so that optimize, ab, and other
tools can compute confidence intervals and energy integrals without
instantiating the full profiler (which requires NVML).
"""
import math
from .profiler import ConfidenceInterval


def t_critical_95(df: int) -> float:
    """
    Two-tailed t critical values for 95% confidence (alpha = 0.05).

    Pre-computed to avoid requiring scipy. Values from standard
    t-distribution tables. For df > 120, converges to z = 1.96.
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000,
        80: 1.990, 100: 1.984, 120: 1.980,
    }

    if df in table:
        return table[df]

    if df > 120:
        return 1.960

    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (df - lo) / (hi - lo)
            return table[lo] + frac * (table[hi] - table[lo])

    return 1.960


def compute_ci(values: list[float]) -> ConfidenceInterval:
    """
    Compute 95% confidence interval using the Student's t-distribution.

    For n < 30 samples, the t-distribution provides correct coverage
    probability (unlike z-based intervals which assume known variance).

    CI = x_bar +/- t_{alpha/2, n-1} * (s / sqrt(n))
    """
    n = len(values)
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, 0.0, 0)

    mean = sum(values) / n

    if n == 1:
        return ConfidenceInterval(mean, 0.0, mean, mean, 1)

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)

    t_crit = t_critical_95(n - 1)
    margin = t_crit * (std / math.sqrt(n))

    return ConfidenceInterval(
        mean=round(mean, 6),
        std=round(std, 6),
        ci_lower=round(mean - margin, 6),
        ci_upper=round(mean + margin, 6),
        n=n,
    )


def trapezoidal_energy(timestamps: list[float], powers: list[float]) -> float:
    """
    Compute total energy (Joules) via trapezoidal numerical integration.

    E = sum [(P_i + P_{i+1}) / 2] * (t_{i+1} - t_i)

    Args:
        timestamps: Monotonic timestamps in seconds.
        powers: Power readings in watts, same length as timestamps.

    Returns:
        Total energy in Joules.
    """
    if len(timestamps) < 2 or len(timestamps) != len(powers):
        return 0.0

    energy = 0.0
    for i in range(len(timestamps) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        avg_power = (powers[i] + powers[i + 1]) / 2.0
        energy += avg_power * dt
    return energy
