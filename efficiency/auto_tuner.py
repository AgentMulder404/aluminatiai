"""
AutoTuner — periodic roofline analysis with optional power cap enforcement.

Integrated into the agent daemon loop. Every AUTO_TUNE_INTERVAL seconds,
samples GPU metrics, runs the optimize analysis, and optionally applies
power cap recommendations via NVML.

Opt-in only: AUTO_TUNE_ENABLED=0 by default.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    _NVML = True
except ImportError:
    _NVML = False


@dataclass
class TuneResult:
    gpu_index: int
    gpu_name: str
    current_power_w: float
    recommended_cap_w: Optional[float]
    estimated_savings_pct: float
    applied: bool
    reason: str


class AutoTuner:
    """Periodic GPU efficiency analysis with optional power cap enforcement."""

    def __init__(
        self,
        interval_s: float = 300,
        min_savings_pct: float = 10.0,
        dry_run: bool = True,
    ):
        self._interval = interval_s
        self._min_savings = min_savings_pct
        self._dry_run = dry_run
        self._last_run: float = 0.0
        self._results: list[TuneResult] = []

    def should_run(self) -> bool:
        return (time.monotonic() - self._last_run) >= self._interval

    def analyze_and_tune(self, metrics: list) -> list[TuneResult]:
        """Analyze current GPU metrics and optionally apply power caps.

        Args:
            metrics: List of GPUMetrics from the latest collection cycle.

        Returns:
            List of TuneResult for each GPU analyzed.
        """
        self._last_run = time.monotonic()
        results: list[TuneResult] = []

        for m in metrics:
            result = self._analyze_gpu(m)
            if result:
                results.append(result)

        self._results = results
        return results

    def _analyze_gpu(self, m) -> Optional[TuneResult]:
        """Analyze a single GPU and optionally apply a power cap."""
        power_w = m.power_draw_w
        util = m.utilization_gpu_pct
        power_limit = getattr(m, 'power_limit_w', None) or 0

        if power_limit <= 0 or power_w <= 0:
            return None

        power_ratio = power_w / power_limit

        if util < 30 and power_ratio > 0.5:
            recommended_cap = power_w * 0.8
            savings_pct = ((power_w - recommended_cap) / power_w) * 100

            if savings_pct < self._min_savings:
                return TuneResult(
                    gpu_index=m.gpu_index,
                    gpu_name=getattr(m, 'gpu_name', 'unknown'),
                    current_power_w=power_w,
                    recommended_cap_w=None,
                    estimated_savings_pct=0,
                    applied=False,
                    reason="savings below threshold",
                )

            applied = False
            if not self._dry_run and _NVML:
                try:
                    from efficiency.power_control import set_power_limit
                    set_power_limit(m.gpu_index, int(recommended_cap))
                    applied = True
                    logger.info(
                        "AutoTune GPU %d: applied power cap %dW (was %.0fW, saving ~%.0f%%)",
                        m.gpu_index, int(recommended_cap), power_w, savings_pct,
                    )
                except Exception as exc:
                    logger.warning("AutoTune GPU %d: failed to set power cap: %s", m.gpu_index, exc)
            else:
                logger.info(
                    "AutoTune GPU %d: recommend cap %dW (current %.0fW, util %d%%, save ~%.0f%%)",
                    m.gpu_index, int(recommended_cap), power_w, util, savings_pct,
                )

            return TuneResult(
                gpu_index=m.gpu_index,
                gpu_name=getattr(m, 'gpu_name', 'unknown'),
                current_power_w=power_w,
                recommended_cap_w=recommended_cap,
                estimated_savings_pct=savings_pct,
                applied=applied,
                reason="low utilization with high power draw",
            )

        return None

    @property
    def last_results(self) -> list[TuneResult]:
        return self._results
