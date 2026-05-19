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
Efficiency Curve Builder.

Aggregates real GPU metrics from the fleet to build per-architecture
efficiency curves: Joules/TFLOP at each utilization bucket.

These curves answer: "How energy-efficient is GPU arch X at Y% utilization?"

Usage:
  As library:
    builder = EfficiencyCurveBuilder(db_connection)
    curves = builder.build_all()

  As CLI:
    python -m efficiency.curve_builder --db-url postgresql://... --arch A100-SXM4-80GB

  Offline mode (no DB, uses local metrics CSV):
    builder = EfficiencyCurveBuilder()
    curves = builder.build_from_local(Path("data/metrics.csv"))
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .gpu_specs import GPU_ARCHITECTURES, ArchSpec, resolve_arch

logger = logging.getLogger(__name__)

# Utilization bucket width (5% increments → 20 buckets)
BUCKET_WIDTH = 5
MIN_SAMPLES_PER_BUCKET = 50


@dataclass
class EfficiencyPoint:
    """Single data point on an efficiency curve."""

    arch_name: str
    utilization_bucket: int     # 5, 10, 15, ..., 100
    avg_power_w: float
    avg_tflops_achieved: float
    joules_per_tflop: float     # Lower is better
    sample_count: int


class EfficiencyCurveBuilder:
    """
    Builds per-architecture efficiency curves from observed fleet data.

    An efficiency curve maps utilization % → Joules/TFLOP, showing where
    each GPU architecture hits its energy efficiency sweet spot.

    Two modes:
    1. DB-backed: queries gpu_metrics table directly
    2. Local: processes a list of metric dicts (for offline/testing)
    """

    def __init__(self, db=None):
        """
        Args:
            db: Database connection with .query() method.
                If None, only local/offline methods are available.
        """
        self._db = db

    def build_all(self) -> dict[str, list[EfficiencyPoint]]:
        """Build efficiency curves for all architectures found in the fleet."""
        if not self._db:
            raise RuntimeError("Database connection required for build_all()")

        # Discover which architectures exist in the fleet
        rows = self._db.query(
            "SELECT DISTINCT gpu_name FROM gpu_metrics WHERE gpu_name IS NOT NULL"
        )

        curves: dict[str, list[EfficiencyPoint]] = {}
        for row in rows:
            arch = resolve_arch(row['gpu_name'])
            if arch and arch.name not in curves:
                curve = self.build_for_arch(arch.name)
                if curve:
                    curves[arch.name] = curve
                    logger.info(
                        f"Built efficiency curve for {arch.name}: "
                        f"{len(curve)} data points"
                    )

        return curves

    def build_for_arch(self, arch_name: str) -> list[EfficiencyPoint]:
        """
        Build efficiency curve for a single GPU architecture from DB.

        Queries gpu_metrics, buckets by utilization, and computes
        Joules/TFLOP at each bucket using the roofline model.
        """
        if not self._db:
            raise RuntimeError("Database connection required")

        spec = GPU_ARCHITECTURES.get(arch_name)
        if not spec:
            logger.warning(f"Unknown architecture: {arch_name}")
            return []

        # Aggregate metrics bucketed by utilization (5% increments)
        rows = self._db.query("""
            SELECT
                (utilization_gpu_pct / %(bw)s) * %(bw)s AS util_bucket,
                AVG(power_draw_w)                        AS avg_power_w,
                COUNT(*)                                  AS sample_count
            FROM gpu_metrics
            WHERE gpu_name LIKE %(pattern)s
              AND utilization_gpu_pct > 0
              AND power_draw_w > 0
            GROUP BY util_bucket
            HAVING COUNT(*) >= %(min_samples)s
            ORDER BY util_bucket
        """, {
            'bw': BUCKET_WIDTH,
            'pattern': f'%{arch_name}%',
            'min_samples': MIN_SAMPLES_PER_BUCKET,
        })

        return self._rows_to_curve(arch_name, spec, rows)

    def build_from_metrics(
        self,
        arch_name: str,
        metrics: list[dict],
    ) -> list[EfficiencyPoint]:
        """
        Build efficiency curve from a list of metric dictionaries.

        Each dict must have at minimum:
          - utilization_gpu_pct: int
          - power_draw_w: float

        Use this for offline analysis or testing without a database.
        """
        spec = GPU_ARCHITECTURES.get(arch_name)
        if not spec:
            logger.warning(f"Unknown architecture: {arch_name}")
            return []

        # Bucket the metrics
        buckets: dict[int, list[float]] = {}
        for m in metrics:
            util = m.get('utilization_gpu_pct', 0)
            power = m.get('power_draw_w', 0)
            if util <= 0 or power <= 0:
                continue

            bucket = (util // BUCKET_WIDTH) * BUCKET_WIDTH
            bucket = max(BUCKET_WIDTH, min(bucket, 100))  # Clamp to [5, 100]
            buckets.setdefault(bucket, []).append(power)

        # Convert to aggregated rows
        rows = []
        for bucket, powers in sorted(buckets.items()):
            if len(powers) >= MIN_SAMPLES_PER_BUCKET:
                rows.append({
                    'util_bucket': bucket,
                    'avg_power_w': sum(powers) / len(powers),
                    'sample_count': len(powers),
                })

        return self._rows_to_curve(arch_name, spec, rows)

    def build_theoretical(self, arch_name: str) -> list[EfficiencyPoint]:
        """
        Generate a theoretical efficiency curve from hardware specs alone.

        Useful when no observed data exists yet (cold start).
        Uses the power model P(u) = P_idle + (P_tdp - P_idle) * u
        and roofline TFLOPS estimation.
        """
        spec = GPU_ARCHITECTURES.get(arch_name)
        if not spec:
            return []

        curve: list[EfficiencyPoint] = []

        for bucket in range(BUCKET_WIDTH, 100 + BUCKET_WIDTH, BUCKET_WIDTH):
            util_frac = bucket / 100.0
            power_w = spec.estimated_power_at_utilization(util_frac)
            tflops = spec.fp16_tflops * util_frac

            if tflops <= 0:
                continue

            jpt = power_w / tflops

            curve.append(EfficiencyPoint(
                arch_name=arch_name,
                utilization_bucket=bucket,
                avg_power_w=round(power_w, 2),
                avg_tflops_achieved=round(tflops, 4),
                joules_per_tflop=round(jpt, 4),
                sample_count=0,  # Theoretical — no real samples
            ))

        return curve

    def persist_curves(self, curves: dict[str, list[EfficiencyPoint]]) -> int:
        """
        Upsert efficiency curves into the gpu_efficiency_curves table.

        Returns the total number of data points written.
        """
        if not self._db:
            raise RuntimeError("Database connection required for persist_curves()")

        total = 0
        for arch_name, points in curves.items():
            for point in points:
                self._db.query("""
                    INSERT INTO gpu_efficiency_curves
                        (arch_name, utilization_bucket, avg_power_w,
                         avg_tflops_achieved, joules_per_tflop, sample_count, updated_at)
                    VALUES (%(arch)s, %(bucket)s, %(power)s,
                            %(tflops)s, %(jpt)s, %(samples)s, NOW())
                    ON CONFLICT (arch_name, utilization_bucket)
                    DO UPDATE SET
                        avg_power_w = EXCLUDED.avg_power_w,
                        avg_tflops_achieved = EXCLUDED.avg_tflops_achieved,
                        joules_per_tflop = EXCLUDED.joules_per_tflop,
                        sample_count = EXCLUDED.sample_count,
                        updated_at = NOW()
                """, {
                    'arch': point.arch_name,
                    'bucket': point.utilization_bucket,
                    'power': point.avg_power_w,
                    'tflops': point.avg_tflops_achieved,
                    'jpt': point.joules_per_tflop,
                    'samples': point.sample_count,
                })
                total += 1

        logger.info(f"Persisted {total} efficiency curve data points")
        return total

    # ── Internal ─────────────────────────────────────────────────────

    def _rows_to_curve(
        self,
        arch_name: str,
        spec: ArchSpec,
        rows: list[dict],
    ) -> list[EfficiencyPoint]:
        """Convert aggregated utilization-bucket rows into EfficiencyPoints."""
        curve: list[EfficiencyPoint] = []

        for row in rows:
            bucket = int(row['util_bucket'])
            avg_power = float(row['avg_power_w'])
            sample_count = int(row['sample_count'])
            util_frac = bucket / 100.0

            # Estimate achieved TFLOPS from utilization × peak
            tflops = spec.fp16_tflops * util_frac

            if tflops <= 0:
                continue

            # Joules per TFLOP = Watts / TFLOPS
            # (Watts = Joules/second, TFLOPS = TeraFLOP/second, so W/TFLOPS = J/TFLOP)
            jpt = avg_power / tflops

            curve.append(EfficiencyPoint(
                arch_name=arch_name,
                utilization_bucket=bucket,
                avg_power_w=round(avg_power, 2),
                avg_tflops_achieved=round(tflops, 4),
                joules_per_tflop=round(jpt, 4),
                sample_count=sample_count,
            ))

        return curve

    def compare_architectures(
        self,
        arch_names: Optional[list[str]] = None,
        utilization_pct: int = 80,
    ) -> list[dict]:
        """
        Compare efficiency across architectures at a given utilization level.

        Returns a ranked list (most efficient first) with:
          - arch_name
          - joules_per_tflop
          - power_w
          - effective_tflops
          - relative_efficiency (percentage of best)
        """
        if arch_names is None:
            arch_names = list(GPU_ARCHITECTURES.keys())

        util_frac = utilization_pct / 100.0
        results = []

        for name in arch_names:
            spec = GPU_ARCHITECTURES.get(name)
            if not spec:
                continue

            power = spec.estimated_power_at_utilization(util_frac)
            tflops = spec.fp16_tflops * util_frac
            jpt = power / tflops if tflops > 0 else float('inf')

            results.append({
                'arch_name': name,
                'family': spec.family,
                'joules_per_tflop': round(jpt, 4),
                'power_w': round(power, 1),
                'effective_tflops': round(tflops, 1),
                'tdp_w': spec.tdp_w,
            })

        # Sort by efficiency (lowest J/TFLOP first)
        results.sort(key=lambda x: x['joules_per_tflop'])

        # Add relative efficiency
        if results:
            best_jpt = results[0]['joules_per_tflop']
            for r in results:
                if r['joules_per_tflop'] > 0:
                    r['relative_efficiency'] = round(
                        100.0 * best_jpt / r['joules_per_tflop'], 1
                    )
                else:
                    r['relative_efficiency'] = 0.0

        return results
