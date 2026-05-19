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
Hardware Match Scorer.

Computes a 0-100 score answering: "Is this model running on the most
energy-efficient GPU for its workload profile?"

The score is grounded in the roofline model — it compares the Joules/TFLOP
of the current GPU to the theoretical best GPU across the fleet for the
same workload's math intensity.

Score interpretation:
  100  = This IS the most efficient hardware. No action needed.
  80+  = Good match. Minor gains possible.
  50-79 = Significant energy waste. Consider migration.
  <50  = Poor match. Hardware is fundamentally wrong for this workload.

Example:
  BERT-large (memory-bound, low math intensity) on A100:
    - A100 has massive FP16 compute that BERT can't saturate
    - Score: ~41 — the A100's compute goes to waste as heat
    - Recommendation: Use a bandwidth-optimized GPU or batch more aggressively

  Llama-3-70B (compute-bound, high math intensity) on H100:
    - H100's Transformer Engine is purpose-built for this
    - Score: 100 — optimal pairing
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .gpu_specs import (
    ArchSpec,
    ModelProfile,
    GPU_ARCHITECTURES,
    MODEL_PROFILES,
    resolve_arch,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a hardware match score computation."""

    score: float                     # 0-100
    current_arch: str
    current_jpt: float               # Joules/TFLOP on current hardware
    best_arch: str                   # Most efficient arch for this workload
    best_jpt: float                  # Joules/TFLOP on best hardware
    energy_savings_pct: float        # % energy reduction if migrated to best
    recommendation: str              # Human-readable recommendation
    model_tag: str
    is_memory_bound: bool
    math_intensity: float


class HardwareMatchScorer:
    """
    Computes Hardware Match Scores for workload-GPU pairings.

    Uses the roofline model to evaluate how well a GPU architecture
    matches a workload's arithmetic intensity profile.

    Two modes:
    1. Known model: looks up ModelProfile from the registry
    2. Observed behavior: infers profile from actual utilization/power data
    """

    def __init__(self, db=None):
        """
        Args:
            db: Optional database connection for persisting scores
                and loading custom model profiles.
        """
        self._db = db

    def score(
        self,
        gpu_arch: str,
        model_tag: str,
        avg_utilization: Optional[float] = None,
        avg_power_w: Optional[float] = None,
    ) -> Optional[MatchResult]:
        """
        Compute the Hardware Match Score for a model-GPU pairing.

        Args:
            gpu_arch: GPU architecture name (e.g., "A100-SXM4-80GB")
                      or NVML gpu_name (e.g., "NVIDIA A100-SXM4-80GB")
            model_tag: Model identifier (e.g., "llama-3-70b")
            avg_utilization: Observed average GPU utilization (0-100).
                             If None, uses the model profile's typical midpoint.
            avg_power_w: Observed average power draw.
                         If None, uses the power model estimate.

        Returns:
            MatchResult with score, recommendation, and comparison data.
            None if the model or architecture is unknown.
        """
        # Resolve GPU architecture
        spec = GPU_ARCHITECTURES.get(gpu_arch) or resolve_arch(gpu_arch)
        if not spec:
            logger.warning(f"Unknown GPU architecture: {gpu_arch}")
            return None

        # Resolve model profile
        model = MODEL_PROFILES.get(model_tag)
        if not model:
            model = self._try_family_match(model_tag)
        if not model:
            logger.debug(f"Unknown model: {model_tag}, cannot compute match score")
            return None

        # Determine utilization
        if avg_utilization is not None:
            util_frac = max(0.01, min(avg_utilization / 100.0, 1.0))
        else:
            util_frac = model.typical_util_mid

        # Compute J/TFLOP for CURRENT hardware
        current_jpt = self._compute_jpt(spec, model, util_frac, avg_power_w)

        # Find BEST hardware across the fleet
        best_arch_name = spec.name
        best_jpt = current_jpt

        for candidate_name, candidate_spec in GPU_ARCHITECTURES.items():
            # Estimate utilization this workload would achieve on candidate
            candidate_util = self._estimate_util_on_arch(model, candidate_spec)
            candidate_jpt = self._compute_jpt(
                candidate_spec, model, candidate_util, None
            )

            if candidate_jpt < best_jpt:
                best_jpt = candidate_jpt
                best_arch_name = candidate_name

        # Score: 100 × (best / current)
        if current_jpt <= 0 or current_jpt == float('inf'):
            raw_score = 0.0
        else:
            raw_score = 100.0 * (best_jpt / current_jpt)

        score = round(min(raw_score, 100.0), 2)
        savings = round(max(0, (1.0 - best_jpt / current_jpt) * 100), 1) if current_jpt > 0 else 0.0

        recommendation = self._generate_recommendation(
            score, spec.name, best_arch_name, model, savings
        )

        return MatchResult(
            score=score,
            current_arch=spec.name,
            current_jpt=round(current_jpt, 4),
            best_arch=best_arch_name,
            best_jpt=round(best_jpt, 4),
            energy_savings_pct=savings,
            recommendation=recommendation,
            model_tag=model.tag,
            is_memory_bound=model.is_memory_bound,
            math_intensity=model.math_intensity,
        )

    def score_all_combinations(self) -> list[MatchResult]:
        """
        Compute match scores for every model × architecture combination.

        Useful for populating the hardware_match_scores table.
        """
        results: list[MatchResult] = []
        for model_tag in MODEL_PROFILES:
            for arch_name in GPU_ARCHITECTURES:
                result = self.score(arch_name, model_tag)
                if result:
                    results.append(result)
        return results

    def persist_scores(self, results: list[MatchResult]) -> int:
        """Upsert match scores into the hardware_match_scores table."""
        if not self._db:
            raise RuntimeError("Database connection required for persist_scores()")

        count = 0
        for r in results:
            self._db.query("""
                INSERT INTO hardware_match_scores
                    (model_tag, gpu_arch, match_score, joules_per_tflop,
                     best_arch, recommendation, computed_at)
                VALUES (%(model)s, %(arch)s, %(score)s, %(jpt)s,
                        %(best)s, %(rec)s, NOW())
                ON CONFLICT (model_tag, gpu_arch)
                DO UPDATE SET
                    match_score = EXCLUDED.match_score,
                    joules_per_tflop = EXCLUDED.joules_per_tflop,
                    best_arch = EXCLUDED.best_arch,
                    recommendation = EXCLUDED.recommendation,
                    computed_at = NOW()
            """, {
                'model': r.model_tag,
                'arch': r.current_arch,
                'score': r.score,
                'jpt': r.current_jpt,
                'best': r.best_arch,
                'rec': r.recommendation,
            })
            count += 1

        logger.info(f"Persisted {count} hardware match scores")
        return count

    def rank_gpus_for_model(self, model_tag: str) -> list[dict]:
        """
        Rank all GPU architectures by efficiency for a given model.

        Returns a list sorted by J/TFLOP (most efficient first) with:
          - arch_name, score, jpt, power_w, recommendation
        """
        model = MODEL_PROFILES.get(model_tag)
        if not model:
            return []

        rankings = []
        for arch_name, spec in GPU_ARCHITECTURES.items():
            result = self.score(arch_name, model_tag)
            if result:
                rankings.append({
                    'arch_name': arch_name,
                    'family': spec.family,
                    'score': result.score,
                    'joules_per_tflop': result.current_jpt,
                    'energy_savings_vs_worst': 0.0,  # Filled below
                    'recommendation': result.recommendation,
                })

        # Sort by efficiency
        rankings.sort(key=lambda x: x['joules_per_tflop'])

        # Compute savings vs worst
        if rankings:
            worst_jpt = rankings[-1]['joules_per_tflop']
            for r in rankings:
                if worst_jpt > 0:
                    r['energy_savings_vs_worst'] = round(
                        (1.0 - r['joules_per_tflop'] / worst_jpt) * 100, 1
                    )

        return rankings

    # ── Internal ─────────────────────────────────────────────────────

    def _compute_jpt(
        self,
        spec: ArchSpec,
        model: ModelProfile,
        util_frac: float,
        observed_power_w: Optional[float],
    ) -> float:
        """
        Compute Joules/TFLOP for a specific arch-model-utilization combination.

        Uses the roofline model for effective TFLOPS and either observed or
        estimated power draw.
        """
        # Effective TFLOPS via roofline model
        effective_tflops = spec.roofline_tflops(
            model.math_intensity, util_frac, model.precision
        )

        if effective_tflops <= 0:
            return float('inf')

        # Power: use observed if available, otherwise estimate
        if observed_power_w is not None and observed_power_w > 0:
            power_w = observed_power_w
        else:
            power_w = spec.estimated_power_at_utilization(util_frac)

        return power_w / effective_tflops

    def _estimate_util_on_arch(
        self,
        model: ModelProfile,
        spec: ArchSpec,
    ) -> float:
        """
        Estimate what utilization a model would achieve on a different GPU.

        Memory-bound workloads: utilization driven by memory bandwidth ratio.
        Compute-bound workloads: use model's typical midpoint utilization.
        """
        if model.is_memory_bound:
            # Memory-bound: higher BW GPUs can keep compute busier
            # Scale utilization relative to A100 baseline (2039 GB/s)
            baseline_bw = 2039.0
            bw_ratio = spec.memory_bw_gbps / baseline_bw
            # But utilization can't exceed typical range
            scaled = model.typical_util_mid * min(bw_ratio, 1.3)
            return min(scaled, model.typical_util_max / 100.0)
        else:
            # Compute-bound: use typical midpoint
            return model.typical_util_mid

    def _try_family_match(self, model_tag: str) -> Optional[ModelProfile]:
        """
        Try to match an unknown model_tag to a known family.

        E.g., "llama-3-13b" isn't in the registry but "llama" family is.
        """
        tag_lower = model_tag.lower()
        for known_tag, profile in MODEL_PROFILES.items():
            # Check if the family name appears in the tag
            if profile.family.lower() in tag_lower:
                logger.debug(
                    f"Matched '{model_tag}' to family '{profile.family}' "
                    f"via profile '{known_tag}'"
                )
                return profile
        return None

    def _generate_recommendation(
        self,
        score: float,
        current_arch: str,
        best_arch: str,
        model: ModelProfile,
        savings_pct: float,
    ) -> str:
        """Generate a human-readable recommendation based on the match score."""
        if score >= 95:
            return (
                f"Optimal. {current_arch} is the best match for "
                f"{model.tag} workloads."
            )

        if score >= 80:
            if current_arch == best_arch:
                return (
                    f"Good match. {current_arch} is well-suited for {model.tag}. "
                    f"Minor efficiency gains possible through utilization tuning."
                )
            return (
                f"Good match. {best_arch} would be ~{savings_pct:.0f}% more "
                f"efficient, but {current_arch} is reasonable."
            )

        if score >= 50:
            reason = self._explain_mismatch(current_arch, best_arch, model)
            return (
                f"Inefficient. Migrate to {best_arch} to reduce energy by "
                f"~{savings_pct:.0f}%. {reason}"
            )

        reason = self._explain_mismatch(current_arch, best_arch, model)
        return (
            f"Poor match. {current_arch} wastes significant energy on "
            f"{model.tag}. {best_arch} would save ~{savings_pct:.0f}%. {reason}"
        )

    def _explain_mismatch(
        self, current: str, best: str, model: ModelProfile
    ) -> str:
        """Explain WHY the hardware is mismatched for this workload."""
        current_spec = GPU_ARCHITECTURES.get(current)
        best_spec = GPU_ARCHITECTURES.get(best)
        if not current_spec or not best_spec:
            return ""

        if model.is_memory_bound:
            if current_spec.memory_bw_gbps < best_spec.memory_bw_gbps:
                return (
                    f"{model.tag} is memory-bandwidth-bound "
                    f"(intensity: {model.math_intensity} FLOP/byte). "
                    f"{best} has {best_spec.memory_bw_gbps} GB/s vs "
                    f"{current_spec.memory_bw_gbps} GB/s."
                )
            return (
                f"{model.tag} is memory-bound — {current}'s compute "
                f"capacity ({current_spec.fp16_tflops} TFLOPS) is underutilized."
            )

        # Compute-bound
        if current_spec.fp16_tflops < best_spec.fp16_tflops:
            return (
                f"{model.tag} is compute-bound "
                f"(intensity: {model.math_intensity} FLOP/byte). "
                f"{best} delivers {best_spec.fp16_tflops} vs "
                f"{current_spec.fp16_tflops} FP16 TFLOPS."
            )

        # Same or more compute, but less efficient (higher TDP per TFLOP)
        current_tflop_per_watt = current_spec.fp16_tflops / current_spec.tdp_w
        best_tflop_per_watt = best_spec.fp16_tflops / best_spec.tdp_w
        return (
            f"{best} achieves {best_tflop_per_watt:.1f} TFLOPS/W vs "
            f"{current_tflop_per_watt:.1f} TFLOPS/W on {current}."
        )
