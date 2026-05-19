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
AttributionEngine: Resolve GPU power attribution per sample.

For each GPU handle + power reading, returns one or more AttributionResult
objects representing each job's fractional share of the GPU power.

Attribution confidence levels (and numeric scores):
  "tagged"         1.00 — ALUMINATAI_TEAM/MODEL env var explicitly set by the user
  "api_tag"        0.95 — job registered via /api/v1/tag REST endpoint
  "scheduler"      0.90 — resolved via SLURM_JOB_ID / RUNAI_JOB_NAME / K8s pod UID
  "scheduler_poll" 0.75 — resolved via scheduler.gpu_to_job() (legacy poll path)
  "rules"          0.60 — matched by a custom attribution rules file
  "heuristic"      0.40 — matched by a built-in cmdline heuristic
  "memory_split"   0.20 — unresolved; power split proportionally by GPU memory usage
  "idle"           0.30 — GPU is idle; billed to ALUMINATAI_IDLE_TEAM

Resolution priority (step 1 → 7):
  1.   ALUMINATAI_TEAM/MODEL env var on the process                    → tagged (1.0)
  1.5  /api/v1/tag REST registration (via TagClient)                   → api_tag (0.95)
  2.   SLURM_JOB_ID / RUNAI_JOB_NAME / K8s pod UID on the process     → scheduler (0.9)
  3.   scheduler.gpu_to_job() poll                                      → scheduler_poll (0.75)
  4.   custom attribution rules file                                    → rules (0.6)
  5.   built-in cmdline heuristics                                      → heuristic (0.4)
  6.   GPU memory split (all unresolved processes)                      → memory_split (0.2)
  7.   ALUMINATAI_IDLE_TEAM env var fallback                            → idle (0.3)
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from .process_probe import ProcessProbe
from .pid_resolver import PidResolver

if TYPE_CHECKING:
    from schedulers.base import SchedulerAdapter, JobMetadata
    from tag_client import TagClient
    from pid_smoother import PidSmoother

logger = logging.getLogger(__name__)

# Numeric confidence scores (0.0–1.0) for each attribution method.
# Higher = more trustworthy attribution.
CONFIDENCE_SCORES: dict[str, float] = {
    "tagged":          1.00,   # explicit ALUMINATAI_TEAM env var on the process
    "api_tag":         0.95,   # job registered via /api/v1/tag REST endpoint
    "scheduler":       0.90,   # SLURM_JOB_ID / Run:ai / K8s pod UID on the process
    "scheduler_poll":  0.75,   # scheduler.gpu_to_job() fallback poll
    "rules":           0.60,   # custom attribution rules file regex match
    "heuristic":       0.40,   # built-in cmdline heuristic (jupyter, vllm, …)
    "memory_split":    0.20,   # unresolved; power split proportionally by GPU memory
    "idle":            0.30,   # ALUMINATAI_IDLE_TEAM fallback
}

# Estimated ± power-attribution uncertainty (percentage of reported power_w).
# Reflects how much the true power share could deviate from the attributed value:
#   tagged/api_tag  — user set the tag explicitly; measurement noise only (~2–5 %)
#   scheduler       — env var on the process; reliable but kernel timing jitter (~10 %)
#   scheduler_poll  — poll lag can span job boundaries (~20 %)
#   rules           — regex heuristics; may hit wrong process (~25 %)
#   heuristic       — cmdline guesses; significant false-positive risk (~35 %)
#   memory_split    — proportional fallback; real share unknown (~50 %)
#   idle            — catch-all; baseline noise dominates (~15 %)
UNCERTAINTY_PCT: dict[str, float] = {
    "tagged":          2.0,
    "api_tag":         5.0,
    "scheduler":      10.0,
    "scheduler_poll": 20.0,
    "rules":          25.0,
    "heuristic":      35.0,
    "memory_split":   50.0,
    "idle":           15.0,
}


@dataclass
class AttributionResult:
    team_id: str
    model_tag: str
    job_id: str
    scheduler_source: str
    power_w: float
    gpu_fraction: float                   # 0.0–1.0
    energy_delta_j: Optional[float]
    confidence: str        # "tagged"|"api_tag"|"scheduler"|"scheduler_poll"|"rules"|"heuristic"|"memory_split"|"idle"
    confidence_score: float = field(default=0.0)   # numeric confidence in [0, 1]
    uncertainty_pct: float = field(default=0.0)    # ± % of power_w the true attribution could deviate


class AttributionEngine:
    def __init__(
        self,
        probe: ProcessProbe,
        resolver: PidResolver,
        scheduler: "SchedulerAdapter",
        tag_client: "Optional[TagClient]" = None,
        smoother: "Optional[PidSmoother]" = None,
    ):
        self._probe = probe
        self._resolver = resolver
        self._scheduler = scheduler
        self._tag_client = tag_client
        self._smoother = smoother

    def resolve(
        self,
        handle,
        gpu_index: int,
        total_power_w: float,
        energy_delta_j: Optional[float],
        sample_time: Optional[datetime] = None,
    ) -> list[AttributionResult]:
        """
        Return attribution result(s) for one GPU at one sample time.

        Steps:
          1.   Query running compute processes via NVML
          1.5  For each resolved process, check TagClient for a REST-registered tag
          2.   Resolve each process to a job via env vars / scheduler / heuristics
          3.   Fallback: scheduler poll (single winner)
          4.   Fallback: idle attribution if ALUMINATAI_IDLE_TEAM is set
          5.   Return [] if no attribution configured (backward compat)
        """
        if sample_time is None:
            sample_time = datetime.now(timezone.utc)
        elif sample_time.tzinfo is None:
            sample_time = sample_time.replace(tzinfo=timezone.utc)

        processes = self._probe.query(handle, gpu_index)

        # Temporal PID smoothing: filter out transient processes (DDP spawn
        # workers still allocating, one-shot CUDA helpers) that appeared in
        # fewer than `stable_threshold` fraction of the sliding window.
        # Falls back to the raw NVML list when:
        #   a) smoother is disabled, or
        #   b) stable_pids() returns ∅ (cold start / no history yet), or
        #   c) filtering would remove *all* current processes (new job)
        if self._smoother and processes:
            stable = self._smoother.stable_pids(gpu_index)
            if stable:
                filtered = [p for p in processes if p.pid in stable]
                if filtered:
                    processes = filtered
                # else: all current PIDs are new → keep full raw list

        # Maps scheduler_source → confidence string
        _SOURCE_CONFIDENCE = {
            "manual":     "tagged",
            "slurm":      "scheduler",
            "runai":      "scheduler",
            "kubernetes": "scheduler",
            "rules":      "rules",
            "heuristic":  "heuristic",
        }

        if processes:
            # Group by resolved job key, accumulate GPU memory bytes
            by_key: dict[str, tuple[Optional["JobMetadata"], int, str]] = {}
            for proc in processes:
                job = self._resolver.resolve(proc)

                # Step 1.5: check REST-registered tag for this (gpu_index, pid, time)
                api_tag = None
                if self._tag_client is not None:
                    api_tag = self._tag_client.match(
                        gpu_index=gpu_index,
                        pid=proc.pid,
                        ts=sample_time,
                    )

                if api_tag is not None:
                    key = f"tag:{api_tag.id}"
                    _, mem, _src = by_key.get(key, (None, 0, "api_tag"))
                    by_key[key] = (api_tag, mem + proc.gpu_memory_bytes, "api_tag")  # type: ignore[assignment]
                    logger.debug(
                        "GPU %d PID %d → api_tag (id=%s, team=%s)",
                        gpu_index, proc.pid, api_tag.id, getattr(api_tag, 'team_id', '?'),
                    )
                else:
                    key = job.job_id if job else f"pid:{proc.pid}"
                    _, mem, _src = by_key.get(key, (job, 0, ""))
                    by_key[key] = (job, mem + proc.gpu_memory_bytes, "")  # type: ignore[assignment]
                    if job:
                        logger.debug(
                            "GPU %d PID %d → %s (job=%s, team=%s, source=%s)",
                            gpu_index, proc.pid,
                            _SOURCE_CONFIDENCE.get(job.scheduler_source, "unknown"),
                            job.job_id, job.team_id, job.scheduler_source,
                        )
                    else:
                        logger.debug(
                            "GPU %d PID %d → unresolved (memory_split fallback)",
                            gpu_index, proc.pid,
                        )

            total_mem = sum(m for _, m, _ in by_key.values()) or 1
            results: list[AttributionResult] = []

            for key, (job_or_tag, mem, forced_confidence) in by_key.items():
                frac = mem / total_mem

                if forced_confidence == "api_tag":
                    # job_or_tag is a TagRecord here
                    tag = job_or_tag  # type: ignore[assignment]
                    results.append(AttributionResult(
                        team_id=tag.team_id or "unresolved",
                        model_tag=tag.model_tag or "untagged",
                        job_id=tag.job_id,
                        scheduler_source="api_tag",
                        power_w=round(total_power_w * frac, 3),
                        gpu_fraction=round(frac, 4),
                        energy_delta_j=round(energy_delta_j * frac, 4) if energy_delta_j is not None else None,
                        confidence="api_tag",
                        confidence_score=CONFIDENCE_SCORES["api_tag"],
                        uncertainty_pct=UNCERTAINTY_PCT["api_tag"],
                    ))
                    continue

                job = job_or_tag  # type: ignore[assignment]
                if job:
                    team_id = job.team_id
                    model_tag = job.model_tag
                    job_id = job.job_id
                    scheduler_source = job.scheduler_source
                    confidence = _SOURCE_CONFIDENCE.get(scheduler_source, "scheduler")
                else:
                    # Unresolved process — power split proportionally by GPU memory
                    team_id = os.getenv("ALUMINATAI_IDLE_TEAM", "unresolved")
                    model_tag = "untagged"
                    job_id = key
                    scheduler_source = "unresolved"
                    confidence = "memory_split"

                results.append(AttributionResult(
                    team_id=team_id,
                    model_tag=model_tag,
                    job_id=job_id,
                    scheduler_source=scheduler_source,
                    power_w=round(total_power_w * frac, 3),
                    gpu_fraction=round(frac, 4),
                    energy_delta_j=round(energy_delta_j * frac, 4) if energy_delta_j is not None else None,
                    confidence=confidence,
                    confidence_score=CONFIDENCE_SCORES.get(confidence, 0.0),
                    uncertainty_pct=UNCERTAINTY_PCT.get(confidence, 50.0),
                ))

            return results

        # Fallback: scheduler poll (current/old behaviour)
        logger.debug("GPU %d → no processes found, trying scheduler poll fallback", gpu_index)
        job = self._scheduler.gpu_to_job(gpu_index)
        if job:
            logger.debug("GPU %d → scheduler_poll (job=%s, team=%s)", gpu_index, job.job_id, job.team_id)
            return [AttributionResult(
                team_id=job.team_id,
                model_tag=job.model_tag,
                job_id=job.job_id,
                scheduler_source=job.scheduler_source,
                power_w=round(total_power_w, 3),
                gpu_fraction=1.0,
                energy_delta_j=energy_delta_j,
                confidence="scheduler_poll",
                confidence_score=CONFIDENCE_SCORES["scheduler_poll"],
                uncertainty_pct=UNCERTAINTY_PCT["scheduler_poll"],
            )]

        # Fallback: idle
        idle_team = os.getenv("ALUMINATAI_IDLE_TEAM")
        if idle_team:
            logger.debug("GPU %d → idle fallback (team=%s)", gpu_index, idle_team)
            return [AttributionResult(
                team_id=idle_team,
                model_tag="idle",
                job_id="idle",
                scheduler_source="manual",
                power_w=round(total_power_w, 3),
                gpu_fraction=1.0,
                energy_delta_j=energy_delta_j,
                confidence="idle",
                confidence_score=CONFIDENCE_SCORES["idle"],
                uncertainty_pct=UNCERTAINTY_PCT["idle"],
            )]

        # No attribution configured — emit raw (backward compat)
        logger.debug("GPU %d → no attribution (no processes, no scheduler, no idle team)", gpu_index)
        return []
