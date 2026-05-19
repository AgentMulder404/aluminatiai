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
Scheduler auto-detection.

Probes the environment to determine which scheduler is managing this node
and returns the appropriate adapter instance.

Detection order (first match wins):
1. Run:ai  — RUNAI_JOB_NAME or RUNAI_PROJECT env var
2. Slurm   — SLURM_JOB_ID env var, or squeue binary on PATH
3. K8s     — KUBERNETES_SERVICE_HOST env var
4. None    — NullAdapter (standalone mode)
"""

import os
import shutil
import logging

from .base import SchedulerAdapter, NullAdapter

logger = logging.getLogger(__name__)


def detect_scheduler() -> SchedulerAdapter:
    """
    Auto-detect the active scheduler and return the appropriate adapter.

    Can be overridden with ALUMINATAI_SCHEDULER env var:
      "kubernetes", "slurm", "runai", "none"
    """
    # Manual override
    override = os.getenv("ALUMINATAI_SCHEDULER", "").lower().strip()
    if override:
        return _create_adapter(override)

    # Auto-detection
    # 1. Run:ai (runs on K8s, so check before K8s)
    if os.getenv("RUNAI_JOB_NAME") or os.getenv("RUNAI_PROJECT"):
        return _create_adapter("runai")

    # 2. Slurm
    if os.getenv("SLURM_JOB_ID") or shutil.which("squeue"):
        return _create_adapter("slurm")

    # 3. Kubernetes
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return _create_adapter("kubernetes")

    # 4. No scheduler detected
    logger.info("No scheduler detected — running in standalone mode (no attribution)")
    return NullAdapter()


def _create_adapter(scheduler_type: str) -> SchedulerAdapter:
    """Instantiate the requested adapter, falling back to NullAdapter on error."""
    try:
        if scheduler_type == "kubernetes":
            from .kubernetes import KubernetesAdapter
            adapter = KubernetesAdapter()
            logger.info(f"Scheduler adapter: {adapter.name}")
            return adapter

        elif scheduler_type == "slurm":
            from .slurm import SlurmAdapter
            adapter = SlurmAdapter()
            logger.info(f"Scheduler adapter: {adapter.name}")
            return adapter

        elif scheduler_type == "runai":
            from .runai import RunaiAdapter
            adapter = RunaiAdapter()
            logger.info(f"Scheduler adapter: {adapter.name}")
            return adapter

        elif scheduler_type == "none":
            logger.info("Scheduler explicitly set to none — standalone mode")
            return NullAdapter()

        else:
            logger.warning(f"Unknown scheduler type '{scheduler_type}', using standalone")
            return NullAdapter()

    except Exception as e:
        logger.error(
            f"Failed to initialize {scheduler_type} adapter: {e}. "
            f"Falling back to standalone mode."
        )
        return NullAdapter()
