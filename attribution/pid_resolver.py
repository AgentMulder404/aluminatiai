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
PidResolver: Map a running process to a JobMetadata via scheduler back-lookup.

Resolution priority (first match wins):
  1. SLURM_JOB_ID in environ  → SlurmAdapter.resolve_job()
  2. RUNAI_JOB_NAME in environ → RunaiAdapter.resolve_job()
  3. KUBERNETES_SERVICE_HOST   → read cgroup for pod UID → KubernetesAdapter.resolve_pod_by_uid()
  4. ALUMINATAI_TEAM + ALUMINATAI_MODEL env vars (manual override, anti-spoof checked)
  5. Custom attribution rules file (ALUMINATAI_ATTRIBUTION_CONFIG)
  6. Built-in cmdline heuristics (jupyter, vllm, torchserve, …)
  7. None (unresolved — power is still tracked under "pid:<pid>")
"""

import logging
import re
import sys
from typing import Optional, TYPE_CHECKING

from .process_probe import ProcessInfo

if TYPE_CHECKING:
    from schedulers.base import SchedulerAdapter, JobMetadata

logger = logging.getLogger(__name__)

_IS_LINUX = sys.platform.startswith("linux")

# Load TRUSTED_UIDS from config (guarded so tests can run without config on path)
try:
    from config import TRUSTED_UIDS as _TRUSTED_UIDS
except ImportError:
    _TRUSTED_UIDS: set[int] = set()

# Built-in cmdline heuristics: (compiled_regex, team_id, model_tag)
# Checked in order; first match wins. Confidence → "heuristic".
_BUILTIN_HEURISTICS: list[tuple[re.Pattern, str, str]] = [
    # Notebooks / IDEs
    (re.compile(r"jupyter",                         re.IGNORECASE), "jupyter",      "notebook"),
    (re.compile(r"ipykernel",                       re.IGNORECASE), "jupyter",      "notebook"),
    (re.compile(r"code.*server|vscode.*server",     re.IGNORECASE), "vscode",       "ide"),
    # Inference servers
    (re.compile(r"vllm",                            re.IGNORECASE), "unknown",      "vllm-serve"),
    (re.compile(r"torchserve",                      re.IGNORECASE), "unknown",      "torchserve"),
    (re.compile(r"tritonserver",                    re.IGNORECASE), "unknown",      "triton"),
    (re.compile(r"text.generation.inference|tgi",   re.IGNORECASE), "unknown",      "tgi-serve"),
    (re.compile(r"ollama",                          re.IGNORECASE), "unknown",      "ollama-serve"),
    (re.compile(r"llama.cpp|llama-server",          re.IGNORECASE), "unknown",      "llamacpp-serve"),
    # Training frameworks
    (re.compile(r"python.*train",                   re.IGNORECASE), "unknown",      "training"),
    (re.compile(r"torchrun|torch\.distributed",     re.IGNORECASE), "unknown",      "distributed-training"),
    (re.compile(r"deepspeed",                       re.IGNORECASE), "unknown",      "deepspeed-training"),
    (re.compile(r"accelerate.*launch",              re.IGNORECASE), "unknown",      "hf-accelerate"),
    (re.compile(r"transformers.*train|run_clm|run_mlm|run_glue", re.IGNORECASE), "unknown", "hf-transformers"),
    (re.compile(r"lightning.*trainer|pl\.trainer",  re.IGNORECASE), "unknown",      "pytorch-lightning"),
    (re.compile(r"ray.*train|ray\.tune",            re.IGNORECASE), "unknown",      "ray-train"),
    (re.compile(r"mlflow.*run",                     re.IGNORECASE), "unknown",      "mlflow-run"),
    (re.compile(r"wandb.*agent",                    re.IGNORECASE), "unknown",      "wandb-sweep"),
    (re.compile(r"tensorflow|tf\.keras|tf2",        re.IGNORECASE), "unknown",      "tensorflow"),
    (re.compile(r"jax.*train|flax.*train",          re.IGNORECASE), "unknown",      "jax-training"),
    # Evaluation / batch jobs
    (re.compile(r"python.*eval|lm.eval|lm_eval",   re.IGNORECASE), "unknown",      "eval"),
    (re.compile(r"python.*inference",               re.IGNORECASE), "unknown",      "inference"),
    (re.compile(r"python.*generate|python.*infer",  re.IGNORECASE), "unknown",      "batch-inference"),
    # Data processing
    (re.compile(r"spark.*submit|pyspark",           re.IGNORECASE), "unknown",      "spark"),
    (re.compile(r"dask.*worker",                    re.IGNORECASE), "unknown",      "dask"),
    # CUDA / GPU utilities (idle-like)
    (re.compile(r"cuda.*memcpy|cudnn",              re.IGNORECASE), "unknown",      "cuda-utility"),
]


class PidResolver:
    def __init__(self, scheduler: "SchedulerAdapter"):
        self._scheduler = scheduler
        # Load custom attribution rules (silently skipped if no config file found)
        from .rules import AttributionRules
        self._rules = AttributionRules()
        self._rules.load()

    def resolve(self, proc: ProcessInfo) -> "Optional[JobMetadata]":
        env = proc.environ

        # 1. Slurm
        slurm_job_id = env.get("SLURM_JOB_ID")
        if slurm_job_id:
            job = self._scheduler.resolve_job(slurm_job_id)
            if job:
                return job

        # 2. Run:ai
        runai_job_name = env.get("RUNAI_JOB_NAME")
        if runai_job_name:
            job = self._scheduler.resolve_job(runai_job_name)
            if job:
                return job

        # 3. Kubernetes (via cgroup → pod UID)
        if env.get("KUBERNETES_SERVICE_HOST") and _IS_LINUX:
            pod_uid = self._read_pod_uid_from_cgroup(proc.pid)
            if pod_uid:
                job = self._scheduler.resolve_pod_by_uid(pod_uid)
                if job:
                    return job

        # 4. Manual ALUMINATAI env vars (with optional anti-spoofing)
        team = env.get("ALUMINATAI_TEAM")
        model = env.get("ALUMINATAI_MODEL", "untagged")
        if team:
            if _TRUSTED_UIDS and proc.owner_uid not in _TRUSTED_UIDS:
                logger.warning(
                    "PID %d claims ALUMINATAI_TEAM=%r but UID %d is not in TRUSTED_UIDS"
                    " — skipping manual tag",
                    proc.pid, team, proc.owner_uid,
                )
                # Fall through to rules / heuristic
            else:
                from schedulers.base import JobMetadata
                return JobMetadata(
                    job_id=f"manual-pid-{proc.pid}",
                    job_name=f"pid-{proc.pid}",
                    team_id=team,
                    model_tag=model,
                    scheduler_source="manual",
                    gpu_indices=[],
                )

        # 5. Custom attribution rules
        rule = self._rules.match(proc.cmdline)
        if rule:
            from schedulers.base import JobMetadata
            return JobMetadata(
                job_id=f"rules-pid-{proc.pid}",
                job_name=f"pid-{proc.pid}",
                team_id=rule.team,
                model_tag=rule.model,
                scheduler_source="rules",
                gpu_indices=[],
            )

        # 6. Built-in heuristics
        return self._resolve_heuristic(proc)

    def _resolve_heuristic(self, proc: ProcessInfo) -> "Optional[JobMetadata]":
        """Match cmdline against built-in heuristic patterns."""
        cmdline = proc.cmdline
        if not cmdline:
            return None
        for pattern, team, model in _BUILTIN_HEURISTICS:
            if pattern.search(cmdline):
                from schedulers.base import JobMetadata
                return JobMetadata(
                    job_id=f"heuristic-pid-{proc.pid}",
                    job_name=f"pid-{proc.pid}",
                    team_id=team,
                    model_tag=model,
                    scheduler_source="heuristic",
                    gpu_indices=[],
                )
        return None

    def _read_pod_uid_from_cgroup(self, pid: int) -> Optional[str]:
        """
        Parse the Kubernetes pod UID from /proc/<pid>/cgroup.

        cgroup v1 example:
          12:devices:/kubepods/burstable/pod<uid>/container-<id>
        cgroup v2 example:
          0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod<uid>.slice/...
        """
        path = f"/proc/{pid}/cgroup"
        try:
            with open(path, "r") as f:
                content = f.read()
        except OSError:
            return None

        # Match "pod<uuid>" pattern (RFC 4122 UUID)
        try:
            match = re.search(
                r"pod([0-9a-f]{8}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{12})",
                content,
                re.IGNORECASE,
            )
        except (TypeError, re.error):
            return None
        if match:
            # Normalise underscores → hyphens (cgroup v2 sometimes uses underscores)
            return match.group(1).replace("_", "-")
        return None
