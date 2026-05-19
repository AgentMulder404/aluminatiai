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
Slurm scheduler adapter.

Intercepts job metadata from Slurm using squeue and scontrol commands.
Maps GPU allocations to jobs via GRES assignments and cgroup inspection.

Attribution convention:
  team_id:   Slurm account (--account=ml-research)
  model_tag: ALUMINATAI_MODEL env var, or --comment="model:llama-3-70b"
  user:      Slurm username

Works in two modes:
  1. Inside a job: reads SLURM_JOB_ID from environment (single job)
  2. On a shared node: polls squeue for all GPU jobs on this node
"""

import os
import re
import subprocess
import socket
import logging
from typing import Optional

from .base import SchedulerAdapter, JobMetadata

logger = logging.getLogger(__name__)


class SlurmAdapter(SchedulerAdapter):
    """
    Intercepts Slurm job metadata for GPU energy attribution.

    Correlation strategy:
    1. Detect if we're inside a Slurm job (SLURM_JOB_ID set)
    2. Otherwise, poll squeue for GPU jobs on this node
    3. Resolve GPU indices from scontrol GRES_IDX or CUDA_VISIBLE_DEVICES
    4. Extract team from Slurm account, model from comment/env
    """

    def __init__(self):
        self._hostname = socket.gethostname().split(".")[0]
        self._gpu_job_map: dict[int, JobMetadata] = {}
        self._inside_job = bool(os.getenv("SLURM_JOB_ID"))

        if self._inside_job:
            logger.info(
                f"SlurmAdapter: running inside job {os.getenv('SLURM_JOB_ID')}"
            )
        else:
            logger.info(
                f"SlurmAdapter: node-level mode on {self._hostname}"
            )

    def discover_jobs(self) -> list[JobMetadata]:
        if self._inside_job:
            return self._discover_local_job()
        return self._discover_node_jobs()

    def gpu_to_job(self, gpu_index: int) -> Optional[JobMetadata]:
        return self._gpu_job_map.get(gpu_index)

    # ── Discovery modes ──────────────────────────────────────────────

    def _discover_local_job(self) -> list[JobMetadata]:
        """When agent runs inside a Slurm job, read env vars directly."""
        job_id = os.getenv("SLURM_JOB_ID", "")
        job_name = os.getenv("SLURM_JOB_NAME", "unknown")
        account = os.getenv("SLURM_JOB_ACCOUNT", "default")
        user = os.getenv("SLURM_JOB_USER", os.getenv("USER", "unknown"))

        gpu_indices = self._resolve_local_gpus()
        model_tag = self._extract_model_tag_from_env()
        start_time = self._get_job_start_time(job_id)

        metadata = JobMetadata(
            job_id=f"slurm-{job_id}",
            job_name=job_name,
            team_id=account,
            model_tag=model_tag,
            scheduler_source="slurm",
            gpu_indices=gpu_indices,
            user_email=f"{user}@cluster",
            start_time=start_time,
        )

        self._gpu_job_map = {idx: metadata for idx in gpu_indices}
        return [metadata]

    def _discover_node_jobs(self) -> list[JobMetadata]:
        """Poll squeue for all GPU jobs running on this node."""
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "--nodelist", self._hostname,
                    "--states", "RUNNING",
                    "--format", "%A|%j|%u|%a|%b|%S|%k",
                    "--noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            logger.error("squeue not found — is Slurm installed?")
            return []
        except subprocess.TimeoutExpired:
            logger.error("squeue timed out")
            return []

        if result.returncode != 0:
            logger.error(f"squeue failed: {result.stderr.strip()}")
            return []

        jobs: list[JobMetadata] = []
        new_map: dict[int, JobMetadata] = {}

        for line in result.stdout.strip().split("
"):
            if not line.strip():
                continue

            parts = line.strip().split("|")
            if len(parts) < 7:
                logger.warning(f"Unexpected squeue output: {line}")
                continue

            job_id, name, user, account, gres, start, comment = parts[:7]

            # Skip jobs without GPU GRES
            if "gpu" not in gres.lower():
                continue

            gpu_indices = self._resolve_job_gpus(job_id)
            model_tag = self._parse_model_from_comment(comment, job_id)

            metadata = JobMetadata(
                job_id=f"slurm-{job_id}",
                job_name=name.strip(),
                team_id=account.strip() or "default",
                model_tag=model_tag,
                scheduler_source="slurm",
                gpu_indices=gpu_indices,
                user_email=f"{user.strip()}@cluster",
                start_time=start.strip(),
            )

            jobs.append(metadata)
            for idx in gpu_indices:
                new_map[idx] = metadata

        self._gpu_job_map = new_map
        logger.debug(f"Discovered {len(jobs)} Slurm GPU jobs")
        return jobs

    # ── GPU index resolution ─────────────────────────────────────────

    def _resolve_local_gpus(self) -> list[int]:
        """Resolve GPU indices when running inside a job."""
        # CUDA_VISIBLE_DEVICES is the most reliable inside a job
        cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if cuda_devs:
            try:
                return [int(i.strip()) for i in cuda_devs.split(",")]
            except ValueError:
                pass

        # Fallback: SLURM_STEP_GPUS or SLURM_JOB_GPUS
        for var in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "GPU_DEVICE_ORDINAL"):
            val = os.getenv(var, "")
            if val:
                try:
                    return [int(i.strip()) for i in val.split(",")]
                except ValueError:
                    continue

        # Last resort: count from GRES
        gres_count = os.getenv("SLURM_GPUS_ON_NODE", "0")
        try:
            count = int(gres_count)
            return list(range(count))
        except ValueError:
            return []

    def _resolve_job_gpus(self, job_id: str) -> list[int]:
        """Resolve GPU indices for a remote job via scontrol."""
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id, "--details"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []

            output = result.stdout

            # Look for GRES_IDX pattern: gpu(IDX:0-3) or gpu(IDX:0,2)
            match = re.search(r"GRES_IDX=.*?gpu\(IDX:([0-9,\-]+)\)", output)
            if match:
                return self._parse_index_range(match.group(1))

            # Fallback: count GPUs from GRES=gpu:N
            match = re.search(r"GRES=.*?gpu[^:]*:(\d+)", output)
            if match:
                count = int(match.group(1))
                return list(range(count))

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return []

    def _parse_index_range(self, range_str: str) -> list[int]:
        """Parse Slurm index ranges like '0-3' or '0,2,4' or '0-1,4'."""
        indices: list[int] = []
        for part in range_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                indices.extend(range(int(start), int(end) + 1))
            else:
                indices.append(int(part))
        return indices

    # ── Model tag extraction ─────────────────────────────────────────

    def _extract_model_tag_from_env(self) -> str:
        """Read model tag from environment (when inside a job)."""
        # Direct env var
        model = os.getenv("ALUMINATAI_MODEL", "")
        if model:
            return model

        # Check Slurm comment
        comment = os.getenv("SLURM_JOB_COMMENT", "")
        return self._parse_model_from_comment(comment)

    def _parse_model_from_comment(
        self, comment: str, job_id: str = ""
    ) -> str:
        """
        Parse model tag from Slurm job comment.

        Conventions:
          --comment="model:llama-3-70b"
          --comment="llama-3-70b"
        """
        comment = comment.strip()
        if not comment:
            # Try fetching from scontrol if we have a job_id
            if job_id:
                return self._fetch_model_from_scontrol(job_id)
            return "untagged"

        if comment.startswith("model:"):
            return comment[len("model:"):].strip()

        # If comment looks like a model name (no spaces, reasonable length)
        if len(comment) < 64 and " " not in comment:
            return comment

        return "untagged"

    def _fetch_model_from_scontrol(self, job_id: str) -> str:
        """Attempt to read ALUMINATAI_MODEL from job's environment via scontrol."""
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Look for Comment= field
                match = re.search(r"Comment=(\S+)", result.stdout)
                if match:
                    return self._parse_model_from_comment(match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "untagged"

    def _get_job_start_time(self, job_id: str) -> str:
        """Get job start time from scontrol."""
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                match = re.search(r"StartTime=(\S+)", result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return ""

    def resolve_job(self, job_id: str) -> Optional[JobMetadata]:
        """
        Back-lookup a Slurm job by its numeric job ID.

        Checks the local gpu_job_map cache first (populated by discover_jobs),
        then falls back to scontrol for jobs discovered via PID environ.
        """
        # Cache hit: strip "slurm-" prefix used internally
        for cached in self._gpu_job_map.values():
            if cached.job_id == f"slurm-{job_id}" or cached.job_id == job_id:
                return cached

        # Not in cache — query scontrol directly
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            output = result.stdout

            name_match = re.search(r"JobName=(\S+)", output)
            account_match = re.search(r"Account=(\S+)", output)
            user_match = re.search(r"UserId=(\w+)", output)
            start_match = re.search(r"StartTime=(\S+)", output)

            name = name_match.group(1) if name_match else "unknown"
            account = account_match.group(1) if account_match else "default"
            user = user_match.group(1) if user_match else "unknown"
            start = start_match.group(1) if start_match else ""
            model_tag = self._fetch_model_from_scontrol(job_id)
            gpu_indices = self._resolve_job_gpus(job_id)

            return JobMetadata(
                job_id=f"slurm-{job_id}",
                job_name=name,
                team_id=account,
                model_tag=model_tag,
                scheduler_source="slurm",
                gpu_indices=gpu_indices,
                user_email=f"{user}@cluster",
                start_time=start,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    @property
    def name(self) -> str:
        if self._inside_job:
            return f"slurm (job={os.getenv('SLURM_JOB_ID')})"
        return f"slurm (node={self._hostname})"
