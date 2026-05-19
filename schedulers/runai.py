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
Run:ai scheduler adapter.

Queries the Run:ai REST API for job metadata and GPU allocations.
Run:ai sits on top of Kubernetes, so this adapter provides richer
metadata than raw K8s pod inspection when Run:ai is the scheduler.

Attribution convention:
  team_id:   Run:ai project name
  model_tag: ALUMINATAI_MODEL env var, or job annotation
  user:      Run:ai job submitter

Configuration via environment:
  RUNAI_API_URL:    Run:ai control plane URL (e.g., https://runai.example.com)
  RUNAI_API_TOKEN:  Bearer token for Run:ai API
  RUNAI_PROJECT:    Current project (set automatically inside Run:ai jobs)
  RUNAI_JOB_NAME:   Current job name (set automatically inside Run:ai jobs)
"""

import os
import socket
import logging
from typing import Optional

from .base import SchedulerAdapter, JobMetadata

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class RunaiAdapter(SchedulerAdapter):
    """
    Intercepts Run:ai job metadata for GPU energy attribution.

    Correlation strategy:
    1. If inside a Run:ai job, read env vars (RUNAI_PROJECT, RUNAI_JOB_NAME)
    2. Otherwise, query Run:ai API for jobs on this node
    3. Map GPU indices from pod spec (Run:ai manages K8s pods)
    """

    def __init__(self):
        self._api_url = os.getenv("RUNAI_API_URL", "").rstrip("/")
        self._api_token = os.getenv("RUNAI_API_TOKEN", "")
        self._hostname = socket.gethostname()
        self._gpu_job_map: dict[int, JobMetadata] = {}
        self._inside_job = bool(os.getenv("RUNAI_JOB_NAME"))

        if self._inside_job:
            logger.info(
                f"RunaiAdapter: running inside job '{os.getenv('RUNAI_JOB_NAME')}' "
                f"in project '{os.getenv('RUNAI_PROJECT')}'"
            )
        elif self._api_url and self._api_token:
            logger.info(f"RunaiAdapter: API mode, endpoint {self._api_url}")
        else:
            logger.warning(
                "RunaiAdapter: no RUNAI_JOB_NAME or RUNAI_API_URL/TOKEN set. "
                "Attribution will be limited."
            )

    def discover_jobs(self) -> list[JobMetadata]:
        if self._inside_job:
            return self._discover_local_job()
        if self._api_url and self._api_token:
            return self._discover_via_api()
        return []

    def gpu_to_job(self, gpu_index: int) -> Optional[JobMetadata]:
        return self._gpu_job_map.get(gpu_index)

    # ── Discovery modes ──────────────────────────────────────────────

    def _discover_local_job(self) -> list[JobMetadata]:
        """Read job metadata from Run:ai-injected environment variables."""
        job_name = os.getenv("RUNAI_JOB_NAME", "unknown")
        project = os.getenv("RUNAI_PROJECT", "default")
        user = os.getenv("RUNAI_USER", os.getenv("USER", "unknown"))
        job_id = os.getenv("RUNAI_JOB_UUID", f"runai-{job_name}")
        model_tag = os.getenv("ALUMINATAI_MODEL", "untagged")

        gpu_indices = self._resolve_local_gpus()

        metadata = JobMetadata(
            job_id=job_id,
            job_name=job_name,
            team_id=project,
            model_tag=model_tag,
            scheduler_source="runai",
            gpu_indices=gpu_indices,
            user_email=f"{user}@cluster",
            start_time="",
        )

        self._gpu_job_map = {idx: metadata for idx in gpu_indices}
        return [metadata]

    def _discover_via_api(self) -> list[JobMetadata]:
        """Query Run:ai API for active jobs on this node."""
        if not REQUESTS_AVAILABLE:
            logger.error("requests package required for Run:ai API mode")
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            }

            response = requests.get(
                f"{self._api_url}/api/v1/jobs",
                headers=headers,
                params={"status": "Running", "nodeName": self._hostname},
                timeout=10,
            )

            if response.status_code != 200:
                logger.error(
                    f"Run:ai API returned {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return []

            data = response.json()

        except requests.RequestException as e:
            logger.error(f"Run:ai API request failed: {e}")
            return []

        jobs: list[JobMetadata] = []
        new_map: dict[int, JobMetadata] = {}

        for job_data in data.get("jobs", data if isinstance(data, list) else []):
            gpu_count = job_data.get("gpuCount", 0) or job_data.get("totalGpuCount", 0)
            if gpu_count == 0:
                continue

            # Run:ai exposes GPU allocation per job
            gpu_indices = self._extract_gpu_indices(job_data, gpu_count)
            annotations = job_data.get("annotations", {})

            metadata = JobMetadata(
                job_id=job_data.get("uid", job_data.get("name", "unknown")),
                job_name=job_data.get("name", "unknown"),
                team_id=job_data.get("project", "default"),
                model_tag=annotations.get("aluminatai.io/model", "untagged"),
                scheduler_source="runai",
                gpu_indices=gpu_indices,
                user_email=job_data.get("user", "unknown"),
                start_time=job_data.get("createdAt", ""),
            )

            jobs.append(metadata)
            for idx in gpu_indices:
                new_map[idx] = metadata

        self._gpu_job_map = new_map
        logger.debug(f"Discovered {len(jobs)} Run:ai GPU jobs")
        return jobs

    # ── GPU index resolution ─────────────────────────────────────────

    def _resolve_local_gpus(self) -> list[int]:
        """Resolve GPU indices when running inside a Run:ai job."""
        # CUDA_VISIBLE_DEVICES is set by Run:ai's device plugin
        cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if cuda_devs:
            try:
                return [int(i.strip()) for i in cuda_devs.split(",")]
            except ValueError:
                pass

        # Fallback: NVIDIA_VISIBLE_DEVICES
        nvidia_devs = os.getenv("NVIDIA_VISIBLE_DEVICES", "")
        if nvidia_devs and nvidia_devs not in ("all", "none", "void"):
            try:
                return [int(i.strip()) for i in nvidia_devs.split(",")]
            except ValueError:
                pass

        # Last resort: RUNAI_NUM_OF_GPUS
        num_gpus = os.getenv("RUNAI_NUM_OF_GPUS", "0")
        try:
            return list(range(int(num_gpus)))
        except ValueError:
            return []

    def _extract_gpu_indices(
        self, job_data: dict, gpu_count: int
    ) -> list[int]:
        """Extract GPU indices from Run:ai API job response."""
        # Some Run:ai versions expose allocated device indices
        alloc = job_data.get("allocatedGpus", [])
        if alloc:
            try:
                return [int(g.get("index", i)) for i, g in enumerate(alloc)]
            except (ValueError, AttributeError):
                pass

        # Fallback: sequential
        return list(range(gpu_count))

    def resolve_job(self, job_name: str) -> Optional[JobMetadata]:
        """
        Back-lookup a Run:ai job by its name for PidResolver attribution.

        Checks the gpu_job_map cache (keyed by job name from API response).
        """
        for cached in self._gpu_job_map.values():
            if cached.job_name == job_name or cached.job_id == job_name:
                return cached
        return None

    @property
    def name(self) -> str:
        if self._inside_job:
            return f"runai (job={os.getenv('RUNAI_JOB_NAME')})"
        return f"runai (api={self._api_url or 'unconfigured'})"
