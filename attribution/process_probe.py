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
ProcessProbe: Query NVML for compute processes on a GPU device and
read their environment variables from /proc/<pid>/environ (Linux).
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)

_IS_LINUX = sys.platform.startswith("linux")

# Only these env var keys are retained from GPU process environments.
# All others (AWS_SECRET_ACCESS_KEY, database URLs, tokens, etc.) are dropped.
_ENVIRON_ALLOWLIST = frozenset({
    "SLURM_JOB_ID",
    "RUNAI_JOB_NAME",
    "KUBERNETES_SERVICE_HOST",
    "ALUMINATAI_TEAM",
    "ALUMINATAI_MODEL",
})
_ALUMINATAI_PREFIX = "ALUMINATAI_"


def _filter_environ(env: dict) -> dict:
    """Return only attribution-relevant env vars; drop secrets and noise."""
    return {k: v for k, v in env.items()
            if k in _ENVIRON_ALLOWLIST or k.startswith(_ALUMINATAI_PREFIX)}


@dataclass
class ProcessInfo:
    pid: int
    gpu_memory_bytes: int
    environ: dict[str, str] = field(default_factory=dict)
    cmdline: str = ""        # /proc/<pid>/cmdline joined with spaces (Linux only)
    owner_uid: int = -1      # effective UID from /proc/<pid>/status (-1 = unknown)


class ProcessProbe:
    """
    Queries NVML for compute processes on a GPU handle and enriches each
    with environment variables read from /proc/<pid>/environ.

    On non-Linux platforms (Windows, macOS) environ is always empty and
    the caller falls back to scheduler-poll attribution gracefully.
    """

    def query(self, handle, gpu_index: int) -> list[ProcessInfo]:
        """
        Return a list of ProcessInfo for each compute process on this GPU.

        Args:
            handle:    pynvml device handle
            gpu_index: for logging only

        Returns:
            List of ProcessInfo (may be empty if no processes or NVML error)
        """
        if not NVML_AVAILABLE:
            return []

        try:
            nvml_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError as e:
            logger.debug(f"GPU {gpu_index}: nvmlDeviceGetComputeRunningProcesses failed: {e}")
            nvml_procs = []

        # Fallback: some MIG configurations and inference servers (TensorRT, graphics APIs)
        # only appear under graphics processes, not compute.
        if not nvml_procs:
            try:
                nvml_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                if nvml_procs:
                    logger.debug(f"GPU {gpu_index}: using graphics processes fallback ({len(nvml_procs)} procs)")
            except pynvml.NVMLError:
                pass

        if not nvml_procs:
            return []

        results: list[ProcessInfo] = []
        for p in nvml_procs:
            if _IS_LINUX:
                environ = self._read_environ(p.pid)
                # Inherit ALUMINATAI_* tags from ancestor processes (e.g. launcher scripts)
                ancestor_env = self._walk_parent_environ(p.pid)
                for k, v in ancestor_env.items():
                    if k not in environ:
                        environ[k] = v
                cmdline = self._read_cmdline(p.pid)
                owner_uid = self._read_owner_uid(p.pid)
            else:
                environ = {}
                cmdline = ""
                owner_uid = -1

            results.append(ProcessInfo(
                pid=p.pid,
                gpu_memory_bytes=p.usedGpuMemory or 0,
                environ=environ,
                cmdline=cmdline,
                owner_uid=owner_uid,
            ))

        return results

    def _read_environ(self, pid: int) -> dict[str, str]:
        """
        Read /proc/<pid>/environ and parse into a key=value dict.

        Returns empty dict on PermissionError (different user) or if
        the process has already exited (FileNotFoundError).
        """
        path = f"/proc/{pid}/environ"
        try:
            with open(path, "rb") as f:
                data = f.read()
        except PermissionError:
            logger.debug(f"PID {pid}: no permission to read environ (different user)")
            return {}
        except FileNotFoundError:
            logger.debug(f"PID {pid}: process exited before environ read")
            return {}
        except OSError as e:
            logger.debug(f"PID {pid}: error reading environ: {e}")
            return {}

        env: dict[str, str] = {}
        for entry in data.split(b" "):
            if b"=" in entry:
                k, _, v = entry.partition(b"=")
                try:
                    env[k.decode("utf-8", errors="replace")] = v.decode("utf-8", errors="replace")
                except (UnicodeDecodeError, ValueError):
                    pass  # malformed env entry — skip
        return _filter_environ(env)

    def _read_cmdline(self, pid: int) -> str:
        """
        Read /proc/<pid>/cmdline (null-separated argv), capped at 4096 bytes.

        Returns empty string on any error.
        """
        path = f"/proc/{pid}/cmdline"
        try:
            with open(path, "rb") as f:
                data = f.read(4096)
        except OSError:
            return ""
        parts = data.rstrip(b"\x00").split(b"\x00")
        try:
            return " ".join(p.decode("utf-8", errors="replace") for p in parts if p)
        except Exception:
            return ""

    def _read_ppid(self, pid: int) -> Optional[int]:
        """
        Read the parent PID from /proc/<pid>/status.

        Returns None on any error or if the process has already exited.
        """
        path = f"/proc/{pid}/status"
        try:
            with open(path, "r") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        return int(line.split()[1])
        except OSError:
            pass
        return None

    def _read_owner_uid(self, pid: int) -> int:
        """
        Read the effective UID from /proc/<pid>/status.

        Format: Uid: <real> <effective> <saved> <filesystem>
        Returns -1 on any error.
        """
        path = f"/proc/{pid}/status"
        try:
            with open(path, "r") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        parts = line.split()
                        if len(parts) >= 3:
                            return int(parts[2])  # effective UID
        except OSError:
            pass
        return -1

    def _walk_parent_environ(self, pid: int, depth: int = 3) -> dict[str, str]:
        """
        Walk up to `depth` levels of the process tree looking for an ancestor
        whose environ contains ALUMINATAI_TEAM.

        Returns the first matching ancestor's full environ dict, or {} if none
        is found. Stops early at PID <= 1 (init/idle).

        This lets a launcher script that sets ALUMINATAI_TEAM propagate the
        tag into GPU child processes that don't set it themselves.
        """
        current_pid = pid
        for _ in range(depth):
            ppid = self._read_ppid(current_pid)
            if ppid is None or ppid <= 1:
                break
            parent_env = self._read_environ(ppid)  # already filtered by _filter_environ
            if "ALUMINATAI_TEAM" in parent_env:
                return parent_env
            current_pid = ppid
        return {}
