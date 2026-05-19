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
IdleBaseline — auto-calibrate per-GPU idle power draw.

Calibration samples GPU power draw for IDLE_BASELINE_WINDOW seconds while
confirming no compute processes are running on any GPU.  The resulting
per-GPU baseline watt values are persisted to
~/.config/aluminatai/baselines.json and loaded on agent startup.

The agent subtracts the baseline from each sample before passing power to
the attribution engine, removing steady-state idle power (driver overhead,
memory ECC scrubbing, link traffic) from job energy accounting.

Typical idle baselines observed in production:
  A100 SXM4  : 55–70 W
  H100 SXM5  : 70–90 W
  RTX 4090   : 15–20 W
  A10G       : 25–35 W
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_BASELINES_PATH = Path.home() / ".config" / "aluminatai" / "baselines.json"
_STALE_AFTER_HOURS = 24


class IdleBaseline:
    """Per-GPU idle power baseline manager."""

    def __init__(self, path: Path = _BASELINES_PATH):
        self._path = path

    # ── Persistence ───────────────────────────────────────────────────────

    def load(self) -> dict[str, float]:
        """
        Load persisted baselines.

        Returns {gpu_uuid: baseline_w}.
        Returns {} if the file doesn't exist, is unreadable, or is corrupt.
        """
        try:
            with open(self._path) as f:
                data = json.load(f)
            return {
                k: float(v["baseline_w"])
                for k, v in data.items()
                if isinstance(v, dict) and "baseline_w" in v
            }
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            return {}

    def save(self, records: dict[str, dict]) -> None:
        """Persist calibration results atomically via .tmp rename."""
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(records, f, indent=2)
                f.flush()
                import os
                os.fsync(f.fileno())
            tmp.replace(self._path)  # atomic on POSIX
            log.debug("Baselines saved → %s", self._path)
        except OSError as exc:
            log.warning("Could not save baselines to %s: %s", self._path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def is_stale(self) -> bool:
        """Return True if the baselines file is absent or older than _STALE_AFTER_HOURS."""
        try:
            age_hours = (time.time() - self._path.stat().st_mtime) / 3600.0
            return age_hours > _STALE_AFTER_HOURS
        except OSError:
            return True

    def maybe_recalibrate(
        self,
        handles: list,
        gpu_uuids: list[str],
    ) -> Optional[dict[int, float]]:
        """Re-calibrate if baselines are stale (>24h) and all GPUs are currently idle.

        Returns new baselines dict on success, None if skipped.
        Intended to be called periodically from the main loop (e.g. every 60s).
        """
        if not self.is_stale():
            return None

        try:
            import pynvml
        except ImportError:
            return None

        # Quick idle check — don't burn 30s if any GPU has work
        for idx, handle in enumerate(handles):
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if procs:
                    return None
            except pynvml.NVMLError:
                return None

        log.info("Baselines stale (>%dh) and all GPUs idle — recalibrating", _STALE_AFTER_HOURS)
        return self.calibrate(handles, gpu_uuids)

    # ── Calibration ───────────────────────────────────────────────────────

    def calibrate(
        self,
        handles: list,
        gpu_uuids: list[str],
        duration_s: int = 30,
    ) -> Optional[dict[int, float]]:
        """
        Sample idle power draw on all GPUs for `duration_s` seconds.

        Returns {gpu_index: baseline_w} on success.
        Returns None if:
          - pynvml is unavailable
          - any GPU has compute processes at the start of the window
          - any GPU acquires a compute process mid-window (aborts immediately)
          - no power readings could be collected

        The caller should only invoke this when GPUs are expected to be idle
        (i.e., on agent startup before jobs are scheduled).
        """
        try:
            import pynvml
        except ImportError:
            log.debug("pynvml unavailable — baseline calibration skipped")
            return None

        # Check upfront that all GPUs are idle
        for idx, handle in enumerate(handles):
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if procs:
                    log.info(
                        "GPU %d has %d running process(es) — "
                        "skipping idle baseline calibration",
                        idx,
                        len(procs),
                    )
                    return None
            except pynvml.NVMLError as exc:
                log.debug("GPU %d NVML check failed: %s — skipping calibration", idx, exc)
                return None

        log.info(
            "All GPUs idle — calibrating power baseline over %ds…",
            duration_s,
        )

        power_samples: dict[int, list[float]] = {i: [] for i in range(len(handles))}

        for tick in range(duration_s):
            aborted = False
            for idx, handle in enumerate(handles):
                # Abort if a process starts mid-window
                try:
                    if pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        log.info(
                            "Process appeared on GPU %d at tick %d/%d — "
                            "aborting baseline calibration",
                            idx,
                            tick + 1,
                            duration_s,
                        )
                        aborted = True
                        break
                except pynvml.NVMLError as exc:
                    log.debug("GPU %d process check failed at tick %d: %s", idx, tick + 1, exc)

                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_samples[idx].append(mw / 1000.0)
                except pynvml.NVMLError as exc:
                    log.debug("GPU %d power read failed at tick %d: %s", idx, tick + 1, exc)

            if aborted:
                return None

            time.sleep(1.0)

        if not all(power_samples[i] for i in range(len(handles))):
            log.warning("No power samples collected — baseline calibration failed")
            return None

        baselines: dict[int, float] = {}
        records: dict[str, dict] = {}
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for idx, samples in power_samples.items():
            avg_w = round(sum(samples) / len(samples), 2)
            baselines[idx] = avg_w
            uuid = gpu_uuids[idx] if idx < len(gpu_uuids) else str(idx)
            records[uuid] = {
                "baseline_w": avg_w,
                "gpu_index": idx,
                "sample_count": len(samples),
                "calibrated_at": ts,
            }
            log.info("  GPU %d  idle baseline: %.1f W  (%d samples)", idx, avg_w, len(samples))

        self.save(records)
        return baselines
