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
Attribution engine unit tests.

Tests:
  1.  DDP: 8 PIDs with same SLURM_JOB_ID → 1 result, gpu_fraction=1.0, confidence="scheduler"
  2.  Multi-tenant: 2 PIDs from different jobs (60/40 mem split) → 2 results summing to 1.0
  3.  No-PID scheduler fallback → confidence="scheduler_poll", gpu_fraction=1.0
  4.  No-PID, no scheduler, ALUMINATAI_IDLE_TEAM set → confidence="idle"
  5.  Heuristic match: jupyter cmdline → confidence="heuristic", scheduler_source="heuristic"
  6.  Parent PID walk: ancestor environ with ALUMINATAI_TEAM → tag inherited
  7.  Rules file match: custom JSON rules file → correct team/model assigned
  8.  Spoofing guard: untrusted UID with ALUMINATAI_TEAM → manual tag ignored
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch, patch as mock_patch

# Allow running from repo root or agent/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attribution.engine import AttributionEngine, AttributionResult
from attribution.process_probe import ProcessInfo, ProcessProbe
from attribution.pid_resolver import PidResolver
from attribution.rules import AttributionRules
from schedulers.base import JobMetadata, NullAdapter, SchedulerAdapter


# ── Test helpers ──────────────────────────────────────────────────────────────


def make_job(job_id: str, team: str = "team-a", model: str = "gpt") -> JobMetadata:
    return JobMetadata(
        job_id=job_id,
        job_name=f"job-{job_id}",
        team_id=team,
        model_tag=model,
        scheduler_source="slurm",
        gpu_indices=[0],
    )


def make_proc(pid: int, gpu_mem: int, slurm_job_id: str = "",
              cmdline: str = "", owner_uid: int = -1) -> ProcessInfo:
    environ = {}
    if slurm_job_id:
        environ["SLURM_JOB_ID"] = slurm_job_id
    return ProcessInfo(pid=pid, gpu_memory_bytes=gpu_mem, environ=environ,
                       cmdline=cmdline, owner_uid=owner_uid)


class MockScheduler(SchedulerAdapter):
    """Minimal scheduler that resolves known job IDs."""

    def __init__(self, jobs: List[JobMetadata]):
        self._jobs = {j.job_id: j for j in jobs}
        self._gpu_map: dict[int, JobMetadata] = {}
        for j in jobs:
            for idx in j.gpu_indices:
                self._gpu_map[idx] = j

    def discover_jobs(self) -> List[JobMetadata]:
        return list(self._jobs.values())

    def gpu_to_job(self, gpu_index: int) -> Optional[JobMetadata]:
        return self._gpu_map.get(gpu_index)

    def resolve_job(self, job_id: str) -> Optional[JobMetadata]:
        return self._jobs.get(job_id)

    @property
    def name(self) -> str:
        return "mock-slurm"


def build_engine(procs: List[ProcessInfo], scheduler: SchedulerAdapter) -> AttributionEngine:
    probe = MagicMock(spec=ProcessProbe)
    probe.query.return_value = procs
    resolver = PidResolver(scheduler)
    return AttributionEngine(probe, resolver, scheduler)


# ── Tests: existing behaviour ─────────────────────────────────────────────────


class TestDDPGrouping(unittest.TestCase):
    """Case 1: Multiple PIDs from the same Slurm job → single attribution result."""

    def test_8_pids_same_job_yields_fraction_1(self):
        job = make_job("JOB_001", team="ml-team", model="llama3")
        scheduler = MockScheduler([job])

        # 8 DDP workers, all in the same Slurm job, equal memory usage
        procs = [make_proc(pid=1000 + i, gpu_mem=10 * 1024**3, slurm_job_id="JOB_001")
                 for i in range(8)]

        engine = build_engine(procs, scheduler)
        results = engine.resolve(handle=None, gpu_index=0, total_power_w=300.0, energy_delta_j=1500.0)

        self.assertEqual(len(results), 1, "8 DDP workers should collapse to 1 attribution result")
        r = results[0]
        self.assertEqual(r.job_id, "JOB_001")
        self.assertEqual(r.team_id, "ml-team")
        self.assertAlmostEqual(r.gpu_fraction, 1.0, places=3)
        self.assertAlmostEqual(r.power_w, 300.0, places=1)
        self.assertEqual(r.confidence, "scheduler")


class TestMultiTenantSplit(unittest.TestCase):
    """Case 2: 2 jobs sharing a GPU with 60/40 memory split."""

    def test_two_jobs_60_40_split(self):
        job_a = make_job("JOB_A", team="team-alpha")
        job_b = make_job("JOB_B", team="team-beta")
        scheduler = MockScheduler([job_a, job_b])

        total_mem = 80 * 1024**3  # 80 GB GPU
        procs = [
            make_proc(pid=2001, gpu_mem=int(total_mem * 0.60), slurm_job_id="JOB_A"),
            make_proc(pid=2002, gpu_mem=int(total_mem * 0.40), slurm_job_id="JOB_B"),
        ]

        engine = build_engine(procs, scheduler)
        results = engine.resolve(handle=None, gpu_index=0, total_power_w=400.0, energy_delta_j=2000.0)

        self.assertEqual(len(results), 2)

        fracs = {r.team_id: r.gpu_fraction for r in results}
        self.assertAlmostEqual(fracs["team-alpha"], 0.60, places=2)
        self.assertAlmostEqual(fracs["team-beta"], 0.40, places=2)

        total_frac = sum(r.gpu_fraction for r in results)
        self.assertAlmostEqual(total_frac, 1.0, places=3, msg="Fractions must sum to 1.0")

        total_energy = sum(r.energy_delta_j for r in results if r.energy_delta_j)
        self.assertAlmostEqual(total_energy, 2000.0, places=2)

        for r in results:
            self.assertEqual(r.confidence, "scheduler")


class TestSchedulerFallback(unittest.TestCase):
    """Case 3: No running processes → fall back to scheduler.gpu_to_job()."""

    def test_no_procs_uses_scheduler_poll(self):
        job = JobMetadata(
            job_id="JOB_SCHED",
            job_name="training",
            team_id="ops-team",
            model_tag="bert",
            scheduler_source="slurm",
            gpu_indices=[0],
        )
        scheduler = MockScheduler([job])

        engine = build_engine([], scheduler)  # no processes
        results = engine.resolve(handle=None, gpu_index=0, total_power_w=200.0, energy_delta_j=1000.0)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.job_id, "JOB_SCHED")
        self.assertEqual(r.confidence, "scheduler_poll")
        self.assertAlmostEqual(r.gpu_fraction, 1.0)


class TestIdleFallback(unittest.TestCase):
    """Case 4: No processes and no scheduler job → idle attribution."""

    def test_idle_attribution(self):
        scheduler = NullAdapter()
        engine = build_engine([], scheduler)

        with patch.dict(os.environ, {"ALUMINATAI_IDLE_TEAM": "infra"}):
            results = engine.resolve(handle=None, gpu_index=0, total_power_w=50.0, energy_delta_j=250.0)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.team_id, "infra")
        self.assertEqual(r.confidence, "idle")
        self.assertEqual(r.model_tag, "idle")
        self.assertAlmostEqual(r.gpu_fraction, 1.0)

    def test_no_idle_team_returns_empty(self):
        scheduler = NullAdapter()
        engine = build_engine([], scheduler)

        env = {k: v for k, v in os.environ.items() if k != "ALUMINATAI_IDLE_TEAM"}
        with patch.dict(os.environ, env, clear=True):
            results = engine.resolve(handle=None, gpu_index=0, total_power_w=50.0, energy_delta_j=250.0)

        self.assertEqual(results, [], "No attribution config → empty list (backward compat)")


# ── Tests: new Phase 2–3 behaviour ────────────────────────────────────────────


class TestHeuristicResolution(unittest.TestCase):
    """Case 5: Untagged process with recognisable cmdline → heuristic attribution."""

    def test_jupyter_cmdline_heuristic(self):
        proc = make_proc(pid=9999, gpu_mem=2 * 1024**3,
                         cmdline="jupyter notebook --no-browser --port=8888")
        scheduler = NullAdapter()
        engine = build_engine([proc], scheduler)

        results = engine.resolve(handle=None, gpu_index=0, total_power_w=80.0, energy_delta_j=400.0)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.confidence, "heuristic")
        self.assertEqual(r.scheduler_source, "heuristic")
        self.assertEqual(r.team_id, "jupyter")
        self.assertEqual(r.model_tag, "notebook")

    def test_vllm_cmdline_heuristic(self):
        proc = make_proc(pid=8888, gpu_mem=10 * 1024**3,
                         cmdline="python -m vllm.entrypoints.openai.api_server --model llama3")
        scheduler = NullAdapter()
        engine = build_engine([proc], scheduler)

        results = engine.resolve(handle=None, gpu_index=0, total_power_w=200.0, energy_delta_j=None)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].confidence, "heuristic")
        self.assertEqual(results[0].model_tag, "vllm-serve")

    def test_empty_cmdline_no_heuristic(self):
        """Processes with no cmdline should not match heuristics → memory_split."""
        proc = make_proc(pid=7777, gpu_mem=1 * 1024**3, cmdline="")
        scheduler = NullAdapter()
        engine = build_engine([proc], scheduler)

        env_without_idle = {k: v for k, v in os.environ.items() if k != "ALUMINATAI_IDLE_TEAM"}
        with patch.dict(os.environ, env_without_idle, clear=True):
            results = engine.resolve(handle=None, gpu_index=0, total_power_w=50.0, energy_delta_j=None)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].confidence, "memory_split")


class TestParentPidWalk(unittest.TestCase):
    """Case 6: Ancestor process has ALUMINATAI_TEAM in its environ."""

    def test_inherits_team_from_parent(self):
        probe = ProcessProbe()
        child_pid, parent_pid = 5000, 4000

        with patch.object(probe, "_read_ppid", side_effect=lambda p: parent_pid if p == child_pid else None), \
             patch.object(probe, "_read_environ", side_effect=lambda p: {"ALUMINATAI_TEAM": "nlp-team"} if p == parent_pid else {}):
            result = probe._walk_parent_environ(child_pid)

        self.assertEqual(result.get("ALUMINATAI_TEAM"), "nlp-team")

    def test_no_match_returns_empty(self):
        probe = ProcessProbe()

        with patch.object(probe, "_read_ppid", return_value=None):
            result = probe._walk_parent_environ(1234)

        self.assertEqual(result, {})

    def test_stops_at_init_pid(self):
        probe = ProcessProbe()

        # Parent is PID 1 (init) → should stop immediately without reading its environ
        with patch.object(probe, "_read_ppid", return_value=1), \
             patch.object(probe, "_read_environ") as mock_environ:
            result = probe._walk_parent_environ(5000)

        mock_environ.assert_not_called()
        self.assertEqual(result, {})


class TestAttributionRulesFile(unittest.TestCase):
    """Case 7: Custom rules JSON file assigns correct team/model."""

    def _write_rules(self, rules_data: dict) -> str:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(rules_data, tmp)
        tmp.close()
        return tmp.name

    def test_rules_file_match(self):
        path = self._write_rules({
            "rules": [
                {"pattern": "python.*gpt4_train", "team": "llm-infra", "model": "gpt4", "priority": 10},
                {"pattern": "jupyter",            "team": "research",  "model": "notebook", "priority": 1},
            ]
        })
        try:
            with patch.dict(os.environ, {"ALUMINATAI_ATTRIBUTION_CONFIG": path}):
                rules = AttributionRules()
                rules.load()

            match = rules.match("python gpt4_train.py --lr 0.001")
            self.assertIsNotNone(match)
            self.assertEqual(match.team, "llm-infra")
            self.assertEqual(match.model, "gpt4")
        finally:
            os.unlink(path)

    def test_priority_ordering(self):
        path = self._write_rules({
            "rules": [
                {"pattern": "python", "team": "low-prio",  "priority": 1},
                {"pattern": "python", "team": "high-prio", "priority": 99},
            ]
        })
        try:
            with patch.dict(os.environ, {"ALUMINATAI_ATTRIBUTION_CONFIG": path}):
                rules = AttributionRules()
                rules.load()

            match = rules.match("python train.py")
            self.assertEqual(match.team, "high-prio")
        finally:
            os.unlink(path)

    def test_no_rules_file_is_noop(self):
        """Missing config file → load() silently no-ops, match() returns None."""
        with patch.dict(os.environ, {"ALUMINATAI_ATTRIBUTION_CONFIG": "/nonexistent/path.json"}):
            rules = AttributionRules()
            rules.load()

        self.assertIsNone(rules.match("python anything.py"))

    def test_rules_file_end_to_end(self):
        """Rules match flows through PidResolver → AttributionEngine → confidence='rules'."""
        path = self._write_rules({
            "rules": [{"pattern": "special_workload", "team": "ops-team", "model": "custom"}]
        })
        try:
            proc = make_proc(pid=3333, gpu_mem=4 * 1024**3,
                             cmdline="python special_workload.py")
            scheduler = NullAdapter()

            with patch.dict(os.environ, {"ALUMINATAI_ATTRIBUTION_CONFIG": path}):
                engine = build_engine([proc], scheduler)
                results = engine.resolve(handle=None, gpu_index=0, total_power_w=100.0, energy_delta_j=None)

            self.assertEqual(len(results), 1)
            r = results[0]
            self.assertEqual(r.confidence, "rules")
            self.assertEqual(r.team_id, "ops-team")
            self.assertEqual(r.model_tag, "custom")
        finally:
            os.unlink(path)


class TestSpoofingGuard(unittest.TestCase):
    """Case 8: Untrusted UID with ALUMINATAI_TEAM → manual tag ignored."""

    def test_untrusted_uid_skips_manual_tag(self):
        """Process from UID 1001 claiming ALUMINATAI_TEAM when only UID 0 is trusted."""
        proc = ProcessInfo(
            pid=1234,
            gpu_memory_bytes=1 * 1024**3,
            environ={"ALUMINATAI_TEAM": "malicious-team"},
            owner_uid=1001,
        )
        scheduler = NullAdapter()

        with patch("attribution.pid_resolver._TRUSTED_UIDS", {0}):
            resolver = PidResolver(scheduler)
            job = resolver.resolve(proc)

        # Manual tag ignored, no heuristic match → unresolved
        self.assertIsNone(job)

    def test_trusted_uid_allows_manual_tag(self):
        """Root process (UID 0) setting ALUMINATAI_TEAM should be honoured."""
        proc = ProcessInfo(
            pid=1235,
            gpu_memory_bytes=1 * 1024**3,
            environ={"ALUMINATAI_TEAM": "trusted-team"},
            owner_uid=0,
        )
        scheduler = NullAdapter()

        with patch("attribution.pid_resolver._TRUSTED_UIDS", {0}):
            resolver = PidResolver(scheduler)
            job = resolver.resolve(proc)

        self.assertIsNotNone(job)
        self.assertEqual(job.team_id, "trusted-team")
        self.assertEqual(job.scheduler_source, "manual")

    def test_empty_trusted_uids_allows_all(self):
        """When TRUSTED_UIDS is empty (default), all UIDs are accepted."""
        proc = ProcessInfo(
            pid=1236,
            gpu_memory_bytes=1 * 1024**3,
            environ={"ALUMINATAI_TEAM": "any-team"},
            owner_uid=9999,
        )
        scheduler = NullAdapter()

        with patch("attribution.pid_resolver._TRUSTED_UIDS", set()):
            resolver = PidResolver(scheduler)
            job = resolver.resolve(proc)

        self.assertIsNotNone(job)
        self.assertEqual(job.team_id, "any-team")


if __name__ == "__main__":
    unittest.main(verbosity=2)
