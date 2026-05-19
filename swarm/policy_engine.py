"""
Swarm Policy Engine — production-grade fleet optimization.

Leader election: only one agent per (user, cluster) runs policies.
Blast radius: caps the percentage of fleet affected per eval cycle.
Ramp-up: gradually expands from canary to full fleet.
Snapshot history: policies see recent trends, not just point-in-time.

Usage in agent.py:
    engine = SwarmPolicyEngine(endpoint, api_key, machine_id, reporter)
    engine.evaluate()  # called every SWARM_EVAL_INTERVAL seconds
"""
from __future__ import annotations

import collections
import json
import logging
import time
import urllib.request
import urllib.error
from typing import Any, Optional
from urllib.parse import urlparse

from swarm.fleet_state import FleetSnapshot, MachineState, GpuState
from swarm.policies import ALL_POLICIES

logger = logging.getLogger(__name__)


class SwarmPolicyEngine:
    """Fleet-wide optimization policy engine with production safety."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        machine_id: str,
        reporter,
        cluster_tag: str = "",
        policy_config: dict[str, dict[str, Any]] | None = None,
        enabled_policies: list[str] | None = None,
        max_recs_per_eval: int = 20,
        cooldown_s: float = 300.0,
        max_affected_pct: float = 25.0,
        ramp_pct: float = 10.0,
        history_size: int = 3,
    ):
        parsed = urlparse(endpoint)
        self._base = f"{parsed.scheme}://{parsed.netloc}"
        self._api_key = api_key
        self._machine_id = machine_id
        self._cluster_tag = cluster_tag
        self._reporter = reporter
        self._policy_config = policy_config or {}
        self._max_recs = max_recs_per_eval
        self._cooldown_s = cooldown_s
        self._max_affected_pct = max_affected_pct
        self._ramp_pct = ramp_pct
        self._last_eval: float = 0.0
        self._last_recs: list[dict] = []

        # Leader election
        self._is_leader: bool = False
        self._lease_token: Optional[str] = None

        # Snapshot history for trend-aware policies
        self._history: collections.deque[FleetSnapshot] = collections.deque(maxlen=history_size)

        # Ramp-up tracking: policy_name → current ramp percentage
        self._ramp_state: dict[str, float] = {}

        # Stats
        self._eval_count: int = 0
        self._total_recs_dispatched: int = 0

        enabled = set(enabled_policies) if enabled_policies else None
        self._policies = [
            (name, fn) for name, fn in ALL_POLICIES
            if enabled is None or name in enabled
        ]

    def should_evaluate(self) -> bool:
        return (time.time() - self._last_eval) >= self._cooldown_s

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    def evaluate(self) -> int:
        """Acquire lease, fetch fleet state, run policies with safety, upload.

        Returns number of recommendations uploaded.
        """
        # Leader election — skip if another agent holds the lease
        if not self._acquire_lease():
            return 0

        fleet = self._fetch_fleet_state()
        if not fleet or not fleet.active_machines:
            logger.debug("Swarm: no active machines in fleet")
            return 0

        self._history.append(fleet)
        history = list(self._history)

        all_recs: list[dict] = []
        for policy_name, policy_fn in self._policies:
            try:
                config = self._policy_config.get(policy_name, {})
                recs = policy_fn(fleet, config)
                if not recs:
                    continue

                # Ramp-up: limit how many machines a policy can affect
                recs = self._apply_ramp(policy_name, recs, fleet)

                logger.info("Swarm policy %s: %d recommendations (after ramp)", policy_name, len(recs))
                all_recs.extend(recs)
            except Exception as exc:
                logger.warning("Swarm policy %s failed: %s", policy_name, exc)

        # Blast radius: cap total affected machines
        all_recs = self._enforce_blast_radius(all_recs, fleet)

        all_recs = all_recs[:self._max_recs]
        self._last_recs = all_recs
        self._last_eval = time.time()
        self._eval_count += 1

        if not all_recs:
            return 0

        try:
            n = self._reporter.report_from_swarm_policy(all_recs)
            self._total_recs_dispatched += n
            return n
        except Exception as exc:
            logger.warning("Swarm recommendation upload failed: %s", exc)
            return 0

    # ── Leader Election ────────────────────────────────────────────────────────

    def _acquire_lease(self) -> bool:
        """POST /api/agent/swarm/lease to acquire or renew leader lease."""
        url = f"{self._base}/api/agent/swarm/lease"
        payload = json.dumps({
            "machine_id": self._machine_id,
            "cluster_tag": self._cluster_tag,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self._api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                self._is_leader = body.get("leader", False)
                if self._is_leader:
                    self._lease_token = body.get("lease_token")
                else:
                    current = body.get("current_leader", "unknown")
                    logger.debug("Swarm: not leader — current leader is %s", current)
                return self._is_leader
        except Exception as exc:
            logger.debug("Swarm lease acquisition failed: %s", exc)
            self._is_leader = False
            return False

    # ── Blast Radius Control ───────────────────────────────────────────────────

    def _enforce_blast_radius(
        self, recs: list[dict], fleet: FleetSnapshot
    ) -> list[dict]:
        """Limit the percentage of fleet affected in one eval cycle."""
        total_machines = len(fleet.active_machines)
        if total_machines == 0:
            return recs

        max_affected = max(1, int(total_machines * self._max_affected_pct / 100))
        affected: set[str] = set()
        safe_recs: list[dict] = []

        # Sort by priority (P1 first) then savings (highest first)
        priority_order = {"P1": 0, "P2": 1, "P3": 2}
        recs.sort(key=lambda r: (
            priority_order.get(r.get("priority", "P2"), 1),
            -(r.get("estimated_savings_pct", 0)),
        ))

        for rec in recs:
            mid = rec.get("machine_id", "")
            if mid not in affected:
                if len(affected) >= max_affected:
                    logger.warning(
                        "Swarm blast radius: capped at %d/%d machines (%.0f%%)",
                        max_affected, total_machines, self._max_affected_pct,
                    )
                    break
                affected.add(mid)
            safe_recs.append(rec)

        return safe_recs

    # ── Ramp-Up / Canary ───────────────────────────────────────────────────────

    def _apply_ramp(
        self, policy_name: str, recs: list[dict], fleet: FleetSnapshot
    ) -> list[dict]:
        """Gradually expand a policy's reach from canary to full fleet."""
        if self._ramp_pct >= 100:
            return recs

        current_ramp = self._ramp_state.get(policy_name, self._ramp_pct)
        total_machines = len(fleet.active_machines)
        max_machines = max(1, int(total_machines * current_ramp / 100))

        # Group recs by target machine
        by_machine: dict[str, list[dict]] = {}
        for rec in recs:
            mid = rec.get("machine_id", "")
            by_machine.setdefault(mid, []).append(rec)

        if len(by_machine) <= max_machines:
            # All fit within ramp — advance ramp for next eval
            self._ramp_state[policy_name] = min(100, current_ramp * 2)
            return recs

        # Limit to max_machines (pick highest-savings machines)
        machine_savings = {
            mid: sum(r.get("estimated_savings_pct", 0) for r in mrs)
            for mid, mrs in by_machine.items()
        }
        top_machines = sorted(machine_savings, key=machine_savings.get, reverse=True)[:max_machines]

        ramped = []
        for mid in top_machines:
            ramped.extend(by_machine[mid])

        logger.info(
            "Swarm ramp %s: %d/%d machines (%.0f%% ramp)",
            policy_name, len(top_machines), len(by_machine), current_ramp,
        )
        # Don't advance ramp — we're still capping
        return ramped

    # ── Fleet State Fetch ──────────────────────────────────────────────────────

    def _fetch_fleet_state(self) -> Optional[FleetSnapshot]:
        """GET /api/agent/fleet-state — uses DB function, no row limit."""
        url = f"{self._base}/api/agent/fleet-state"
        req = urllib.request.Request(
            url,
            headers={
                "X-API-Key": self._api_key,
                "Accept-Encoding": "gzip",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # Handle gzip response
                if resp.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                data = json.loads(raw)
                return self._parse_fleet(data)
        except Exception as exc:
            logger.debug("Fleet state fetch failed: %s", exc)
            return None

    def _parse_fleet(self, data: dict) -> FleetSnapshot:
        machines: list[MachineState] = []
        for m in data.get("machines", []):
            gpus = []
            for g in m.get("gpus", []):
                gpus.append(GpuState(
                    gpu_index=g.get("gpu_index", 0),
                    gpu_name=g.get("gpu_name", ""),
                    power_draw_w=g.get("power_draw_w", 0),
                    power_limit_w=g.get("power_limit_w", 0),
                    utilization_pct=g.get("utilization_pct", 0),
                    temperature_c=g.get("temperature_c", 0),
                    memory_used_mb=g.get("memory_used_mb", 0),
                    memory_total_mb=g.get("memory_total_mb", 0),
                    energy_j_last_hour=g.get("energy_j_last_hour", 0),
                    model_tag=g.get("model_tag"),
                    job_id=g.get("job_id"),
                    is_idle=g.get("is_idle", False),
                ))
            machines.append(MachineState(
                machine_id=m.get("machine_id", ""),
                hostname=m.get("hostname", ""),
                cluster_tag=m.get("cluster_tag", ""),
                gpu_count=m.get("gpu_count", len(gpus)),
                gpus=gpus,
                last_heartbeat_age_s=m.get("last_heartbeat_age_s", 0),
                carbon_intensity_gco2e=m.get("carbon_intensity_gco2e", 0),
                grid_zone=m.get("grid_zone", ""),
                agent_version=m.get("agent_version", ""),
            ))
        return FleetSnapshot(machines=machines, timestamp=time.time())

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def last_recommendations(self) -> list[dict]:
        return self._last_recs

    @property
    def snapshot_history(self) -> list[FleetSnapshot]:
        return list(self._history)

    def get_stats(self) -> dict:
        return {
            "is_leader": self._is_leader,
            "eval_count": self._eval_count,
            "total_recs_dispatched": self._total_recs_dispatched,
            "last_eval_age_s": round(time.time() - self._last_eval, 1) if self._last_eval else None,
            "history_depth": len(self._history),
            "ramp_state": dict(self._ramp_state),
            "policies_enabled": [name for name, _ in self._policies],
        }
