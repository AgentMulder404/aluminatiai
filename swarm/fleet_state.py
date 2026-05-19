"""
Fleet State — data model for cross-node GPU telemetry.

The policy engine consumes a FleetSnapshot to evaluate optimization
opportunities that span multiple machines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GpuState:
    gpu_index: int
    gpu_name: str
    power_draw_w: float
    power_limit_w: float
    utilization_pct: float
    temperature_c: float
    memory_used_mb: float
    memory_total_mb: float
    energy_j_last_hour: float = 0.0
    model_tag: Optional[str] = None
    job_id: Optional[str] = None
    is_idle: bool = False


@dataclass
class MachineState:
    machine_id: str
    hostname: str
    cluster_tag: str
    gpu_count: int
    gpus: list[GpuState] = field(default_factory=list)
    last_heartbeat_age_s: float = 0.0
    carbon_intensity_gco2e: float = 0.0
    grid_zone: str = ""
    agent_version: str = ""

    @property
    def total_power_w(self) -> float:
        return sum(g.power_draw_w for g in self.gpus)

    @property
    def total_power_limit_w(self) -> float:
        return sum(g.power_limit_w for g in self.gpus)

    @property
    def avg_utilization(self) -> float:
        if not self.gpus:
            return 0.0
        return sum(g.utilization_pct for g in self.gpus) / len(self.gpus)

    @property
    def idle_gpus(self) -> list[GpuState]:
        return [g for g in self.gpus if g.is_idle]

    @property
    def hot_gpus(self) -> list[GpuState]:
        return [g for g in self.gpus if g.temperature_c >= 80]

    @property
    def is_stale(self) -> bool:
        return self.last_heartbeat_age_s > 600


@dataclass
class FleetSnapshot:
    """Point-in-time view of all machines for a user."""
    machines: list[MachineState] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def total_gpus(self) -> int:
        return sum(m.gpu_count for m in self.machines)

    @property
    def active_machines(self) -> list[MachineState]:
        return [m for m in self.machines if not m.is_stale]

    @property
    def total_power_w(self) -> float:
        return sum(m.total_power_w for m in self.active_machines)

    @property
    def total_idle_gpus(self) -> int:
        return sum(len(m.idle_gpus) for m in self.active_machines)

    @property
    def fleet_avg_utilization(self) -> float:
        active = self.active_machines
        if not active:
            return 0.0
        total_gpus = sum(len(m.gpus) for m in active)
        if total_gpus == 0:
            return 0.0
        total_util = sum(g.utilization_pct for m in active for g in m.gpus)
        return total_util / total_gpus

    def machines_in_zone(self, zone: str) -> list[MachineState]:
        return [m for m in self.active_machines if m.grid_zone == zone]

    def machines_in_cluster(self, cluster_tag: str) -> list[MachineState]:
        return [m for m in self.active_machines if m.cluster_tag == cluster_tag]

    @property
    def clusters(self) -> dict[str, list[MachineState]]:
        groups: dict[str, list[MachineState]] = {}
        for m in self.active_machines:
            groups.setdefault(m.cluster_tag, []).append(m)
        return groups

    @property
    def zones(self) -> dict[str, list[MachineState]]:
        groups: dict[str, list[MachineState]] = {}
        for m in self.active_machines:
            if m.grid_zone:
                groups.setdefault(m.grid_zone, []).append(m)
        return groups
