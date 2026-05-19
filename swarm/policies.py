"""
Built-in Swarm policies — each evaluates a FleetSnapshot and returns
optimization recommendations for the policy engine to dispatch.

Policies are stateless functions: (FleetSnapshot, config) → list[dict].
Each dict follows the recommendation format expected by
RecommendationReporter.report_from_swarm_policy().
"""
from __future__ import annotations

from typing import Any
from swarm.fleet_state import FleetSnapshot, MachineState, GpuState


def idle_gpu_power_cap(
    fleet: FleetSnapshot,
    config: dict[str, Any] | None = None,
) -> list[dict]:
    """Cap power on idle GPUs across the fleet.

    Default thresholds:
      - idle = utilization < 10% for 5+ minutes
      - cap to 40% of power limit (e.g., 300W TDP → 120W cap)
    """
    cfg = config or {}
    idle_threshold_pct = cfg.get("idle_threshold_pct", 10)
    cap_fraction = cfg.get("cap_fraction", 0.40)
    min_cap_w = cfg.get("min_cap_w", 100)

    recs: list[dict] = []
    for machine in fleet.active_machines:
        for gpu in machine.gpus:
            if not gpu.is_idle and gpu.utilization_pct >= idle_threshold_pct:
                continue
            cap_w = max(min_cap_w, int(gpu.power_limit_w * cap_fraction))
            if gpu.power_draw_w <= cap_w + 20:
                continue

            savings_pct = round((1 - cap_w / gpu.power_draw_w) * 100, 1) if gpu.power_draw_w > 0 else 0

            recs.append({
                "machine_id": machine.machine_id,
                "gpu_index": gpu.gpu_index,
                "gpu_name": gpu.gpu_name,
                "category": "power_cap",
                "priority": "P1",
                "title": f"Idle GPU {gpu.gpu_index} on {machine.hostname} — cap to {cap_w}W",
                "description": (
                    f"GPU {gpu.gpu_index} ({gpu.gpu_name}) is idle at "
                    f"{gpu.utilization_pct:.0f}% util, drawing {gpu.power_draw_w:.0f}W. "
                    f"Capping to {cap_w}W saves ~{savings_pct}% power on this GPU."
                ),
                "action": f"Set power limit to {cap_w}W",
                "estimated_savings_pct": savings_pct,
                "effort_score": 1,
                "action_payload": {
                    "command": "apply_power_cap",
                    "gpu_index": gpu.gpu_index,
                    "watts": cap_w,
                },
            })
    return recs


def thermal_balancing(
    fleet: FleetSnapshot,
    config: dict[str, Any] | None = None,
) -> list[dict]:
    """Reduce power on thermally stressed GPUs to prevent throttling.

    If a GPU is above the hot threshold while the machine average is below,
    recommend capping that GPU to reduce thermal load.
    """
    cfg = config or {}
    hot_threshold_c = cfg.get("hot_threshold_c", 83)
    target_temp_c = cfg.get("target_temp_c", 75)
    cap_step_pct = cfg.get("cap_step_pct", 15)

    recs: list[dict] = []
    for machine in fleet.active_machines:
        if not machine.gpus:
            continue
        avg_temp = sum(g.temperature_c for g in machine.gpus) / len(machine.gpus)

        for gpu in machine.gpus:
            if gpu.temperature_c < hot_threshold_c:
                continue
            if avg_temp >= hot_threshold_c:
                continue

            cap_w = int(gpu.power_limit_w * (1 - cap_step_pct / 100))
            recs.append({
                "machine_id": machine.machine_id,
                "gpu_index": gpu.gpu_index,
                "gpu_name": gpu.gpu_name,
                "category": "thermal",
                "priority": "P1",
                "title": f"GPU {gpu.gpu_index} on {machine.hostname} overheating ({gpu.temperature_c:.0f}°C)",
                "description": (
                    f"GPU {gpu.gpu_index} at {gpu.temperature_c:.0f}°C while machine average is "
                    f"{avg_temp:.0f}°C. Reducing power limit by {cap_step_pct}% to {cap_w}W "
                    f"should bring temperature toward {target_temp_c}°C."
                ),
                "action": f"Set power limit to {cap_w}W",
                "estimated_savings_pct": round(cap_step_pct * 0.8, 1),
                "effort_score": 1,
                "action_payload": {
                    "command": "apply_power_cap",
                    "gpu_index": gpu.gpu_index,
                    "watts": cap_w,
                },
            })
    return recs


def carbon_aware_fleet_cap(
    fleet: FleetSnapshot,
    config: dict[str, Any] | None = None,
) -> list[dict]:
    """During high-carbon periods, cap non-critical workloads fleet-wide.

    Zones with intensity above the threshold get power-capped on GPUs
    that aren't running high-priority jobs.
    """
    cfg = config or {}
    high_carbon_gco2e = cfg.get("high_carbon_gco2e", 400)
    cap_fraction = cfg.get("cap_fraction", 0.65)
    min_cap_w = cfg.get("min_cap_w", 100)

    recs: list[dict] = []
    for machine in fleet.active_machines:
        if machine.carbon_intensity_gco2e < high_carbon_gco2e:
            continue

        for gpu in machine.gpus:
            if gpu.utilization_pct > 90:
                continue
            cap_w = max(min_cap_w, int(gpu.power_limit_w * cap_fraction))
            if gpu.power_draw_w <= cap_w + 20:
                continue

            savings_pct = round((1 - cap_w / gpu.power_draw_w) * 100, 1) if gpu.power_draw_w > 0 else 0

            recs.append({
                "machine_id": machine.machine_id,
                "gpu_index": gpu.gpu_index,
                "gpu_name": gpu.gpu_name,
                "category": "carbon_schedule",
                "priority": "P2",
                "title": (
                    f"High carbon ({machine.carbon_intensity_gco2e:.0f} gCO2e/kWh) — "
                    f"cap GPU {gpu.gpu_index} on {machine.hostname}"
                ),
                "description": (
                    f"Grid zone {machine.grid_zone} is at {machine.carbon_intensity_gco2e:.0f} "
                    f"gCO2e/kWh. GPU {gpu.gpu_index} is at {gpu.utilization_pct:.0f}% util. "
                    f"Capping to {cap_w}W reduces emissions during this high-carbon window."
                ),
                "action": f"Set power limit to {cap_w}W until carbon intensity drops",
                "estimated_savings_pct": savings_pct,
                "effort_score": 1,
                "action_payload": {
                    "command": "apply_power_cap",
                    "gpu_index": gpu.gpu_index,
                    "watts": cap_w,
                },
            })
    return recs


def fleet_gpu_rightsizing(
    fleet: FleetSnapshot,
    config: dict[str, Any] | None = None,
) -> list[dict]:
    """Flag GPUs that are consistently underutilized across the fleet.

    If a GPU has been drawing <30% of its power limit for a sustained
    period, recommend downsizing or consolidating workloads.
    """
    cfg = config or {}
    underutil_pct = cfg.get("underutil_pct", 30)
    power_ratio_threshold = cfg.get("power_ratio_threshold", 0.30)

    recs: list[dict] = []
    for machine in fleet.active_machines:
        for gpu in machine.gpus:
            if gpu.is_idle:
                continue
            if gpu.utilization_pct >= underutil_pct:
                continue
            if gpu.power_limit_w <= 0:
                continue
            power_ratio = gpu.power_draw_w / gpu.power_limit_w
            if power_ratio >= power_ratio_threshold:
                continue

            recs.append({
                "machine_id": machine.machine_id,
                "gpu_index": gpu.gpu_index,
                "gpu_name": gpu.gpu_name,
                "category": "utilization",
                "priority": "P2",
                "title": f"GPU {gpu.gpu_index} on {machine.hostname} underutilized ({gpu.utilization_pct:.0f}%)",
                "description": (
                    f"GPU {gpu.gpu_index} ({gpu.gpu_name}) averaging {gpu.utilization_pct:.0f}% "
                    f"utilization and {gpu.power_draw_w:.0f}W / {gpu.power_limit_w:.0f}W power. "
                    f"Consider consolidating workloads or downsizing GPU allocation."
                ),
                "action": "Consolidate workload onto fewer GPUs or reduce allocation",
                "estimated_savings_pct": round((1 - power_ratio) * 50, 1),
                "effort_score": 3,
                "action_payload": {},
            })
    return recs


ALL_POLICIES = [
    ("idle_gpu_power_cap", idle_gpu_power_cap),
    ("thermal_balancing", thermal_balancing),
    ("carbon_aware_fleet_cap", carbon_aware_fleet_cap),
    ("fleet_gpu_rightsizing", fleet_gpu_rightsizing),
]
