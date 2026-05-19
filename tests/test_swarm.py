"""Tests for swarm policy engine and built-in policies."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
from swarm.fleet_state import FleetSnapshot, MachineState, GpuState
from swarm.policies import (
    idle_gpu_power_cap,
    thermal_balancing,
    carbon_aware_fleet_cap,
    fleet_gpu_rightsizing,
)
from swarm.policy_engine import SwarmPolicyEngine


def _gpu(index=0, name="A100", power=300, limit=400, util=80, temp=65,
         mem_used=20000, mem_total=81920, idle=False):
    return GpuState(
        gpu_index=index, gpu_name=name,
        power_draw_w=power, power_limit_w=limit,
        utilization_pct=util, temperature_c=temp,
        memory_used_mb=mem_used, memory_total_mb=mem_total,
        is_idle=idle,
    )


def _machine(mid="m1", hostname="node-1", gpus=None, carbon=0, zone=""):
    return MachineState(
        machine_id=mid, hostname=hostname, cluster_tag="test",
        gpu_count=len(gpus or []), gpus=gpus or [],
        carbon_intensity_gco2e=carbon, grid_zone=zone,
    )


def _fleet(machines):
    return FleetSnapshot(machines=machines)


# ── Fleet State tests ──────────────────────────────────────────────────────────

class TestFleetState:
    def test_total_gpus(self):
        f = _fleet([_machine(gpus=[_gpu(), _gpu(1)]), _machine("m2", gpus=[_gpu()])])
        assert f.total_gpus == 3

    def test_fleet_avg_utilization(self):
        f = _fleet([_machine(gpus=[_gpu(util=60), _gpu(1, util=40)])])
        assert f.fleet_avg_utilization == 50.0

    def test_stale_machines_excluded(self):
        m = _machine(gpus=[_gpu()])
        m.last_heartbeat_age_s = 700
        f = _fleet([m])
        assert len(f.active_machines) == 0
        assert f.total_power_w == 0

    def test_idle_gpu_count(self):
        f = _fleet([_machine(gpus=[_gpu(idle=True), _gpu(1, idle=False)])])
        assert f.total_idle_gpus == 1


# ── Idle GPU Power Cap ─────────────────────────────────────────────────────────

class TestIdleGpuPowerCap:
    def test_caps_idle_gpus(self):
        f = _fleet([_machine(gpus=[
            _gpu(0, power=350, limit=400, util=3, idle=True),
            _gpu(1, power=300, limit=400, util=85),
        ])])
        recs = idle_gpu_power_cap(f)
        assert len(recs) == 1
        assert recs[0]["gpu_index"] == 0
        assert recs[0]["action_payload"]["command"] == "apply_power_cap"
        assert recs[0]["action_payload"]["watts"] == 160  # 40% of 400

    def test_skips_already_low_power(self):
        f = _fleet([_machine(gpus=[
            _gpu(0, power=100, limit=400, util=2, idle=True),
        ])])
        recs = idle_gpu_power_cap(f)
        assert len(recs) == 0

    def test_skips_active_gpus(self):
        f = _fleet([_machine(gpus=[_gpu(0, power=350, util=80)])])
        recs = idle_gpu_power_cap(f)
        assert len(recs) == 0

    def test_cross_machine(self):
        f = _fleet([
            _machine("m1", gpus=[_gpu(0, power=300, limit=400, util=2, idle=True)]),
            _machine("m2", "node-2", gpus=[_gpu(0, power=350, limit=400, util=5, idle=True)]),
        ])
        recs = idle_gpu_power_cap(f)
        assert len(recs) == 2
        machines = {r["machine_id"] for r in recs}
        assert machines == {"m1", "m2"}


# ── Thermal Balancing ──────────────────────────────────────────────────────────

class TestThermalBalancing:
    def test_caps_hot_gpu(self):
        f = _fleet([_machine(gpus=[
            _gpu(0, temp=88, power=350, limit=400),
            _gpu(1, temp=62, power=300, limit=400),
        ])])
        recs = thermal_balancing(f)
        assert len(recs) == 1
        assert recs[0]["gpu_index"] == 0
        assert recs[0]["category"] == "thermal"

    def test_skips_when_all_hot(self):
        f = _fleet([_machine(gpus=[
            _gpu(0, temp=88), _gpu(1, temp=85),
        ])])
        recs = thermal_balancing(f)
        assert len(recs) == 0

    def test_skips_cool_gpus(self):
        f = _fleet([_machine(gpus=[_gpu(0, temp=65), _gpu(1, temp=70)])])
        recs = thermal_balancing(f)
        assert len(recs) == 0


# ── Carbon-Aware Fleet Cap ─────────────────────────────────────────────────────

class TestCarbonAwareFleetCap:
    def test_caps_during_high_carbon(self):
        f = _fleet([_machine(gpus=[
            _gpu(0, power=350, limit=400, util=50),
        ], carbon=500, zone="US-CAL")])
        recs = carbon_aware_fleet_cap(f)
        assert len(recs) == 1
        assert recs[0]["category"] == "carbon_schedule"

    def test_skips_low_carbon(self):
        f = _fleet([_machine(gpus=[_gpu(0, power=350, util=50)], carbon=200)])
        recs = carbon_aware_fleet_cap(f)
        assert len(recs) == 0

    def test_skips_high_util_gpus(self):
        f = _fleet([_machine(gpus=[_gpu(0, power=350, util=95)], carbon=500)])
        recs = carbon_aware_fleet_cap(f)
        assert len(recs) == 0


# ── Fleet GPU Right-Sizing ─────────────────────────────────────────────────────

class TestFleetGpuRightsizing:
    def test_flags_underutilized(self):
        f = _fleet([_machine(gpus=[
            _gpu(0, power=80, limit=400, util=15),
        ])])
        recs = fleet_gpu_rightsizing(f)
        assert len(recs) == 1
        assert recs[0]["category"] == "utilization"

    def test_skips_idle(self):
        f = _fleet([_machine(gpus=[_gpu(0, power=50, limit=400, util=5, idle=True)])])
        recs = fleet_gpu_rightsizing(f)
        assert len(recs) == 0

    def test_skips_well_utilized(self):
        f = _fleet([_machine(gpus=[_gpu(0, power=300, limit=400, util=80)])])
        recs = fleet_gpu_rightsizing(f)
        assert len(recs) == 0


# ── Policy Engine ──────────────────────────────────────────────────────────────

class TestPolicyEngine:
    def _make_engine(self, reporter=None, **kwargs):
        reporter = reporter or MagicMock()
        reporter.report_from_swarm_policy = MagicMock(return_value=0)
        defaults = dict(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="leader",
            reporter=reporter,
            cooldown_s=0,
            max_affected_pct=100,
            ramp_pct=100,
        )
        defaults.update(kwargs)
        return SwarmPolicyEngine(**defaults)

    def test_evaluate_with_fleet(self):
        engine = self._make_engine()
        fleet = _fleet([_machine(gpus=[
            _gpu(0, power=350, limit=400, util=3, idle=True),
        ])])
        with patch.object(engine, "_acquire_lease", return_value=True), \
             patch.object(engine, "_fetch_fleet_state", return_value=fleet):
            engine.evaluate()
            engine._reporter.report_from_swarm_policy.assert_called_once()
            recs = engine._reporter.report_from_swarm_policy.call_args[0][0]
            assert len(recs) >= 1

    def test_evaluate_no_fleet(self):
        engine = self._make_engine()
        with patch.object(engine, "_acquire_lease", return_value=True), \
             patch.object(engine, "_fetch_fleet_state", return_value=None):
            n = engine.evaluate()
            assert n == 0

    def test_evaluate_not_leader(self):
        engine = self._make_engine()
        with patch.object(engine, "_acquire_lease", return_value=False):
            n = engine.evaluate()
            assert n == 0

    def test_cooldown(self):
        engine = self._make_engine()
        engine._cooldown_s = 300
        engine._last_eval = __import__("time").time()
        assert not engine.should_evaluate()

    def test_max_recs_limit(self):
        engine = self._make_engine(max_recs_per_eval=2)
        fleet = _fleet([
            _machine("m1", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)]),
            _machine("m2", "n2", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)]),
            _machine("m3", "n3", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)]),
        ])
        with patch.object(engine, "_acquire_lease", return_value=True), \
             patch.object(engine, "_fetch_fleet_state", return_value=fleet):
            engine.evaluate()
            recs = engine._reporter.report_from_swarm_policy.call_args[0][0]
            assert len(recs) <= 2

    def test_blast_radius_caps_machines(self):
        engine = self._make_engine(max_affected_pct=25, ramp_pct=100)
        # 4 machines with idle GPUs — 25% limit means only 1 machine affected
        fleet = _fleet([
            _machine("m1", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)]),
            _machine("m2", "n2", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)]),
            _machine("m3", "n3", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)]),
            _machine("m4", "n4", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)]),
        ])
        with patch.object(engine, "_acquire_lease", return_value=True), \
             patch.object(engine, "_fetch_fleet_state", return_value=fleet):
            engine.evaluate()
            recs = engine._reporter.report_from_swarm_policy.call_args[0][0]
            affected = {r["machine_id"] for r in recs}
            assert len(affected) <= 1

    def test_ramp_limits_initial_reach(self):
        engine = self._make_engine(ramp_pct=10, max_affected_pct=100)
        # 10 machines — 10% ramp means only 1 machine on first eval
        machines = [
            _machine(f"m{i}", f"n{i}", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)])
            for i in range(10)
        ]
        fleet = _fleet(machines)
        with patch.object(engine, "_acquire_lease", return_value=True), \
             patch.object(engine, "_fetch_fleet_state", return_value=fleet):
            engine.evaluate()
            recs = engine._reporter.report_from_swarm_policy.call_args[0][0]
            affected = {r["machine_id"] for r in recs}
            assert len(affected) <= 1

    def test_ramp_doubles_on_success(self):
        engine = self._make_engine(ramp_pct=50, max_affected_pct=100)
        # 1 machine with idle GPU, 50% ramp = max 1 machine — fits, so ramp doubles
        fleet = _fleet([
            _machine("m1", gpus=[_gpu(0, power=350, limit=400, util=2, idle=True)]),
            _machine("m2", "n2", gpus=[_gpu(0, power=300, limit=400, util=80)]),
        ])
        with patch.object(engine, "_acquire_lease", return_value=True), \
             patch.object(engine, "_fetch_fleet_state", return_value=fleet):
            engine.evaluate()
            assert engine._ramp_state.get("idle_gpu_power_cap", 0) >= 100

    def test_snapshot_history(self):
        engine = self._make_engine()
        fleet1 = _fleet([_machine(gpus=[_gpu(util=80)])])
        fleet2 = _fleet([_machine(gpus=[_gpu(util=30)])])
        with patch.object(engine, "_acquire_lease", return_value=True):
            with patch.object(engine, "_fetch_fleet_state", return_value=fleet1):
                engine.evaluate()
            with patch.object(engine, "_fetch_fleet_state", return_value=fleet2):
                engine.evaluate()
        assert len(engine.snapshot_history) == 2

    def test_get_stats(self):
        engine = self._make_engine()
        stats = engine.get_stats()
        assert stats["is_leader"] is False
        assert stats["eval_count"] == 0
        assert "policies_enabled" in stats

    def test_parse_fleet(self):
        engine = self._make_engine()
        data = {
            "machines": [{
                "machine_id": "m1",
                "hostname": "node-1",
                "cluster_tag": "prod",
                "gpu_count": 2,
                "gpus": [
                    {"gpu_index": 0, "gpu_name": "A100", "power_draw_w": 300,
                     "power_limit_w": 400, "utilization_pct": 80, "temperature_c": 65,
                     "memory_used_mb": 20000, "memory_total_mb": 81920, "is_idle": False},
                ],
                "last_heartbeat_age_s": 30,
                "carbon_intensity_gco2e": 150,
                "grid_zone": "US-CAL-CISO",
            }],
        }
        fleet = engine._parse_fleet(data)
        assert len(fleet.machines) == 1
        assert fleet.machines[0].gpus[0].gpu_name == "A100"
        assert fleet.machines[0].carbon_intensity_gco2e == 150

    def test_enabled_policies_filter(self):
        engine = SwarmPolicyEngine(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="leader",
            reporter=MagicMock(),
            enabled_policies=["idle_gpu_power_cap"],
            cooldown_s=0,
        )
        assert len(engine._policies) == 1
        assert engine._policies[0][0] == "idle_gpu_power_cap"


class TestFleetStateGrouping:
    def test_clusters(self):
        f = _fleet([
            _machine("m1", gpus=[_gpu()]),
            _machine("m2", gpus=[_gpu()]),
        ])
        f.machines[0].cluster_tag = "prod"
        f.machines[1].cluster_tag = "staging"
        assert len(f.clusters) == 2
        assert "prod" in f.clusters

    def test_zones(self):
        f = _fleet([
            _machine("m1", gpus=[_gpu()], zone="US-CAL"),
            _machine("m2", gpus=[_gpu()], zone="US-CAL"),
            _machine("m3", gpus=[_gpu()], zone="EU-DE"),
        ])
        assert len(f.zones) == 2
        assert len(f.zones["US-CAL"]) == 2


class TestAdaptivePolling:
    def test_interval_increases_on_empty(self):
        from command_receiver import CommandReceiver
        cr = CommandReceiver(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="test",
            base_interval=60,
            max_interval=300,
        )
        with patch.object(cr, "_fetch_commands", return_value=[]):
            for _ in range(4):
                cr.poll_and_execute()
        assert cr.poll_interval > 60

    def test_interval_resets_on_command(self):
        from command_receiver import CommandReceiver
        cr = CommandReceiver(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="test",
            dry_run=True,
            base_interval=60,
        )
        # Drive up the interval
        with patch.object(cr, "_fetch_commands", return_value=[]):
            for _ in range(5):
                cr.poll_and_execute()
        assert cr.poll_interval > 60

        # Command resets it
        cmd = {"id": "c1", "command_type": "apply_power_cap", "params": {"gpu_index": 0, "watts": 300}}
        with patch.object(cr, "_fetch_commands", return_value=[cmd]), \
             patch.object(cr, "_report_result"):
            cr.poll_and_execute()
        assert cr.poll_interval == 60
