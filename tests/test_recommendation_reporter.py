"""Tests for agent/recommendation_reporter.py."""

import time
from unittest.mock import patch, MagicMock
from recommendation_reporter import RecommendationReporter


class _FakeTuneResult:
    def __init__(self, gpu_index, gpu_name, current_power_w, recommended_cap_w, savings_pct, reason):
        self.gpu_index = gpu_index
        self.gpu_name = gpu_name
        self.current_power_w = current_power_w
        self.recommended_cap_w = recommended_cap_w
        self.estimated_savings_pct = savings_pct
        self.reason = reason
        self.applied = False


class TestRecommendationReporter:
    def _make(self):
        return RecommendationReporter(
            endpoint="https://example.com/api/metrics/ingest",
            api_key="alum_test",
            machine_id="test-machine",
        )

    def test_dedup_same_recommendation(self):
        rr = self._make()
        tr = _FakeTuneResult(0, "A100", 300, 240, 20.0, "low util")
        with patch.object(rr, "_upload", return_value=1) as mock_upload:
            rr.report_from_auto_tuner([tr])
            assert mock_upload.call_count == 1
            # Second call within dedup window should be empty
            rr.report_from_auto_tuner([tr])
            assert mock_upload.call_count == 2
            # Second call should have empty list
            assert mock_upload.call_args[0][0] == []

    def test_skips_none_cap(self):
        rr = self._make()
        tr = _FakeTuneResult(0, "A100", 300, None, 0, "no savings")
        with patch.object(rr, "_upload", return_value=0) as mock_upload:
            rr.report_from_auto_tuner([tr])
            assert mock_upload.call_args[0][0] == []

    def test_priority_p1_for_high_savings(self):
        rr = self._make()
        tr = _FakeTuneResult(0, "A100", 300, 200, 33.0, "low util")
        with patch.object(rr, "_upload", return_value=1) as mock_upload:
            rr.report_from_auto_tuner([tr])
            recs = mock_upload.call_args[0][0]
            assert len(recs) == 1
            assert recs[0]["priority"] == "P1"

    def test_priority_p2_for_low_savings(self):
        rr = self._make()
        tr = _FakeTuneResult(0, "A100", 300, 260, 13.0, "low util")
        with patch.object(rr, "_upload", return_value=1) as mock_upload:
            rr.report_from_auto_tuner([tr])
            recs = mock_upload.call_args[0][0]
            assert recs[0]["priority"] == "P2"

    def test_action_payload_has_command(self):
        rr = self._make()
        tr = _FakeTuneResult(2, "H100", 700, 560, 20.0, "low util")
        with patch.object(rr, "_upload", return_value=1) as mock_upload:
            rr.report_from_auto_tuner([tr])
            recs = mock_upload.call_args[0][0]
            payload = recs[0]["action_payload"]
            assert payload["command"] == "apply_power_cap"
            assert payload["gpu_index"] == 2
            assert payload["watts"] == 560

    def test_carbon_scheduler_no_savings_skipped(self):
        rr = self._make()
        count = rr.report_from_carbon_scheduler(None)
        assert count == 0

    def test_workload_analyzer_integration(self):
        rr = self._make()

        class _FakeRec:
            priority = "P2"
            category = "utilization"
            description = "Low GPU utilization"
            estimated_savings_pct = 15.0
            action = "Increase batch size"
            detail = ""

        class _FakeResult:
            gpu_index = 0
            gpu_name = "A100"
            recommendations = [_FakeRec()]

        with patch.object(rr, "_upload", return_value=1) as mock_upload:
            rr.report_from_workload_analyzer(_FakeResult())
            recs = mock_upload.call_args[0][0]
            assert len(recs) == 1
            assert recs[0]["source"] == "workload_analyzer"


class TestCommandReceiver:
    def test_apply_power_cap_validates_range(self):
        from command_receiver import CommandReceiver
        cr = CommandReceiver(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="test",
            dry_run=True,
        )
        ok, msg = cr._apply_power_cap({"gpu_index": 0, "watts": 50})
        assert not ok
        assert "safe range" in msg

    def test_apply_power_cap_dry_run(self):
        from command_receiver import CommandReceiver
        cr = CommandReceiver(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="test",
            dry_run=True,
        )
        ok, msg = cr._apply_power_cap({"gpu_index": 0, "watts": 300})
        assert ok
        assert "Dry run" in msg

    def test_missing_params(self):
        from command_receiver import CommandReceiver
        cr = CommandReceiver(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="test",
        )
        ok, msg = cr._apply_power_cap({})
        assert not ok
        assert "Missing" in msg

    def test_unknown_command(self):
        from command_receiver import CommandReceiver
        cr = CommandReceiver(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="test",
        )
        ok, msg = cr._execute({"command_type": "nuke_gpu", "params": {}})
        assert not ok
        assert "Unknown" in msg

    def test_set_precision_valid(self):
        from command_receiver import CommandReceiver
        cr = CommandReceiver(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="test",
            dry_run=True,
        )
        ok, msg = cr._set_precision({"gpu_index": 0, "precision": "bf16"})
        assert ok
        assert "bf16" in msg

    def test_set_precision_invalid(self):
        from command_receiver import CommandReceiver
        cr = CommandReceiver(
            endpoint="https://example.com",
            api_key="alum_test",
            machine_id="test",
        )
        ok, msg = cr._set_precision({"gpu_index": 0, "precision": "fp8"})
        assert not ok
        assert "Unsupported" in msg


class TestRecommendationReporterExtended:
    def _make(self):
        return RecommendationReporter(
            endpoint="https://example.com/api/metrics/ingest",
            api_key="alum_test",
            machine_id="test-machine",
        )

    def test_hardware_match_skips_low_savings(self):
        rr = self._make()

        class _FakeMatch:
            energy_savings_pct = 3.0
            current_arch = "A100"
            best_arch = "H100"
            model_tag = "llama-3-70b"
            recommendation = "test"

        count = rr.report_from_hardware_match(_FakeMatch())
        assert count == 0

    def test_hardware_match_skips_same_arch(self):
        rr = self._make()

        class _FakeMatch:
            energy_savings_pct = 20.0
            current_arch = "A100"
            best_arch = "A100"
            model_tag = "llama-3-70b"
            recommendation = "already optimal"

        count = rr.report_from_hardware_match(_FakeMatch())
        assert count == 0

    def test_hardware_match_uploads(self):
        rr = self._make()

        class _FakeMatch:
            energy_savings_pct = 25.0
            current_arch = "A100"
            best_arch = "H100"
            model_tag = "llama-3-70b"
            recommendation = "Migrate to H100"

        with patch.object(rr, "_upload", return_value=1) as mock_upload:
            rr.report_from_hardware_match(_FakeMatch(), gpu_index=2)
            recs = mock_upload.call_args[0][0]
            assert len(recs) == 1
            assert recs[0]["source"] == "workload_analyzer"
            assert recs[0]["category"] == "gpu_match"
            assert recs[0]["gpu_index"] == 2

    def test_swarm_policy_upload(self):
        rr = self._make()
        policy_recs = [{
            "category": "power_cap",
            "gpu_index": 0,
            "gpu_name": "A100",
            "priority": "P1",
            "title": "Fleet-wide power cap reduction",
            "description": "3 GPUs underutilized across cluster",
            "action": "Cap all idle GPUs to 150W",
            "estimated_savings_pct": 30.0,
            "effort_score": 1,
            "action_payload": {"command": "apply_power_cap", "gpu_index": 0, "watts": 150},
        }]
        with patch.object(rr, "_upload", return_value=1) as mock_upload:
            rr.report_from_swarm_policy(policy_recs)
            recs = mock_upload.call_args[0][0]
            assert len(recs) == 1
            assert recs[0]["source"] == "swarm_policy"
