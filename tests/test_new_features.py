"""Tests for U1-U3 unicorn features: auto-tuner, fleet, chargeback, cloud, RAPL, MIG, TSDB, carbon scheduler."""

import json
import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
#  U1.1 — AutoTuner
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoTuner(unittest.TestCase):
    def _make_metric(self, gpu_index=0, power_w=250.0, util=20, power_limit=300.0):
        m = MagicMock()
        m.gpu_index = gpu_index
        m.gpu_name = "Test GPU"
        m.power_draw_w = power_w
        m.utilization_gpu_pct = util
        m.power_limit_w = power_limit
        return m

    def test_recommends_cap_for_low_util_high_power(self):
        from efficiency.auto_tuner import AutoTuner
        tuner = AutoTuner(interval_s=0, min_savings_pct=5.0, dry_run=True)
        metrics = [self._make_metric(power_w=250, util=15, power_limit=300)]
        results = tuner.analyze_and_tune(metrics)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].recommended_cap_w)
        self.assertGreater(results[0].estimated_savings_pct, 0)
        self.assertFalse(results[0].applied)

    def test_no_recommendation_for_high_util(self):
        from efficiency.auto_tuner import AutoTuner
        tuner = AutoTuner(interval_s=0, min_savings_pct=5.0, dry_run=True)
        metrics = [self._make_metric(power_w=250, util=80, power_limit=300)]
        results = tuner.analyze_and_tune(metrics)
        self.assertEqual(len(results), 0)

    def test_should_run_respects_interval(self):
        from efficiency.auto_tuner import AutoTuner
        tuner = AutoTuner(interval_s=300, dry_run=True)
        self.assertTrue(tuner.should_run())
        tuner._last_run = time.monotonic()
        self.assertFalse(tuner.should_run())


# ═══════════════════════════════════════════════════════════════════════════════
#  U1.3 — Alert Rules (file validity)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlertRules(unittest.TestCase):
    def test_alert_rules_yaml_is_valid(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("pyyaml not installed")
        rules_path = Path(__file__).parent.parent / "deploy" / "alerts.rules.yml"
        self.assertTrue(rules_path.exists(), f"Missing {rules_path}")
        with open(rules_path) as f:
            data = yaml.safe_load(f)
        self.assertIn("groups", data)
        for group in data["groups"]:
            self.assertIn("name", group)
            self.assertIn("rules", group)
            for rule in group["rules"]:
                self.assertIn("alert", rule)
                self.assertIn("expr", rule)


# ═══════════════════════════════════════════════════════════════════════════════
#  U2.1 — Fleet Aggregator
# ═══════════════════════════════════════════════════════════════════════════════


class TestFleetAggregator(unittest.TestCase):
    def test_ingest_and_stats(self):
        from fleet_aggregator import FleetAggregator
        agg = FleetAggregator()
        agg.ingest({
            "machine_id": "node-1",
            "cluster_tag": "us-west",
            "gpu_count": 4,
            "total_power_w": 1200.0,
            "total_energy_kwh": 5.5,
            "uptime_s": 3600,
            "timestamp": time.time(),
        })
        agg.ingest({
            "machine_id": "node-2",
            "cluster_tag": "us-west",
            "gpu_count": 8,
            "total_power_w": 2400.0,
            "total_energy_kwh": 12.0,
            "uptime_s": 7200,
            "timestamp": time.time(),
        })
        stats = agg.get_fleet_stats()
        self.assertEqual(stats["active_nodes"], 2)
        self.assertEqual(stats["total_gpus"], 12)
        self.assertAlmostEqual(stats["total_power_w"], 3600.0)
        self.assertAlmostEqual(stats["total_energy_kwh"], 17.5)

    def test_stale_nodes_pruned(self):
        from fleet_aggregator import FleetAggregator
        agg = FleetAggregator()
        agg.ingest({
            "machine_id": "stale-node",
            "gpu_count": 2,
            "total_power_w": 500,
            "timestamp": time.time() - 700,  # older than STALE_THRESHOLD_S
        })
        stats = agg.get_fleet_stats()
        self.assertEqual(stats["active_nodes"], 0)

    def test_overwrite_same_machine(self):
        from fleet_aggregator import FleetAggregator
        agg = FleetAggregator()
        agg.ingest({"machine_id": "n1", "gpu_count": 4, "total_power_w": 100, "timestamp": time.time()})
        agg.ingest({"machine_id": "n1", "gpu_count": 4, "total_power_w": 200, "timestamp": time.time()})
        stats = agg.get_fleet_stats()
        self.assertEqual(stats["active_nodes"], 1)
        self.assertAlmostEqual(stats["total_power_w"], 200.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  U2.2 — Chargeback Report
# ═══════════════════════════════════════════════════════════════════════════════


class TestChargebackReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "manifest.csv")
        with open(self.csv_path, "w") as f:
            f.write("timestamp,team_id,model_tag,energy_delta_j,gpu_index\n")
            f.write("2026-04-01T10:00:00+00:00,ml-team,gpt4,360000,0\n")
            f.write("2026-04-01T10:00:05+00:00,ml-team,gpt4,360000,0\n")
            f.write("2026-04-01T10:00:10+00:00,infra,serving,180000,1\n")

    def test_json_report(self):
        from reports.chargeback import generate_report
        result = generate_report(self.csv_path, rate_per_kwh=0.12, output_format="json")
        data = json.loads(result)
        self.assertEqual(len(data["teams"]), 2)
        self.assertGreater(data["total_cost_usd"], 0)
        self.assertGreater(data["total_energy_kwh"], 0)

    def test_csv_report(self):
        from reports.chargeback import generate_report
        result = generate_report(self.csv_path, rate_per_kwh=0.12, output_format="csv")
        lines = result.strip().split("\n")
        self.assertEqual(lines[0], "team_id,energy_kwh,cost_usd,co2_grams,samples")
        self.assertEqual(len(lines), 3)  # header + 2 teams

    def test_html_report(self):
        from reports.chargeback import generate_report
        result = generate_report(self.csv_path, rate_per_kwh=0.12, output_format="html")
        self.assertIn("GPU Energy Chargeback", result)
        self.assertIn("ml-team", result)
        self.assertIn("infra", result)

    def test_with_carbon(self):
        from reports.chargeback import generate_report
        result = generate_report(self.csv_path, rate_per_kwh=0.12, carbon_intensity=400.0, output_format="json")
        data = json.loads(result)
        for team in data["teams"]:
            if team["co2_grams"]:
                self.assertGreater(team["co2_grams"], 0)

    def test_missing_csv(self):
        from reports.chargeback import generate_report
        result = generate_report("/nonexistent/path.csv", output_format="json")
        data = json.loads(result)
        self.assertIn("error", data)

    def test_date_filtering(self):
        from reports.chargeback import generate_report
        result = generate_report(
            self.csv_path,
            from_date="2026-04-01T10:00:03+00:00",
            output_format="json",
        )
        data = json.loads(result)
        # Only rows after 10:00:03 should be included (2 of 3)
        total_samples = sum(t["samples"] for t in data["teams"])
        self.assertEqual(total_samples, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  U2.3 — Cloud Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestCloudDetect(unittest.TestCase):
    def test_detect_returns_unknown_on_local(self):
        from efficiency.cloud_detect import detect
        result = detect()
        self.assertEqual(result.provider, "unknown")
        self.assertEqual(result.gpu_cost_per_hour, 0.0)

    def test_cost_table_has_entries(self):
        from efficiency.cloud_detect import GPU_COST_TABLE
        self.assertGreater(len(GPU_COST_TABLE), 10)
        self.assertIn("p4d.24xlarge", GPU_COST_TABLE)
        self.assertIn("a2-highgpu-8g", GPU_COST_TABLE)


# ═══════════════════════════════════════════════════════════════════════════════
#  U2.4 — RAPL Reader
# ═══════════════════════════════════════════════════════════════════════════════


class TestRaplReader(unittest.TestCase):
    def test_unavailable_on_macos(self):
        from efficiency.rapl import RaplReader
        reader = RaplReader()
        # On macOS (test environment), RAPL should not be available
        if not os.path.exists("/sys/class/powercap/intel-rapl:0/energy_uj"):
            self.assertFalse(reader.available)
            self.assertIsNone(reader.read())

    def test_overflow_handling(self):
        from efficiency.rapl import RaplReader
        self.assertEqual(RaplReader._delta_with_overflow(900, 100, 1000), 200)
        self.assertEqual(RaplReader._delta_with_overflow(100, 500, 1000), 400)


# ═══════════════════════════════════════════════════════════════════════════════
#  U3.1 — MIG Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestMIG(unittest.TestCase):
    def test_no_mig_without_nvml(self):
        from efficiency.mig import detect_mig
        # On test machine without MIG-capable GPU
        info = detect_mig(0)
        self.assertFalse(info.enabled)
        self.assertEqual(len(info.instances), 0)

    def test_split_power_no_mig(self):
        from efficiency.mig import split_power_by_mig, MigInfo
        info = MigInfo(enabled=False, instances=[], total_slices=0)
        result = split_power_by_mig(300.0, info)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (0, 300.0))

    def test_split_power_with_mig(self):
        from efficiency.mig import split_power_by_mig, MigInfo, MigInstance
        instances = [
            MigInstance(index=0, gpu_instance_id=0, compute_instance_id=0,
                       profile_id=3, slice_count=3, memory_mb=20480, power_fraction=3/7),
            MigInstance(index=1, gpu_instance_id=1, compute_instance_id=0,
                       profile_id=2, slice_count=2, memory_mb=10240, power_fraction=2/7),
            MigInstance(index=2, gpu_instance_id=2, compute_instance_id=0,
                       profile_id=2, slice_count=2, memory_mb=10240, power_fraction=2/7),
        ]
        info = MigInfo(enabled=True, instances=instances, total_slices=7)
        result = split_power_by_mig(350.0, info)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0][1], 150.0, places=1)  # 3/7 * 350
        self.assertAlmostEqual(result[1][1], 100.0, places=1)  # 2/7 * 350
        self.assertAlmostEqual(result[2][1], 100.0, places=1)  # 2/7 * 350
        self.assertAlmostEqual(sum(r[1] for r in result), 350.0, places=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  U3.2 — Carbon Scheduler
# ═══════════════════════════════════════════════════════════════════════════════


class TestCarbonScheduler(unittest.TestCase):
    def test_parse_duration_hours(self):
        from efficiency.carbon_scheduler import _parse_duration
        self.assertAlmostEqual(_parse_duration("4h"), 4.0)
        self.assertAlmostEqual(_parse_duration("1.5h"), 1.5)

    def test_parse_duration_minutes(self):
        from efficiency.carbon_scheduler import _parse_duration
        self.assertAlmostEqual(_parse_duration("30m"), 0.5)
        self.assertAlmostEqual(_parse_duration("90m"), 1.5)

    def test_parse_duration_seconds(self):
        from efficiency.carbon_scheduler import _parse_duration
        self.assertAlmostEqual(_parse_duration("3600s"), 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  U3.3 — Local TSDB
# ═══════════════════════════════════════════════════════════════════════════════


class TestLocalTSDB(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")

    def test_insert_and_query(self):
        from storage.tsdb import LocalTSDB
        db = LocalTSDB(db_path=self.db_path, retention_days=7)
        db.insert(0, {"power_w": 250.5, "utilization_pct": 85.0})
        results = db.query("power_w", gpu_index=0)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][1], 250.5)
        db.close()

    def test_batch_insert(self):
        from storage.tsdb import LocalTSDB
        db = LocalTSDB(db_path=self.db_path, retention_days=7)
        metrics = []
        for i in range(3):
            m = MagicMock()
            m.gpu_index = i
            m.power_draw_w = 200 + i * 50
            m.utilization_gpu_pct = 70 + i * 10
            m.temperature_c = 60 + i * 5
            m.memory_used_mb = 8000 + i * 1000
            m.energy_delta_j = 1000 + i * 100
            metrics.append(m)
        db.insert_batch(metrics)
        # Should have 5 metrics per GPU * 3 GPUs = 15 rows
        results_0 = db.query("power_w", gpu_index=0)
        self.assertEqual(len(results_0), 1)
        self.assertAlmostEqual(results_0[0][1], 200.0)
        results_2 = db.query("power_w", gpu_index=2)
        self.assertAlmostEqual(results_2[0][1], 300.0)
        db.close()

    def test_stats(self):
        from storage.tsdb import LocalTSDB
        db = LocalTSDB(db_path=self.db_path, retention_days=7)
        db.insert(0, {"power_w": 100})
        db.insert(0, {"power_w": 200})
        stats = db.stats()
        self.assertEqual(stats["rows"], 2)
        self.assertGreaterEqual(stats["size_mb"], 0)
        self.assertIn("path", stats)
        db.close()

    def test_retention_pruning(self):
        from storage.tsdb import LocalTSDB
        db = LocalTSDB(db_path=self.db_path, retention_days=1)
        # Insert a row with an old timestamp using the db's own connection
        old_ts = int(time.time()) - 200000  # ~2.3 days ago
        db._conn.execute("INSERT INTO metrics VALUES (?, 0, 'power_w', 100.0)", (old_ts,))
        db._conn.commit()
        db.close()
        # Reopen — should prune the old row on init
        db2 = LocalTSDB(db_path=self.db_path, retention_days=1)
        results = db2.query("power_w")
        self.assertEqual(len(results), 0)
        db2.close()

    def test_query_with_time_range(self):
        from storage.tsdb import LocalTSDB
        db = LocalTSDB(db_path=self.db_path, retention_days=7)
        now = int(time.time())
        db._conn.execute("INSERT INTO metrics VALUES (?, 0, 'power_w', 100.0)", (now - 3600,))
        db._conn.execute("INSERT INTO metrics VALUES (?, 0, 'power_w', 200.0)", (now - 1800,))
        db._conn.execute("INSERT INTO metrics VALUES (?, 0, 'power_w', 300.0)", (now,))
        db._conn.commit()
        results = db.query("power_w", gpu_index=0, from_ts=now - 2000, to_ts=now - 100)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][1], 200.0)
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  P0.3 — Prometheus Staleness (MetricsServer.mark_collection_failed)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricsServerStaleness(unittest.TestCase):
    def test_mark_collection_failed_no_crash_when_not_started(self):
        from metrics_server import MetricsServer
        server = MetricsServer()
        # Should not crash even when not started
        server.mark_collection_failed(["GPU-UUID-1", "GPU-UUID-2"])

    def test_update_carbon_no_crash_when_not_started(self):
        from metrics_server import MetricsServer
        server = MetricsServer()
        server.update_carbon("US-CAL-CISO", 200.0, 45.0, 0.001)


# ═══════════════════════════════════════════════════════════════════════════════
#  P1.4 — Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker(unittest.TestCase):
    def test_circuit_opens_after_threshold(self):
        with patch.dict(os.environ, {"ALUMINATAI_API_KEY": "test_key_123"}):
            import importlib
            import config
            importlib.reload(config)
            import uploader
            importlib.reload(uploader)

            up = uploader.MetricsUploader(api_endpoint="http://localhost:99999", api_key="test")
            up.CIRCUIT_OPEN_THRESHOLD = 2
            up.CIRCUIT_COOLDOWN_SECONDS = 1

            # Simulate consecutive failures
            up._consecutive_failures = 2
            up._circuit_open_since = time.time()

            # Circuit is open — should go straight to WAL
            result = up.upload_batch([{"test": 1}])
            self.assertFalse(result)

    def test_get_status_includes_circuit(self):
        with patch.dict(os.environ, {"ALUMINATAI_API_KEY": "test_key_123"}):
            import importlib
            import config
            importlib.reload(config)
            import uploader
            importlib.reload(uploader)

            up = uploader.MetricsUploader(api_endpoint="http://localhost:99999", api_key="test")
            status = up.get_status()
            self.assertIn("circuit_breaker", status)
            self.assertEqual(status["circuit_breaker"], "closed")
            self.assertEqual(status["consecutive_failures"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ring Buffer (Multi-Agent Architecture)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRingBuffer(unittest.TestCase):
    def test_append_and_summarize(self):
        from ring_buffer import GPURingBuffer, GPUSample
        buf = GPURingBuffer(gpu_index=0, max_samples=50)
        now = time.time()
        for i in range(20):
            buf.append(GPUSample(
                timestamp=now + i * 0.2,
                power_w=200.0 + i * 5,
                utilization_pct=80,
                temperature_c=70,
                memory_used_mb=8000.0,
            ))
        summary = buf.summarize()
        self.assertIsNotNone(summary)
        self.assertEqual(summary.sample_count, 20)
        self.assertAlmostEqual(summary.power_mean_w, 247.5, places=1)
        self.assertEqual(summary.power_min_w, 200.0)
        self.assertEqual(summary.power_max_w, 295.0)

    def test_summarize_empty_returns_none(self):
        from ring_buffer import GPURingBuffer
        buf = GPURingBuffer(gpu_index=0)
        self.assertIsNone(buf.summarize())

    def test_summarize_too_few_samples_returns_none(self):
        from ring_buffer import GPURingBuffer, GPUSample
        buf = GPURingBuffer(gpu_index=0)
        buf.append(GPUSample(time.time(), 200.0, 80, 70, 8000.0))
        buf.append(GPUSample(time.time(), 210.0, 82, 71, 8100.0))
        self.assertIsNone(buf.summarize())

    def test_window_seconds_filter(self):
        from ring_buffer import GPURingBuffer, GPUSample
        buf = GPURingBuffer(gpu_index=0, max_samples=100)
        now = time.time()
        for i in range(20):
            buf.append(GPUSample(now - 10 + i, 100.0 + i, 50, 60, 4000.0))
        summary = buf.summarize(window_seconds=5.0)
        self.assertIsNotNone(summary)
        self.assertLess(summary.sample_count, 20)

    def test_eviction_at_max(self):
        from ring_buffer import GPURingBuffer, GPUSample
        buf = GPURingBuffer(gpu_index=0, max_samples=10)
        now = time.time()
        for i in range(20):
            buf.append(GPUSample(now + i, float(i), 50, 60, 4000.0))
        self.assertEqual(len(buf), 10)
        summary = buf.summarize()
        self.assertEqual(summary.power_min_w, 10.0)

    def test_p95_p99_with_enough_samples(self):
        from ring_buffer import GPURingBuffer, GPUSample
        buf = GPURingBuffer(gpu_index=0, max_samples=200)
        now = time.time()
        for i in range(100):
            buf.append(GPUSample(now + i * 0.1, float(i), 50, 60, 4000.0))
        summary = buf.summarize()
        self.assertEqual(summary.sample_count, 100)
        self.assertGreater(summary.power_p95_w, summary.power_p50_w)
        self.assertGreaterEqual(summary.power_p99_w, summary.power_p95_w)


# ═══════════════════════════════════════════════════════════════════════════════
#  Memory Leak Detector
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryLeakDetector(unittest.TestCase):
    def test_detects_monotonic_increase(self):
        from memory_leak_detector import MemoryLeakDetector
        det = MemoryLeakDetector(window_size=20, threshold_pct=0.85)
        triggered = False
        for i in range(25):
            if det.update(0, 1000.0 + i * 10):
                triggered = True
        self.assertTrue(triggered)

    def test_constant_memory_no_trigger(self):
        from memory_leak_detector import MemoryLeakDetector
        det = MemoryLeakDetector(window_size=20, threshold_pct=0.85)
        for i in range(30):
            self.assertFalse(det.update(0, 8000.0))

    def test_sawtooth_no_trigger(self):
        from memory_leak_detector import MemoryLeakDetector
        det = MemoryLeakDetector(window_size=20, threshold_pct=0.85)
        for i in range(30):
            val = 8000.0 + (i % 5) * 100
            self.assertFalse(det.update(0, val))

    def test_leak_score(self):
        from memory_leak_detector import MemoryLeakDetector
        det = MemoryLeakDetector(window_size=20)
        for i in range(20):
            det.update(0, 1000.0 + i * 10)
        score = det.get_leak_score(0)
        self.assertGreater(score, 0.8)

    def test_no_alert_spam(self):
        from memory_leak_detector import MemoryLeakDetector
        det = MemoryLeakDetector(window_size=10, threshold_pct=0.85)
        triggers = 0
        for i in range(30):
            if det.update(0, 1000.0 + i * 10):
                triggers += 1
        self.assertEqual(triggers, 1)

    def test_multi_gpu(self):
        from memory_leak_detector import MemoryLeakDetector
        det = MemoryLeakDetector(window_size=15)
        for i in range(20):
            det.update(0, 8000.0)
            det.update(1, 1000.0 + i * 10)
        self.assertLess(det.get_leak_score(0), 0.5)
        self.assertGreater(det.get_leak_score(1), 0.8)


# ═══════════════════════════════════════════════════════════════════════════════
#  TSDB Bug Fix — stats() after close()
# ═══════════════════════════════════════════════════════════════════════════════


class TestTSDBStatsFix(unittest.TestCase):
    def test_stats_returns_data_before_close(self):
        from storage.tsdb import LocalTSDB
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_fix.db")
        db = LocalTSDB(db_path=db_path, retention_days=7)
        db.insert(0, {"power_w": 100.0})
        db.insert(0, {"power_w": 200.0})
        stats = db.stats()
        self.assertEqual(stats["rows"], 2)
        self.assertIn("path", stats)
        db.close()
        empty_stats = db.stats()
        self.assertEqual(empty_stats, {})


# ═══════════════════════════════════════════════════════════════════════════════
#  K8s Healthz Endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthzEndpoint(unittest.TestCase):
    def test_healthz_exists_on_metrics_server(self):
        from metrics_server import MetricsServer
        server = MetricsServer()
        self.assertTrue(hasattr(server, '_health_middleware'))
        self.assertTrue(hasattr(server, '_health_response'))

    def test_update_mem_leak_score_no_crash(self):
        from metrics_server import MetricsServer
        server = MetricsServer()
        server.update_mem_leak_score("GPU-UUID-1", "0", 0.5)


if __name__ == "__main__":
    unittest.main()
