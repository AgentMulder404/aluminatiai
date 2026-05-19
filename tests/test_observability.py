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
Error Handling & Observability unit tests.

Tests:
  1.  JSON formatter includes extra={} fields in output
  2.  JSON formatter does not include standard log record noise
  3.  Config hash is stable across two identical calls
  4.  Config hash changes when a config value changes
  5.  send_heartbeat payload includes uptime_sec and config_hash
  6.  WAL pending counter increments on _wal_append, resets on _wal_clear
  7.  WAL pending counter resets to len(rows) on _wal_rewrite
  8.  Structured error fields present in upload_batch warning log
  9.  MetricsServer registers new metrics when prometheus-client is available
  10. get_status() returns wal_entries_pending and replay counters
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


def _load_agent_module():
    """Load agent.py bypassing the agent/ package and collector dep."""
    if "collector" not in sys.modules:
        sys.modules["collector"] = types.ModuleType("collector")
    spec = importlib.util.spec_from_file_location("_agent_script", _AGENT_DIR / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1 & 2. _JsonFormatter — extra fields included, noise excluded
# ---------------------------------------------------------------------------

class TestJsonFormatter(unittest.TestCase):
    def _make_record(self, msg: str, extra: dict = None) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test.logger", level=logging.WARNING,
            pathname="", lineno=0, msg=msg, args=(), exc_info=None,
        )
        if extra:
            for k, v in extra.items():
                setattr(record, k, v)
        return record

    def test_extra_fields_merged_into_json(self):
        agent = _load_agent_module()
        fmt = agent._JsonFormatter()
        record = self._make_record(
            "Upload timeout",
            extra={"event": "upload_timeout", "attempt": 2, "max_retries": 5},
        )
        out = json.loads(fmt.format(record))
        self.assertEqual(out["event"], "upload_timeout")
        self.assertEqual(out["attempt"], 2)
        self.assertEqual(out["max_retries"], 5)
        self.assertEqual(out["level"], "WARNING")
        self.assertEqual(out["msg"], "Upload timeout")

    def test_standard_attrs_not_duplicated(self):
        agent = _load_agent_module()
        fmt = agent._JsonFormatter()
        record = self._make_record("Hello")
        out = json.loads(fmt.format(record))
        # Standard LogRecord internals should not leak into JSON
        self.assertNotIn("lineno", out)
        self.assertNotIn("pathname", out)
        self.assertNotIn("args", out)
        self.assertNotIn("exc_text", out)
        # But our standard keys must be present
        self.assertIn("ts", out)
        self.assertIn("level", out)
        self.assertIn("logger", out)
        self.assertIn("msg", out)

    def test_json_parseable_with_int_extra(self):
        agent = _load_agent_module()
        fmt = agent._JsonFormatter()
        record = self._make_record("msg", extra={"count": 42, "ratio": 0.95})
        out = json.loads(fmt.format(record))
        self.assertEqual(out["count"], 42)
        self.assertAlmostEqual(out["ratio"], 0.95)


# ---------------------------------------------------------------------------
# 3 & 4. Config hash stability and sensitivity
# ---------------------------------------------------------------------------

class TestConfigHash(unittest.TestCase):
    def test_stable_across_calls(self):
        agent = _load_agent_module()
        h1 = agent._compute_config_hash()
        h2 = agent._compute_config_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 8)
        self.assertRegex(h1, r"^[0-9a-f]{8}$")

    def test_changes_when_config_changes(self):
        import config as config_mod
        agent = _load_agent_module()

        original = config_mod.SAMPLE_INTERVAL
        try:
            h_before = agent._compute_config_hash()
            config_mod.SAMPLE_INTERVAL = original + 99.0
            h_after = agent._compute_config_hash()
            self.assertNotEqual(h_before, h_after)
        finally:
            config_mod.SAMPLE_INTERVAL = original


# ---------------------------------------------------------------------------
# 5. send_heartbeat includes uptime_sec and config_hash
# ---------------------------------------------------------------------------

class TestHeartbeatPayload(unittest.TestCase):
    def test_payload_includes_uptime_and_config_hash(self):
        agent = _load_agent_module()
        captured = {}

        class _FakeResp:
            def read(self): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *a): pass

        def _fake_urlopen(req, timeout=None):
            import json as _j
            captured["payload"] = _j.loads(req.data)
            return _FakeResp()

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            agent.send_heartbeat(
                endpoint="http://localhost",
                api_key="alum_test",
                gpu_count=2,
                gpu_uuids=["GPU-0", "GPU-1"],
                scheduler_name="slurm",
                uptime_sec=300.5,
                config_hash="abcd1234",
            )

        self.assertAlmostEqual(captured["payload"]["uptime_sec"], 300.5)
        self.assertEqual(captured["payload"]["config_hash"], "abcd1234")
        self.assertEqual(captured["payload"]["agent_version"], agent.AGENT_VERSION)


# ---------------------------------------------------------------------------
# 6 & 7. WAL pending counter
# ---------------------------------------------------------------------------

class TestWALPendingCounter(unittest.TestCase):
    def test_increments_on_append_resets_on_clear(self):
        import uploader as u

        original_fernet = u._FERNET
        original_wal = u.WAL_FILE
        original_pending = u._WAL_PENDING

        with tempfile.TemporaryDirectory() as td:
            wal_path = Path(td) / "metrics.wal"
            try:
                u._FERNET = None
                u.WAL_FILE = wal_path
                u._WAL_PENDING = 0

                batch = [{"gpu_uuid": "GPU-0"}, {"gpu_uuid": "GPU-1"}]
                u._wal_append(batch)
                self.assertEqual(u._WAL_PENDING, 2)

                u._wal_append([{"gpu_uuid": "GPU-2"}])
                self.assertEqual(u._WAL_PENDING, 3)

                u._wal_clear()
                self.assertEqual(u._WAL_PENDING, 0)
            finally:
                u._FERNET = original_fernet
                u.WAL_FILE = original_wal
                u._WAL_PENDING = original_pending

    def test_rewrite_sets_pending_to_row_count(self):
        import uploader as u

        original_fernet = u._FERNET
        original_wal = u.WAL_FILE
        original_pending = u._WAL_PENDING

        with tempfile.TemporaryDirectory() as td:
            wal_path = Path(td) / "metrics.wal"
            try:
                u._FERNET = None
                u.WAL_FILE = wal_path
                u._WAL_PENDING = 100  # some initial value

                remaining = [{"gpu_uuid": f"GPU-{i}"} for i in range(7)]
                u._wal_rewrite(remaining)
                self.assertEqual(u._WAL_PENDING, 7)
            finally:
                u._FERNET = original_fernet
                u.WAL_FILE = original_wal
                u._WAL_PENDING = original_pending


# ---------------------------------------------------------------------------
# 8. Structured error fields in upload_batch log output
# ---------------------------------------------------------------------------

class TestStructuredUploadErrors(unittest.TestCase):
    def test_timeout_log_has_structured_extra(self):
        import uploader as uploader_mod

        original_dry = uploader_mod.DRY_RUN
        original_offline = uploader_mod.OFFLINE_MODE
        try:
            uploader_mod.DRY_RUN = False
            uploader_mod.OFFLINE_MODE = False
            inst = uploader_mod.MetricsUploader(
                api_endpoint="http://unused.invalid", api_key="alum_test"
            )
        finally:
            uploader_mod.DRY_RUN = original_dry
            uploader_mod.OFFLINE_MODE = original_offline

        with tempfile.TemporaryDirectory() as td:
            wal_path = Path(td) / "metrics.wal"
            original_wal = uploader_mod.WAL_FILE
            try:
                uploader_mod.WAL_FILE = wal_path

                log_records = []

                class _Capture(logging.Handler):
                    def emit(self, record):
                        log_records.append(record)

                handler = _Capture()
                logging.getLogger("uploader").addHandler(handler)
                try:
                    import requests
                    with patch.object(inst.session, "post",
                                      side_effect=requests.Timeout("simulated")):
                        inst.upload_batch([{"gpu_uuid": "GPU-0"}])
                finally:
                    logging.getLogger("uploader").removeHandler(handler)
            finally:
                uploader_mod.WAL_FILE = original_wal

        timeout_records = [r for r in log_records if "timeout" in r.getMessage().lower()]
        self.assertTrue(len(timeout_records) > 0, "Expected at least one timeout log")
        first = timeout_records[0]
        self.assertTrue(hasattr(first, "event"), "Expected 'event' extra field")
        self.assertEqual(first.event, "upload_timeout")
        self.assertTrue(hasattr(first, "attempt"))
        self.assertTrue(hasattr(first, "next_delay_s"))


# ---------------------------------------------------------------------------
# 9. MetricsServer new metric attributes registered
# ---------------------------------------------------------------------------

class TestMetricsServerNewMetrics(unittest.TestCase):
    def test_new_metrics_registered_when_prom_available(self):
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            self.skipTest("prometheus-client not installed")

        # Import fresh to avoid re-registration errors from other tests
        import metrics_server as ms_mod
        srv = ms_mod.MetricsServer()

        if not ms_mod._PROM or srv._port == 0:
            self.skipTest("prometheus-client not active (port=0 or not imported)")

        self.assertTrue(hasattr(srv, "_wal_size_bytes"), "Expected _wal_size_bytes gauge")
        self.assertTrue(hasattr(srv, "_wal_entries_pending"), "Expected _wal_entries_pending gauge")
        self.assertTrue(hasattr(srv, "_wal_replay_uploaded"), "Expected _wal_replay_uploaded counter")
        self.assertTrue(hasattr(srv, "_wal_replay_failed"), "Expected _wal_replay_failed counter")
        self.assertTrue(hasattr(srv, "_attribution_unresolved"), "Expected _attribution_unresolved counter")
        self.assertTrue(hasattr(srv, "_agent_uptime"), "Expected _agent_uptime gauge")
        self.assertTrue(hasattr(srv, "_agent_info"), "Expected _agent_info gauge")

    def test_update_wal_stats_no_crash_when_not_started(self):
        import metrics_server as ms_mod
        srv = ms_mod.MetricsServer()
        # Should be a no-op when not started
        srv.update_wal_stats(wal_size_bytes=1024, wal_entries_pending=10)

    def test_record_attribution_unresolved_no_crash_when_not_started(self):
        import metrics_server as ms_mod
        srv = ms_mod.MetricsServer()
        srv.record_attribution_unresolved(5)  # should not raise


# ---------------------------------------------------------------------------
# 10. get_status returns new fields
# ---------------------------------------------------------------------------

class TestGetStatus(unittest.TestCase):
    def test_get_status_includes_wal_and_replay_fields(self):
        import uploader as uploader_mod

        original_dry = uploader_mod.DRY_RUN
        original_offline = uploader_mod.OFFLINE_MODE
        try:
            uploader_mod.DRY_RUN = False
            uploader_mod.OFFLINE_MODE = False
            inst = uploader_mod.MetricsUploader(
                api_endpoint="http://unused.invalid", api_key="alum_test"
            )
        finally:
            uploader_mod.DRY_RUN = original_dry
            uploader_mod.OFFLINE_MODE = original_offline

        status = inst.get_status()
        self.assertIn("wal_entries_pending", status)
        self.assertIn("wal_replay_uploaded_total", status)
        self.assertIn("wal_replay_failed_total", status)
        self.assertIsInstance(status["wal_entries_pending"], int)
        self.assertEqual(status["wal_replay_uploaded_total"], 0)
        self.assertEqual(status["wal_replay_failed_total"], 0)


if __name__ == "__main__":
    unittest.main()
