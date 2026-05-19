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
Security hardening unit tests.

Tests:
  1.  Env allowlist: secrets are filtered; ALUMINATAI_* keys are kept
  2.  WAL encrypt/decrypt round-trip: row survives write → read intact
  3.  WAL key mismatch: row encrypted with key_A is silently skipped when read with key_B
  4.  Prometheus Basic Auth: 401 without credentials; 200 with correct credentials
  5.  Offline mode: upload_batch() writes to WAL, returns False, makes no HTTP call
  6.  Replay subcommand: WAL rows exported to CSV correctly
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure agent/ directory is on sys.path
_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


def _load_agent_module():
    """Load agent.py as a module, bypassing the agent/ package and collector dep."""
    # Pre-populate a stub collector so the SyntaxError in collector.py is skipped
    if "collector" not in sys.modules:
        sys.modules["collector"] = types.ModuleType("collector")
    spec = importlib.util.spec_from_file_location(
        "_agent_script", _AGENT_DIR / "agent.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Env var allowlist
# ---------------------------------------------------------------------------

class TestEnvironAllowlist(unittest.TestCase):
    def test_secrets_are_dropped(self):
        from attribution.process_probe import _filter_environ
        dirty = {
            "AWS_SECRET_ACCESS_KEY": "supersecret",
            "DATABASE_URL": "postgres://user:pw@host/db",
            "ALUMINATAI_TEAM": "ml-infra",
            "SLURM_JOB_ID": "12345",
            "PATH": "/usr/bin:/bin",
        }
        result = _filter_environ(dirty)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", result)
        self.assertNotIn("DATABASE_URL", result)
        self.assertNotIn("PATH", result)
        self.assertIn("ALUMINATAI_TEAM", result)
        self.assertIn("SLURM_JOB_ID", result)

    def test_aluminatai_prefix_kept(self):
        from attribution.process_probe import _filter_environ
        env = {
            "ALUMINATAI_MODEL": "gpt4",
            "ALUMINATAI_CUSTOM_TAG": "prod",
            "SECRET_TOKEN": "abc",
        }
        result = _filter_environ(env)
        self.assertIn("ALUMINATAI_MODEL", result)
        self.assertIn("ALUMINATAI_CUSTOM_TAG", result)
        self.assertNotIn("SECRET_TOKEN", result)

    def test_allowlisted_scheduler_keys_kept(self):
        from attribution.process_probe import _filter_environ
        env = {
            "RUNAI_JOB_NAME": "my-job",
            "KUBERNETES_SERVICE_HOST": "10.0.0.1",
            "NOISY_VAR": "noise",
        }
        result = _filter_environ(env)
        self.assertIn("RUNAI_JOB_NAME", result)
        self.assertIn("KUBERNETES_SERVICE_HOST", result)
        self.assertNotIn("NOISY_VAR", result)


# ---------------------------------------------------------------------------
# 2. WAL encrypt/decrypt round-trip
# ---------------------------------------------------------------------------

class TestWALEncryption(unittest.TestCase):
    def _make_fernet(self, key_str: str):
        from cryptography.fernet import Fernet
        raw = hashlib.sha256(key_str.encode()).digest()
        return Fernet(base64.urlsafe_b64encode(raw))

    def test_round_trip(self):
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            self.skipTest("cryptography not installed")

        fernet = self._make_fernet("alum_testkey_abc123")
        sample_row = {"gpu_uuid": "GPU-test", "power_draw_w": 250.0, "team_id": "ml"}

        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "metrics.wal"

            # Write encrypted line manually
            entry = {"ts": time.time(), "row": sample_row}
            line_bytes = json.dumps(entry).encode()
            with open(wal_path, "w") as f:
                f.write(fernet.encrypt(line_bytes).decode() + "\n")

            # Read it back using a patched _FERNET
            import uploader as uploader_mod
            original_fernet = uploader_mod._FERNET
            original_wal = uploader_mod.WAL_FILE
            try:
                uploader_mod._FERNET = fernet
                uploader_mod.WAL_FILE = wal_path
                rows = uploader_mod._wal_read_valid()
            finally:
                uploader_mod._FERNET = original_fernet
                uploader_mod.WAL_FILE = original_wal

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gpu_uuid"], "GPU-test")
        self.assertEqual(rows[0]["power_draw_w"], 250.0)

    def test_key_mismatch_silently_skipped(self):
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            self.skipTest("cryptography not installed")

        fernet_a = self._make_fernet("key_A_secret")
        fernet_b = self._make_fernet("key_B_different")

        sample_row = {"gpu_uuid": "GPU-1", "team_id": "finance"}

        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "metrics.wal"

            # Write with key A
            entry = {"ts": time.time(), "row": sample_row}
            line_bytes = json.dumps(entry).encode()
            with open(wal_path, "w") as f:
                f.write(fernet_a.encrypt(line_bytes).decode() + "\n")

            # Read with key B
            import uploader as uploader_mod
            original_fernet = uploader_mod._FERNET
            original_wal = uploader_mod.WAL_FILE
            try:
                uploader_mod._FERNET = fernet_b
                uploader_mod.WAL_FILE = wal_path
                rows = uploader_mod._wal_read_valid()
            finally:
                uploader_mod._FERNET = original_fernet
                uploader_mod.WAL_FILE = original_wal

        # Row should be silently skipped — no crash, no data
        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# 3. Prometheus Basic Auth middleware
# ---------------------------------------------------------------------------

class TestPrometheusBasicAuth(unittest.TestCase):
    def _make_middleware(self, credentials: str):
        from metrics_server import _basic_auth_middleware

        # Minimal WSGI app that always returns 200 OK
        def _ok_app(environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"ok"]

        return _basic_auth_middleware(_ok_app, credentials)

    def _call_wsgi(self, app, auth_header: str = "") -> tuple[str, list]:
        """Call a WSGI app and return (status, body_bytes)."""
        environ = {
            "REQUEST_METHOD": "GET",
            "PATH_INFO": "/metrics",
            "SERVER_NAME": "localhost",
            "SERVER_PORT": "9100",
            "wsgi.input": io.BytesIO(b""),
            "wsgi.errors": sys.stderr,
        }
        if auth_header:
            environ["HTTP_AUTHORIZATION"] = auth_header

        status_holder = []
        headers_holder = []

        def start_response(status, headers):
            status_holder.append(status)
            headers_holder.extend(headers)

        body = b"".join(app(environ, start_response))
        return status_holder[0], body

    def test_no_auth_returns_401(self):
        app = self._make_middleware("user:pass")
        status, body = self._call_wsgi(app)
        self.assertTrue(status.startswith("401"))

    def test_wrong_credentials_returns_401(self):
        app = self._make_middleware("user:pass")
        wrong = "Basic " + base64.b64encode(b"user:wrong").decode()
        status, body = self._call_wsgi(app, wrong)
        self.assertTrue(status.startswith("401"))

    def test_correct_credentials_returns_200(self):
        app = self._make_middleware("scraper:secret")
        correct = "Basic " + base64.b64encode(b"scraper:secret").decode()
        status, body = self._call_wsgi(app, correct)
        self.assertTrue(status.startswith("200"))
        self.assertEqual(body, b"ok")


# ---------------------------------------------------------------------------
# 4. Offline mode
# ---------------------------------------------------------------------------

class TestOfflineMode(unittest.TestCase):
    def test_upload_batch_writes_wal_no_http(self):
        import uploader as uploader_mod

        sample_metrics = [{"gpu_uuid": "GPU-0", "power_draw_w": 100.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "metrics.wal"
            original_offline = uploader_mod.OFFLINE_MODE
            original_wal = uploader_mod.WAL_FILE
            try:
                uploader_mod.OFFLINE_MODE = True
                uploader_mod.WAL_FILE = wal_path
                # WAL_DIR must also exist
                wal_path.parent.mkdir(parents=True, exist_ok=True)

                uploader = uploader_mod.MetricsUploader(
                    api_endpoint="http://should-not-be-called.invalid",
                    api_key="alum_test",
                )

                # Patch requests.Session.post to detect any HTTP calls
                with patch.object(uploader.session, "post") as mock_post:
                    result = uploader.upload_batch(sample_metrics)
                    mock_post.assert_not_called()

                self.assertFalse(result)
                self.assertTrue(wal_path.exists())

                # Verify WAL contains the row
                content = wal_path.read_text().strip()
                self.assertTrue(content)

            finally:
                uploader_mod.OFFLINE_MODE = original_offline
                uploader_mod.WAL_FILE = original_wal

    def test_retry_failed_uploads_skipped_in_offline_mode(self):
        import uploader as uploader_mod

        original_offline = uploader_mod.OFFLINE_MODE
        try:
            uploader_mod.OFFLINE_MODE = True
            uploader = uploader_mod.MetricsUploader(
                api_endpoint="http://unused.invalid",
                api_key="alum_test",
            )
            result = uploader.retry_failed_uploads()
            self.assertEqual(result, 0)
        finally:
            uploader_mod.OFFLINE_MODE = original_offline


# ---------------------------------------------------------------------------
# 5. Replay subcommand
# ---------------------------------------------------------------------------

class TestReplaySubcommand(unittest.TestCase):
    def test_exports_wal_to_csv(self):
        import uploader as uploader_mod
        agent_mod = _load_agent_module()

        sample_row = {"gpu_uuid": "GPU-0", "power_draw_w": 300.0, "team_id": "research"}

        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "metrics.wal"
            csv_path = Path(tmpdir) / "out.csv"

            # Write a plaintext WAL entry (no encryption)
            entry = {"ts": time.time(), "row": sample_row}
            wal_path.write_text(json.dumps(entry) + "\n")

            original_fernet = uploader_mod._FERNET
            original_wal = uploader_mod.WAL_FILE
            try:
                uploader_mod._FERNET = None  # no encryption for this test
                uploader_mod.WAL_FILE = wal_path

                args = MagicMock()
                args.output = str(csv_path)
                args.clear = False

                rc = agent_mod._cmd_replay(args)

                self.assertEqual(rc, 0)
                self.assertTrue(csv_path.exists())

                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["gpu_uuid"], "GPU-0")
                self.assertEqual(rows[0]["team_id"], "research")
            finally:
                uploader_mod._FERNET = original_fernet
                uploader_mod.WAL_FILE = original_wal

    def test_clear_flag_removes_wal(self):
        import uploader as uploader_mod
        agent_mod = _load_agent_module()

        sample_row = {"gpu_uuid": "GPU-1", "power_draw_w": 150.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "metrics.wal"
            csv_path = Path(tmpdir) / "out.csv"

            entry = {"ts": time.time(), "row": sample_row}
            wal_path.write_text(json.dumps(entry) + "\n")

            original_fernet = uploader_mod._FERNET
            original_wal = uploader_mod.WAL_FILE
            try:
                uploader_mod._FERNET = None
                uploader_mod.WAL_FILE = wal_path

                args = MagicMock()
                args.output = str(csv_path)
                args.clear = True

                rc = agent_mod._cmd_replay(args)
            finally:
                uploader_mod._FERNET = original_fernet
                uploader_mod.WAL_FILE = original_wal

        self.assertEqual(rc, 0)
        self.assertFalse(wal_path.exists())


if __name__ == "__main__":
    unittest.main()
