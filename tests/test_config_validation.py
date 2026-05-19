"""Tests for config validation and bounds clamping."""

import importlib
import os
import unittest
from unittest.mock import patch


class TestConfigValidation(unittest.TestCase):

    def _reload_config(self, env_overrides: dict):
        """Reload config module with custom env vars."""
        env = {
            "ALUMINATAI_API_KEY": "",
            "ALUMINATAI_CONFIG": "",
            **env_overrides,
        }
        with patch.dict(os.environ, env, clear=False):
            import config
            importlib.reload(config)
            return config

    def test_sample_interval_too_low(self):
        cfg = self._reload_config({"SAMPLE_INTERVAL": "0.001"})
        self.assertGreaterEqual(cfg.SAMPLE_INTERVAL, 0.1)

    def test_sample_interval_too_high(self):
        cfg = self._reload_config({"SAMPLE_INTERVAL": "9999"})
        self.assertLessEqual(cfg.SAMPLE_INTERVAL, 300)

    def test_upload_batch_size_zero(self):
        cfg = self._reload_config({"UPLOAD_BATCH_SIZE": "0"})
        self.assertGreaterEqual(cfg.UPLOAD_BATCH_SIZE, 1)

    def test_metrics_port_negative(self):
        cfg = self._reload_config({"METRICS_PORT": "-1"})
        self.assertGreaterEqual(cfg.METRICS_PORT, 0)

    def test_metrics_port_too_high(self):
        cfg = self._reload_config({"METRICS_PORT": "70000"})
        self.assertLessEqual(cfg.METRICS_PORT, 65535)

    def test_pid_stable_threshold_clamped(self):
        cfg = self._reload_config({"PID_STABLE_THRESHOLD": "1.5"})
        self.assertLessEqual(cfg.PID_STABLE_THRESHOLD, 1.0)

    def test_warmup_adjusted_when_less_than_baseline(self):
        cfg = self._reload_config({
            "WARMUP_DISCARD_SECONDS": "20",
            "IDLE_BASELINE_WINDOW": "30",
        })
        self.assertGreater(cfg.WARMUP_DISCARD_SECONDS, cfg.IDLE_BASELINE_WINDOW)

    def test_valid_config_unchanged(self):
        cfg = self._reload_config({
            "SAMPLE_INTERVAL": "5.0",
            "UPLOAD_BATCH_SIZE": "100",
            "METRICS_PORT": "9100",
        })
        self.assertEqual(cfg.SAMPLE_INTERVAL, 5.0)
        self.assertEqual(cfg.UPLOAD_BATCH_SIZE, 100)
        self.assertEqual(cfg.METRICS_PORT, 9100)


if __name__ == "__main__":
    unittest.main()
