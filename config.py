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
Configuration for AluminatAI GPU Agent v0.2.2

Priority (highest to lowest):
  1. Environment variables (ALUMINATAI_*, SAMPLE_INTERVAL, …)
  2. Config file  (ALUMINATAI_CONFIG=/path/to/file.json|yaml)
  3. Built-in defaults

Config file is JSON or YAML (YAML requires pip install aluminatiai[observability]).
Default search order when ALUMINATAI_CONFIG is unset:
  ./aluminatai.json → ./aluminatai.yaml → ~/.config/aluminatai/config.json
"""
import json
import logging
import os
import sys
from pathlib import Path

# ── Version ───────────────────────────────────────────────────────────────────

AGENT_VERSION = "0.2.2"

# ── Config-file → env-var mapping ─────────────────────────────────────────────

_CONFIG_KEY_TO_ENV: dict[str, str] = {
    "api_key":                  "ALUMINATAI_API_KEY",
    "api_endpoint":             "ALUMINATAI_API_ENDPOINT",
    "sample_interval":          "SAMPLE_INTERVAL",
    "upload_interval":          "UPLOAD_INTERVAL",
    "upload_batch_size":        "UPLOAD_BATCH_SIZE",
    "upload_max_retries":       "UPLOAD_MAX_RETRIES",
    "upload_max_retry_delay":   "UPLOAD_MAX_RETRY_DELAY",
    "wal_max_age_hours":        "WAL_MAX_AGE_HOURS",
    "wal_max_mb":               "WAL_MAX_MB",
    "scheduler_poll_interval":  "SCHEDULER_POLL_INTERVAL",
    "tag_poll_interval":        "TAG_POLL_INTERVAL",
    "heartbeat_interval":       "HEARTBEAT_INTERVAL",
    "metrics_port":             "METRICS_PORT",
    "metrics_bind_host":        "METRICS_BIND_HOST",
    "metrics_basic_auth":       "METRICS_BASIC_AUTH",
    "offline_mode":             "OFFLINE_MODE",
    "dry_run":                  "DRY_RUN",
    "prometheus_only":          "PROMETHEUS_ONLY",
    "log_level":                "LOG_LEVEL",
    "log_format":               "LOG_FORMAT",
    "data_dir":                 "DATA_DIR",
    "log_dir":                  "LOG_DIR",
    "https_proxy":              "HTTPS_PROXY",
    "ca_bundle":                "ALUMINATAI_CA_BUNDLE",
    "client_cert":              "ALUMINATAI_CLIENT_CERT",
    "client_key":               "ALUMINATAI_CLIENT_KEY",
    "attribution_config":       "ALUMINATAI_ATTRIBUTION_CONFIG",
    "trusted_uids":             "ALUMINATAI_TRUSTED_UIDS",
    "cluster_tag":              "ALUMINATAI_CLUSTER_TAG",
    "location_hint":            "ALUMINATAI_LOCATION_HINT",
    "grid_zone":                "ALUMINATAI_GRID_ZONE",
    "idle_baseline_window":     "IDLE_BASELINE_WINDOW",
    "warmup_discard_seconds":   "WARMUP_DISCARD_SECONDS",
    "dcgm_enabled":             "DCGM_ENABLED",
    "pid_smooth_window":        "PID_SMOOTH_WINDOW",
    "pid_stable_threshold":     "PID_STABLE_THRESHOLD",
}


def _load_config_file() -> None:
    """
    Read a JSON or YAML config file and apply values as env-var fallbacks.

    Env vars already set in the process environment are never overridden —
    explicit env vars always take precedence over the config file.
    """
    path = os.getenv("ALUMINATAI_CONFIG", "")
    if not path:
        candidates = [
            "aluminatai.json",
            "aluminatai.yaml",
            "aluminatai.yml",
            os.path.expanduser("~/.config/aluminatai/config.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    if not path:
        return

    data: dict = {}
    try:
        with open(path) as f:
            raw = f.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore[import-untyped]
                data = yaml.safe_load(raw) or {}
            except ImportError:
                print(
                    "[aluminatai] YAML config requires PyYAML — "
                    "install with: pip install 'aluminatiai[observability]'",
                    file=sys.stderr,
                )
                return
        else:
            data = json.loads(raw)
    except Exception as exc:
        print(f"[aluminatai] Failed to load config file {path!r}: {exc}", file=sys.stderr)
        return

    applied = []
    for key, value in data.items():
        env_var = _CONFIG_KEY_TO_ENV.get(key)
        if not env_var:
            continue
        if env_var in os.environ:
            continue  # env var wins
        # Booleans → "1" or "" so downstream bool() and .lower() parsing works
        if isinstance(value, bool):
            os.environ[env_var] = "1" if value else ""
        else:
            os.environ[env_var] = str(value)
        applied.append(f"{key}={value!r}")

    if applied:
        print(f"[aluminatai] Config file {path!r}: applied {', '.join(applied)}", file=sys.stderr)


# Apply file-based config before any constants are evaluated so that all
# os.getenv() calls below see the merged environment.
_load_config_file()

# ── API Configuration ─────────────────────────────────────────────────────────

API_ENDPOINT = os.getenv("ALUMINATAI_API_ENDPOINT", "https://www.aluminatiai.com/v1/metrics/ingest")
API_KEY = os.getenv("ALUMINATAI_API_KEY", "")

# ── Upload Configuration ──────────────────────────────────────────────────────

UPLOAD_ENABLED = bool(API_KEY)
UPLOAD_INTERVAL = int(os.getenv("UPLOAD_INTERVAL", "60"))        # seconds between flush calls
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))   # metrics per HTTP request

# Exponential backoff
UPLOAD_MAX_RETRIES = int(os.getenv("UPLOAD_MAX_RETRIES", "5"))
UPLOAD_MAX_RETRY_DELAY = int(os.getenv("UPLOAD_MAX_RETRY_DELAY", "60"))  # seconds cap
UPLOAD_TIMEOUT = int(os.getenv("UPLOAD_TIMEOUT", "30"))                 # HTTP request timeout

# ── WAL (Write-Ahead Log) ─────────────────────────────────────────────────────

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
WAL_DIR = DATA_DIR / "wal"
WAL_MAX_AGE_HOURS = int(os.getenv("WAL_MAX_AGE_HOURS", "24"))
WAL_MAX_MB = int(os.getenv("WAL_MAX_MB", "512"))

# Legacy — kept for backward compat with old uploader.py imports
ENABLE_LOCAL_BACKUP = True  # WAL is always active when DATA_DIR is writable

# ── Sampling ──────────────────────────────────────────────────────────────────

SAMPLE_INTERVAL = float(os.getenv("SAMPLE_INTERVAL", "5.0"))     # seconds between NVML reads
NVML_TIMEOUT = float(os.getenv("NVML_TIMEOUT", "2.0"))          # per-GPU collection timeout

# ── Multi-Agent High-Frequency Sampling ──────────────────────────────────────
MULTI_AGENT_ENABLED = os.getenv("MULTI_AGENT_ENABLED", "").lower() in ("1", "true", "yes")
FAST_SAMPLE_INTERVAL = float(os.getenv("FAST_SAMPLE_INTERVAL", "0.2"))
FAST_SAMPLE_BUFFER_SIZE = int(os.getenv("FAST_SAMPLE_BUFFER_SIZE", "100"))

# ── PID Temporal Smoothing ────────────────────────────────────────────────────

# Width (seconds) of the sliding window used to determine PID stability.
# At SAMPLE_INTERVAL=5s this gives ~6 observations per GPU in steady state.
PID_SMOOTH_WINDOW = float(os.getenv("PID_SMOOTH_WINDOW", "30.0"))

# Fraction [0–1] of window samples a PID must appear in to be "stable".
# PIDs below this threshold are filtered from attribution (transient helpers,
# DDP spawn workers still allocating memory, one-shot CUDA utilities).
# The attribution engine always falls back to the raw NVML list when filtering
# would remove *all* current processes (ensures new jobs are never dropped).
PID_STABLE_THRESHOLD = float(os.getenv("PID_STABLE_THRESHOLD", "0.60"))

# ── DCGM Phase Decomposition ──────────────────────────────────────────────────

# Set to "0" to disable DCGM even when pydcgm is installed.
# When enabled, the agent tries to connect to the local nv-hostengine daemon
# for tensor/fp16/fp32/DRAM activity counters; falls back to NVML utilization
# rates automatically if DCGM is unavailable.
DCGM_ENABLED = os.getenv("DCGM_ENABLED", "1").lower() not in ("0", "false", "no")

# ── Idle Baseline + Warmup ────────────────────────────────────────────────────

# Seconds to sample GPU power during startup calibration (requires all GPUs idle).
# Set to 0 to disable automatic calibration.
IDLE_BASELINE_WINDOW = int(os.getenv("IDLE_BASELINE_WINDOW", "30"))

# Samples collected in this window after agent start are excluded from uploads
# and energy accounting to avoid skewing job attribution with warm-up transients.
# Set to 0 to disable.  Must be > IDLE_BASELINE_WINDOW when calibration is enabled
# (the agent finishes calibrating, then discards the overlap).
WARMUP_DISCARD_SECONDS = int(os.getenv("WARMUP_DISCARD_SECONDS", "45"))

# ── Scheduler Integration ─────────────────────────────────────────────────────

SCHEDULER_POLL_INTERVAL = int(os.getenv("SCHEDULER_POLL_INTERVAL", "30"))

# ── Auto-Tuning ──────────────────────────────────────────────────────────────

AUTO_TUNE_ENABLED = os.getenv("AUTO_TUNE_ENABLED", "").lower() in ("1", "true", "yes")
AUTO_TUNE_INTERVAL = int(os.getenv("AUTO_TUNE_INTERVAL", "300"))     # seconds between analysis
AUTO_TUNE_MIN_SAVINGS_PCT = float(os.getenv("AUTO_TUNE_MIN_SAVINGS_PCT", "10"))

# ── Power Budget Enforcement ─────────────────────────────────────────────────

POWER_BUDGET_ENABLED = os.getenv("POWER_BUDGET_ENABLED", "").lower() in ("1", "true", "yes")
POWER_BUDGET_WATTS = int(os.getenv("POWER_BUDGET_WATTS", "0"))     # per-GPU cap, 0 = disabled

# ── Fleet Aggregation ────────────────────────────────────────────────────────

FLEET_AGGREGATOR_ENABLED = os.getenv("FLEET_AGGREGATOR_ENABLED", "").lower() in ("1", "true", "yes")
FLEET_AGGREGATOR_PORT = int(os.getenv("FLEET_AGGREGATOR_PORT", "9101"))
FLEET_AGGREGATOR_PEERS = os.getenv("FLEET_AGGREGATOR_PEERS", "")  # comma-separated URLs

# ── RAPL (CPU + RAM Energy) ──────────────────────────────────────────────────

RAPL_ENABLED = os.getenv("RAPL_ENABLED", "1").lower() not in ("0", "false", "no")

# Explicit opt-in for CPU-only monitoring when no GPU is present.
CPU_ONLY_MODE = os.getenv("CPU_ONLY_MODE", "").lower() in ("1", "true", "yes")

# Override the auto-detected CPU model name (useful for heterogeneous clusters).
RAPL_CPU_MODEL_OVERRIDE = os.getenv("RAPL_CPU_MODEL_OVERRIDE", "")

# ── Intel Gaudi ──────────────────────────────────────────────────────────────

# Enable/disable Gaudi collector (auto-detects pyhlml or hl-smi).
GAUDI_ENABLED = os.getenv("GAUDI_ENABLED", "1").lower() not in ("0", "false", "no")

# Custom path to hl-smi binary (if not in PATH).
HL_SMI_PATH = os.getenv("HL_SMI_PATH", "hl-smi")

# ── Intel Arc / Data Center GPU ──────────────────────────────────────────────

# Enable/disable Intel Arc collector (auto-detects xpu-smi or hwmon sysfs).
INTEL_ARC_ENABLED = os.getenv("INTEL_ARC_ENABLED", "1").lower() not in ("0", "false", "no")

# Custom path to xpu-smi binary (if not in PATH).
XPU_SMI_PATH = os.getenv("XPU_SMI_PATH", "xpu-smi")

# ── Apple Silicon ────────────────────────────────────────────────────────────

# Enable/disable powermetrics backend (requires sudo -n for NOPASSWD).
APPLE_POWERMETRICS_ENABLED = os.getenv("APPLE_POWERMETRICS_ENABLED", "1").lower() not in ("0", "false", "no")

# Sampling interval for powermetrics subprocess (milliseconds).
APPLE_POWERMETRICS_INTERVAL_MS = int(os.getenv("APPLE_POWERMETRICS_INTERVAL_MS", "1000"))

# Override GPU TDP estimate for unknown Apple chips (watts).
APPLE_CHIP_TDP_OVERRIDE = os.getenv("APPLE_CHIP_TDP_OVERRIDE", "")

# ── Cloud Cost Detection ─────────────────────────────────────────────────────

CLOUD_COST_ENABLED = os.getenv("CLOUD_COST_ENABLED", "1").lower() not in ("0", "false", "no")

# ── Cluster Identity ───────────────────────────────────────────────────────────

CLUSTER_TAG   = os.getenv("ALUMINATAI_CLUSTER_TAG", "")    # e.g. "aws-us-west-2"
LOCATION_HINT = os.getenv("ALUMINATAI_LOCATION_HINT", "")  # free-text, shown in UI
GRID_ZONE     = os.getenv("ALUMINATAI_GRID_ZONE", "")      # Electricity Maps zone, e.g. "US-CAL-CISO"

# How often the agent polls GET /api/v1/tag for user-registered job tags.
# Defaults to the same cadence as the scheduler poll.
TAG_POLL_INTERVAL = int(os.getenv("TAG_POLL_INTERVAL", str(SCHEDULER_POLL_INTERVAL)))

# ── Heartbeat ─────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "300"))  # 5 min

# ── Error Upload ─────────────────────────────────────────────────────────────

ERROR_UPLOAD_INTERVAL = int(os.getenv("ERROR_UPLOAD_INTERVAL", "300"))  # 5 min

# ── Command Polling (Advisor / Swarm) ────────────────────────────────────────

COMMAND_POLL_ENABLED = os.getenv("COMMAND_POLL_ENABLED", "").lower() in ("1", "true", "yes")
COMMAND_POLL_INTERVAL = int(os.getenv("COMMAND_POLL_INTERVAL", "60"))

# ── Swarm ────────────────────────────────────────────────────────────────────

SWARM_ENABLED = os.getenv("SWARM_ENABLED", "").lower() in ("1", "true", "yes")
SWARM_EVAL_INTERVAL = int(os.getenv("SWARM_EVAL_INTERVAL", "300"))  # 5 min between policy evals
SWARM_MAX_RECS = int(os.getenv("SWARM_MAX_RECS", "20"))

# ── TLS / Proxy ───────────────────────────────────────────────────────────────

HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")
CA_BUNDLE = os.getenv("ALUMINATAI_CA_BUNDLE", "")      # path to company CA PEM
CLIENT_CERT = os.getenv("ALUMINATAI_CLIENT_CERT", "")  # mTLS client cert path
CLIENT_KEY = os.getenv("ALUMINATAI_CLIENT_KEY", "")    # mTLS client key path

# ── Prometheus Metrics Server ─────────────────────────────────────────────────

METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))         # 0 = disabled
METRICS_BIND_HOST = os.getenv("METRICS_BIND_HOST", "")        # "" = 0.0.0.0 (all interfaces)
METRICS_BASIC_AUTH = os.getenv("METRICS_BASIC_AUTH", "")      # "user:pass" or "" (no auth)

# ── Run Modes ─────────────────────────────────────────────────────────────────

OFFLINE_MODE = os.getenv("OFFLINE_MODE", "").lower() in ("1", "true", "yes")

# Collect + attribute + Prometheus, but skip all HTTP uploads and WAL writes.
# Useful for debugging attribution and config without sending data.
DRY_RUN = os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")

# Disable cloud uploads entirely; run only local Prometheus metrics.
# Implies no WAL writes (unlike OFFLINE_MODE which writes to WAL).
PROMETHEUS_ONLY = os.getenv("PROMETHEUS_ONLY", "").lower() in ("1", "true", "yes")

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# "text" (human-readable, default) or "json" (newline-delimited JSON for ELK/Loki)
LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

# ── Memory Leak Detection ────────────────────────────────────────────────────

MEM_LEAK_DETECTION = os.getenv("MEM_LEAK_DETECTION", "1").lower() not in ("0", "false", "no")
MEM_LEAK_WINDOW = int(os.getenv("MEM_LEAK_WINDOW", "60"))

# ── Attribution ───────────────────────────────────────────────────────────────

ATTRIBUTION_CONFIG = os.getenv("ALUMINATAI_ATTRIBUTION_CONFIG", "")


def _parse_trusted_uids() -> set:
    """Parse ALUMINATAI_TRUSTED_UIDS=0,1000,1001 into a set of ints."""
    _log = logging.getLogger(__name__)
    result: set[int] = set()
    for part in os.getenv("ALUMINATAI_TRUSTED_UIDS", "").split(","):
        part = part.strip()
        if part:
            try:
                result.add(int(part))
            except ValueError:
                _log.warning("Ignoring invalid UID in ALUMINATAI_TRUSTED_UIDS: %r", part)
    return result


TRUSTED_UIDS: set[int] = _parse_trusted_uids()


# ── Config Validation ────────────────────────────────────────────────────────

def _clamp(name: str, value, lo, hi):
    """Clamp a numeric value to [lo, hi], warn if adjusted."""
    if value < lo or value > hi:
        clamped = max(lo, min(hi, value))
        logging.getLogger(__name__).warning(
            "Config %s=%r out of range [%s, %s] — clamped to %s",
            name, value, lo, hi, clamped,
        )
        return clamped
    return value


def _validate_config() -> None:
    """Validate all config constants and clamp to safe ranges."""
    global SAMPLE_INTERVAL, NVML_TIMEOUT, UPLOAD_BATCH_SIZE, UPLOAD_INTERVAL
    global UPLOAD_MAX_RETRIES, UPLOAD_MAX_RETRY_DELAY, UPLOAD_TIMEOUT
    global WAL_MAX_AGE_HOURS, WAL_MAX_MB
    global METRICS_PORT, SCHEDULER_POLL_INTERVAL, TAG_POLL_INTERVAL
    global HEARTBEAT_INTERVAL, PID_SMOOTH_WINDOW, PID_STABLE_THRESHOLD
    global IDLE_BASELINE_WINDOW, WARMUP_DISCARD_SECONDS
    global MEM_LEAK_WINDOW
    global FAST_SAMPLE_INTERVAL, FAST_SAMPLE_BUFFER_SIZE

    SAMPLE_INTERVAL       = _clamp("SAMPLE_INTERVAL",       SAMPLE_INTERVAL,       0.1, 300)
    NVML_TIMEOUT          = _clamp("NVML_TIMEOUT",          NVML_TIMEOUT,          0.5, 30.0)
    UPLOAD_BATCH_SIZE     = int(_clamp("UPLOAD_BATCH_SIZE",  UPLOAD_BATCH_SIZE,     1,   10000))
    UPLOAD_INTERVAL       = int(_clamp("UPLOAD_INTERVAL",    UPLOAD_INTERVAL,       1,   3600))
    UPLOAD_MAX_RETRIES    = int(_clamp("UPLOAD_MAX_RETRIES", UPLOAD_MAX_RETRIES,    0,   20))
    UPLOAD_MAX_RETRY_DELAY = int(_clamp("UPLOAD_MAX_RETRY_DELAY", UPLOAD_MAX_RETRY_DELAY, 1, 600))
    UPLOAD_TIMEOUT        = int(_clamp("UPLOAD_TIMEOUT",        UPLOAD_TIMEOUT,        5,   120))
    WAL_MAX_AGE_HOURS     = int(_clamp("WAL_MAX_AGE_HOURS",  WAL_MAX_AGE_HOURS,    1,   720))
    WAL_MAX_MB            = int(_clamp("WAL_MAX_MB",          WAL_MAX_MB,           1,   10240))
    METRICS_PORT          = int(_clamp("METRICS_PORT",        METRICS_PORT,         0,   65535))
    SCHEDULER_POLL_INTERVAL = int(_clamp("SCHEDULER_POLL_INTERVAL", SCHEDULER_POLL_INTERVAL, 5, 600))
    TAG_POLL_INTERVAL     = int(_clamp("TAG_POLL_INTERVAL",   TAG_POLL_INTERVAL,    5,   600))
    HEARTBEAT_INTERVAL    = int(_clamp("HEARTBEAT_INTERVAL",  HEARTBEAT_INTERVAL,   30,  3600))
    PID_SMOOTH_WINDOW     = _clamp("PID_SMOOTH_WINDOW",       PID_SMOOTH_WINDOW,    1.0, 300.0)
    PID_STABLE_THRESHOLD  = _clamp("PID_STABLE_THRESHOLD",    PID_STABLE_THRESHOLD, 0.0, 1.0)
    IDLE_BASELINE_WINDOW  = int(_clamp("IDLE_BASELINE_WINDOW", IDLE_BASELINE_WINDOW, 0, 300))
    WARMUP_DISCARD_SECONDS = int(_clamp("WARMUP_DISCARD_SECONDS", WARMUP_DISCARD_SECONDS, 0, 600))
    MEM_LEAK_WINDOW       = int(_clamp("MEM_LEAK_WINDOW",        MEM_LEAK_WINDOW,       10, 600))
    FAST_SAMPLE_INTERVAL  = _clamp("FAST_SAMPLE_INTERVAL",      FAST_SAMPLE_INTERVAL,  0.05, 2.0)
    FAST_SAMPLE_BUFFER_SIZE = int(_clamp("FAST_SAMPLE_BUFFER_SIZE", FAST_SAMPLE_BUFFER_SIZE, 10, 1000))

    if WARMUP_DISCARD_SECONDS > 0 and IDLE_BASELINE_WINDOW > 0:
        if WARMUP_DISCARD_SECONDS <= IDLE_BASELINE_WINDOW:
            logging.getLogger(__name__).warning(
                "WARMUP_DISCARD_SECONDS (%d) should be > IDLE_BASELINE_WINDOW (%d) "
                "— adjusting warmup to %d",
                WARMUP_DISCARD_SECONDS, IDLE_BASELINE_WINDOW,
                IDLE_BASELINE_WINDOW + 15,
            )
            WARMUP_DISCARD_SECONDS = IDLE_BASELINE_WINDOW + 15


_validate_config()

# ── Ensure directories exist ──────────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
WAL_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
