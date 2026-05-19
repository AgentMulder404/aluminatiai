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
#!/usr/bin/env python3
"""
AluminatAI GPU Agent v0.2.0 — unified production daemon.

Combines the signal-handling / CSV reliability of aluminati_agent.py with
the attribution engine, scheduler detection, and API uploader from main.py.

Usage:
    aluminatai-agent                        # reads env vars, runs forever
    aluminatai-agent --interval 2           # 0.5 Hz sampling
    aluminatai-agent --duration 3600        # run 1 h then exit 0
    aluminatai-agent --output /data/m.csv   # local CSV manifest too
    aluminatai-agent --help

Signal handling:
    SIGINT / SIGTERM → flush buffer → close CSV → signal_job_complete → exit 0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import shutil
import subprocess
import io
import json
import logging
import os
import signal
import socket
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── Logging helpers ────────────────────────────────────────────────────────────

# Standard LogRecord attributes that should not be treated as user "extra" fields.
_STANDARD_LOG_ATTRS: frozenset = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "process", "processName", "taskName", "message", "asctime",
})


class _JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single-line JSON object for ELK / Grafana Loki.

    Standard fields: ts, level, logger, msg.
    Extra fields passed via logger.xxx(..., extra={...}) are merged in at the
    top level, enabling structured events like:
      {"ts":"…","level":"WARNING","event":"upload_timeout","attempt":2, …}
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra={} fields the caller attached
        for key, val in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """
    Configure the root logger with the requested level and format.

    Args:
        level: Standard logging level name ("DEBUG", "INFO", "WARNING", …).
        fmt:   "text" (human-readable) or "json" (newline-delimited JSON).
    """
    from logging.handlers import RotatingFileHandler as _RFH

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    if fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    # Always log to stderr
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Optional file handler with rotation (10MB, 5 backups)
    try:
        from config import LOG_DIR
        if LOG_DIR and str(LOG_DIR) != ".":
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = _RFH(
                LOG_DIR / "agent.log",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
    except (ImportError, OSError) as exc:
        root.warning("Could not set up file logging: %s", exc)


# ── Config hash ───────────────────────────────────────────────────────────────


def _compute_config_hash() -> str:
    """
    Return an 8-char hex digest that changes when key agent config values drift.

    Useful for fleet-wide drift detection: if two agent nodes report the same
    version but different config_hash values in their heartbeats, their configs
    have diverged (e.g., one has a stale SAMPLE_INTERVAL).
    """
    try:
        from config import (
            API_ENDPOINT, SAMPLE_INTERVAL, UPLOAD_INTERVAL, UPLOAD_BATCH_SIZE,
            METRICS_PORT, WAL_MAX_MB, LOG_LEVEL, DRY_RUN, PROMETHEUS_ONLY, OFFLINE_MODE,
        )
    except ImportError:
        return "00000000"
    canonical = json.dumps({
        "api_endpoint":    API_ENDPOINT,
        "sample_interval": SAMPLE_INTERVAL,
        "upload_interval": UPLOAD_INTERVAL,
        "batch_size":      UPLOAD_BATCH_SIZE,
        "metrics_port":    METRICS_PORT,
        "wal_max_mb":      WAL_MAX_MB,
        "log_level":       LOG_LEVEL,
        "dry_run":         DRY_RUN,
        "prometheus_only": PROMETHEUS_ONLY,
        "offline_mode":    OFFLINE_MODE,
    }, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:8]


# Minimal early setup so import-time warnings are visible; overridden in main().
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("aluminatai-agent")

# ── Optional rich console ──────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False

# ── Local module imports (all optional for graceful degradation) ───────────────

try:
    from collector import GPUCollector, CSV_HEADER
    _COLLECTOR = True
except (ImportError, SyntaxError) as _e:
    _COLLECTOR = False
    log.warning("collector.py unavailable (%s) — NVML collection disabled", type(_e).__name__)

try:
    from amd_collector import AMDGPUCollector
    _AMD_COLLECTOR = True
except (ImportError, SyntaxError) as _e:
    _AMD_COLLECTOR = False

try:
    from gaudi_collector import GaudiCollector
    _GAUDI_COLLECTOR = True
except (ImportError, SyntaxError) as _e:
    _GAUDI_COLLECTOR = False

try:
    from intel_arc_collector import IntelArcCollector
    _INTEL_ARC_COLLECTOR = True
except (ImportError, SyntaxError) as _e:
    _INTEL_ARC_COLLECTOR = False

try:
    from apple_collector import AppleSiliconCollector
    _APPLE_COLLECTOR = True
except (ImportError, SyntaxError) as _e:
    _APPLE_COLLECTOR = False

try:
    from rapl_collector import RAPLCollector
    _RAPL_COLLECTOR = True
except (ImportError, SyntaxError) as _e:
    _RAPL_COLLECTOR = False

try:
    from error_tracker import ErrorTracker
    _ERROR_TRACKER = True
except ImportError:
    _ERROR_TRACKER = False

try:
    from recommendation_reporter import RecommendationReporter
    _REC_REPORTER = True
except ImportError:
    _REC_REPORTER = False

try:
    from command_receiver import CommandReceiver
    _CMD_RECEIVER = True
except ImportError:
    _CMD_RECEIVER = False

try:
    from swarm.policy_engine import SwarmPolicyEngine
    _SWARM_ENGINE = True
except ImportError:
    _SWARM_ENGINE = False

try:
    from uploader import MetricsUploader
    from config import (
        UPLOAD_ENABLED, UPLOAD_INTERVAL, API_KEY, API_ENDPOINT,
        SCHEDULER_POLL_INTERVAL, SAMPLE_INTERVAL,
        AGENT_VERSION, HEARTBEAT_INTERVAL,
        DRY_RUN, PROMETHEUS_ONLY, LOG_LEVEL, LOG_FORMAT,
        CLUSTER_TAG, LOCATION_HINT,
        IDLE_BASELINE_WINDOW, WARMUP_DISCARD_SECONDS,
        DCGM_ENABLED,
        PID_SMOOTH_WINDOW, PID_STABLE_THRESHOLD,
        GRID_ZONE, MULTI_AGENT_ENABLED,
        ERROR_UPLOAD_INTERVAL,
        COMMAND_POLL_ENABLED, COMMAND_POLL_INTERVAL,
        SWARM_ENABLED, SWARM_EVAL_INTERVAL, SWARM_MAX_RECS,
    )
    _UPLOADER = True
except ImportError:
    _UPLOADER = False
    UPLOAD_ENABLED = False
    API_KEY = ""
    API_ENDPOINT = "https://aluminatiai.com/api/metrics/ingest"
    UPLOAD_INTERVAL = 60
    SCHEDULER_POLL_INTERVAL = 30
    SAMPLE_INTERVAL = 5.0
    AGENT_VERSION = "0.2.2"
    HEARTBEAT_INTERVAL = 300
    DRY_RUN = False
    PROMETHEUS_ONLY = False
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "text"
    CLUSTER_TAG = ""
    LOCATION_HINT = ""
    IDLE_BASELINE_WINDOW = 30
    WARMUP_DISCARD_SECONDS = 45
    DCGM_ENABLED = True
    PID_SMOOTH_WINDOW = 30.0
    PID_STABLE_THRESHOLD = 0.60
    GRID_ZONE = ""
    MULTI_AGENT_ENABLED = False
    ERROR_UPLOAD_INTERVAL = 300
    COMMAND_POLL_ENABLED = False
    COMMAND_POLL_INTERVAL = 60
    SWARM_ENABLED = False
    SWARM_EVAL_INTERVAL = 300
    SWARM_MAX_RECS = 20

try:
    from baseline import IdleBaseline
    _BASELINE = True
except ImportError:
    _BASELINE = False

try:
    from dcgm_probe import DcgmProbe
    _DCGM = True
except ImportError:
    _DCGM = False

try:
    from pid_smoother import PidSmoother
    _PID_SMOOTHER = True
except ImportError:
    _PID_SMOOTHER = False

try:
    from schedulers import detect_scheduler
    _SCHEDULER = True
except ImportError:
    _SCHEDULER = False

try:
    from attribution import AttributionEngine
    from attribution.process_probe import ProcessProbe
    from attribution.pid_resolver import PidResolver
    _ATTRIBUTION = True
except ImportError:
    _ATTRIBUTION = False

try:
    from tag_client import TagClient
    _TAG_CLIENT = True
except ImportError:
    _TAG_CLIENT = False

try:
    from metrics_server import MetricsServer
    _METRICS_SERVER = True
except ImportError:
    _METRICS_SERVER = False

try:
    from integrations.otel_exporter import AluminatAIOtelExporter
    _OTEL = True
except ImportError:
    _OTEL = False

try:
    from machine_id import get_machine_id
    _MACHINE_ID = True
except ImportError:
    _MACHINE_ID = False
    def get_machine_id() -> str:  # type: ignore[misc]
        import uuid
        return str(uuid.uuid4())

# ── ManifestWriter — atomic-flush CSV output ──────────────────────────────────

CSV_MANIFEST_COLUMNS = [
    "timestamp", "job_id", "gpu_uuid", "gpu_index", "gpu_name",
    "power_w", "energy_delta_j", "util_pct", "temp_c",
    "mem_used_mb", "team_id", "model_tag", "gpu_fraction",
    "attribution_confidence",
]


class ManifestWriter:
    """Append-only CSV manifest with line-buffering + fsync on close."""

    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[io.TextIOWrapper] = None
        self._writer = None
        self._row_count = 0

    def open(self):
        self._file = open(self._path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_MANIFEST_COLUMNS)
        self._file.flush()
        log.info("Manifest CSV: %s", self._path)

    def write_row(self, row: list):
        if self._writer:
            self._writer.writerow(row)
            self._row_count += 1

    def flush(self):
        if self._file and not self._file.closed:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass

    def close(self):
        if self._file and not self._file.closed:
            self.flush()
            self._file.close()
            log.info("Manifest closed: %s (%d rows)", self._path, self._row_count)

    @property
    def row_count(self) -> int:
        return self._row_count


# ── Job completion signal ─────────────────────────────────────────────────────


def signal_job_complete(endpoint: str, api_key: str, job_uuid: str,
                        end_time: Optional[str] = None) -> bool:
    url = endpoint.rstrip("/") + "/api/metrics/jobs/complete"
    payload: dict = {"job_id": job_uuid}
    if end_time:
        payload["end_time"] = end_time
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            log.info("Job complete signalled: job=%s msg=%s", job_uuid, body.get("message", ""))
            return True
    except Exception as exc:
        log.warning("Job complete signal failed (non-fatal): %s", exc)
        return False


# ── Heartbeat sender ──────────────────────────────────────────────────────────


def send_heartbeat(
    endpoint: str,
    api_key: str,
    gpu_count: int,
    gpu_uuids: List[str],
    scheduler_name: str,
    uptime_sec: float = 0.0,
    config_hash: str = "",
    machine_id: str = "",
    cluster_tag: str = "",
    location_hint: str = "",
    gpu_names: Optional[List[str]] = None,
    error_stats: Optional[dict] = None,
    gpu_backend: str = "",
    agent_mode: str = "normal",
) -> None:
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    base = f"{parsed.scheme}://{parsed.netloc}"
    url = base + "/api/agent/heartbeat"
    payload = {
        "agent_version": AGENT_VERSION,
        "hostname": socket.gethostname(),
        "gpu_count": gpu_count,
        "gpu_uuids": gpu_uuids,
        "scheduler": scheduler_name,
        "uptime_sec": round(uptime_sec, 1),
        "config_hash": config_hash,
        "machine_id": machine_id,
        "cluster_tag": cluster_tag,
        "location_hint": location_hint,
        "gpu_names": list(dict.fromkeys(gpu_names)) if gpu_names else [],
        "os_info": f"{sys.platform} {os.uname().release}" if hasattr(os, "uname") else sys.platform,
        "python_version": sys.version.split()[0],
        "gpu_backend": gpu_backend,
        "agent_mode": agent_mode,
    }
    if error_stats:
        payload.update(error_stats)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as exc:
        log.debug("Heartbeat failed (non-fatal): %s", exc)


def _upload_errors(
    endpoint: str,
    api_key: str,
    machine_id: str,
    errors: list[dict],
) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    base = f"{parsed.scheme}://{parsed.netloc}"
    url = base + "/api/agent/errors"
    payload = {
        "machine_id": machine_id,
        "hostname": socket.gethostname(),
        "errors": errors,
    }
    data = json.dumps(payload, default=str).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception as exc:
        log.debug("Error upload failed (non-fatal): %s", exc)
        return False


# ── Unified Agent ─────────────────────────────────────────────────────────────


class Agent:
    """
    Production GPU energy monitoring daemon.

    Poll loop:
      1. Poll scheduler every SCHEDULER_POLL_INTERVAL
      2. Collect NVML metrics via GPUCollector
      3. AttributionEngine.resolve() per GPU handle
      4. Append fractional rows to uploader buffer + CSV manifest
      5. Flush uploader every UPLOAD_INTERVAL
      6. Send heartbeat every HEARTBEAT_INTERVAL
      7. On SIGTERM/SIGINT: flush → close CSV → signal_job_complete → exit 0
    """

    def __init__(
        self,
        interval: float = SAMPLE_INTERVAL,
        output_csv: Optional[str] = None,
        duration: Optional[float] = None,
        quiet: bool = False,
        job_uuid: Optional[str] = None,
        dry_run: bool = DRY_RUN,
        prometheus_only: bool = PROMETHEUS_ONLY,
    ):
        self.interval = interval
        self.output_csv = output_csv
        self.duration = duration
        self.quiet = quiet
        self.job_uuid = job_uuid or os.getenv("ALUMINATAI_JOB_UUID")
        self.dry_run = dry_run
        self.prometheus_only = prometheus_only

        self.running = False
        self.sample_count = 0
        self.total_energy: dict[int, float] = {}
        self._start_time = time.monotonic()
        self._config_hash = _compute_config_hash()

        # Error tracker
        self.error_tracker = ErrorTracker() if _ERROR_TRACKER else None
        self._last_error_upload = 0.0

        if self.dry_run:
            log.warning("DRY RUN — collecting and attributing, but no data will be uploaded or written to WAL")
        if self.prometheus_only:
            log.warning("PROMETHEUS ONLY — cloud uploads disabled; Prometheus metrics served locally")

        # Upload — disabled in prometheus_only mode (dry_run still creates uploader for logging)
        self.uploader: Optional[MetricsUploader] = None
        self.last_upload_time = 0.0
        if self.prometheus_only:
            pass  # no uploader; Prometheus is the only sink
        elif _UPLOADER and UPLOAD_ENABLED and API_KEY:
            self.uploader = MetricsUploader()
            log.info("API upload enabled → %s", API_ENDPOINT)
        elif not quiet:
            log.info("API upload disabled (no API key)")

        # Scheduler
        self.scheduler = None
        self.last_scheduler_poll = 0.0
        if _SCHEDULER:
            self.scheduler = detect_scheduler()
            log.info("Scheduler: %s", self.scheduler.name)
        elif not quiet:
            log.info("Scheduler integration unavailable")

        # Tag client (polls /api/v1/tag for REST-registered job tags)
        self.tag_client = None
        if _TAG_CLIENT and UPLOAD_ENABLED and API_KEY:
            from config import TAG_POLL_INTERVAL
            self.tag_client = TagClient(API_ENDPOINT, API_KEY, poll_interval=TAG_POLL_INTERVAL)
            self.tag_client.start()
            log.info("TagClient: polling %s every %ds", API_ENDPOINT, TAG_POLL_INTERVAL)

        # PID temporal smoother (filters transient spawn workers before attribution)
        self._smoother = None
        if _PID_SMOOTHER and PID_SMOOTH_WINDOW > 0:
            self._smoother = PidSmoother(
                window_s=PID_SMOOTH_WINDOW,
                stable_threshold=PID_STABLE_THRESHOLD,
            )
            log.info(
                "PidSmoother: window=%.0fs threshold=%.0f%%",
                PID_SMOOTH_WINDOW, PID_STABLE_THRESHOLD * 100,
            )

        # Attribution engine
        self.attribution_engine = None
        if _ATTRIBUTION and self.scheduler:
            probe = ProcessProbe()
            resolver = PidResolver(self.scheduler)
            self.attribution_engine = AttributionEngine(
                probe, resolver, self.scheduler,
                tag_client=self.tag_client,
                smoother=self._smoother,
            )
            log.info("Attribution: process-level GPU attribution enabled")

        # Prometheus metrics server
        self.metrics_server = None
        if _METRICS_SERVER:
            self.metrics_server = MetricsServer()
            self.metrics_server.start()

        # DCGM phase-decomposition probe
        self.dcgm_probe = None
        if _DCGM and DCGM_ENABLED:
            self.dcgm_probe = DcgmProbe()
            self.dcgm_probe.start()

        # OpenTelemetry exporter (auto-enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set)
        self.otel_exporter = None
        if _OTEL and os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            self.otel_exporter = AluminatAIOtelExporter()
            self.otel_exporter.start()
            log.info("OTel exporter wired → %s", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))

        # GPU memory leak detector
        self._mem_leak_detector = None
        try:
            from config import MEM_LEAK_DETECTION, MEM_LEAK_WINDOW
            if MEM_LEAK_DETECTION:
                from memory_leak_detector import MemoryLeakDetector
                self._mem_leak_detector = MemoryLeakDetector(window_size=MEM_LEAK_WINDOW)
                log.info("Memory leak detection enabled (window=%d samples)", MEM_LEAK_WINDOW)
        except ImportError:
            pass

        # Multi-agent fast collector (opt-in)
        self._fast_collector = None
        self._ring_buffers: list = []
        self._multi_agent_ready = False
        if MULTI_AGENT_ENABLED:
            try:
                from ring_buffer import GPURingBuffer  # noqa: PLC0415
                from fast_collector import FastCollector  # noqa: PLC0415
                from config import FAST_SAMPLE_INTERVAL, FAST_SAMPLE_BUFFER_SIZE
                self._multi_agent_ready = True
                log.info("Multi-agent mode enabled (fast interval=%.0fms, buffer=%d)",
                         FAST_SAMPLE_INTERVAL * 1000, FAST_SAMPLE_BUFFER_SIZE)
            except ImportError as exc:
                log.warning("Multi-agent mode requested but unavailable: %s", exc)

        # Recommendation reporter (uploads optimization recs to cloud)
        self.rec_reporter = None
        if _REC_REPORTER and UPLOAD_ENABLED and API_KEY and not self.prometheus_only:
            self.rec_reporter = RecommendationReporter(
                endpoint=API_ENDPOINT,
                api_key=API_KEY,
                machine_id=get_machine_id(),
            )
            log.info("RecommendationReporter: enabled")

        # Command receiver (polls cloud for pending commands)
        self.cmd_receiver = None
        self._last_cmd_poll: float = 0.0
        if (_CMD_RECEIVER and COMMAND_POLL_ENABLED
                and UPLOAD_ENABLED and API_KEY
                and not self.prometheus_only):
            self.cmd_receiver = CommandReceiver(
                endpoint=API_ENDPOINT,
                api_key=API_KEY,
                machine_id=get_machine_id(),
                dry_run=self.dry_run,
            )
            log.info("CommandReceiver: polling every %ds", COMMAND_POLL_INTERVAL)

        # Swarm policy engine (fleet-wide optimization — leader mode)
        self.swarm_engine = None
        self._last_swarm_eval: float = 0.0
        if (_SWARM_ENGINE and SWARM_ENABLED
                and self.rec_reporter
                and UPLOAD_ENABLED and API_KEY
                and not self.prometheus_only):
            self.swarm_engine = SwarmPolicyEngine(
                endpoint=API_ENDPOINT,
                api_key=API_KEY,
                machine_id=get_machine_id(),
                reporter=self.rec_reporter,
                cluster_tag=CLUSTER_TAG,
                max_recs_per_eval=SWARM_MAX_RECS,
                cooldown_s=SWARM_EVAL_INTERVAL,
            )
            log.info("SwarmPolicyEngine: leader mode, eval every %ds", SWARM_EVAL_INTERVAL)

        # Rich console
        self.console = Console() if (_RICH and not quiet) else None

        # Machine identity
        self.machine_id = get_machine_id()

        # Heartbeat state
        self.last_heartbeat = 0.0
        self._last_gpu_names: List[str] = []

        # Idle baselines: {gpu_index: idle_power_w} — populated in run() after NVML init
        self._baselines: dict[int, float] = {}

        # Signal handlers
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ── Signal handling ──────────────────────────────────────────────────

    def _on_signal(self, signum, _frame):
        sig_name = signal.Signals(signum).name
        log.info("Received %s — shutting down gracefully.", sig_name)
        self.running = False

    # ── Main run loop ────────────────────────────────────────────────────

    def run(self) -> int:
        collector = None
        gpu_backend = "unknown"

        # CPU_ONLY_MODE skips GPU detection and goes straight to RAPL
        cpu_only = os.getenv("CPU_ONLY_MODE", "").lower() in ("1", "true", "yes")

        if not cpu_only and _COLLECTOR:
            try:
                collector = GPUCollector(collect_clocks=False)
                gpu_backend = "NVIDIA (NVML)"
            except Exception as exc:
                log.warning("NVIDIA collector failed: %s — trying AMD", exc)

        if not cpu_only and collector is None and _AMD_COLLECTOR:
            try:
                collector = AMDGPUCollector(collect_clocks=False)
                gpu_backend = "AMD (amdsmi)" if getattr(collector, "_use_amdsmi", False) else "AMD (rocm-smi)"
            except Exception as exc:
                log.warning("AMD collector failed: %s", exc)

        if not cpu_only and collector is None and _GAUDI_COLLECTOR:
            try:
                collector = GaudiCollector(collect_clocks=False)
                gpu_backend = "Intel Gaudi (pyhlml)" if getattr(collector, "_use_pyhlml", False) else "Intel Gaudi (hl-smi)"
            except Exception as exc:
                log.warning("Gaudi collector failed: %s", exc)

        if not cpu_only and collector is None and _INTEL_ARC_COLLECTOR:
            try:
                collector = IntelArcCollector(collect_clocks=False)
                gpu_backend = f"Intel Arc ({collector.backend})"
            except Exception as exc:
                log.warning("Intel Arc collector failed: %s", exc)

        if not cpu_only and collector is None and _APPLE_COLLECTOR:
            try:
                collector = AppleSiliconCollector(collect_clocks=False)
                gpu_backend = f"Apple Silicon ({collector.backend})"
            except Exception as exc:
                log.warning("Apple Silicon collector failed: %s", exc)

        if collector is None and _RAPL_COLLECTOR:
            try:
                collector = RAPLCollector(collect_clocks=False)
                gpu_backend = "CPU (RAPL)"
            except Exception as exc:
                log.warning("RAPL collector failed: %s", exc)

        if collector is None:
            log.error(
                "No collector available — install nvidia-ml-py3 (NVIDIA), "
                "amdsmi/rocm-smi (AMD), xpu-smi (Intel Arc), "
                "run on macOS Apple Silicon, "
                "or run on Linux with RAPL sysfs access (CPU-only)"
            )
            return 3

        log.info("GPU backend: %s", gpu_backend)

        gpu_count = collector.get_gpu_count()
        gpu_uuids = [info["uuid"] for info in collector.get_gpu_info()]
        gpu_info  = collector.get_gpu_info()
        scheduler_name = self.scheduler.name if self.scheduler else "none"

        for i in range(gpu_count):
            self.total_energy[i] = 0.0

        # Start multi-agent fast collector if enabled
        if self._multi_agent_ready:
            from ring_buffer import GPURingBuffer
            from fast_collector import FastCollector
            from config import FAST_SAMPLE_INTERVAL, FAST_SAMPLE_BUFFER_SIZE
            self._ring_buffers = [
                GPURingBuffer(i, max_samples=FAST_SAMPLE_BUFFER_SIZE)
                for i in range(gpu_count)
            ]
            self._fast_collector = FastCollector(
                gpu_handles=collector.gpu_handles,
                ring_buffers=self._ring_buffers,
                sample_interval=FAST_SAMPLE_INTERVAL,
            )
            self._fast_collector.start()

        # Load or calibrate idle power baselines (runs before WAL replay so
        # the 30s calibration window doesn't delay the first upload flush).
        self._baselines = self._load_or_calibrate_baselines(
            collector.gpu_handles, gpu_uuids, gpu_info
        )

        # Replay WAL on startup
        if self.uploader:
            retried = self.uploader.retry_failed_uploads()
            if retried > 0:
                log.info("WAL replay: %d metrics re-uploaded", retried)
            # Push initial WAL stats to Prometheus
            if self.metrics_server:
                status = self.uploader.get_status()
                self.metrics_server.update_wal_stats(
                    wal_size_bytes=status["wal_bytes"],
                    wal_entries_pending=status["wal_entries_pending"],
                    replay_uploaded_delta=self.uploader._wal_replay_uploaded,
                    replay_failed_delta=self.uploader._wal_replay_failed,
                )

        # CSV manifest
        manifest: Optional[ManifestWriter] = None
        if self.output_csv:
            manifest = ManifestWriter(Path(self.output_csv))
            manifest.open()

        # Local TSDB (opt-in)
        _tsdb = None
        try:
            from config import DATA_DIR
            _tsdb_enabled = os.getenv("LOCAL_TSDB_ENABLED", "0").lower() in ("1", "true", "yes")
            if _tsdb_enabled:
                _tsdb_retention = int(os.getenv("LOCAL_TSDB_RETENTION_DAYS", "7"))
                _tsdb_path = os.getenv("LOCAL_TSDB_PATH", str(DATA_DIR / "metrics.db"))
                from storage.tsdb import LocalTSDB
                _tsdb = LocalTSDB(db_path=_tsdb_path, retention_days=_tsdb_retention)
                log.info("Local TSDB enabled: %s (retention=%dd)", _tsdb_path, _tsdb_retention)
        except (ImportError, Exception) as exc:
            log.debug("Local TSDB unavailable: %s", exc)

        # Auto-tuner (opt-in)
        _auto_tuner = None
        try:
            from config import AUTO_TUNE_ENABLED, AUTO_TUNE_INTERVAL, AUTO_TUNE_MIN_SAVINGS_PCT
            if AUTO_TUNE_ENABLED:
                from efficiency.auto_tuner import AutoTuner
                _auto_tuner = AutoTuner(
                    interval_s=AUTO_TUNE_INTERVAL,
                    min_savings_pct=AUTO_TUNE_MIN_SAVINGS_PCT,
                    dry_run=self.dry_run,
                )
                log.info("AutoTuner enabled (interval=%ds, min_savings=%.0f%%)",
                         AUTO_TUNE_INTERVAL, AUTO_TUNE_MIN_SAVINGS_PCT)
        except (ImportError, AttributeError):
            pass

        # Carbon tracking (if grid zone is configured)
        _carbon_client = None
        _last_carbon_fetch: float = 0.0
        _carbon_intensity: float = 0.0
        _carbon_renewable: float = 0.0
        _carbon_zone: str = ""
        if GRID_ZONE:
            try:
                from efficiency.carbon import ElectricityMapsClient
                _carbon_client = ElectricityMapsClient(zone=GRID_ZONE)
                _carbon_zone = GRID_ZONE
                log.info("Carbon tracking enabled for zone: %s", GRID_ZONE)
            except ImportError:
                log.debug("Carbon tracking unavailable — efficiency.carbon module not found")

        if not self.quiet:
            self._print_banner(gpu_count, scheduler_name, gpu_backend)

        # SIGHUP hot-reload: re-read mutable config settings without restart.
        # Only available on POSIX (Linux/macOS); silently skipped on Windows.
        def _handle_sighup(signum, frame):
            try:
                import importlib
                import config as _cfg_mod
                importlib.reload(_cfg_mod)
                log.info("Config reloaded via SIGHUP")
            except Exception as exc:
                log.warning("SIGHUP config reload failed: %s", exc)

        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _handle_sighup)

        self.running = True
        start_time = time.monotonic()
        self.last_upload_time = time.time()

        # Initial heartbeat (skipped in dry-run / prometheus-only modes)
        if self.uploader and API_KEY and not self.dry_run and not self.prometheus_only:
            gpu_info = collector.get_gpu_info()
            self._last_gpu_names = [g.get("name", "") for g in gpu_info if g.get("name")]
            send_heartbeat(
                API_ENDPOINT, API_KEY, gpu_count, gpu_uuids, scheduler_name,
                uptime_sec=0.0,
                config_hash=self._config_hash,
                machine_id=self.machine_id,
                cluster_tag=CLUSTER_TAG,
                location_hint=LOCATION_HINT,
                gpu_names=self._last_gpu_names,
                gpu_backend=gpu_backend,
                agent_mode=_mode,
            )
            self.last_heartbeat = time.time()

        # Determine run mode label for Prometheus agent_info
        _mode = "dry_run" if self.dry_run else ("prometheus_only" if self.prometheus_only else "normal")
        _hostname = socket.gethostname()

        try:
            while self.running:
                loop_start = time.monotonic()
                now = time.time()

                # Scheduler poll — runs in a thread with 10s timeout to prevent
                # a hung scheduler adapter from blocking the entire collection loop.
                _sched_interval = getattr(self, '_sched_backoff_interval', SCHEDULER_POLL_INTERVAL)
                if self.scheduler and (now - self.last_scheduler_poll >= _sched_interval):
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            future = pool.submit(self.scheduler.discover_jobs)
                            future.result(timeout=10)
                        self._sched_fail_count = 0
                        self._sched_backoff_interval = SCHEDULER_POLL_INTERVAL
                    except concurrent.futures.TimeoutError:
                        self._sched_fail_count = getattr(self, '_sched_fail_count', 0) + 1
                        if self._sched_fail_count >= 5:
                            log.warning("Scheduler poll failed %d times — cached job data may be stale",
                                        self._sched_fail_count)
                        else:
                            log.warning("Scheduler poll timed out after 10s — skipping this cycle")
                        if self.error_tracker:
                            self.error_tracker.record("scheduler", "Scheduler poll timed out after 10s")
                        if self._sched_fail_count >= 3:
                            self._sched_backoff_interval = min(
                                self._sched_backoff_interval * 2, 300
                            )
                    except Exception as exc:
                        self._sched_fail_count = getattr(self, '_sched_fail_count', 0) + 1
                        log.warning("Scheduler poll failed: %s", exc)
                        if self.error_tracker:
                            self.error_tracker.record("scheduler", str(exc), exc=exc)
                        if self._sched_fail_count >= 3:
                            self._sched_backoff_interval = min(
                                self._sched_backoff_interval * 2, 300
                            )
                    finally:
                        self.last_scheduler_poll = now

                # Collect
                try:
                    metrics = collector.collect()
                except Exception as exc:
                    log.warning("Collection error: %s", exc)
                    if self.error_tracker:
                        self.error_tracker.record("collection", str(exc), exc=exc)
                    if self.metrics_server and hasattr(collector, "gpu_uuids"):
                        self.metrics_server.mark_collection_failed(collector.gpu_uuids)
                    time.sleep(self.interval)
                    continue

                self.sample_count += 1

                # Multi-agent: overlay ring buffer statistical summaries
                if self._fast_collector and self._ring_buffers:
                    for m in metrics:
                        summary = self._ring_buffers[m.gpu_index].summarize(
                            window_seconds=self.interval,
                        )
                        if summary and summary.sample_count >= 3:
                            m.power_draw_w = summary.power_mean_w
                            m._power_p95_w = summary.power_p95_w
                            m._power_p99_w = summary.power_p99_w

                # PID smoother update — feed current NVML PID sets into the
                # sliding window *before* engine.resolve() consults stable_pids().
                if self._smoother:
                    _now_ts = time.monotonic()
                    for _m in metrics:
                        _raw_pids = frozenset(
                            p["pid"] for p in (_m.processes or [])
                            if isinstance(p, dict)
                        )
                        self._smoother.update(_m.gpu_index, _now_ts, _raw_pids)

                # DCGM phase decomposition — runs every tick regardless of warmup.
                # Decomposes total GPU power into tensor / fp16 / memory / idle
                # components and pushes them to Prometheus.
                if self.dcgm_probe and self.metrics_server:
                    for m in metrics:
                        activity = self.dcgm_probe.get_activity(m.gpu_index, fallback=m)
                        idle_w   = self._baselines.get(m.gpu_index, 0.0)
                        decomp   = self.dcgm_probe.decompose_power(
                            m.power_draw_w, activity, m.gpu_name, idle_w
                        )
                        self.metrics_server.update_dcgm(
                            m.gpu_uuid, str(m.gpu_index), decomp
                        )

                # Warmup gate: discard uploads + CSV writes for the first N seconds
                # to avoid skewing attribution with warm-up transients.
                # Prometheus is still updated (so dashboards see live data).
                _in_warmup = (
                    WARMUP_DISCARD_SECONDS > 0
                    and (time.monotonic() - start_time) < WARMUP_DISCARD_SECONDS
                )

                # Update gpu_names from latest metrics (for heartbeat)
                self._last_gpu_names = list(dict.fromkeys(
                    m.gpu_name for m in metrics if getattr(m, "gpu_name", None)
                ))

                # Attribution + upload buffering
                attributed_rows: list[dict] = []
                for m in metrics:
                    # Baseline-corrected power for attribution.
                    # Raw m.power_draw_w is kept unchanged for Prometheus gauges.
                    idle_w = self._baselines.get(m.gpu_index, 0.0)
                    effective_power_w = max(0.0, m.power_draw_w - idle_w)
                    # Scale energy delta by the same ratio to stay consistent.
                    if m.energy_delta_j is not None and m.power_draw_w > 0:
                        adj_energy_j: Optional[float] = m.energy_delta_j * (effective_power_w / m.power_draw_w)
                    else:
                        adj_energy_j = m.energy_delta_j

                    handle = collector.gpu_handles[m.gpu_index]
                    if self.attribution_engine:
                        attributions = self.attribution_engine.resolve(
                            handle=handle,
                            gpu_index=m.gpu_index,
                            total_power_w=effective_power_w,
                            energy_delta_j=adj_energy_j,
                        )
                        if attributions:
                            for attr in attributions:
                                d = m.to_dict()
                                d.update(
                                    team_id=attr.team_id,
                                    model_tag=attr.model_tag,
                                    job_id=attr.job_id,
                                    scheduler_source=attr.scheduler_source,
                                    power_draw_w=attr.power_w,
                                    energy_delta_j=attr.energy_delta_j,
                                    gpu_fraction=attr.gpu_fraction,
                                    attribution_confidence=attr.confidence,
                                    attribution_confidence_score=attr.confidence_score,
                                    attribution_uncertainty_pct=attr.uncertainty_pct,
                                    machine_id=self.machine_id,
                                    cluster_tag=CLUSTER_TAG,
                                )
                                attributed_rows.append(d)
                                if manifest and not _in_warmup:
                                    manifest.write_row([
                                        d.get("timestamp"), d.get("job_id"),
                                        d.get("gpu_uuid"), d.get("gpu_index"),
                                        d.get("gpu_name"), d.get("power_draw_w"),
                                        d.get("energy_delta_j"), d.get("utilization_gpu_pct"),
                                        d.get("temperature_c"), d.get("memory_used_mb"),
                                        attr.team_id, attr.model_tag,
                                        attr.gpu_fraction, attr.confidence,
                                    ])
                        else:
                            d = m.to_dict()
                            d["machine_id"] = self.machine_id
                            d["cluster_tag"] = CLUSTER_TAG
                            attributed_rows.append(d)
                            if manifest and not _in_warmup:
                                manifest.write_row([
                                    d.get("timestamp"), d.get("job_id"),
                                    d.get("gpu_uuid"), d.get("gpu_index"),
                                    d.get("gpu_name"), d.get("power_draw_w"),
                                    d.get("energy_delta_j"), d.get("utilization_gpu_pct"),
                                    d.get("temperature_c"), d.get("memory_used_mb"),
                                    None, None, None, None,
                                ])
                            if self.metrics_server:
                                self.metrics_server.record_attribution_unresolved()
                    else:
                        if self.scheduler:
                            job = self.scheduler.gpu_to_job(m.gpu_index)
                            if job:
                                m.job_id = job.job_id
                                m.team_id = job.team_id
                                m.model_tag = job.model_tag
                                m.scheduler_source = job.scheduler_source
                        d = m.to_dict()
                        d["machine_id"] = self.machine_id
                        d["cluster_tag"] = CLUSTER_TAG
                        attributed_rows.append(d)
                        if manifest and not _in_warmup:
                            manifest.write_row([
                                d.get("timestamp"), d.get("job_id"),
                                d.get("gpu_uuid"), d.get("gpu_index"),
                                d.get("gpu_name"), d.get("power_draw_w"),
                                d.get("energy_delta_j"), d.get("utilization_gpu_pct"),
                                d.get("temperature_c"), d.get("memory_used_mb"),
                                d.get("team_id"), d.get("model_tag"), None, None,
                            ])

                if not _in_warmup:
                    if self.uploader and attributed_rows:
                        self.uploader.add_metrics(attributed_rows)
                elif self.sample_count == 1:
                    log.info(
                        "Warmup window active — discarding first %ds of samples",
                        WARMUP_DISCARD_SECONDS,
                    )

                # Prometheus metrics update (GPU + attribution) — always, even during warmup
                if self.metrics_server:
                    self.metrics_server.update(metrics, attributed_rows)

                # OTel metrics export
                if self.otel_exporter:
                    self.otel_exporter.record(metrics, attributed_rows)
                    # Agent uptime and info — updated every collection cycle
                    self.metrics_server.update_agent_stats(
                        uptime_sec=time.monotonic() - self._start_time,
                        version=AGENT_VERSION,
                        hostname=_hostname,
                        mode=_mode,
                    )

                # Energy accumulation (skipped during warmup to avoid inflating totals)
                if not _in_warmup:
                    for m in metrics:
                        if m.energy_delta_j:
                            self.total_energy[m.gpu_index] += m.energy_delta_j / 3_600_000

                # GPU memory leak detection
                if self._mem_leak_detector and not _in_warmup:
                    for m in metrics:
                        if self._mem_leak_detector.update(m.gpu_index, m.memory_used_mb):
                            log.warning(
                                "Potential GPU memory leak on GPU %d: "
                                "memory_used_mb increasing for %d consecutive samples "
                                "(current: %.0f MB)",
                                m.gpu_index,
                                self._mem_leak_detector._window_size,
                                m.memory_used_mb,
                            )
                        if self.metrics_server:
                            score = self._mem_leak_detector.get_leak_score(m.gpu_index)
                            self.metrics_server.update_mem_leak_score(
                                m.gpu_uuid, str(m.gpu_index), score,
                            )

                # TSDB insert (if enabled)
                if _tsdb and not _in_warmup:
                    try:
                        _tsdb.insert_batch(metrics)
                    except Exception as exc:
                        log.debug("TSDB insert failed: %s", exc)

                # Power budget enforcement — cap GPUs exceeding budget for 3+ consecutive samples
                try:
                    from config import POWER_BUDGET_ENABLED, POWER_BUDGET_WATTS
                    if POWER_BUDGET_ENABLED and POWER_BUDGET_WATTS > 0 and not _in_warmup:
                        if not hasattr(self, '_power_budget_violations'):
                            self._power_budget_violations: dict[int, int] = {}
                        for m in metrics:
                            if m.power_draw_w > POWER_BUDGET_WATTS:
                                self._power_budget_violations[m.gpu_index] = \
                                    self._power_budget_violations.get(m.gpu_index, 0) + 1
                                if self._power_budget_violations[m.gpu_index] >= 3:
                                    if not self.dry_run:
                                        try:
                                            from efficiency.power_control import set_power_limit
                                            set_power_limit(m.gpu_index, POWER_BUDGET_WATTS)
                                            log.warning(
                                                "Power budget enforced: GPU %d capped to %dW (was %.0fW)",
                                                m.gpu_index, POWER_BUDGET_WATTS, m.power_draw_w,
                                            )
                                        except Exception as exc:
                                            log.warning("Power budget enforcement failed GPU %d: %s",
                                                        m.gpu_index, exc)
                                    else:
                                        log.info(
                                            "Power budget: GPU %d exceeds %dW (%.0fW, %d consecutive) — dry run",
                                            m.gpu_index, POWER_BUDGET_WATTS, m.power_draw_w,
                                            self._power_budget_violations[m.gpu_index],
                                        )
                            else:
                                self._power_budget_violations[m.gpu_index] = 0
                except (ImportError, AttributeError):
                    pass

                # Auto-tuner — periodic roofline analysis + optional power cap
                if _auto_tuner and not _in_warmup and _auto_tuner.should_run():
                    try:
                        tune_results = _auto_tuner.analyze_and_tune(metrics)
                        for tr in tune_results:
                            if tr.recommended_cap_w:
                                log.info(
                                    "AutoTune: GPU %d %s cap=%dW savings=%.0f%% applied=%s",
                                    tr.gpu_index, tr.gpu_name,
                                    int(tr.recommended_cap_w), tr.estimated_savings_pct, tr.applied,
                                )
                        if self.rec_reporter and tune_results:
                            try:
                                self.rec_reporter.report_from_auto_tuner(tune_results)
                            except Exception as _rec_exc:
                                log.debug("Recommendation upload failed: %s", _rec_exc)
                    except Exception as exc:
                        log.warning("AutoTuner error: %s", exc)
                        if self.error_tracker:
                            self.error_tracker.record("auto_tuner", str(exc), exc=exc)

                # Carbon tracking — fetch intensity every 5 minutes, update CO2 counter
                if _carbon_client and self.metrics_server and not _in_warmup:
                    _now_carbon = time.time()
                    if _now_carbon - _last_carbon_fetch >= 300:
                        try:
                            _ci = _carbon_client.get_current()
                            _carbon_intensity = _ci.carbon_intensity_gco2e
                            _carbon_renewable = _ci.renewable_pct
                            _last_carbon_fetch = _now_carbon

                            # Carbon schedule recommendation (every fetch cycle)
                            if self.rec_reporter and _carbon_intensity > 0:
                                try:
                                    from efficiency.carbon_scheduler import find_optimal_window
                                    schedule_rec = find_optimal_window(
                                        zone=_carbon_zone, duration_hours=4.0,
                                    )
                                    if schedule_rec:
                                        self.rec_reporter.report_from_carbon_scheduler(schedule_rec)
                                except ImportError:
                                    pass
                                except Exception as _cs_exc:
                                    log.debug("Carbon schedule rec failed: %s", _cs_exc)
                        except Exception as exc:
                            log.debug("Carbon fetch failed: %s", exc)
                    if _carbon_intensity > 0:
                        _cycle_energy_kwh = sum(
                            (m.energy_delta_j or 0) / 3_600_000 for m in metrics
                        )
                        self.metrics_server.update_carbon(
                            zone=_carbon_zone,
                            carbon_intensity_gco2e=_carbon_intensity,
                            renewable_pct=_carbon_renewable,
                            energy_delta_kwh=_cycle_energy_kwh,
                        )

                # Periodic upload flush
                if self.uploader and (time.time() - self.last_upload_time >= UPLOAD_INTERVAL):
                    n = self.uploader.flush()
                    self.last_upload_time = time.time()
                    if n and not self.quiet:
                        log.info("Uploaded %d metrics", n)
                    # Update WAL + upload stats in Prometheus after every flush
                    if self.metrics_server:
                        status = self.uploader.get_status()
                        self.metrics_server.update_upload_stats(
                            success_delta=0,   # counters cumulative — delta tracked by uploader
                            failure_delta=0,
                            buffer_size=status["buffer_size"],
                        )
                        self.metrics_server.update_wal_stats(
                            wal_size_bytes=status["wal_bytes"],
                            wal_entries_pending=status["wal_entries_pending"],
                        )

                # Periodic heartbeat (skipped in dry-run / prometheus-only modes)
                if (self.uploader and API_KEY
                        and not self.dry_run and not self.prometheus_only
                        and time.time() - self.last_heartbeat >= HEARTBEAT_INTERVAL):
                    _err_stats = self.error_tracker.get_stats() if self.error_tracker else None
                    send_heartbeat(
                        API_ENDPOINT, API_KEY, gpu_count, gpu_uuids, scheduler_name,
                        uptime_sec=time.monotonic() - self._start_time,
                        config_hash=self._config_hash,
                        machine_id=self.machine_id,
                        cluster_tag=CLUSTER_TAG,
                        location_hint=LOCATION_HINT,
                        gpu_names=self._last_gpu_names,
                        error_stats=_err_stats,
                        gpu_backend=gpu_backend,
                        agent_mode=_mode,
                    )
                    self.last_heartbeat = time.time()

                # Periodic error upload
                if (self.error_tracker and self.uploader and API_KEY
                        and not self.dry_run and not self.prometheus_only
                        and time.time() - self._last_error_upload >= ERROR_UPLOAD_INTERVAL):
                    unsent = self.error_tracker.get_unsent(self._last_error_upload)
                    if unsent:
                        _upload_errors(
                            API_ENDPOINT, API_KEY, self.machine_id,
                            [e.to_dict() for e in unsent],
                        )
                    self._last_error_upload = time.time()

                # Command polling (Advisor / Swarm) — adaptive interval
                _cmd_interval = (
                    self.cmd_receiver.poll_interval
                    if self.cmd_receiver else COMMAND_POLL_INTERVAL
                )
                if (self.cmd_receiver
                        and time.time() - self._last_cmd_poll >= _cmd_interval):
                    try:
                        n = self.cmd_receiver.poll_and_execute()
                        if n:
                            log.info("Executed %d remote commands", n)
                    except Exception as exc:
                        log.debug("Command poll error: %s", exc)
                        if self.error_tracker:
                            self.error_tracker.record("command_receiver", str(exc), exc=exc)
                    self._last_cmd_poll = time.time()

                # Swarm policy engine (fleet-wide optimization)
                if self.swarm_engine and self.swarm_engine.should_evaluate():
                    try:
                        n = self.swarm_engine.evaluate()
                        if n:
                            log.info("Swarm: dispatched %d fleet recommendations", n)
                    except Exception as exc:
                        log.debug("Swarm eval error: %s", exc)
                        if self.error_tracker:
                            self.error_tracker.record("swarm_engine", str(exc), exc=exc)

                # Console display
                if not self.quiet and self.sample_count % 1 == 0:
                    self._display(metrics)

                # Duration guard
                if self.duration and (time.monotonic() - start_time) >= self.duration:
                    log.info("Duration limit reached (%.0fs).", self.duration)
                    break

                # Sleep remainder
                elapsed = time.monotonic() - loop_start
                sleep_s = max(0.0, self.interval - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except Exception as exc:
            log.exception("Unhandled error in agent loop: %s", exc)
            if self.error_tracker:
                self.error_tracker.record("unhandled", str(exc), exc=exc)
            return 1

        finally:
            # Flush remaining metrics
            if self.uploader:
                remaining = len(self.uploader.buffer)
                if remaining > 0:
                    log.info("Flushing %d remaining metrics…", remaining)
                    self.uploader.flush()

            # Close CSV manifest (fsync)
            if manifest:
                manifest.close()

            # Stop fast collector (multi-agent mode)
            if self._fast_collector:
                self._fast_collector.stop()

            # Shutdown collector
            try:
                collector.shutdown()
            except Exception as exc:
                log.debug("Collector shutdown error: %s", exc)

            # Stop DCGM probe
            if self.dcgm_probe:
                self.dcgm_probe.shutdown()

            # Stop Prometheus server
            if self.metrics_server:
                self.metrics_server.stop()

            # Stop tag client polling
            if self.tag_client:
                self.tag_client.stop()

            # Shutdown OTel exporter (flushes remaining spans)
            if self.otel_exporter:
                self.otel_exporter.stop()

            # Signal job completion (skipped in dry-run / prometheus-only modes)
            if API_KEY and self.job_uuid and not self.dry_run and not self.prometheus_only:
                signal_job_complete(
                    endpoint=API_ENDPOINT,
                    api_key=API_KEY,
                    job_uuid=self.job_uuid,
                    end_time=datetime.now(timezone.utc).isoformat(),
                )

            if not self.quiet:
                self._print_summary(time.monotonic() - start_time)

        return 0

    # ── Idle baseline ────────────────────────────────────────────────────

    def _load_or_calibrate_baselines(
        self,
        handles: list,
        gpu_uuids: list[str],
        gpu_info: list[dict],
    ) -> dict[int, float]:
        """
        Load persisted idle baselines matched by GPU UUID, or run a fresh
        calibration if the baselines file is absent / stale and all GPUs
        are currently idle.

        Returns {gpu_index: baseline_w}. Always safe to call; returns {}
        if baselines are unavailable (calibration skipped or failed).
        """
        if not _BASELINE or IDLE_BASELINE_WINDOW == 0:
            return {}

        ib = IdleBaseline()

        # Try loading from cache first (avoids 30s startup delay on busy nodes)
        if not ib.is_stale():
            by_uuid = ib.load()
            if by_uuid:
                result: dict[int, float] = {}
                for info in gpu_info:
                    uuid = info["uuid"]
                    if uuid in by_uuid:
                        result[info["index"]] = by_uuid[uuid]
                if result:
                    log.info(
                        "Loaded idle baselines from cache: %s",
                        {k: f"{v:.1f}W" for k, v in result.items()},
                    )
                    return result

        # Cache miss or stale — attempt live calibration
        calibrated = ib.calibrate(handles, gpu_uuids, duration_s=IDLE_BASELINE_WINDOW)
        if calibrated:
            return calibrated

        log.info("No idle baseline available — baseline subtraction disabled")
        return {}

    # ── Display helpers ──────────────────────────────────────────────────

    def _print_banner(self, gpu_count: int, scheduler: str, gpu_backend: str = "NVIDIA"):
        log.info("=" * 60)
        log.info("  AluminatAI GPU Agent v%s", AGENT_VERSION)
        log.info("=" * 60)
        log.info("  GPUs        : %d (%s)", gpu_count, gpu_backend)
        log.info("  Interval    : %.2fs", self.interval)
        log.info("  Scheduler   : %s", scheduler)
        log.info("  Attribution : %s", "process-level" if self.attribution_engine else "scheduler-poll")
        if self._baselines:
            baseline_str = "  ".join(f"GPU{k}={v:.1f}W" for k, v in sorted(self._baselines.items()))
            log.info("  Baseline    : %s", baseline_str)
        else:
            log.info("  Baseline    : disabled")
        if WARMUP_DISCARD_SECONDS > 0:
            log.info("  Warmup      : %ds discarded", WARMUP_DISCARD_SECONDS)
        if self._smoother:
            log.info(
                "  PID smooth  : %.0fs window / %.0f%% threshold",
                PID_SMOOTH_WINDOW, PID_STABLE_THRESHOLD * 100,
            )
        else:
            log.info("  PID smooth  : disabled")
        if self.dcgm_probe:
            log.info("  DCGM        : %s", self.dcgm_probe.mode)
        else:
            log.info("  DCGM        : disabled")
        if self.dry_run:
            log.info("  Mode        : DRY RUN (no uploads, no WAL)")
        elif self.prometheus_only:
            log.info("  Mode        : PROMETHEUS ONLY (no cloud uploads)")
        else:
            log.info("  Upload      : %s", "enabled" if self.uploader else "disabled")
        if self.duration:
            log.info("  Duration    : %.0fs", self.duration)
        if self.output_csv:
            log.info("  Manifest    : %s", self.output_csv)
        log.info("=" * 60)

    def _display(self, metrics):
        if self.console and _RICH:
            table = Table(title=f"Sample #{self.sample_count}")
            table.add_column("GPU", style="cyan")
            table.add_column("Power", justify="right")
            table.add_column("Util", justify="right")
            table.add_column("Temp", justify="right")
            table.add_column("Energy Δ", justify="right")
            table.add_column("Total kWh", justify="right")
            for m in metrics:
                e_str = f"{m.energy_delta_j:.1f}J" if m.energy_delta_j else "N/A"
                table.add_row(
                    f"GPU {m.gpu_index}",
                    f"{m.power_draw_w:.1f}W",
                    f"{m.utilization_gpu_pct}%",
                    f"{m.temperature_c}°C",
                    e_str,
                    f"{self.total_energy.get(m.gpu_index, 0):.4f}",
                )
            self.console.clear()
            self.console.print(table)

    def _print_summary(self, runtime: float):
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        log.info("  Runtime  : %.1fs  Samples: %d", runtime, self.sample_count)
        total_all = 0.0
        for idx, kwh in sorted(self.total_energy.items()):
            log.info("  GPU %-2d   : %.6f kWh (%.1f J)", idx, kwh, kwh * 3_600_000)
            total_all += kwh
        log.info("  TOTAL    : %.6f kWh ($%.4f @ $0.12/kWh)", total_all, total_all * 0.12)
        log.info("=" * 60)


# ── Replay subcommand ─────────────────────────────────────────────────────────


_SERVICE_UNIT = """\
[Unit]
Description=AluminatAI GPU Energy Monitoring Agent
Documentation=https://aluminatiai.com/docs/agent
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
User=aluminatai
Group=aluminatai
EnvironmentFile=/etc/aluminatai/agent.env
Environment=DATA_DIR=/var/lib/aluminatai
Environment=LOG_DIR=/var/log/aluminatai
ExecStart={bin_path}
Restart=on-failure
RestartSec=10s
TimeoutStopSec=30s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/aluminatai /var/log/aluminatai
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM
CapabilityBoundingSet=
MemoryMax=256M
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
"""

_ENV_TEMPLATE = """\
# AluminatAI Agent Configuration
# Edit this file, then restart: sudo systemctl restart aluminatai-agent
# Full reference: https://aluminatiai.com/docs/agent#configuration

ALUMINATAI_API_KEY={api_key}
ALUMINATAI_API_ENDPOINT=https://aluminatiai.com/api/metrics/ingest
SAMPLE_INTERVAL=5.0
UPLOAD_INTERVAL=60
METRICS_PORT=9100
LOG_LEVEL=INFO
"""

_UNIT_PATH = Path("/etc/systemd/system/aluminatai-agent.service")
_ENV_PATH  = Path("/etc/aluminatai/agent.env")


def _cmd_service(args) -> int:
    """service install | uninstall | status."""
    action = args.service_action

    if action == "status":
        ret = subprocess.run(["systemctl", "status", "aluminatai-agent"]).returncode
        return 0 if ret == 0 else 1

    if action == "uninstall":
        if os.geteuid() != 0:
            print("error: 'service uninstall' must be run as root (try: sudo aluminatiai service uninstall)")
            return 1
        subprocess.run(["systemctl", "stop", "aluminatai-agent"], stderr=subprocess.DEVNULL)
        subprocess.run(["systemctl", "disable", "aluminatai-agent"], stderr=subprocess.DEVNULL)
        for path in [_UNIT_PATH]:
            if path.exists():
                path.unlink()
                print(f"Removed {path}")
        subprocess.run(["systemctl", "daemon-reload"])
        print("Service uninstalled.  Config and data directories were NOT removed.")
        print(f"  Config: {_ENV_PATH}  (remove manually if desired)")
        return 0

    # install
    if os.geteuid() != 0:
        print("error: 'service install' must be run as root (try: sudo aluminatiai service install)")
        return 1

    if not hasattr(args, "api_key") or not args.api_key:
        existing_key = ""
        if _ENV_PATH.exists():
            for line in _ENV_PATH.read_text().splitlines():
                if line.startswith("ALUMINATAI_API_KEY="):
                    existing_key = line.split("=", 1)[1].strip()
                    break
        if existing_key:
            print(f"Found existing API key in {_ENV_PATH}")
            api_key = existing_key
        else:
            print("Get your API key at: https://aluminatiai.com/dashboard/setup")
            try:
                api_key = input("Enter API Key: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 1
            if not api_key:
                print("error: API key cannot be empty")
                return 1

    else:
        api_key = args.api_key

    # Find the installed binary
    bin_path = sys.executable.replace("python", "aluminatiai").replace("python3", "aluminatiai")
    bin_path = shutil.which("aluminatiai") or bin_path

    # Create user if missing
    if subprocess.run(["id", "aluminatai"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        subprocess.run([
            "useradd", "--system", "--no-create-home",
            "--shell", "/usr/sbin/nologin",
            "--comment", "AluminatAI GPU agent", "aluminatai",
        ])
        print("Created system user 'aluminatai'")

    # Directories
    for d, mode in [
        (Path("/var/lib/aluminatai"), 0o700),
        (Path("/var/log/aluminatai"), 0o755),
        (Path("/etc/aluminatai"),     0o750),
    ]:
        d.mkdir(parents=True, exist_ok=True, mode=mode)

    # Env file (write only if not present or explicitly updating)
    if not _ENV_PATH.exists() or getattr(args, "update_env", False):
        _ENV_PATH.write_text(_ENV_TEMPLATE.format(api_key=api_key))
        _ENV_PATH.chmod(0o600)
        print(f"Wrote {_ENV_PATH}")
    else:
        print(f"Keeping existing {_ENV_PATH} (pass --update-env to overwrite)")

    # Unit file
    _UNIT_PATH.write_text(_SERVICE_UNIT.format(bin_path=bin_path))
    _UNIT_PATH.chmod(0o644)
    print(f"Wrote {_UNIT_PATH}")

    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "aluminatai-agent"])
    subprocess.run(["systemctl", "restart", "aluminatai-agent"])

    import time as _t
    _t.sleep(3)

    active = subprocess.run(["systemctl", "is-active", "--quiet", "aluminatai-agent"]).returncode == 0
    if active:
        print("\nAluminatAI Agent is running!")
        print("  Status:    sudo systemctl status aluminatai-agent")
        print("  Logs:      sudo journalctl -u aluminatai-agent -f")
        print("  Metrics:   curl -s localhost:9100/metrics | head -20")
        print("  Dashboard: https://aluminatiai.com/dashboard")
        return 0
    else:
        print("\nService failed to start.")
        print("  Logs: sudo journalctl -u aluminatai-agent -n 50")
        return 1


def _cmd_replay(args) -> int:
    """Export WAL contents to a CSV file, optionally clearing the WAL."""
    try:
        from uploader import _wal_read_valid, _wal_clear
    except ImportError:
        log.error("uploader.py not available — cannot replay WAL")
        return 1

    rows = _wal_read_valid()
    if not rows:
        print("WAL is empty.")
        return 0

    out = Path(args.output)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows → {out}")

    if args.clear:
        _wal_clear()
        print("WAL cleared.")

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aluminatai-agent",
        description="AluminatAI GPU Energy Agent v0.2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables and config file (lowest to highest priority):
  Config file  (--config path.json or ALUMINATAI_CONFIG env var)
  ALUMINATAI_API_KEY        API key (alum_...)
  ALUMINATAI_API_ENDPOINT   Ingest URL
  ALUMINATAI_JOB_UUID       DB job UUID for completion signal
  SAMPLE_INTERVAL           Sampling interval in seconds (default: 5.0)
  UPLOAD_INTERVAL           Upload flush interval in seconds (default: 60)
  LOG_LEVEL                 DEBUG / INFO / WARNING / ERROR (default: INFO)
  LOG_FORMAT                text | json  (default: text)
  DRY_RUN                   1 — collect/attribute but do not upload
  PROMETHEUS_ONLY           1 — disable cloud uploads; serve Prometheus only
  OFFLINE_MODE              1 — write WAL only, no HTTP uploads

Config file keys (JSON/YAML): sample_interval, upload_interval,
  metrics_port, wal_max_mb, log_level, log_format, dry_run,
  prometheus_only, offline_mode, … (see docs)

Examples:
  aluminatiai
  aluminatiai --interval 1 --duration 3600
  aluminatiai --dry-run --log-format json
  aluminatiai --prometheus-only --interval 2
  aluminatiai --config /etc/aluminatai.json
  aluminatiai replay --output /data/metrics.csv --clear
  aluminatiai service install
  aluminatiai service status
  aluminatiai service uninstall
        """,
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="JSON or YAML config file path (also: ALUMINATAI_CONFIG env var)")
    parser.add_argument("--interval", "-i", type=float, default=None,
                        help="Sampling interval in seconds (default: SAMPLE_INTERVAL or 5.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV manifest path")
    parser.add_argument("--duration", "-d", type=float, default=None,
                        help="Run for N seconds then exit 0 (default: infinite)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress console output")
    parser.add_argument("--job-uuid", type=str, default=None,
                        help="DB job UUID — signals completion on exit")
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN,
                        help="Collect and attribute but skip all uploads/WAL writes")
    parser.add_argument("--prometheus-only", action="store_true", default=PROMETHEUS_ONLY,
                        help="Disable cloud uploads; serve Prometheus metrics only")
    parser.add_argument("--log-format", choices=["text", "json"], default=LOG_FORMAT,
                        help="Log output format: 'text' (default) or 'json' for ELK/Loki")
    parser.add_argument("--log-level", default=None,
                        help="Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: LOG_LEVEL or INFO)")
    parser.add_argument("--integrations", default="",
                        help="Comma-separated ML integrations to auto-register: mlflow,wandb")
    parser.add_argument("--version", action="version", version=f"aluminatai-agent {AGENT_VERSION}")

    subparsers = parser.add_subparsers(dest="command")

    # replay subcommand: export WAL → CSV
    replay_parser = subparsers.add_parser(
        "replay",
        help="Export WAL contents to CSV (for offline/air-gapped clusters)",
    )
    replay_parser.add_argument(
        "--output", "-o", default="metrics.csv",
        help="Output CSV file path (default: metrics.csv)",
    )
    replay_parser.add_argument(
        "--clear", action="store_true",
        help="Clear the WAL after successful export",
    )

    # service subcommand: install / uninstall / status for systemd
    service_parser = subparsers.add_parser(
        "service",
        help="Manage the aluminatai-agent systemd service (requires root for install/uninstall)",
    )
    service_sub = service_parser.add_subparsers(dest="service_action")
    service_sub.required = True

    svc_install = service_sub.add_parser("install", help="Install and start the systemd service")
    svc_install.add_argument(
        "--api-key", dest="api_key", default=os.environ.get("ALUMINATAI_API_KEY", ""),
        help="API key (default: ALUMINATAI_API_KEY env var)",
    )
    svc_install.add_argument(
        "--update-env", action="store_true",
        help="Overwrite existing /etc/aluminatai/agent.env",
    )

    service_sub.add_parser("uninstall", help="Stop, disable, and remove the systemd service")
    service_sub.add_parser("status", help="Show systemd service status")

    args = parser.parse_args()

    # Apply --config if provided (already applied in cli.py for installed entry-point;
    # this handles the case where agent.py is run directly with python agent.py).
    if args.command not in ("replay", "service") and getattr(args, "config", None):
        if not os.environ.get("ALUMINATAI_CONFIG"):
            os.environ["ALUMINATAI_CONFIG"] = args.config

    if args.command == "replay":
        return _cmd_replay(args)

    if args.command == "service":
        return _cmd_service(args)

    # Re-configure logging now that we have the final level + format
    effective_level = args.log_level or LOG_LEVEL
    effective_fmt = args.log_format
    _setup_logging(level=effective_level, fmt=effective_fmt)

    if not UPLOAD_ENABLED and not args.dry_run and not args.prometheus_only:
        log.warning(
            "ALUMINATAI_API_KEY is not set — metrics will NOT be uploaded to the dashboard. "
            "Get your API key at https://aluminatiai.com/dashboard"
        )

    interval = args.interval if args.interval is not None else SAMPLE_INTERVAL
    if interval < 0.1:
        log.error("Interval must be >= 0.1s")
        return 2

    # Auto-register ML integrations
    _integrations = [s.strip() for s in (args.integrations or "").split(",") if s.strip()]
    for _integ in _integrations:
        if _integ == "mlflow":
            try:
                from integrations.mlflow_callback import AluminatAIMLflowCallback
                log.info("MLflow integration enabled")
            except ImportError:
                log.warning("MLflow integration requested but mlflow package not installed")
        elif _integ == "wandb":
            try:
                from integrations.wandb_callback import AluminatAIWandbCallback
                log.info("W&B integration enabled")
            except ImportError:
                log.warning("W&B integration requested but wandb package not installed")
        else:
            log.warning("Unknown integration: %s (available: mlflow, wandb)", _integ)

    agent = Agent(
        interval=interval,
        output_csv=args.output,
        duration=args.duration,
        quiet=args.quiet,
        job_uuid=args.job_uuid,
        dry_run=args.dry_run,
        prometheus_only=args.prometheus_only,
    )
    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
