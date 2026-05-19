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
Prometheus /metrics endpoint for the AluminatAI GPU agent.

Activated when METRICS_PORT != 0 (default 9100).
Optional dep: pip install aluminatiai[prometheus]

GPU metrics:
  aluminatai_gpu_power_watts{gpu_uuid, gpu_index}
  aluminatai_gpu_energy_joules_total{gpu_uuid}
  aluminatai_gpu_utilization_pct{gpu_uuid, gpu_index}
  aluminatai_gpu_temperature_c{gpu_uuid, gpu_index}

Phase decomposition (requires DCGM or NVML fallback):
  aluminatai_gpu_tensor_power_watts{gpu_uuid, gpu_index}
  aluminatai_gpu_fp16_power_watts{gpu_uuid, gpu_index}
  aluminatai_gpu_memory_power_watts{gpu_uuid, gpu_index}
  aluminatai_gpu_idle_power_watts{gpu_uuid, gpu_index}

Upload / WAL health:
  aluminatai_upload_success_total
  aluminatai_upload_failure_total
  aluminatai_buffer_size
  aluminatai_wal_size_bytes
  aluminatai_wal_entries_pending
  aluminatai_wal_replay_uploaded_total
  aluminatai_wal_replay_failed_total

Attribution:
  aluminatai_attribution_confidence{gpu_index, job_id, method}
  aluminatai_attribution_uncertainty_pct{gpu_index, job_id, method}
  aluminatai_attribution_unresolved_total

Agent health:
  aluminatai_agent_uptime_seconds
  aluminatai_agent_info{version, hostname, mode}
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Any, List
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server

logger = logging.getLogger(__name__)

try:
    import prometheus_client as prom
    from prometheus_client import Gauge, Counter, REGISTRY, make_wsgi_app
    _PROM = True
except ImportError:
    _PROM = False


class _QuietHandler(WSGIRequestHandler):
    """Suppress per-request stdout logs from wsgiref."""

    def log_message(self, *a, **kw):
        pass


def _basic_auth_middleware(app, credentials: str):
    """WSGI wrapper enforcing HTTP Basic Auth."""
    expected = b"Basic " + base64.b64encode(credentials.encode())

    def _inner(environ, start_response):
        auth = environ.get("HTTP_AUTHORIZATION", "").encode()
        if auth != expected:
            start_response("401 Unauthorized", [
                ("WWW-Authenticate", 'Basic realm="aluminatai"'),
                ("Content-Type", "text/plain"),
            ])
            return [b"Unauthorized\n"]
        return app(environ, start_response)

    return _inner


class MetricsServer:
    """Background thread serving Prometheus metrics."""

    def __init__(self):
        self._port: int = 0
        self._bind_host: str = ""
        self._basic_auth: str = ""
        self._started = False
        self._srv = None
        self._start_time = time.monotonic()
        self._last_collection_ts: float = 0.0
        self._last_upload_ok: bool = True
        self._gpu_count: int = 0

        try:
            from config import METRICS_PORT, METRICS_BIND_HOST, METRICS_BASIC_AUTH
            self._port = METRICS_PORT
            self._bind_host = METRICS_BIND_HOST
            self._basic_auth = METRICS_BASIC_AUTH
        except ImportError:
            self._port = 9100

        if not _PROM:
            logger.info("prometheus-client not installed — metrics server disabled")
            return

        if self._port == 0:
            logger.info("METRICS_PORT=0 — metrics server disabled")
            return

        # Define metrics
        labels = ["gpu_uuid", "gpu_index"]

        self._power = Gauge(
            "aluminatai_gpu_power_watts",
            "GPU power draw in watts",
            labels,
        )
        self._energy = Counter(
            "aluminatai_gpu_energy_joules_total",
            "Cumulative GPU energy in joules",
            ["gpu_uuid"],
        )
        self._util = Gauge(
            "aluminatai_gpu_utilization_pct",
            "GPU compute utilization percent",
            labels,
        )
        self._temp = Gauge(
            "aluminatai_gpu_temperature_c",
            "GPU temperature in Celsius",
            labels,
        )
        self._upload_success = Counter(
            "aluminatai_upload_success_total",
            "Total metrics successfully uploaded",
        )
        self._upload_failure = Counter(
            "aluminatai_upload_failure_total",
            "Total metric batches that failed upload",
        )
        self._buffer_size = Gauge(
            "aluminatai_buffer_size",
            "Current in-memory upload buffer size",
        )
        self._wal_size_bytes = Gauge(
            "aluminatai_wal_size_bytes",
            "WAL file size in bytes",
        )
        self._wal_entries_pending = Gauge(
            "aluminatai_wal_entries_pending",
            "Approximate number of metric rows waiting in the WAL",
        )
        self._wal_replay_uploaded = Counter(
            "aluminatai_wal_replay_uploaded_total",
            "Total WAL rows successfully re-uploaded during replay",
        )
        self._wal_replay_failed = Counter(
            "aluminatai_wal_replay_failed_total",
            "Total WAL rows that failed replay and remain pending",
        )
        self._confidence = Gauge(
            "aluminatai_attribution_confidence",
            "Attribution confidence score (0.0–1.0), labelled by resolution method",
            ["gpu_index", "job_id", "method"],
        )
        self._uncertainty = Gauge(
            "aluminatai_attribution_uncertainty_pct",
            "Estimated ± power attribution uncertainty as a percentage of reported power_w; "
            "reflects how much the true attribution could deviate based on resolution method",
            ["gpu_index", "job_id", "method"],
        )
        self._attribution_unresolved = Counter(
            "aluminatai_attribution_unresolved_total",
            "Collection cycles where the attribution engine returned no result for a GPU",
        )

        # Phase decomposition gauges (populated by DcgmProbe)
        self._tensor_power = Gauge(
            "aluminatai_gpu_tensor_power_watts",
            "Estimated tensor-core power draw in watts (DCGM mode); "
            "0 when DCGM unavailable",
            labels,
        )
        self._fp16_power = Gauge(
            "aluminatai_gpu_fp16_power_watts",
            "Estimated FP16 pipeline power draw in watts (DCGM mode); "
            "0 when DCGM unavailable",
            labels,
        )
        self._memory_power = Gauge(
            "aluminatai_gpu_memory_power_watts",
            "Estimated memory-subsystem power draw in watts",
            labels,
        )
        self._idle_power = Gauge(
            "aluminatai_gpu_idle_power_watts",
            "Estimated idle / baseline power draw in watts",
            labels,
        )

        # Carbon tracking gauges
        self._carbon_intensity = Gauge(
            "aluminatai_carbon_intensity_gco2e",
            "Current grid carbon intensity in gCO2e/kWh",
            ["zone"],
        )
        self._carbon_renewable_pct = Gauge(
            "aluminatai_carbon_renewable_pct",
            "Percentage of grid power from renewable sources",
            ["zone"],
        )
        self._co2_grams = Counter(
            "aluminatai_co2_grams_total",
            "Cumulative CO2 emissions in grams",
        )

        self._mem_leak_score = Gauge(
            "aluminatai_gpu_memory_leak_score",
            "Memory leak probability (0.0-1.0) based on monotonic increase detection",
            labels,
        )

        self._agent_uptime = Gauge(
            "aluminatai_agent_uptime_seconds",
            "Seconds since the agent process started",
        )
        # Info-pattern gauge: always 1.0, metadata in labels
        self._agent_info = Gauge(
            "aluminatai_agent_info",
            "Agent metadata (version, hostname, run mode); value is always 1",
            ["version", "hostname", "mode"],
        )

    def _health_response(self) -> tuple[str, bytes]:
        """Build JSON health check response."""
        uptime = time.monotonic() - self._start_time
        since_collection = time.monotonic() - self._last_collection_ts if self._last_collection_ts else None

        # Determine status: healthy / degraded / unhealthy
        if since_collection is None:
            status = "unhealthy"  # never collected
        elif since_collection > 120:
            status = "unhealthy"  # no data for 2+ minutes
        elif since_collection > 30 or not self._last_upload_ok:
            status = "degraded"
        else:
            status = "healthy"

        body = json.dumps({
            "status": status,
            "uptime_seconds": round(uptime, 1),
            "last_collection_ago_seconds": round(since_collection, 1) if since_collection else None,
            "last_upload_ok": self._last_upload_ok,
            "gpu_count": self._gpu_count,
        }).encode()
        return status, body

    def _health_middleware(self, app):
        """WSGI middleware that intercepts GET /health and GET /healthz."""
        def _inner(environ, start_response):
            path = environ.get("PATH_INFO", "")
            if path == "/health":
                status, body = self._health_response()
                http_status = "200 OK" if status != "unhealthy" else "503 Service Unavailable"
                start_response(http_status, [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(body))),
                ])
                return [body]
            if path == "/healthz":
                status, _ = self._health_response()
                if status == "unhealthy":
                    body = b"not ok\n"
                    start_response("503 Service Unavailable", [
                        ("Content-Type", "text/plain"),
                        ("Content-Length", str(len(body))),
                    ])
                    return [body]
                body = b"ok\n"
                start_response("200 OK", [
                    ("Content-Type", "text/plain"),
                    ("Content-Length", str(len(body))),
                ])
                return [body]
            return app(environ, start_response)
        return _inner

    def start(self) -> None:
        if not _PROM or self._port == 0 or self._started:
            return
        app = make_wsgi_app()
        app = self._health_middleware(app)
        if self._basic_auth:
            app = _basic_auth_middleware(app, self._basic_auth)
            logger.warning(
                "Prometheus Basic Auth active — use a TLS proxy in production"
            )
        bind = self._bind_host or "127.0.0.1"
        for offset in range(5):
            port = self._port + offset
            try:
                srv = make_server(bind, port, app, WSGIServer, _QuietHandler)
                self._srv = srv
                self._port = port
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                self._started = True
                if offset > 0:
                    logger.info("Port %d in use — bound to %d instead", self._port - offset, port)
                logger.info("Prometheus metrics server on %s:%d/metrics", bind, port)
                logger.info("Health endpoint on %s:%d/health", bind, port)
                return
            except OSError as exc:
                if offset < 4:
                    logger.debug("Port %d unavailable: %s — trying %d", port, exc, port + 1)
                else:
                    logger.error(
                        "Could not start metrics server on ports %d–%d: %s",
                        self._port - 4, port, exc,
                    )

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.shutdown()

    def mark_collection_failed(self, gpu_uuids: List[str]) -> None:
        """Reset GPU gauges to NaN so Prometheus sees absent data, not stale values."""
        if not _PROM or not self._started:
            return
        for i, uuid in enumerate(gpu_uuids):
            idx = str(i)
            self._power.labels(gpu_uuid=uuid, gpu_index=idx).set(float("nan"))
            self._util.labels(gpu_uuid=uuid, gpu_index=idx).set(float("nan"))
            self._temp.labels(gpu_uuid=uuid, gpu_index=idx).set(float("nan"))

    def update(self, metrics: List[Any], attributed_rows: List[dict]) -> None:
        """Called from the main loop after each collection cycle."""
        self._last_collection_ts = time.monotonic()
        self._gpu_count = len(metrics)
        if not _PROM or not self._started:
            return

        for m in metrics:
            uuid = m.gpu_uuid
            idx = str(m.gpu_index)
            self._power.labels(gpu_uuid=uuid, gpu_index=idx).set(m.power_draw_w)
            self._util.labels(gpu_uuid=uuid, gpu_index=idx).set(m.utilization_gpu_pct)
            self._temp.labels(gpu_uuid=uuid, gpu_index=idx).set(m.temperature_c)
            if m.energy_delta_j:
                self._energy.labels(gpu_uuid=uuid).inc(m.energy_delta_j)

        # Fallback score map for agents that don't yet send attribution_confidence_score
        _CONF_SCORES = {
            "tagged":          1.00,
            "api_tag":         0.95,
            "scheduler":       0.90,
            "scheduler_poll":  0.75,
            "rules":           0.60,
            "heuristic":       0.40,
            "memory_split":    0.20,
            "idle":            0.30,
        }
        for row in attributed_rows:
            conf = row.get("attribution_confidence", "unknown")
            gpu_idx = str(row.get("gpu_index", ""))
            job_id = str(row.get("job_id", ""))
            conf_score = row.get("attribution_confidence_score")
            if conf_score is None:
                conf_score = _CONF_SCORES.get(conf, 0.0)
            self._confidence.labels(gpu_index=gpu_idx, job_id=job_id, method=conf).set(conf_score)
            uncertainty = row.get("attribution_uncertainty_pct")
            if uncertainty is not None:
                self._uncertainty.labels(gpu_index=gpu_idx, job_id=job_id, method=conf).set(uncertainty)

    def update_upload_stats(self, success_delta: int, failure_delta: int, buffer_size: int) -> None:
        self._last_upload_ok = failure_delta == 0
        if not _PROM or not self._started:
            return
        if success_delta > 0:
            self._upload_success.inc(success_delta)
        if failure_delta > 0:
            self._upload_failure.inc(failure_delta)
        self._buffer_size.set(buffer_size)

    def update_wal_stats(
        self,
        wal_size_bytes: int,
        wal_entries_pending: int,
        replay_uploaded_delta: int = 0,
        replay_failed_delta: int = 0,
    ) -> None:
        """Update WAL health metrics. Called after each flush/replay cycle."""
        if not _PROM or not self._started:
            return
        self._wal_size_bytes.set(wal_size_bytes)
        self._wal_entries_pending.set(wal_entries_pending)
        if replay_uploaded_delta > 0:
            self._wal_replay_uploaded.inc(replay_uploaded_delta)
        if replay_failed_delta > 0:
            self._wal_replay_failed.inc(replay_failed_delta)

    def update_agent_stats(
        self,
        uptime_sec: float,
        version: str = "",
        hostname: str = "",
        mode: str = "normal",
    ) -> None:
        """Update agent uptime and info label. Called once per collection cycle."""
        if not _PROM or not self._started:
            return
        self._agent_uptime.set(uptime_sec)
        if version and hostname:
            self._agent_info.labels(version=version, hostname=hostname, mode=mode).set(1.0)

    def record_attribution_unresolved(self, count: int = 1) -> None:
        """Increment the unresolved attribution counter."""
        if not _PROM or not self._started or count <= 0:
            return
        self._attribution_unresolved.inc(count)

    def update_mem_leak_score(
        self, gpu_uuid: str, gpu_index: str, score: float,
    ) -> None:
        """Update memory leak probability gauge for a GPU."""
        if not _PROM or not self._started:
            return
        self._mem_leak_score.labels(gpu_uuid=gpu_uuid, gpu_index=gpu_index).set(score)

    def update_carbon(
        self,
        zone: str,
        carbon_intensity_gco2e: float,
        renewable_pct: float,
        energy_delta_kwh: float,
    ) -> None:
        """Update carbon tracking metrics."""
        if not _PROM or not self._started:
            return
        self._carbon_intensity.labels(zone=zone).set(carbon_intensity_gco2e)
        self._carbon_renewable_pct.labels(zone=zone).set(renewable_pct)
        if energy_delta_kwh > 0:
            co2_grams = energy_delta_kwh * carbon_intensity_gco2e
            self._co2_grams.inc(co2_grams)

    def update_dcgm(
        self,
        gpu_uuid: str,
        gpu_index: str,
        decomp: dict,
    ) -> None:
        """
        Update phase-decomposition power gauges from DcgmProbe.decompose_power().

        decomp keys: tensor_power_w, fp32_power_w, fp16_power_w,
                     memory_power_w, idle_power_w
        """
        if not _PROM or not self._started:
            return
        lbl = {"gpu_uuid": gpu_uuid, "gpu_index": gpu_index}
        self._tensor_power.labels(**lbl).set(decomp.get("tensor_power_w", 0.0))
        self._fp16_power.labels(**lbl).set(decomp.get("fp16_power_w", 0.0))
        self._memory_power.labels(**lbl).set(decomp.get("memory_power_w", 0.0))
        self._idle_power.labels(**lbl).set(decomp.get("idle_power_w", 0.0))
