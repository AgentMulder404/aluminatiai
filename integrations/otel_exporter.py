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
AluminatAI OpenTelemetry exporter.

Routes GPU telemetry into existing Datadog / Grafana / Jaeger stacks via OTLP.
Activated by standard OTEL_EXPORTER_OTLP_ENDPOINT env var.

Optional dep: pip install aluminatai-agent[otel]

Usage — attach to an existing Agent instance:
    from agent.agent import Agent
    from agent.integrations.otel_exporter import AluminatAIOtelExporter

    exporter = AluminatAIOtelExporter()
    exporter.start()
    # ... agent runs and calls exporter.record(metrics) each tick ...
    exporter.stop()

Or let agent.py auto-detect it (future integration point).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    _OTEL = True
except ImportError:
    _OTEL = False


class AluminatAIOtelExporter:
    """
    Emits GPU telemetry as OpenTelemetry gauges via OTLP.

    Instruments:
      aluminatai.gpu.power_watts         (gauge)
      aluminatai.gpu.energy_joules       (counter)
      aluminatai.gpu.utilization_pct     (gauge)
      aluminatai.gpu.temperature_c       (gauge)
      aluminatai.gpu.attribution_fraction (gauge)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        export_interval_ms: int = 10_000,
    ):
        self._endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        self._export_interval_ms = export_interval_ms
        self._provider: Optional[Any] = None
        self._meter: Optional[Any] = None
        self._instruments: dict[str, Any] = {}

    def start(self) -> None:
        if not _OTEL:
            logger.info("opentelemetry packages not installed — OTel export disabled")
            return
        if not self._endpoint:
            logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — OTel export disabled")
            return

        try:
            exporter = OTLPMetricExporter(endpoint=self._endpoint, insecure=True)
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self._export_interval_ms,
            )
            self._provider = MeterProvider(metric_readers=[reader])
            otel_metrics.set_meter_provider(self._provider)
            self._meter = otel_metrics.get_meter("aluminatai.agent")

            self._instruments["power"] = self._meter.create_gauge(
                "aluminatai.gpu.power_watts",
                description="GPU power draw in watts",
                unit="W",
            )
            self._instruments["util"] = self._meter.create_gauge(
                "aluminatai.gpu.utilization_pct",
                description="GPU compute utilization",
                unit="%",
            )
            self._instruments["temp"] = self._meter.create_gauge(
                "aluminatai.gpu.temperature_c",
                description="GPU temperature",
                unit="Cel",
            )
            self._instruments["fraction"] = self._meter.create_gauge(
                "aluminatai.gpu.attribution_fraction",
                description="Fractional GPU attribution per job",
            )
            self._instruments["energy"] = self._meter.create_counter(
                "aluminatai.gpu.energy_joules",
                description="Cumulative GPU energy",
                unit="J",
            )

            logger.info("OTel exporter started → %s", self._endpoint)
        except Exception as exc:
            logger.warning("OTel exporter failed to start: %s", exc)

    def record(self, metrics: List[Any], attributed_rows: List[dict]) -> None:
        """Call from the agent main loop after each collection cycle."""
        if not _OTEL or not self._meter:
            return

        for m in metrics:
            attrs = {"gpu_uuid": m.gpu_uuid, "gpu_index": str(m.gpu_index)}
            self._instruments["power"].set(m.power_draw_w, attrs)
            self._instruments["util"].set(m.utilization_gpu_pct, attrs)
            self._instruments["temp"].set(m.temperature_c, attrs)
            if m.energy_delta_j:
                self._instruments["energy"].add(m.energy_delta_j, attrs)

        for row in attributed_rows:
            attrs = {
                "gpu_uuid": row.get("gpu_uuid", "unknown"),
                "team_id": row.get("team_id", "unknown"),
                "job_id": row.get("job_id", "unknown"),
                "confidence": row.get("attribution_confidence", "unknown"),
            }
            frac = row.get("gpu_fraction")
            if frac is not None:
                self._instruments["fraction"].set(float(frac), attrs)

    def stop(self) -> None:
        if self._provider:
            try:
                self._provider.shutdown()
            except Exception as exc:
                logger.debug("OTel provider shutdown error: %s", exc)
