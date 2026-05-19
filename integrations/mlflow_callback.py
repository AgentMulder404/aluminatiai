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
AluminatAI MLflow integration.

Auto-tags MLflow runs with energy consumption and cost.

Usage:
    import mlflow
    from agent.integrations.mlflow_callback import AluminatAIMLflowCallback

    callback = AluminatAIMLflowCallback()

    with mlflow.start_run() as run:
        callback.on_run_start(run)
        # ... training code ...
        callback.on_run_end(run)

Or use as a context manager:
    with mlflow.start_run() as run, callback.track(run):
        # ... training code ...

Env vars:
    ALUMINATAI_API_KEY        Required for cost lookup
    ALUMINATAI_API_ENDPOINT   API base URL (default: production)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False


class AluminatAIMLflowCallback:
    """
    Logs GPU energy and cost to MLflow runs.

    On run start:
      - Sets ALUMINATAI_MODEL to the run name
      - Sets ALUMINATAI_TEAM to the experiment name

    On run end:
      - Fetches energy metrics from AluminatAI API (last 24h filtered by job UUID)
      - Logs energy_kwh, cost_usd, co2_kg as MLflow metrics
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ALUMINATAI_API_KEY", "")
        self.api_endpoint = api_endpoint or os.getenv(
            "ALUMINATAI_API_ENDPOINT", "https://aluminatiai.com/api/metrics/ingest"
        )
        self._start_times: dict[str, float] = {}

    def on_run_start(self, run) -> None:
        """Call at the beginning of an MLflow run."""
        if not _MLFLOW:
            return

        run_id = run.info.run_id
        self._start_times[run_id] = time.time()

        # Inject attribution env vars so the agent can tag metric rows
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment(run.info.experiment_id)
            run_name = run.info.run_name or run_id[:8]

            os.environ["ALUMINATAI_MODEL"] = run_name
            os.environ["ALUMINATAI_TEAM"] = experiment.name
            logger.info(
                "AluminatAI: tracking run '%s' in experiment '%s'",
                run_name, experiment.name
            )
        except Exception as exc:
            logger.warning("AluminatAI MLflow on_run_start: %s", exc)

    def on_run_end(self, run) -> None:
        """Call at the end of an MLflow run to log energy metrics."""
        if not _MLFLOW or not self.api_key:
            return

        run_id = run.info.run_id
        start_time = self._start_times.pop(run_id, None)
        if start_time is None:
            return

        try:
            energy = self._fetch_energy(start_time)
            if energy:
                mlflow.log_metrics({
                    "aluminatai_energy_kwh": energy.get("energy_kwh", 0),
                    "aluminatai_cost_usd": energy.get("cost_usd", 0),
                    "aluminatai_co2_kg": energy.get("co2_kg", 0),
                })
                logger.info(
                    "AluminatAI: logged %.4f kWh / $%.4f to run %s",
                    energy.get("energy_kwh", 0),
                    energy.get("cost_usd", 0),
                    run_id,
                )
        except Exception as exc:
            logger.warning("AluminatAI MLflow on_run_end: %s", exc)
        finally:
            # Clean up env vars
            os.environ.pop("ALUMINATAI_MODEL", None)
            os.environ.pop("ALUMINATAI_TEAM", None)

    @contextmanager
    def track(self, run) -> Generator[None, None, None]:
        """Context manager that wraps on_run_start / on_run_end."""
        self.on_run_start(run)
        try:
            yield
        finally:
            self.on_run_end(run)

    def _fetch_energy(self, since: float) -> Optional[dict]:
        """Fetch aggregated energy for metrics since `since` (epoch seconds)."""
        import urllib.request
        import urllib.error
        import json
        from datetime import datetime, timezone

        since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()
        base = self.api_endpoint.removesuffix("/api/metrics/ingest").rstrip("/")
        url = f"{base}/api/metrics/cost-estimate"

        req = urllib.request.Request(
            url,
            headers={"X-API-Key": self.api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.debug("energy fetch failed: %s", exc)
            return None
