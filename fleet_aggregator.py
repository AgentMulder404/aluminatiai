"""
Fleet edge aggregator — lightweight HTTP server for multi-node GPU metric rollups.

Agents on peer nodes POST their per-node summaries here. The aggregator
maintains fleet-wide totals and exposes them via Prometheus-compatible metrics.

Usage:
    Set FLEET_AGGREGATOR_ENABLED=1 on the designated aggregator node.
    Set FLEET_AGGREGATOR_PEERS=http://aggregator:9101 on all other nodes.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

logger = logging.getLogger(__name__)


class NodeSummary:
    """Summary received from one agent node."""
    __slots__ = (
        "machine_id", "cluster_tag", "gpu_count", "total_power_w",
        "total_energy_kwh", "uptime_s", "timestamp",
    )

    def __init__(self, data: dict):
        self.machine_id: str = data.get("machine_id", "unknown")
        self.cluster_tag: str = data.get("cluster_tag", "")
        self.gpu_count: int = data.get("gpu_count", 0)
        self.total_power_w: float = data.get("total_power_w", 0.0)
        self.total_energy_kwh: float = data.get("total_energy_kwh", 0.0)
        self.uptime_s: float = data.get("uptime_s", 0.0)
        self.timestamp: float = data.get("timestamp", time.time())


class FleetAggregator:
    """Collects and aggregates per-node summaries from peer agents."""

    STALE_THRESHOLD_S = 600  # nodes not reporting for 10min are stale

    def __init__(self, port: int = 9101, bind: str = "0.0.0.0"):
        self._port = port
        self._bind = bind
        self._nodes: dict[str, NodeSummary] = {}
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None

    def ingest(self, data: dict) -> None:
        """Ingest a node summary."""
        summary = NodeSummary(data)
        with self._lock:
            self._nodes[summary.machine_id] = summary

    def get_fleet_stats(self) -> dict[str, Any]:
        """Return aggregated fleet statistics, pruning stale nodes."""
        now = time.time()
        with self._lock:
            active = {
                k: v for k, v in self._nodes.items()
                if (now - v.timestamp) < self.STALE_THRESHOLD_S
            }
        return {
            "active_nodes": len(active),
            "total_gpus": sum(n.gpu_count for n in active.values()),
            "total_power_w": sum(n.total_power_w for n in active.values()),
            "total_energy_kwh": sum(n.total_energy_kwh for n in active.values()),
            "nodes": [
                {
                    "machine_id": n.machine_id,
                    "cluster_tag": n.cluster_tag,
                    "gpu_count": n.gpu_count,
                    "power_w": n.total_power_w,
                    "energy_kwh": n.total_energy_kwh,
                    "uptime_s": n.uptime_s,
                    "age_s": round(now - n.timestamp, 1),
                }
                for n in active.values()
            ],
        }

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        aggregator = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/api/fleet/ingest":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        data = json.loads(body)
                        aggregator.ingest(data)
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'{"ok":true}')
                    except (json.JSONDecodeError, KeyError) as exc:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": str(exc)}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_GET(self):
                if self.path == "/api/fleet/stats":
                    stats = aggregator.get_fleet_stats()
                    body = json.dumps(stats).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args, **kwargs):
                pass

        try:
            self._server = HTTPServer((self._bind, self._port), Handler)
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            logger.info("Fleet aggregator on %s:%d", self._bind, self._port)
        except OSError as exc:
            logger.error("Fleet aggregator failed to bind port %d: %s", self._port, exc)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()


def post_node_summary(peer_url: str, summary: dict, timeout: float = 5.0) -> bool:
    """POST a node summary to a peer fleet aggregator."""
    import requests
    try:
        resp = requests.post(
            f"{peer_url}/api/fleet/ingest",
            json=summary,
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.debug("Fleet post to %s failed: %s", peer_url, exc)
        return False
