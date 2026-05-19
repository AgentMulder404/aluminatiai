"""
Recommendation Reporter — collects optimization recommendations from
local analyzers and uploads them to the AluminatAI cloud for dashboard display.

Runs on the same interval as AutoTuner (every AUTO_TUNE_INTERVAL seconds).
Deduplicates by hashing (machine_id, category, gpu_index) to avoid
re-uploading the same recommendation within an hour.
"""
from __future__ import annotations

import hashlib
import json
import logging
import socket
import time
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)


class RecommendationReporter:
    """Collects and uploads optimization recommendations."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        machine_id: str,
        dedup_window: float = 3600.0,
    ):
        self._endpoint = endpoint
        self._api_key = api_key
        self._machine_id = machine_id
        self._hostname = socket.gethostname()
        self._dedup_window = dedup_window
        self._sent_hashes: dict[str, float] = {}

    def _dedup_key(self, category: str, gpu_index: int) -> str:
        raw = f"{self._machine_id}:{category}:{gpu_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _is_duplicate(self, key: str) -> bool:
        last_sent = self._sent_hashes.get(key, 0)
        return (time.time() - last_sent) < self._dedup_window

    def _mark_sent(self, key: str) -> None:
        self._sent_hashes[key] = time.time()
        now = time.time()
        self._sent_hashes = {
            k: v for k, v in self._sent_hashes.items()
            if now - v < self._dedup_window * 2
        }

    def report_from_auto_tuner(self, tune_results: list) -> int:
        """Convert TuneResult list to recommendations and upload."""
        recs = []
        for tr in tune_results:
            if tr.recommended_cap_w is None:
                continue
            key = self._dedup_key("power_cap", tr.gpu_index)
            if self._is_duplicate(key):
                continue

            recs.append({
                "machine_id": self._machine_id,
                "hostname": self._hostname,
                "gpu_index": tr.gpu_index,
                "gpu_name": tr.gpu_name,
                "source": "auto_tuner",
                "category": "power_cap",
                "priority": "P1" if tr.estimated_savings_pct >= 20 else "P2",
                "title": f"Reduce power cap on GPU {tr.gpu_index}",
                "description": tr.reason,
                "action": f"Set power limit to {int(tr.recommended_cap_w)}W (currently {int(tr.current_power_w)}W)",
                "estimated_savings_pct": round(tr.estimated_savings_pct, 1),
                "effort_score": 1,
                "action_payload": {
                    "command": "apply_power_cap",
                    "gpu_index": tr.gpu_index,
                    "watts": int(tr.recommended_cap_w),
                },
            })
            self._mark_sent(key)

        return self._upload(recs)

    def report_from_workload_analyzer(self, result) -> int:
        """Convert OptimizeResult recommendations to cloud format."""
        recs = []
        for rec in getattr(result, "recommendations", []):
            key = self._dedup_key(rec.category, result.gpu_index)
            if self._is_duplicate(key):
                continue

            effort = 1
            if rec.category in ("precision", "gpu_match"):
                effort = 3
            elif rec.category in ("utilization",):
                effort = 2

            recs.append({
                "machine_id": self._machine_id,
                "hostname": self._hostname,
                "gpu_index": result.gpu_index,
                "gpu_name": result.gpu_name,
                "source": "workload_analyzer",
                "category": rec.category,
                "priority": rec.priority,
                "title": rec.description[:120],
                "description": rec.detail or rec.description,
                "action": rec.action,
                "estimated_savings_pct": round(rec.estimated_savings_pct, 1),
                "effort_score": effort,
                "action_payload": {},
            })
            self._mark_sent(key)

        return self._upload(recs)

    def report_from_carbon_scheduler(self, schedule_rec) -> int:
        """Convert ScheduleRecommendation to cloud format."""
        if schedule_rec is None or schedule_rec.savings_pct <= 0:
            return 0

        key = self._dedup_key("carbon_schedule", 0)
        if self._is_duplicate(key):
            return 0

        start_str = schedule_rec.recommended_start.strftime("%Y-%m-%d %H:%M UTC")
        recs = [{
            "machine_id": self._machine_id,
            "hostname": self._hostname,
            "gpu_index": None,
            "gpu_name": None,
            "source": "carbon_scheduler",
            "category": "carbon_schedule",
            "priority": "P2" if schedule_rec.savings_pct >= 10 else "P3",
            "title": f"Defer job start to {start_str} for lower carbon",
            "description": (
                f"Current grid intensity: {schedule_rec.current_intensity_gco2e} gCO2e/kWh. "
                f"Optimal window at {start_str} averages {schedule_rec.avg_intensity_gco2e} gCO2e/kWh "
                f"(~{schedule_rec.savings_pct}% less CO2)."
            ),
            "action": f"Schedule job start for {start_str}",
            "estimated_savings_pct": round(schedule_rec.savings_pct, 1),
            "effort_score": 2,
            "action_payload": {
                "command": "defer_job",
                "recommended_start": schedule_rec.recommended_start.isoformat(),
                "zone": schedule_rec.zone,
            },
        }]
        self._mark_sent(key)
        return self._upload(recs)

    def report_from_hardware_match(self, match_result, gpu_index: int = 0) -> int:
        """Convert HardwareMatchScorer MatchResult to recommendation."""
        if match_result is None or match_result.energy_savings_pct <= 5:
            return 0
        if match_result.current_arch == match_result.best_arch:
            return 0

        key = self._dedup_key("gpu_match", gpu_index)
        if self._is_duplicate(key):
            return 0

        recs = [{
            "machine_id": self._machine_id,
            "hostname": self._hostname,
            "gpu_index": gpu_index,
            "gpu_name": match_result.current_arch,
            "source": "workload_analyzer",
            "category": "gpu_match",
            "priority": "P2" if match_result.energy_savings_pct >= 15 else "P3",
            "title": f"Better GPU match: {match_result.best_arch} for {match_result.model_tag}",
            "description": match_result.recommendation,
            "action": f"Migrate to {match_result.best_arch} (~{match_result.energy_savings_pct}% less energy)",
            "estimated_savings_pct": round(match_result.energy_savings_pct, 1),
            "effort_score": 4,
            "action_payload": {},
        }]
        self._mark_sent(key)
        return self._upload(recs)

    def report_from_swarm_policy(self, policy_recs: list[dict]) -> int:
        """Upload recommendations from a Swarm policy engine (Tier 3)."""
        recs = []
        for pr in policy_recs:
            target_mid = pr.get("machine_id", self._machine_id)
            raw = f"{target_mid}:{pr.get('category', 'swarm')}:{pr.get('gpu_index', 0)}"
            key = hashlib.sha256(raw.encode()).hexdigest()[:16]
            if self._is_duplicate(key):
                continue
            recs.append({
                "machine_id": target_mid,
                "hostname": pr.get("hostname", self._hostname),
                "gpu_index": pr.get("gpu_index"),
                "gpu_name": pr.get("gpu_name"),
                "source": "swarm_policy",
                "category": pr.get("category", "power_cap"),
                "priority": pr.get("priority", "P1"),
                "title": pr.get("title", ""),
                "description": pr.get("description", ""),
                "action": pr.get("action", ""),
                "estimated_savings_pct": pr.get("estimated_savings_pct", 0),
                "effort_score": pr.get("effort_score", 1),
                "action_payload": pr.get("action_payload", {}),
            })
            self._mark_sent(key)
        return self._upload(recs)

    def _upload(self, recs: list[dict]) -> int:
        if not recs:
            return 0

        from urllib.parse import urlparse
        parsed = urlparse(self._endpoint)
        base = f"{parsed.scheme}://{parsed.netloc}"
        url = base + "/api/agent/recommendations"

        payload = json.dumps({"recommendations": recs}, default=str).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "X-API-Key": self._api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                count = body.get("inserted", 0)
                if count:
                    logger.info("Uploaded %d recommendations", count)
                return count
        except Exception as exc:
            logger.debug("Recommendation upload failed (non-fatal): %s", exc)
            return 0
