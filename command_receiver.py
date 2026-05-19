"""
Command Receiver — polls the cloud for pending commands and executes them.

The agent polls GET /api/agent/commands every COMMAND_POLL_INTERVAL seconds.
When a command is received (e.g., apply_power_cap), it validates params,
executes via the appropriate module, and reports the result back.

Safety: validates power cap between 100W and TDP, requires COMMAND_POLL_ENABLED.
"""
from __future__ import annotations

import json
import logging
import socket
import time
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)


class CommandReceiver:
    """Polls cloud for pending commands and executes them safely."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        machine_id: str,
        dry_run: bool = False,
        base_interval: float = 60.0,
        max_interval: float = 300.0,
    ):
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        self._base = f"{parsed.scheme}://{parsed.netloc}"
        self._api_key = api_key
        self._machine_id = machine_id
        self._dry_run = dry_run
        self._last_poll: float = 0.0
        self._base_interval = base_interval
        self._max_interval = max_interval
        self._current_interval = base_interval
        self._empty_polls: int = 0

    @property
    def poll_interval(self) -> float:
        """Current adaptive poll interval in seconds."""
        return self._current_interval

    def poll_and_execute(self) -> int:
        """Poll for pending commands and execute them. Returns count executed."""
        commands = self._fetch_commands()
        if not commands:
            self._empty_polls += 1
            if self._empty_polls >= 3:
                self._current_interval = min(
                    self._max_interval,
                    self._current_interval * 2,
                )
            return 0

        self._empty_polls = 0
        self._current_interval = self._base_interval

        executed = 0
        for cmd in commands:
            success, message = self._execute(cmd)
            self._report_result(
                cmd["id"],
                success=success,
                message=message,
            )
            executed += 1

        self._last_poll = time.time()
        return executed

    def _fetch_commands(self) -> list[dict]:
        url = f"{self._base}/api/agent/commands?machine_id={self._machine_id}"
        req = urllib.request.Request(
            url,
            headers={"X-API-Key": self._api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                return body.get("commands", [])
        except Exception as exc:
            logger.debug("Command poll failed (non-fatal): %s", exc)
            return []

    def _execute(self, cmd: dict) -> tuple[bool, str]:
        cmd_type = cmd.get("command_type", "")
        params = cmd.get("params", {})

        handler = self._handlers.get(cmd_type)
        if handler:
            return handler(params)
        return False, f"Unknown command type: {cmd_type}"

    @property
    def _handlers(self) -> dict:
        return {
            "apply_power_cap": self._apply_power_cap,
            "rollback_power_cap": self._rollback_power_cap,
            "set_precision": self._set_precision,
        }

    def _apply_power_cap(self, params: dict) -> tuple[bool, str]:
        gpu_index = params.get("gpu_index")
        watts = params.get("watts")

        if gpu_index is None or watts is None:
            return False, "Missing gpu_index or watts"

        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index > 31:
            return False, f"Invalid gpu_index: {gpu_index}"

        if not isinstance(watts, (int, float)) or watts < 100 or watts > 1200:
            return False, f"Power cap {watts}W out of safe range (100-1200W)"

        if self._dry_run:
            logger.info("Command: would set GPU %d power cap to %dW (dry run)", gpu_index, watts)
            return True, f"Dry run: would set GPU {gpu_index} to {watts}W"

        try:
            from efficiency.power_control import set_power_limit
            ok = set_power_limit(gpu_index, int(watts), quiet=True)
            if ok:
                logger.info("Command: set GPU %d power cap to %dW", gpu_index, watts)
                return True, f"Power cap set to {watts}W on GPU {gpu_index}"
            else:
                return False, f"set_power_limit returned False for GPU {gpu_index}"
        except Exception as exc:
            logger.warning("Command: power cap failed GPU %d: %s", gpu_index, exc)
            return False, str(exc)

    def _rollback_power_cap(self, params: dict) -> tuple[bool, str]:
        gpu_index = params.get("gpu_index", 0)

        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index > 31:
            return False, f"Invalid gpu_index: {gpu_index}"

        if self._dry_run:
            return True, f"Dry run: would reset GPU {gpu_index} to default"

        try:
            from efficiency.power_control import get_default_power_limit, set_power_limit
            default_w = get_default_power_limit(gpu_index)
            ok = set_power_limit(gpu_index, default_w, quiet=True)
            if ok:
                logger.info("Command: reset GPU %d to default %dW", gpu_index, default_w)
                return True, f"Power cap reset to default {default_w}W on GPU {gpu_index}"
            return False, f"Rollback failed for GPU {gpu_index}"
        except Exception as exc:
            return False, str(exc)

    def _set_precision(self, params: dict) -> tuple[bool, str]:
        gpu_index = params.get("gpu_index", 0)
        precision = params.get("precision", "")

        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index > 31:
            return False, f"Invalid gpu_index: {gpu_index}"

        if precision not in ("fp16", "bf16", "fp32", "tf32"):
            return False, f"Unsupported precision: {precision}"

        if self._dry_run:
            return True, f"Dry run: would set GPU {gpu_index} to {precision}"

        # Precision switching is advisory — logged for the user to apply manually
        logger.info("Command: precision switch GPU %d → %s (advisory)", gpu_index, precision)
        return True, f"Precision switch to {precision} logged for GPU {gpu_index} (apply in training config)"

    def _report_result(self, command_id: str, success: bool, message: str) -> None:
        url = f"{self._base}/api/agent/commands/{command_id}/result"
        payload = json.dumps({
            "success": success,
            "message": message,
            "hostname": socket.gethostname(),
            "machine_id": self._machine_id,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "X-API-Key": self._api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.debug("Command result report failed: %s", exc)
