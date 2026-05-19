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
DcgmProbe — Phase-aware GPU power decomposition.

Two operating modes, selected automatically at startup:

  dcgm  — NVIDIA DCGM (Data Center GPU Manager) is installed and the
           nv-hostengine daemon is running.  Uses hardware profiling
           counters (DCGM_FI_PROF_*) for accurate tensor / fp32 / fp16 /
           DRAM activity fractions.

  nvml  — DCGM not available.  Falls back to the NVML utilization rates
           already collected by GPUCollector (SM util → compute proxy,
           memory util → DRAM proxy).  tensor_power and fp16_power are
           reported as 0 (indistinguishable from general compute in
           NVML-only mode).

Power model:
  available_power = total_power_w − idle_baseline_w
  Each component receives a share of available_power proportional to its
  weighted activity:
    tensor_power  = available × (α · a_tensor) / Σ
    fp32_power    = available × (β · a_fp32)   / Σ
    fp16_power    = available × (γ · a_fp16)   / Σ
    memory_power  = available × (δ · a_dram)   / Σ
    idle_power    = idle_baseline_w

  Coefficients (α, β, γ, δ) are relative weights loaded from
  ~/.config/aluminatai/dcgm_coefficients.json, keyed by GPU arch prefix.
  Built-in defaults are used when the file is absent.

Installing DCGM Python bindings:
  # On Ubuntu/Debian with DCGM system package installed:
  pip install nvidia-dcgm          # or: pydcgm if available on PyPI
  # Alternatively, bindings ship with DCGM at:
  # /usr/lib/dcgm/bindings/python3/
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_COEFFICIENTS_PATH = Path.home() / ".config" / "aluminatai" / "dcgm_coefficients.json"

# Default relative weights per GPU architecture prefix.
# Higher weight = that component receives proportionally more power
# when its activity counter is at the same level as another component.
# Sources: NVIDIA GTC power modelling talks + empirical tuning.
_DEFAULT_COEFFICIENTS: dict[str, dict[str, float]] = {
    "NVIDIA A100": {"tensor": 3.5, "fp32": 2.5, "fp16": 3.0, "dram": 1.5},
    "NVIDIA H100": {"tensor": 4.0, "fp32": 2.5, "fp16": 3.5, "dram": 1.5},
    "NVIDIA H200": {"tensor": 4.0, "fp32": 2.5, "fp16": 3.5, "dram": 2.0},
    "NVIDIA V100": {"tensor": 3.0, "fp32": 3.0, "fp16": 2.5, "dram": 1.5},
    "NVIDIA A10":  {"tensor": 2.5, "fp32": 2.5, "fp16": 2.5, "dram": 1.5},
    "Tesla":       {"tensor": 2.5, "fp32": 3.0, "fp16": 2.5, "dram": 1.5},
    "RTX":         {"tensor": 2.0, "fp32": 2.5, "fp16": 2.0, "dram": 1.5},
    "default":     {"tensor": 3.0, "fp32": 2.5, "fp16": 2.5, "dram": 1.5},
}

# DCGM field IDs used for profiling (numeric values stable across DCGM versions)
_DCGM_FI_PROF_TENSOR_ACTIVE = 1001
_DCGM_FI_PROF_SM_ACTIVE     = 1002
_DCGM_FI_PROF_DRAM_ACTIVE   = 1005
_DCGM_FI_PROF_FP16_ACTIVE   = 1019
_DCGM_FI_PROF_FP32_ACTIVE   = 1021


class DcgmProbe:
    """
    Phase-aware GPU power decomposition.

    Usage::

        probe = DcgmProbe()
        probe.start()           # initialises DCGM or falls back to NVML

        # In the main loop:
        for m in metrics:
            activity = probe.get_activity(m.gpu_index, fallback=m)
            idle_w   = baselines.get(m.gpu_index, 0.0)
            decomp   = probe.decompose_power(m.power_draw_w, activity,
                                             m.gpu_name, idle_w)
            metrics_server.update_dcgm(m.gpu_uuid, str(m.gpu_index), decomp)

        probe.shutdown()
    """

    def __init__(self):
        self._mode: str = "unavailable"   # "dcgm" | "nvml" | "unavailable"
        self._coefficients: dict[str, dict[str, float]] = {}
        self._dcgm_handle = None
        self._dcgm_group = None
        self._field_group = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise DCGM or fall back to NVML proxy mode."""
        self._coefficients = self._load_coefficients()
        if self._try_init_dcgm():
            self._mode = "dcgm"
            log.info("DcgmProbe: DCGM mode active (tensor/fp16/fp32/DRAM counters)")
        else:
            self._mode = "nvml"
            log.info(
                "DcgmProbe: NVML fallback mode (SM util → compute proxy, "
                "tensor/fp16 unavailable without DCGM)"
            )

    def shutdown(self) -> None:
        if self._dcgm_handle is not None:
            try:
                self._dcgm_handle.Disconnect()
            except Exception as exc:
                log.debug("DCGM disconnect error: %s", exc)
            self._dcgm_handle = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def active(self) -> bool:
        return self._mode in ("dcgm", "nvml")

    # ── Activity query ────────────────────────────────────────────────────

    def get_activity(self, gpu_index: int, fallback=None) -> dict[str, float]:
        """
        Return per-phase activity fractions in [0.0, 1.0].

        Keys: tensor_activity, fp32_activity, fp16_activity, memory_activity.

        fallback: a GPUMetrics object — used in NVML mode to derive compute
                  and memory activity from utilization_gpu_pct / utilization_memory_pct.
        """
        if self._mode == "dcgm":
            return self._get_dcgm_activity(gpu_index)

        if self._mode == "nvml" and fallback is not None:
            sm_frac  = getattr(fallback, "utilization_gpu_pct",    0) / 100.0
            mem_frac = getattr(fallback, "utilization_memory_pct", 0) / 100.0
            return {
                "tensor_activity": 0.0,        # indistinguishable in NVML mode
                "fp32_activity":   sm_frac,    # SM util is the best compute proxy
                "fp16_activity":   0.0,        # indistinguishable in NVML mode
                "memory_activity": mem_frac,
            }

        return {"tensor_activity": 0.0, "fp32_activity": 0.0,
                "fp16_activity": 0.0, "memory_activity": 0.0}

    # ── Power decomposition ───────────────────────────────────────────────

    def decompose_power(
        self,
        total_power_w: float,
        activity: dict[str, float],
        gpu_name: str = "",
        idle_baseline_w: float = 0.0,
    ) -> dict[str, float]:
        """
        Split total_power_w into per-phase watt estimates.

        Returns a dict with keys:
          tensor_power_w, fp32_power_w, fp16_power_w, memory_power_w, idle_power_w

        All values are >= 0 and sum to total_power_w (within float precision).
        """
        coeffs  = self._get_coefficients(gpu_name)
        a_t     = max(0.0, min(1.0, activity.get("tensor_activity", 0.0)))
        a_f32   = max(0.0, min(1.0, activity.get("fp32_activity",   0.0)))
        a_f16   = max(0.0, min(1.0, activity.get("fp16_activity",   0.0)))
        a_dram  = max(0.0, min(1.0, activity.get("memory_activity", 0.0)))

        idle_w     = min(idle_baseline_w, total_power_w)
        available  = max(0.0, total_power_w - idle_w)

        # Weighted activity fractions
        w_t   = coeffs["tensor"] * a_t
        w_f32 = coeffs["fp32"]   * a_f32
        w_f16 = coeffs["fp16"]   * a_f16
        w_d   = coeffs["dram"]   * a_dram
        total_w = w_t + w_f32 + w_f16 + w_d

        if total_w < 1e-9:
            # All counters are zero — all active power is attributed to idle
            return {
                "tensor_power_w": 0.0,
                "fp32_power_w":   0.0,
                "fp16_power_w":   0.0,
                "memory_power_w": 0.0,
                "idle_power_w":   round(total_power_w, 3),
            }

        return {
            "tensor_power_w": round(available * w_t   / total_w, 3),
            "fp32_power_w":   round(available * w_f32 / total_w, 3),
            "fp16_power_w":   round(available * w_f16 / total_w, 3),
            "memory_power_w": round(available * w_d   / total_w, 3),
            "idle_power_w":   round(idle_w, 3),
        }

    # ── Coefficient management ────────────────────────────────────────────

    def _get_coefficients(self, gpu_name: str) -> dict[str, float]:
        """Return coefficients for the closest matching GPU arch prefix."""
        # Try loaded / persisted coefficients first
        for prefix, coeffs in self._coefficients.items():
            if prefix != "default" and gpu_name.startswith(prefix):
                return coeffs

        # Built-in defaults
        for prefix, coeffs in _DEFAULT_COEFFICIENTS.items():
            if prefix != "default" and gpu_name.startswith(prefix):
                return coeffs

        return _DEFAULT_COEFFICIENTS["default"]

    def _load_coefficients(self) -> dict[str, dict[str, float]]:
        try:
            with open(_COEFFICIENTS_PATH) as f:
                data = json.load(f)
            # Validate: each entry must have tensor/fp32/fp16/dram keys
            out: dict[str, dict[str, float]] = {}
            for arch, coeffs in data.items():
                if all(k in coeffs for k in ("tensor", "fp32", "fp16", "dram")):
                    out[arch] = {k: float(coeffs[k]) for k in ("tensor", "fp32", "fp16", "dram")}
            if out:
                log.debug("DCGM coefficients loaded from %s (%d arches)", _COEFFICIENTS_PATH, len(out))
            return out
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            return {}

    def save_coefficients(self, arch: str, coeffs: dict[str, float]) -> None:
        """Persist fitted coefficients for a GPU architecture."""
        existing = self._load_coefficients()
        existing[arch] = coeffs
        _COEFFICIENTS_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with open(_COEFFICIENTS_PATH, "w") as f:
                json.dump(existing, f, indent=2)
            log.info("DCGM coefficients saved for arch %r → %s", arch, _COEFFICIENTS_PATH)
        except OSError as exc:
            log.warning("Could not save DCGM coefficients: %s", exc)

    # ── DCGM initialisation ───────────────────────────────────────────────

    def _try_init_dcgm(self) -> bool:
        """
        Attempt to connect to DCGM in embedded mode (nv-hostengine).
        Returns True on success, False on any failure (ImportError,
        connection refused, missing daemon, etc.).
        """
        try:
            import pydcgm          # type: ignore[import]
            import dcgm_fields     # type: ignore[import]

            handle = pydcgm.DcgmHandle(ipAddress=None)  # embedded / local daemon
            system = handle.GetSystem()
            group  = system.GetDefaultGroup()
            fgroup = system.GetEmptyFieldGroup()

            field_ids = [
                dcgm_fields.DCGM_FI_PROF_TENSOR_ACTIVE,
                dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
                dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
                dcgm_fields.DCGM_FI_PROF_FP16_ACTIVE,
            ]
            fgroup.AddFieldIds(field_ids)

            # Watch at 1 Hz (1_000_000 µs), keep 1 sample, no max age
            group.samples.WatchFields(fgroup, 1_000_000, 1.0, 0)

            self._dcgm_handle = handle
            self._dcgm_group  = group
            self._dcgm_fgroup = fgroup
            self._dcgm_fields = dcgm_fields
            return True

        except ImportError:
            log.debug("pydcgm not installed — DCGM mode unavailable")
            return False
        except Exception as exc:
            log.debug("DCGM init failed (%s) — falling back to NVML proxy", exc)
            return False

    def _get_dcgm_activity(self, gpu_index: int) -> dict[str, float]:
        """Query DCGM profiling counters for one GPU. Returns 0.0 on any error."""
        blank = {"tensor_activity": 0.0, "fp32_activity": 0.0,
                 "fp16_activity": 0.0, "memory_activity": 0.0}
        try:
            latest = self._dcgm_group.samples.GetLatestValues(self._dcgm_fgroup)
            df = self._dcgm_fields

            def _val(field_id: int) -> float:
                try:
                    v = latest[gpu_index][field_id].value
                    return max(0.0, min(1.0, float(v)))
                except (KeyError, TypeError, ValueError):
                    return 0.0

            return {
                "tensor_activity": _val(df.DCGM_FI_PROF_TENSOR_ACTIVE),
                "fp32_activity":   _val(df.DCGM_FI_PROF_SM_ACTIVE),   # SM as fp32 proxy
                "fp16_activity":   _val(df.DCGM_FI_PROF_FP16_ACTIVE),
                "memory_activity": _val(df.DCGM_FI_PROF_DRAM_ACTIVE),
            }
        except Exception as exc:
            log.debug("DCGM activity query failed for GPU %d: %s", gpu_index, exc)
            return blank
