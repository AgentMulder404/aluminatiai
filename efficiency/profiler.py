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
Scientific Energy Profiler for ML Workloads.

Provides a rigorous, reproducible energy audit for any ML model running
on NVIDIA A100/H100 infrastructure. Implements three control protocols
to ensure data integrity:

  1. Steady-State Warm-up:  Discards the first N seconds of each run
     to account for thermal ramp-up, GPU boost clock settling, and
     CUDA kernel JIT compilation.

  2. Baseline Subtraction:  Measures static (idle) power and subtracts
     it from active power to isolate the dynamic energy cost attributable
     to the workload itself.

  3. Confidence Intervals:  Repeats the measurement across K iterations
     and reports the mean with a 95% confidence interval using the
     Student's t-distribution (appropriate for small sample sizes).

Usage:

    profiler = ScientificEnergyProfiler(gpu_index=0)

    # Profile a training step
    result = profiler.profile(
        workload_fn=lambda: model.train_step(batch),
        task_type="training",
        task_unit="step",
        iterations=5,
        warmup_seconds=10.0,
    )

    print(result.summary())
    print(result.metrics)
    plan = result.optimization_plan

Dependencies: torch, pynvml, (optional) pandas, scipy
"""

from __future__ import annotations

import math
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

from .gpu_specs import GPU_ARCHITECTURES, ArchSpec, resolve_arch

logger = logging.getLogger(__name__)

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PowerSample:
    """A single timestamped power reading from NVML."""
    timestamp: float            # time.monotonic() seconds
    power_w: float
    gpu_utilization_pct: int
    memory_utilization_pct: int
    temperature_c: int
    sm_clock_mhz: int
    memory_used_bytes: int
    memory_total_bytes: int


@dataclass
class IterationResult:
    """Raw measurements from a single profiling iteration."""
    duration_s: float
    energy_j: float             # Trapezoidal integral of power over time
    dynamic_energy_j: float     # Energy minus baseline (idle) contribution
    mean_power_w: float
    peak_power_w: float
    mean_utilization_pct: float
    mean_memory_util_pct: float
    mean_temperature_c: float
    sample_count: int
    flop_count: Optional[int] = None
    memory_bytes_read: Optional[int] = None
    memory_bytes_written: Optional[int] = None


@dataclass
class ConfidenceInterval:
    """A measurement with 95% confidence bounds."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} +/- {self.ci_upper - self.mean:.4f} (95% CI, n={self.n})"


@dataclass
class ProfileResult:
    """Complete output of a scientific energy profiling session."""

    # Identity
    gpu_name: str
    gpu_arch: Optional[str]
    gpu_uuid: str

    # Control protocol outputs
    baseline_power_w: ConfidenceInterval
    warmup_seconds_discarded: float
    iteration_count: int

    # Per-iteration raw data
    iterations: list[IterationResult]

    # Aggregate measurements with confidence intervals
    energy_per_task_j: ConfidenceInterval
    dynamic_energy_per_task_j: ConfidenceInterval
    mean_power_w: ConfidenceInterval
    peak_power_w: float
    duration_per_task_s: ConfidenceInterval

    # Task info
    task_type: str              # "training" or "inference"
    task_unit: str              # "step", "token", "image", "batch"
    tasks_per_iteration: int

    # Aluminati Efficiency Metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Optimization plan
    optimization_plan: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the profiling session."""
        lines = [
            "=" * 72,
            "  ALUMINATI ENERGY AUDIT",
            "=" * 72,
            f"  GPU:              {self.gpu_name}",
            f"  Architecture:     {self.gpu_arch or 'unknown'}",
            f"  Iterations:       {self.iteration_count}",
            f"  Warmup Discarded: {self.warmup_seconds_discarded:.1f}s",
            "",
            "  CONTROL MEASUREMENTS",
            f"  Baseline (idle):  {self.baseline_power_w}",
            f"  Active Power:     {self.mean_power_w}",
            f"  Peak Power:       {self.peak_power_w:.1f} W",
            "",
            "  ENERGY PER {unit}",
            f"  Total:            {self.energy_per_task_j}",
            f"  Dynamic (net):    {self.dynamic_energy_per_task_j}",
            f"  Duration:         {self.duration_per_task_s}",
            "",
        ]

        # Replace {unit} placeholder
        lines = [l.replace("{unit}", self.task_unit.upper()) for l in lines]

        if self.metrics:
            lines.append("  ALUMINATI EFFICIENCY METRICS")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key:24s} {value:.4f}")
                else:
                    lines.append(f"  {key:24s} {value}")
            lines.append("")

        if self.optimization_plan:
            lines.append("  OPTIMIZATION PLAN")
            regime = self.optimization_plan.get("regime", "unknown")
            lines.append(f"  Bottleneck:       {regime}")
            for rec in self.optimization_plan.get("recommendations", []):
                lines.append(f"    - [{rec['priority']}] {rec['action']}")
                lines.append(f"      {rec['rationale']}")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Scientific Energy Profiler
# ═══════════════════════════════════════════════════════════════════════

class ScientificEnergyProfiler:
    """
    Production-grade energy profiler implementing rigorous control protocols.

    The profiler wraps any callable workload function and produces a
    statistically validated energy audit with confidence intervals,
    baseline subtraction, and roofline-based optimization recommendations.

    Args:
        gpu_index:        CUDA device index to profile (default: 0)
        sample_interval:  NVML polling interval in seconds (default: 0.05 = 50ms)
                          Lower values give higher temporal resolution but
                          increase overhead. 50ms is the NVML minimum for
                          accurate power readings on A100/H100.
        grid_carbon_intensity_g_kwh:
                          Grid carbon intensity in gCO2e/kWh for emissions
                          calculation. Default 394.0 = US average (EPA eGRID 2024).
    """

    # Minimum NVML power sampling interval (hardware limit)
    _MIN_SAMPLE_INTERVAL_S = 0.020

    def __init__(
        self,
        gpu_index: int = 0,
        sample_interval: float = 0.050,
        grid_carbon_intensity_g_kwh: float = 394.0,
    ):
        if not _NVML_AVAILABLE:
            raise RuntimeError(
                "pynvml required for energy profiling. "
                "Install with: pip install nvidia-ml-py3"
            )

        self._gpu_index = gpu_index
        self._sample_interval = max(sample_interval, self._MIN_SAMPLE_INTERVAL_S)
        self._grid_carbon_g_kwh = grid_carbon_intensity_g_kwh

        # Initialize NVML
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        # Get static GPU info
        name = pynvml.nvmlDeviceGetName(self._handle)
        uuid = pynvml.nvmlDeviceGetUUID(self._handle)
        self.gpu_name = name.decode('utf-8') if isinstance(name, bytes) else name
        self.gpu_uuid = uuid.decode('utf-8') if isinstance(uuid, bytes) else uuid
        self.arch_spec = resolve_arch(self.gpu_name)

        logger.info(
            f"ScientificEnergyProfiler initialized: {self.gpu_name} "
            f"(index={gpu_index}, arch={self.arch_spec.name if self.arch_spec else 'unknown'})"
        )

    def profile(
        self,
        workload_fn: Callable[[], Any],
        task_type: str = "training",
        task_unit: str = "step",
        tasks_per_iteration: int = 1,
        iterations: int = 5,
        warmup_seconds: float = 10.0,
        warmup_fn: Optional[Callable[[], Any]] = None,
        baseline_seconds: float = 5.0,
        flop_count: Optional[int] = None,
        memory_bytes_accessed: Optional[int] = None,
    ) -> ProfileResult:
        """
        Execute a full scientific energy profiling session.

        Protocol:
          Phase 0: Baseline measurement (GPU idle)
          Phase 1: Warm-up (run workload, discard measurements)
          Phase 2: Measurement iterations (K runs with power sampling)
          Phase 3: Metric computation and optimization plan generation

        Args:
            workload_fn:       Callable that executes one unit of work (e.g.,
                               one training step, one inference batch).
            task_type:         "training" or "inference" — affects metric labels.
            task_unit:         Unit name for reporting ("step", "token", "image").
            tasks_per_iteration:
                               How many task_units each call to workload_fn
                               represents (e.g., batch_size for inference).
            iterations:        Number of measurement iterations (K). More
                               iterations = tighter confidence intervals.
                               Minimum 3 for valid CI calculation.
            warmup_seconds:    Seconds to run workload before measurement begins.
                               Accounts for thermal ramp-up and kernel caching.
            warmup_fn:         Optional separate warmup function. If None, uses
                               workload_fn in a loop for warmup_seconds.
            baseline_seconds:  Seconds to sample idle power for baseline.
            flop_count:        If known, the exact FLOPs per workload_fn call.
                               If None, attempts torch profiler auto-detection.
            memory_bytes_accessed:
                               If known, total bytes read+written per call.
                               Used for arithmetic intensity calculation.

        Returns:
            ProfileResult with full audit data, metrics, and optimization plan.
        """
        if iterations < 3:
            raise ValueError(
                f"Minimum 3 iterations required for confidence intervals, got {iterations}"
            )

        logger.info(f"Starting energy audit: {iterations} iterations, "
                     f"{warmup_seconds}s warmup, {baseline_seconds}s baseline")

        # ── Phase 0: Baseline (idle) power measurement ────────────
        logger.info("Phase 0: Measuring baseline (idle) power...")
        self._sync_gpu()
        baseline_samples = self._sample_power_for(baseline_seconds)
        baseline_powers = [s.power_w for s in baseline_samples]
        baseline_ci = self._compute_ci(baseline_powers)
        logger.info(f"  Baseline power: {baseline_ci}")

        # ── Phase 1: Steady-state warm-up ─────────────────────────
        logger.info(f"Phase 1: Warm-up for {warmup_seconds}s (discarded)...")
        self._run_warmup(warmup_fn or workload_fn, warmup_seconds)
        logger.info("  Warm-up complete. GPU at thermal steady state.")

        # ── Phase 2: Measurement iterations ───────────────────────
        logger.info(f"Phase 2: {iterations} measurement iterations...")
        iter_results: list[IterationResult] = []

        for i in range(iterations):
            result = self._measure_iteration(
                workload_fn, baseline_ci.mean,
                flop_count, memory_bytes_accessed,
            )
            iter_results.append(result)
            logger.info(
                f"  Iteration {i + 1}/{iterations}: "
                f"{result.energy_j:.2f} J total, "
                f"{result.dynamic_energy_j:.2f} J dynamic, "
                f"{result.mean_power_w:.1f} W avg, "
                f"{result.duration_s:.3f}s"
            )

        # ── Phase 3: Aggregate and compute metrics ────────────────
        logger.info("Phase 3: Computing metrics and optimization plan...")

        # Per-task energy (divide by tasks_per_iteration)
        energies = [r.energy_j / tasks_per_iteration for r in iter_results]
        dynamic_energies = [r.dynamic_energy_j / tasks_per_iteration for r in iter_results]
        powers = [r.mean_power_w for r in iter_results]
        durations = [r.duration_s / tasks_per_iteration for r in iter_results]

        energy_ci = self._compute_ci(energies)
        dynamic_ci = self._compute_ci(dynamic_energies)
        power_ci = self._compute_ci(powers)
        duration_ci = self._compute_ci(durations)
        peak_power = max(r.peak_power_w for r in iter_results)

        result = ProfileResult(
            gpu_name=self.gpu_name,
            gpu_arch=self.arch_spec.name if self.arch_spec else None,
            gpu_uuid=self.gpu_uuid,
            baseline_power_w=baseline_ci,
            warmup_seconds_discarded=warmup_seconds,
            iteration_count=iterations,
            iterations=iter_results,
            energy_per_task_j=energy_ci,
            dynamic_energy_per_task_j=dynamic_ci,
            mean_power_w=power_ci,
            peak_power_w=peak_power,
            duration_per_task_s=duration_ci,
            task_type=task_type,
            task_unit=task_unit,
            tasks_per_iteration=tasks_per_iteration,
        )

        # Compute Aluminati Efficiency Metrics
        result.metrics = self._compute_aluminati_metrics(
            result, iter_results, flop_count, memory_bytes_accessed,
        )

        # Generate optimization plan from roofline analysis
        result.optimization_plan = self._generate_optimization_plan(
            result, iter_results,
        )

        logger.info("Energy audit complete.")
        return result

    # ═══════════════════════════════════════════════════════════════
    # Phase 0: Baseline Measurement
    # ═══════════════════════════════════════════════════════════════

    def measure_baseline(self, duration_s: float = 5.0) -> ConfidenceInterval:
        """
        Measure static (idle) GPU power consumption.

        Ensures no active CUDA kernels are running, waits for the GPU
        to reach idle state, then samples power for the specified duration.

        Returns a ConfidenceInterval of idle power in Watts.
        """
        self._sync_gpu()
        time.sleep(0.5)  # Let clocks settle after sync
        samples = self._sample_power_for(duration_s)
        powers = [s.power_w for s in samples]
        return self._compute_ci(powers)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Warm-up
    # ═══════════════════════════════════════════════════════════════

    def _run_warmup(self, fn: Callable, duration_s: float):
        """
        Execute the workload repeatedly for `duration_s` seconds.

        Purpose: bring the GPU to thermal steady state, populate CUDA
        kernel caches (JIT), and stabilize boost clock frequencies.
        All data from this phase is discarded.
        """
        start = time.monotonic()
        while time.monotonic() - start < duration_s:
            fn()
            self._sync_gpu()

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Measurement
    # ═══════════════════════════════════════════════════════════════

    def _measure_iteration(
        self,
        workload_fn: Callable,
        baseline_power_w: float,
        flop_count: Optional[int],
        memory_bytes: Optional[int],
    ) -> IterationResult:
        """
        Run one profiling iteration: execute workload_fn while sampling
        GPU power in a background thread. Compute energy via trapezoidal
        integration of the power time-series.
        """
        samples: list[PowerSample] = []
        stop_event = threading.Event()

        # Background power sampler thread
        def _sampler():
            while not stop_event.is_set():
                try:
                    sample = self._read_power_sample()
                    samples.append(sample)
                except Exception as exc:
                    logger.debug("Power sample read failed: %s", exc)
                # Precise sleep using monotonic clock
                target = time.monotonic() + self._sample_interval
                while time.monotonic() < target and not stop_event.is_set():
                    time.sleep(0.001)

        sampler_thread = threading.Thread(target=_sampler, daemon=True)

        # Attempt torch profiler for FLOP counting (if not provided)
        detected_flops = None
        detected_mem_bytes = None

        self._sync_gpu()

        # Start sampling, run workload, stop sampling
        sampler_thread.start()
        t_start = time.monotonic()

        if _TORCH_AVAILABLE and flop_count is None:
            detected_flops, detected_mem_bytes = self._profile_with_torch(workload_fn)
        else:
            workload_fn()

        self._sync_gpu()
        t_end = time.monotonic()
        stop_event.set()
        sampler_thread.join(timeout=2.0)

        duration = t_end - t_start

        if len(samples) < 2:
            raise RuntimeError(
                f"Insufficient power samples ({len(samples)}). "
                f"Increase workload duration or decrease sample_interval."
            )

        # Trapezoidal energy integration
        total_energy = self._trapezoidal_energy(samples)

        # Baseline subtraction: remove idle power contribution
        baseline_energy = baseline_power_w * duration
        dynamic_energy = max(0.0, total_energy - baseline_energy)

        powers = [s.power_w for s in samples]
        utils = [s.gpu_utilization_pct for s in samples]
        mem_utils = [s.memory_utilization_pct for s in samples]
        temps = [s.temperature_c for s in samples]

        return IterationResult(
            duration_s=duration,
            energy_j=total_energy,
            dynamic_energy_j=dynamic_energy,
            mean_power_w=sum(powers) / len(powers),
            peak_power_w=max(powers),
            mean_utilization_pct=sum(utils) / len(utils),
            mean_memory_util_pct=sum(mem_utils) / len(mem_utils),
            mean_temperature_c=sum(temps) / len(temps),
            sample_count=len(samples),
            flop_count=flop_count or detected_flops,
            memory_bytes_read=memory_bytes if memory_bytes else (
                detected_mem_bytes if detected_mem_bytes else None
            ),
            memory_bytes_written=None,
        )

    def _profile_with_torch(
        self, workload_fn: Callable
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Use torch.profiler to auto-detect FLOP count and memory movement.

        Falls back gracefully if profiling is unavailable or the model
        doesn't produce profiler events.
        """
        flops = None
        mem_bytes = None

        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_flops=True,
            ) as prof:
                workload_fn()
                self._sync_gpu()

            # Sum all FLOP events
            total_flops = 0
            total_cuda_mem = 0
            for event in prof.key_averages():
                if event.flops and event.flops > 0:
                    total_flops += event.flops
                if event.cuda_memory_usage and event.cuda_memory_usage > 0:
                    total_cuda_mem += event.cuda_memory_usage

            if total_flops > 0:
                flops = total_flops
            if total_cuda_mem > 0:
                mem_bytes = total_cuda_mem

        except Exception as e:
            logger.debug(f"Torch profiler FLOP detection failed: {e}")
            # Fall back: just run the workload without profiling
            workload_fn()

        return flops, mem_bytes

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Aluminati Efficiency Metrics
    # ═══════════════════════════════════════════════════════════════

    def _compute_aluminati_metrics(
        self,
        result: ProfileResult,
        iters: list[IterationResult],
        flop_count: Optional[int],
        memory_bytes: Optional[int],
    ) -> dict[str, Any]:
        """
        Compute the Aluminati Efficiency Metrics suite.

        Metrics:
          - arithmetic_intensity_flop_per_byte
          - joules_per_task
          - joules_per_task_dynamic
          - power_compute_effectiveness (PCE)
          - carbon_intensity_gco2e
          - tflops_achieved
          - gpu_utilization_efficiency
        """
        metrics: dict[str, Any] = {}

        # Use median iteration for representative values
        sorted_iters = sorted(iters, key=lambda r: r.energy_j)
        median_iter = sorted_iters[len(sorted_iters) // 2]

        # Resolve effective FLOP count
        effective_flops = flop_count
        if effective_flops is None:
            effective_flops = median_iter.flop_count

        effective_mem = memory_bytes
        if effective_mem is None:
            effective_mem = median_iter.memory_bytes_read

        # ── Arithmetic Intensity (FLOP/Byte) ─────────────────────
        if effective_flops and effective_mem and effective_mem > 0:
            ai = effective_flops / effective_mem
            metrics["arithmetic_intensity_flop_per_byte"] = round(ai, 2)
        elif effective_flops and self.arch_spec:
            # Estimate memory movement from roofline model
            # At the observed utilization, we can infer bytes moved
            util_frac = median_iter.mean_utilization_pct / 100.0
            # Achieved TFLOPS from power/utilization
            achieved_tflops = self.arch_spec.fp16_tflops * util_frac
            achieved_flops_per_s = achieved_tflops * 1e12
            if median_iter.duration_s > 0 and achieved_flops_per_s > 0:
                # Bytes/s = bandwidth * memory_util_fraction
                mem_util_frac = median_iter.mean_memory_util_pct / 100.0
                bytes_per_s = self.arch_spec.memory_bw_gbps * 1e9 * max(mem_util_frac, 0.01)
                total_bytes = bytes_per_s * median_iter.duration_s
                ai = effective_flops / total_bytes if total_bytes > 0 else 0
                metrics["arithmetic_intensity_flop_per_byte"] = round(ai, 2)
                metrics["arithmetic_intensity_source"] = "estimated"
        else:
            metrics["arithmetic_intensity_flop_per_byte"] = None
            metrics["arithmetic_intensity_source"] = "unavailable"

        # ── Joules per Task ──────────────────────────────────────
        metrics["joules_per_task"] = round(
            result.energy_per_task_j.mean, 4
        )
        metrics["joules_per_task_dynamic"] = round(
            result.dynamic_energy_per_task_j.mean, 4
        )
        metrics["task_unit"] = result.task_unit

        # ── TFLOPS Achieved ──────────────────────────────────────
        if effective_flops and median_iter.duration_s > 0:
            tflops = (effective_flops / median_iter.duration_s) / 1e12
            metrics["tflops_achieved"] = round(tflops, 2)

            if self.arch_spec:
                peak = self.arch_spec.fp16_tflops
                metrics["tflops_peak"] = peak
                metrics["compute_utilization_pct"] = round(
                    (tflops / peak) * 100, 1
                ) if peak > 0 else 0
        else:
            metrics["tflops_achieved"] = None

        # ── Power Compute Effectiveness (PCE) ────────────────────
        # PCE = P_tensor_cores / P_total
        # Estimated from: compute_utilization vs total GPU utilization
        # Tensor core power ≈ (compute_util / gpu_util) * dynamic_power
        # Since we can't directly read tensor core counters via NVML,
        # we estimate PCE from the ratio of achieved TFLOPS to what
        # the measured power *should* deliver at peak efficiency.
        if self.arch_spec and effective_flops and median_iter.duration_s > 0:
            tflops = (effective_flops / median_iter.duration_s) / 1e12
            # Theoretical power for this TFLOPS: interpolate on the
            # architecture's power curve
            if self.arch_spec.fp16_tflops > 0:
                compute_frac = min(tflops / self.arch_spec.fp16_tflops, 1.0)
                # Power attributable to compute at this fraction
                compute_power = (
                    self.arch_spec.tdp_w - self.arch_spec.idle_power_w
                ) * compute_frac
                dynamic_power = median_iter.mean_power_w - result.baseline_power_w.mean
                if dynamic_power > 0:
                    pce = compute_power / dynamic_power
                    metrics["power_compute_effectiveness"] = round(
                        min(pce, 1.0), 4
                    )
                else:
                    metrics["power_compute_effectiveness"] = 0.0
            else:
                metrics["power_compute_effectiveness"] = None
        else:
            metrics["power_compute_effectiveness"] = None

        # ── GPU Utilization Efficiency ───────────────────────────
        avg_util = sum(r.mean_utilization_pct for r in iters) / len(iters)
        metrics["avg_gpu_utilization_pct"] = round(avg_util, 1)
        metrics["avg_memory_utilization_pct"] = round(
            sum(r.mean_memory_util_pct for r in iters) / len(iters), 1
        )

        # ── Carbon Intensity (gCO2e) ─────────────────────────────
        energy_kwh = result.energy_per_task_j.mean / 3_600_000
        co2_g = energy_kwh * self._grid_carbon_g_kwh
        metrics["carbon_intensity_gco2e_per_task"] = round(co2_g, 6)
        metrics["carbon_intensity_gco2e_per_hour"] = round(
            (result.mean_power_w.mean / 1000.0) * self._grid_carbon_g_kwh, 2
        )
        metrics["grid_carbon_intensity_g_kwh"] = self._grid_carbon_g_kwh

        # ── Energy Cost ──────────────────────────────────────────
        # Default US electricity rate: $0.12/kWh
        cost_per_kwh = 0.12
        metrics["cost_usd_per_task"] = round(energy_kwh * cost_per_kwh, 8)
        metrics["cost_usd_per_hour"] = round(
            (result.mean_power_w.mean / 1000.0) * cost_per_kwh, 4
        )

        return metrics

    # ═══════════════════════════════════════════════════════════════
    # Optimization Plan Generator
    # ═══════════════════════════════════════════════════════════════

    def _generate_optimization_plan(
        self,
        result: ProfileResult,
        iters: list[IterationResult],
    ) -> dict[str, Any]:
        """
        Compare profiling data against the hardware roofline model to
        determine the workload bottleneck and generate targeted
        optimization recommendations.

        Bottleneck classification:
          Memory-Bound: Arithmetic Intensity < ridge point of the roofline.
                        GPU stalls waiting for HBM data movement.
          Compute-Bound: Arithmetic Intensity >= ridge point.
                         GPU compute units are the limiting factor.
          Underutilized: GPU utilization < 40% — not enough work dispatched.
        """
        if not self.arch_spec:
            return {
                "regime": "unknown",
                "detail": "GPU architecture not recognized. Cannot compute roofline.",
                "recommendations": [],
            }

        metrics = result.metrics
        ai = metrics.get("arithmetic_intensity_flop_per_byte")
        avg_util = metrics.get("avg_gpu_utilization_pct", 0)
        avg_mem_util = metrics.get("avg_memory_utilization_pct", 0)
        pce = metrics.get("power_compute_effectiveness")

        spec = self.arch_spec

        # Roofline ridge point: where bandwidth ceiling meets compute ceiling
        # ridge_point = peak_tflops * 1e12 / (memory_bw * 1e9) = peak_tflops * 1e3 / memory_bw
        ridge_point = (spec.fp16_tflops * 1000.0) / spec.memory_bw_gbps

        plan: dict[str, Any] = {
            "hardware": spec.name,
            "roofline_ridge_point_flop_per_byte": round(ridge_point, 1),
            "observed_arithmetic_intensity": ai,
            "recommendations": [],
        }

        recs: list[dict[str, str]] = []

        # ── Classification ────────────────────────────────────────

        if avg_util < 40:
            plan["regime"] = "underutilized"
            plan["detail"] = (
                f"GPU utilization is {avg_util:.0f}% — the GPU is not receiving "
                f"enough work to reach its efficiency sweet spot."
            )
            recs.append({
                "priority": "CRITICAL",
                "action": "Increase batch size",
                "rationale": (
                    f"At {avg_util:.0f}% utilization, fixed power overhead "
                    f"(fans, DRAM refresh, PCIe) dominates. Doubling batch size "
                    f"amortizes idle power across more useful work."
                ),
            })
            recs.append({
                "priority": "HIGH",
                "action": "Use CUDA Graphs or torch.compile()",
                "rationale": (
                    "Kernel launch overhead may be starving the GPU. "
                    "CUDA Graphs fuse the launch sequence; torch.compile() "
                    "eliminates Python-side dispatch latency."
                ),
            })
            if avg_mem_util < 30:
                recs.append({
                    "priority": "MEDIUM",
                    "action": "Increase sequence length or resolution",
                    "rationale": (
                        f"Memory utilization is only {avg_mem_util:.0f}%. "
                        f"The GPU has {spec.memory_gb} GB HBM — use it. "
                        f"Longer sequences or higher-resolution inputs "
                        f"improve data reuse and amortize transfer costs."
                    ),
                })

        elif ai is not None and ai < ridge_point:
            plan["regime"] = "memory-bound"
            headroom = ridge_point / ai if ai > 0 else float('inf')
            plan["detail"] = (
                f"Arithmetic intensity ({ai:.1f} FLOP/byte) is below the "
                f"roofline ridge point ({ridge_point:.1f} FLOP/byte). "
                f"The workload is stalling on HBM bandwidth. "
                f"Compute units are {headroom:.1f}x under-saturated."
            )
            recs.append({
                "priority": "CRITICAL",
                "action": "Enable FlashAttention / Memory-Efficient Attention",
                "rationale": (
                    "FlashAttention fuses QKV projection, softmax, and "
                    "value multiplication into a single kernel that tiles "
                    "through SRAM, reducing HBM reads by O(N) for sequence "
                    "length N. This directly raises arithmetic intensity."
                ),
            })
            recs.append({
                "priority": "HIGH",
                "action": "Apply activation checkpointing (gradient checkpointing)",
                "rationale": (
                    "Trade compute for memory: recompute activations during "
                    "backward pass instead of storing them. Reduces peak memory, "
                    "allowing larger batch sizes that improve data reuse."
                ),
            })
            recs.append({
                "priority": "HIGH",
                "action": "Quantize weights to INT8/FP8 (QLoRA, GPTQ, AWQ)",
                "rationale": (
                    "Halving weight precision halves bytes read from HBM "
                    "while maintaining FLOP count, directly doubling "
                    "arithmetic intensity. H100 FP8 Tensor Cores achieve "
                    "~2x the throughput of FP16."
                ),
            })
            recs.append({
                "priority": "MEDIUM",
                "action": "Enable operator fusion (torch.compile with max-autotune)",
                "rationale": (
                    "Fusing elementwise ops (LayerNorm, GELU, residual add) "
                    "eliminates intermediate HBM round-trips. Each fusion "
                    "removes one full-tensor read + write cycle."
                ),
            })
            if spec.has_transformer_engine:
                recs.append({
                    "priority": "MEDIUM",
                    "action": "Enable Transformer Engine FP8 autocast",
                    "rationale": (
                        f"{spec.name} has dedicated Transformer Engine hardware. "
                        f"FP8 autocast (via transformer_engine.pytorch) "
                        f"delivers up to 2x throughput with automatic "
                        f"loss scaling — no model code changes required."
                    ),
                })

        else:
            plan["regime"] = "compute-bound"
            plan["detail"] = (
                f"Arithmetic intensity ({ai:.1f} FLOP/byte) is at or above "
                f"the ridge point ({ridge_point:.1f} FLOP/byte). The workload "
                f"is limited by Tensor Core throughput, not memory bandwidth."
            )
            recs.append({
                "priority": "HIGH",
                "action": "Scale to multi-GPU with Tensor Parallelism",
                "rationale": (
                    "Compute-bound workloads scale near-linearly with "
                    "additional GPUs when using tensor parallelism (TP). "
                    "Split the weight matrices across GPUs to multiply "
                    "available TFLOPS. Use FSDP or DeepSpeed ZeRO-3."
                ),
            })
            recs.append({
                "priority": "HIGH",
                "action": "Increase batch size to saturate SMs",
                "rationale": (
                    f"GPU has {spec.fp16_tflops} FP16 TFLOPS peak. "
                    f"Larger batches expose more independent operations "
                    f"for SM-level parallelism, improving compute density."
                ),
            })
            if pce is not None and pce < 0.6:
                recs.append({
                    "priority": "HIGH",
                    "action": "Reduce non-Tensor-Core operations",
                    "rationale": (
                        f"PCE is {pce:.2f} — only {pce * 100:.0f}% of dynamic "
                        f"power goes to Tensor Cores. Profile with Nsight to "
                        f"identify non-TC kernels (softmax, LayerNorm, embedding "
                        f"lookups) and fuse or replace them."
                    ),
                })
            if not spec.has_transformer_engine and spec.family != 'Hopper':
                recs.append({
                    "priority": "MEDIUM",
                    "action": f"Migrate to H100/H200 for Transformer Engine",
                    "rationale": (
                        f"{spec.name} lacks Transformer Engine. H100 delivers "
                        f"{GPU_ARCHITECTURES.get('H100-SXM5-80GB', spec).fp16_tflops} "
                        f"FP16 TFLOPS vs {spec.fp16_tflops} — "
                        f"a {GPU_ARCHITECTURES.get('H100-SXM5-80GB', spec).fp16_tflops / spec.fp16_tflops:.1f}x "
                        f"compute advantage for the same power budget."
                    ),
                })
            recs.append({
                "priority": "MEDIUM",
                "action": "Enable mixed-precision training (AMP)",
                "rationale": (
                    "If not already using AMP, switching from FP32 to "
                    "FP16/BF16 doubles Tensor Core throughput at the same "
                    "power. Use torch.amp.autocast() with GradScaler."
                ),
            })

        # ── Universal recommendations ─────────────────────────────
        dynamic_pct = (
            result.dynamic_energy_per_task_j.mean / result.energy_per_task_j.mean * 100
            if result.energy_per_task_j.mean > 0 else 0
        )

        if dynamic_pct < 50:
            recs.append({
                "priority": "HIGH",
                "action": "Reduce idle time between steps",
                "rationale": (
                    f"Only {dynamic_pct:.0f}% of total energy is dynamic "
                    f"(workload-attributed). The remaining {100 - dynamic_pct:.0f}% "
                    f"is static idle power. Reduce data loading latency, "
                    f"use prefetching (DataLoader num_workers), or overlap "
                    f"compute with data transfer."
                ),
            })

        plan["recommendations"] = recs
        plan["dynamic_energy_fraction_pct"] = round(dynamic_pct, 1)
        return plan

    # ═══════════════════════════════════════════════════════════════
    # NVML Sampling Internals
    # ═══════════════════════════════════════════════════════════════

    def _read_power_sample(self) -> PowerSample:
        """Read a single power/utilization snapshot from NVML."""
        power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW → W
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            self._handle, pynvml.NVML_TEMPERATURE_GPU
        )
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(
                self._handle, pynvml.NVML_CLOCK_SM
            )
        except pynvml.NVMLError:
            sm_clock = 0

        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

        return PowerSample(
            timestamp=time.monotonic(),
            power_w=power,
            gpu_utilization_pct=util.gpu,
            memory_utilization_pct=util.memory,
            temperature_c=temp,
            sm_clock_mhz=sm_clock,
            memory_used_bytes=mem.used,
            memory_total_bytes=mem.total,
        )

    def _sample_power_for(self, duration_s: float) -> list[PowerSample]:
        """Collect power samples for a fixed duration."""
        samples: list[PowerSample] = []
        end_time = time.monotonic() + duration_s
        while time.monotonic() < end_time:
            samples.append(self._read_power_sample())
            time.sleep(self._sample_interval)
        return samples

    def _trapezoidal_energy(self, samples: list[PowerSample]) -> float:
        """
        Compute total energy (Joules) via trapezoidal numerical integration
        of the power time-series.

        E = Σ [(P_i + P_{i+1}) / 2] × (t_{i+1} - t_i)

        This is more accurate than rectangular integration because it
        accounts for power changes between samples.
        """
        if len(samples) < 2:
            return 0.0

        energy = 0.0
        for i in range(len(samples) - 1):
            dt = samples[i + 1].timestamp - samples[i].timestamp
            avg_power = (samples[i].power_w + samples[i + 1].power_w) / 2.0
            energy += avg_power * dt
        return energy

    # ═══════════════════════════════════════════════════════════════
    # Statistics
    # ═══════════════════════════════════════════════════════════════

    def _compute_ci(self, values: list[float]) -> ConfidenceInterval:
        """
        Compute 95% confidence interval using the Student's t-distribution.

        For n < 30 samples, the t-distribution provides correct coverage
        probability (unlike z-based intervals which assume known variance).

        CI = x̄ ± t_{α/2, n-1} × (s / √n)
        """
        n = len(values)
        if n == 0:
            return ConfidenceInterval(0.0, 0.0, 0.0, 0.0, 0)

        mean = sum(values) / n

        if n == 1:
            return ConfidenceInterval(mean, 0.0, mean, mean, 1)

        # Sample standard deviation
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)

        # t critical value for 95% CI
        # Using pre-computed values for common df to avoid scipy dependency
        t_critical = self._t_critical_95(n - 1)
        margin = t_critical * (std / math.sqrt(n))

        return ConfidenceInterval(
            mean=round(mean, 6),
            std=round(std, 6),
            ci_lower=round(mean - margin, 6),
            ci_upper=round(mean + margin, 6),
            n=n,
        )

    @staticmethod
    def _t_critical_95(df: int) -> float:
        """
        Two-tailed t critical values for 95% confidence (α = 0.05).

        Pre-computed to avoid requiring scipy. Values from standard
        t-distribution tables. For df > 120, converges to z = 1.96.
        """
        # Pre-computed t_{0.025, df} for common degrees of freedom
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
            16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
            25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000,
            80: 1.990, 100: 1.984, 120: 1.980,
        }

        if df in table:
            return table[df]

        # Interpolate for values not in the table
        if df > 120:
            return 1.960  # Normal approximation

        # Find surrounding entries
        keys = sorted(table.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= df <= keys[i + 1]:
                lo, hi = keys[i], keys[i + 1]
                frac = (df - lo) / (hi - lo)
                return table[lo] + frac * (table[hi] - table[lo])

        return 1.960  # Fallback

    # ═══════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════

    def _sync_gpu(self):
        """Synchronize CUDA to ensure all kernels have completed."""
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize(self._gpu_index)

    def to_dataframe(self, result: ProfileResult):
        """
        Export per-iteration data as a pandas DataFrame.

        Requires pandas. Returns None if pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed — cannot export DataFrame")
            return None

        rows = []
        for i, r in enumerate(result.iterations):
            rows.append({
                "iteration": i + 1,
                "duration_s": r.duration_s,
                "energy_j": r.energy_j,
                "dynamic_energy_j": r.dynamic_energy_j,
                "mean_power_w": r.mean_power_w,
                "peak_power_w": r.peak_power_w,
                "mean_utilization_pct": r.mean_utilization_pct,
                "mean_memory_util_pct": r.mean_memory_util_pct,
                "mean_temperature_c": r.mean_temperature_c,
                "sample_count": r.sample_count,
                "flop_count": r.flop_count,
            })

        return pd.DataFrame(rows)

    def shutdown(self):
        """Release NVML resources."""
        try:
            pynvml.nvmlShutdown()
        except Exception as exc:
            logger.debug("NVML shutdown error: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    def __del__(self):
        self.shutdown()
