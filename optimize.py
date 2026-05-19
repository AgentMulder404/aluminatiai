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
aluminatiai optimize — GPU energy efficiency recommendations.

Samples live GPU metrics, analyzes workload characteristics against the
roofline model, and outputs ranked suggestions to reduce energy cost.

Usage:
    aluminatiai optimize [--gpu N] [--duration SECONDS] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import pynvml  # type: ignore[import-untyped]
except ImportError:
    pynvml = None  # type: ignore[assignment]

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _rich = True
except ImportError:
    _rich = False

from efficiency.gpu_specs import ArchSpec, GPU_ARCHITECTURES, resolve_arch
from efficiency.curve_builder import EfficiencyCurveBuilder
from efficiency.carbon import ElectricityMapsClient, CarbonIntensity


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Recommendation:
    priority: str               # "P1", "P2", "P3"
    category: str               # utilization, precision, gpu_match, power_cap, idle, thermal, memory
    description: str
    estimated_savings_pct: float
    action: str
    detail: str = ""


@dataclass
class OptimizeResult:
    gpu_name: str
    gpu_index: int
    arch_spec: Optional[str]    # arch name or None
    duration_s: float

    # Observed metrics
    avg_power_w: float
    avg_util_pct: float
    avg_mem_util_pct: float
    avg_temp_c: float
    idle_fraction: float        # fraction of samples with util < 5%
    power_limit_w: float

    # Detected workload characteristics
    detected_precision: Optional[str] = None
    is_memory_bound: Optional[bool] = None

    # Carbon
    carbon_intensity_gco2e: Optional[float] = None
    carbon_zone: Optional[str] = None
    estimated_co2_g_per_hour: Optional[float] = None
    renewable_pct: Optional[float] = None
    is_green_hour: Optional[bool] = None

    # Recommendations
    recommendations: list[Recommendation] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Workload Analyzer
# ═══════════════════════════════════════════════════════════════════════════════


class WorkloadAnalyzer:
    """
    Stateless analyzer that takes GPU samples and produces ranked
    efficiency recommendations.
    """

    def __init__(self, arch_spec: Optional[ArchSpec] = None, carbon_client: Optional[ElectricityMapsClient] = None):
        self.arch = arch_spec
        self.carbon = carbon_client

    def analyze(
        self,
        samples: list[dict],
        gpu_name: str,
        gpu_index: int,
        duration_s: float,
    ) -> OptimizeResult:
        """
        Analyze a list of GPU metric samples and produce recommendations.

        Each sample dict should have keys: power_draw_w, utilization_gpu_pct,
        utilization_memory_pct, temperature_c, power_limit_w, and optionally
        processes (list of dicts with cmdline).
        """
        if not samples:
            return OptimizeResult(
                gpu_name=gpu_name, gpu_index=gpu_index,
                arch_spec=self.arch.name if self.arch else None,
                duration_s=duration_s, avg_power_w=0, avg_util_pct=0,
                avg_mem_util_pct=0, avg_temp_c=0, idle_fraction=1.0,
                power_limit_w=0, recommendations=[],
            )

        # Aggregate metrics
        avg_power = sum(s["power_draw_w"] for s in samples) / len(samples)
        avg_util = sum(s["utilization_gpu_pct"] for s in samples) / len(samples)
        avg_mem_util = sum(s["utilization_memory_pct"] for s in samples) / len(samples)
        avg_temp = sum(s["temperature_c"] for s in samples) / len(samples)
        power_limit = samples[0].get("power_limit_w", 0)
        idle_frac = sum(1 for s in samples if s["utilization_gpu_pct"] < 5) / len(samples)

        # Detect workload characteristics
        precision = self._detect_precision(samples)
        is_mem_bound = self._classify_memory_bound(avg_util, avg_mem_util)

        # Carbon intensity
        carbon_info: Optional[CarbonIntensity] = None
        if self.carbon:
            try:
                carbon_info = self.carbon.get_current()
            except Exception:
                pass

        # Run all detectors
        recs: list[Recommendation] = []
        for check in [
            self._check_low_utilization,
            self._check_memory_bound,
            self._check_precision,
            self._check_idle_time,
            self._check_thermal_throttling,
            self._check_gpu_match,
            self._check_power_cap,
            self._check_carbon,
        ]:
            rec = check(
                avg_power=avg_power, avg_util=avg_util, avg_mem_util=avg_mem_util,
                avg_temp=avg_temp, power_limit=power_limit, idle_frac=idle_frac,
                precision=precision, is_mem_bound=is_mem_bound,
                carbon_info=carbon_info,
            )
            if rec is not None:
                recs.append(rec)

        # Sort: P1 first, then by estimated savings descending
        priority_order = {"P1": 0, "P2": 1, "P3": 2}
        recs.sort(key=lambda r: (priority_order.get(r.priority, 9), -r.estimated_savings_pct))

        # Carbon fields
        co2_g_per_hr: Optional[float] = None
        if carbon_info and avg_power > 0:
            energy_kwh_per_hr = avg_power / 1000.0
            co2_g_per_hr = round(energy_kwh_per_hr * carbon_info.carbon_intensity_gco2e, 2)

        return OptimizeResult(
            gpu_name=gpu_name,
            gpu_index=gpu_index,
            arch_spec=self.arch.name if self.arch else None,
            duration_s=duration_s,
            avg_power_w=round(avg_power, 1),
            avg_util_pct=round(avg_util, 1),
            avg_mem_util_pct=round(avg_mem_util, 1),
            avg_temp_c=round(avg_temp, 1),
            idle_fraction=round(idle_frac, 3),
            power_limit_w=power_limit,
            detected_precision=precision,
            is_memory_bound=is_mem_bound,
            carbon_intensity_gco2e=carbon_info.carbon_intensity_gco2e if carbon_info else None,
            carbon_zone=carbon_info.zone if carbon_info else None,
            estimated_co2_g_per_hour=co2_g_per_hr,
            renewable_pct=carbon_info.renewable_pct if carbon_info else None,
            is_green_hour=carbon_info.renewable_pct >= 50.0 if carbon_info else None,
            recommendations=recs,
        )

    # ── Workload classification ───────────────────────────────────────────

    def _detect_precision(self, samples: list[dict]) -> Optional[str]:
        """Scan process cmdlines for precision hints."""
        for s in samples:
            for proc in s.get("processes") or []:
                cmdline = (proc.get("cmdline") or "").lower()
                if "bf16" in cmdline or "bfloat16" in cmdline:
                    return "bf16"
                if "fp16" in cmdline or "float16" in cmdline or "half" in cmdline:
                    return "fp16"
                if "int8" in cmdline or "int4" in cmdline:
                    return "int8"
                if "fp32" in cmdline or "float32" in cmdline:
                    return "fp32"
        return None

    def _classify_memory_bound(self, avg_util: float, avg_mem_util: float) -> Optional[bool]:
        """Estimate if workload is memory-bandwidth-bound."""
        if avg_util < 5:
            return None  # GPU idle, can't classify
        # High memory utilization relative to compute utilization suggests memory-bound
        if avg_mem_util > 0 and avg_util > 0:
            ratio = avg_mem_util / avg_util
            return ratio > 1.5
        return None

    # ── Detector methods ──────────────────────────────────────────────────
    # Each detector receives all aggregated metrics as kwargs and returns
    # a Recommendation or None.

    def _check_low_utilization(self, *, avg_util, idle_frac, **_) -> Optional[Recommendation]:
        if avg_util >= 40:
            return None
        # At low utilization, idle power dominates. Estimate savings from
        # doubling utilization (halving wall-clock time).
        idle_power_frac = 0.30  # ~30% of TDP is idle overhead
        if self.arch:
            idle_power_frac = self.arch.idle_power_w / self.arch.tdp_w
        # If util is 20%, dynamic power is ~20% of TDP. Doubling batch size
        # roughly halves wall-clock, saving ~idle_frac of total energy.
        savings = min(50, round(idle_power_frac * (1 - avg_util / 100) * 100))
        return Recommendation(
            priority="P1",
            category="utilization",
            description=f"GPU utilization is only {avg_util:.0f}% — increase batch size or workload concurrency",
            estimated_savings_pct=savings,
            action="Increase batch size 2-4x or run multiple data-parallel workers",
            detail=f"At {avg_util:.0f}% utilization, ~{idle_power_frac*100:.0f}% of power draw is idle overhead. "
                   f"Doubling utilization saves ~{savings}% energy per unit of work.",
        )

    def _check_memory_bound(self, *, is_mem_bound, avg_util, avg_mem_util, **_) -> Optional[Recommendation]:
        if not is_mem_bound or not self.arch:
            return None
        # Only flag if on a compute-heavy GPU (fp16 > 300 TFLOPS)
        if self.arch.fp16_tflops < 300:
            return None
        return Recommendation(
            priority="P2",
            category="memory",
            description=f"Workload is memory-bandwidth-bound (mem_util {avg_mem_util:.0f}% vs gpu_util {avg_util:.0f}%)",
            estimated_savings_pct=20,
            action=f"Consider a bandwidth-optimized GPU or data-parallel split across more GPUs",
            detail=f"The {self.arch.name} has {self.arch.fp16_tflops} FP16 TFLOPS but your workload "
                   f"is bottlenecked on {self.arch.memory_bw_gbps} GB/s memory bandwidth. "
                   f"A GPU with higher memory bandwidth per dollar may be more cost-effective.",
        )

    def _check_precision(self, *, precision, avg_util, **_) -> Optional[Recommendation]:
        if precision != "fp32" or not self.arch:
            return None
        # Check if GPU has tensor cores (fp16 >> fp32)
        if self.arch.fp16_tflops < self.arch.fp32_tflops * 1.5:
            return None
        speedup = self.arch.fp16_tflops / self.arch.fp32_tflops
        savings = min(60, round((1 - 1 / speedup) * 100))
        best_precision = "BF16" if self.arch.bf16_tflops > 0 else "FP16"
        return Recommendation(
            priority="P1",
            category="precision",
            description=f"FP32 detected on tensor-core GPU — switch to {best_precision} for {speedup:.1f}x throughput",
            estimated_savings_pct=savings,
            action=f"Enable {best_precision} mixed precision (torch.autocast or --bf16 flag)",
            detail=f"The {self.arch.name} delivers {self.arch.fp16_tflops} TFLOPS at FP16 vs "
                   f"{self.arch.fp32_tflops} TFLOPS at FP32. Switching to {best_precision} can "
                   f"cut energy per training step by ~{savings}% with minimal accuracy loss.",
        )

    def _check_idle_time(self, *, idle_frac, **_) -> Optional[Recommendation]:
        if idle_frac <= 0.20:
            return None
        pct = round(idle_frac * 100)
        priority = "P1" if idle_frac > 0.50 else "P2"
        return Recommendation(
            priority=priority,
            category="idle",
            description=f"GPU idle {pct}% of the time — configure auto-shutdown or job scheduling",
            estimated_savings_pct=min(pct, 80),
            action="Enable auto-shutdown after idle timeout or batch jobs into continuous runs",
            detail=f"During the {pct}% idle time, the GPU still draws ~30% of TDP as idle power. "
                   f"Shutting down idle GPUs or packing jobs tighter eliminates this waste.",
        )

    def _check_thermal_throttling(self, *, avg_power, power_limit, avg_util, avg_temp, **_) -> Optional[Recommendation]:
        if power_limit <= 0 or avg_util >= 60:
            return None
        if avg_power < power_limit * 0.90:
            return None
        return Recommendation(
            priority="P2",
            category="thermal",
            description=f"Power at {avg_power:.0f}W ({avg_power/power_limit*100:.0f}% of limit) but utilization only {avg_util:.0f}% — possible thermal throttling",
            estimated_savings_pct=10,
            action="Check GPU cooling, airflow, and thermal paste. Consider reducing power limit.",
            detail=f"High power draw ({avg_power:.0f}W) with low utilization ({avg_util:.0f}%) at "
                   f"{avg_temp:.0f}C suggests the GPU is thermally limited, wasting energy without "
                   f"proportional compute throughput.",
        )

    def _check_gpu_match(self, *, avg_util, is_mem_bound, **_) -> Optional[Recommendation]:
        if not self.arch or avg_util < 10:
            return None
        try:
            builder = EfficiencyCurveBuilder()
            ranked = builder.compare_architectures(utilization_pct=max(10, int(avg_util)))
            if not ranked:
                return None

            # Find current GPU rank
            current_rank = None
            best = ranked[0]
            for i, entry in enumerate(ranked):
                if entry["arch_name"] == self.arch.name:
                    current_rank = i
                    break

            if current_rank is None or current_rank == 0:
                return None  # Already the best or not in the list

            # How much more efficient is the best option?
            current_entry = ranked[current_rank]
            savings = round(100 - current_entry.get("relative_efficiency", 100))
            if savings < 10:
                return None  # Not worth switching

            return Recommendation(
                priority="P2",
                category="gpu_match",
                description=f"The {best['arch_name']} is {savings}% more energy-efficient for this workload",
                estimated_savings_pct=savings,
                action=f"Consider migrating to {best['arch_name']} ({best.get('family', '')}) for this workload profile",
                detail=f"At {avg_util:.0f}% utilization, {best['arch_name']} achieves "
                       f"{best['joules_per_tflop']:.1f} J/TFLOP vs {current_entry['joules_per_tflop']:.1f} J/TFLOP "
                       f"for {self.arch.name}.",
            )
        except Exception:
            return None

    def _check_power_cap(self, *, avg_power, avg_util, power_limit, **_) -> Optional[Recommendation]:
        if not self.arch or avg_util < 30 or power_limit <= 0:
            return None
        # Estimate: capping to 70% of TDP typically loses <5% throughput
        # but saves ~20% energy on compute-bound workloads.
        cap_target = round(self.arch.tdp_w * 0.70)
        if avg_power <= cap_target:
            return None  # Already drawing less than the proposed cap

        energy_savings = round((1 - cap_target / avg_power) * 100)
        if energy_savings < 10:
            return None

        return Recommendation(
            priority="P3",
            category="power_cap",
            description=f"Power capping to {cap_target}W could save ~{energy_savings}% energy with <5% throughput loss",
            estimated_savings_pct=energy_savings,
            action=f"Set power limit: nvidia-smi -i {0} -pl {cap_target}",
            detail=f"Current draw is {avg_power:.0f}W. The efficiency sweet spot for "
                   f"{self.arch.name} is typically 65-75% of TDP ({self.arch.tdp_w:.0f}W). "
                   f"Power capping exploits the non-linear power-performance curve.",
        )

    def _check_carbon(self, *, avg_power, carbon_info, idle_frac, **_) -> Optional[Recommendation]:
        if not carbon_info or carbon_info.is_fallback:
            return None

        # Calculate CO2 rate
        energy_kwh_per_hr = avg_power / 1000.0
        co2_g_per_hr = energy_kwh_per_hr * carbon_info.carbon_intensity_gco2e

        # High carbon intensity — suggest deferring non-urgent work
        if carbon_info.carbon_intensity_gco2e > 500:
            return Recommendation(
                priority="P2",
                category="carbon",
                description=f"Grid carbon intensity is high ({carbon_info.carbon_intensity_gco2e:.0f} gCO2e/kWh, "
                            f"{carbon_info.renewable_pct:.0f}% renewable) — defer batch jobs if possible",
                estimated_savings_pct=round(min(40, (carbon_info.carbon_intensity_gco2e - 200) / 10)),
                action="Schedule non-urgent training jobs during off-peak / high-renewable hours",
                detail=f"Current grid ({carbon_info.zone}) emits {carbon_info.carbon_intensity_gco2e:.0f} gCO2e/kWh. "
                       f"This GPU is producing ~{co2_g_per_hr:.0f}g CO2/hr. "
                       f"Running during greener hours (>50% renewable) could cut emissions by 30-50%.",
            )

        # Low carbon — positive signal, no action needed
        if carbon_info.renewable_pct >= 70:
            return Recommendation(
                priority="P3",
                category="carbon",
                description=f"Grid is {carbon_info.renewable_pct:.0f}% renewable ({carbon_info.carbon_intensity_gco2e:.0f} gCO2e/kWh) — good time to run compute-heavy jobs",
                estimated_savings_pct=0,
                action="Current grid mix is clean — no carbon-related changes needed",
                detail=f"Zone {carbon_info.zone} is running {carbon_info.renewable_pct:.0f}% renewable. "
                       f"This GPU produces ~{co2_g_per_hr:.0f}g CO2/hr at current intensity.",
            )

        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Live GPU Sampling
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_samples(gpu_index: int, duration_s: int) -> tuple[str, list[dict]]:
    """
    Collect GPU metric samples at ~1Hz for the specified duration.

    Returns (gpu_name, samples_list).
    """
    if pynvml is None:
        print("ERROR: pynvml is required. Install with: pip install pynvml")
        sys.exit(1)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    samples = []
    end_time = time.monotonic() + duration_s

    while time.monotonic() < end_time:
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

            # Attempt to get process list
            processes = []
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    processes.append({"pid": p.pid, "used_gpu_memory": p.usedGpuMemory})
            except pynvml.NVMLError:
                pass

            samples.append({
                "power_draw_w": power,
                "utilization_gpu_pct": util.gpu,
                "utilization_memory_pct": util.memory,
                "temperature_c": temp,
                "power_limit_w": power_limit,
                "processes": processes,
            })
        except pynvml.NVMLError:
            pass

        time.sleep(1.0)

    return gpu_name, samples


# ═══════════════════════════════════════════════════════════════════════════════
#  Output Rendering
# ═══════════════════════════════════════════════════════════════════════════════


def _print_rich(result: OptimizeResult) -> None:
    console = Console()

    # Header
    console.print()
    console.print(f"[bold]GPU Efficiency Analysis[/bold] — {result.gpu_name}", style="cyan")
    if result.arch_spec:
        console.print(f"  Architecture: {result.arch_spec}", style="dim")
    console.print(f"  Sampled for {result.duration_s:.0f}s | "
                  f"Avg power: {result.avg_power_w:.0f}W | "
                  f"Avg util: {result.avg_util_pct:.0f}% | "
                  f"Idle: {result.idle_fraction*100:.0f}%", style="dim")
    if result.detected_precision:
        console.print(f"  Detected precision: {result.detected_precision.upper()}", style="dim")
    if result.carbon_zone and result.carbon_intensity_gco2e is not None:
        green = "[green]" if result.is_green_hour else "[yellow]"
        console.print(f"  Carbon: {result.carbon_intensity_gco2e:.0f} gCO2e/kWh "
                      f"({result.carbon_zone}, {result.renewable_pct:.0f}% renewable) | "
                      f"~{result.estimated_co2_g_per_hour:.0f}g CO2/hr | "
                      f"{green}{'Green hour' if result.is_green_hour else 'Not green hour'}[/]", style="dim")
    console.print()

    if not result.recommendations:
        console.print("[green]No efficiency issues detected. GPU workload looks well-optimized.[/green]")
        console.print()
        return

    # Recommendations table
    table = Table(title=f"{len(result.recommendations)} Recommendation(s)")
    table.add_column("Priority", style="bold", width=4)
    table.add_column("Category", width=12)
    table.add_column("Recommendation", min_width=40)
    table.add_column("Est. Savings", justify="right", width=10)
    table.add_column("Action", min_width=30)

    priority_colors = {"P1": "red", "P2": "yellow", "P3": "blue"}

    for rec in result.recommendations:
        color = priority_colors.get(rec.priority, "white")
        table.add_row(
            Text(rec.priority, style=color),
            rec.category,
            rec.description,
            f"{rec.estimated_savings_pct}%",
            rec.action,
        )

    console.print(table)
    console.print()

    # Detail section
    for rec in result.recommendations:
        if rec.detail:
            console.print(f"  [{priority_colors.get(rec.priority, 'white')}]{rec.priority}[/] {rec.category}: {rec.detail}", style="dim")
    console.print()


def _print_plain(result: OptimizeResult) -> None:
    print()
    print(f"GPU Efficiency Analysis — {result.gpu_name}")
    if result.arch_spec:
        print(f"  Architecture: {result.arch_spec}")
    print(f"  Sampled for {result.duration_s:.0f}s | "
          f"Avg power: {result.avg_power_w:.0f}W | "
          f"Avg util: {result.avg_util_pct:.0f}% | "
          f"Idle: {result.idle_fraction*100:.0f}%")
    if result.detected_precision:
        print(f"  Detected precision: {result.detected_precision.upper()}")
    if result.carbon_zone and result.carbon_intensity_gco2e is not None:
        status = "Green hour" if result.is_green_hour else "Not green hour"
        print(f"  Carbon: {result.carbon_intensity_gco2e:.0f} gCO2e/kWh "
              f"({result.carbon_zone}, {result.renewable_pct:.0f}% renewable) | "
              f"~{result.estimated_co2_g_per_hour:.0f}g CO2/hr | {status}")
    print()

    if not result.recommendations:
        print("  No efficiency issues detected. GPU workload looks well-optimized.")
        print()
        return

    print(f"  {len(result.recommendations)} Recommendation(s):")
    print(f"  {'Priority':<8} {'Category':<14} {'Est. Savings':<12} Description")
    print(f"  {'-'*8:<8} {'-'*14:<14} {'-'*12:<12} {'-'*40}")
    for rec in result.recommendations:
        print(f"  {rec.priority:<8} {rec.category:<14} {rec.estimated_savings_pct:>10}%  {rec.description}")
    print()
    for rec in result.recommendations:
        if rec.detail:
            print(f"  [{rec.priority}] {rec.detail}")
    print()


def _print_json(result: OptimizeResult) -> None:
    d = asdict(result)
    print(json.dumps(d, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aluminatiai optimize",
        description="Analyze GPU workload and suggest energy efficiency improvements.",
    )
    p.add_argument(
        "--gpu", type=int, default=0, metavar="N",
        help="GPU index to analyze (default: 0)",
    )
    p.add_argument(
        "--duration", type=int, default=30, metavar="S",
        help="Sampling duration in seconds (default: 30)",
    )
    p.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output results as JSON",
    )
    return p


def run_optimize(args: argparse.Namespace) -> int:
    """Entry point for the optimize subcommand."""
    gpu_index = args.gpu
    duration = max(5, args.duration)

    if not args.json_output:
        print(f"Sampling GPU {gpu_index} for {duration}s...")

    gpu_name, samples = _collect_samples(gpu_index, duration)

    if not samples:
        print("ERROR: No GPU metrics collected. Is the GPU accessible?")
        return 1

    arch = resolve_arch(gpu_name)
    carbon_client = ElectricityMapsClient()  # uses ALUMINATAI_GRID_ZONE from env
    analyzer = WorkloadAnalyzer(arch_spec=arch, carbon_client=carbon_client)
    result = analyzer.analyze(samples, gpu_name, gpu_index, float(duration))

    if args.json_output:
        _print_json(result)
    elif _rich:
        _print_rich(result)
    else:
        _print_plain(result)

    return 0
