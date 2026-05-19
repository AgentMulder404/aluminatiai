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
aluminatiai ab — A/B energy efficiency experiment runner.

Runs two workload configurations (or the same workload at two power limits),
profiles both with confidence intervals, and outputs a diff report proving
whether the optimization reduced energy.

Usage:
    aluminatiai ab --baseline "CMD" --optimized "CMD" [--duration 120] [--iterations 3]
    aluminatiai ab --powercap --baseline-watts 400 --optimized-watts 250 --workload "CMD"
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import threading
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

from efficiency.profiler import ConfidenceInterval
from efficiency.stats import compute_ci, trapezoidal_energy
from efficiency.gpu_specs import resolve_arch, ArchSpec
from efficiency.power_control import set_power_limit, get_power_limit
from efficiency.carbon import ElectricityMapsClient


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PhaseMetrics:
    """Metrics for one A/B phase with confidence intervals."""
    label: str
    energy_j: ConfidenceInterval
    mean_power_w: ConfidenceInterval
    duration_s: ConfidenceInterval
    throughput: Optional[ConfidenceInterval] = None
    peak_power_w: float = 0.0
    avg_util_pct: float = 0.0
    avg_temp_c: float = 0.0
    sample_count: int = 0


@dataclass
class ABResult:
    """Complete A/B experiment result."""
    gpu_name: str
    gpu_index: int
    arch_spec: Optional[str] = None

    idle_baseline_w: float = 0.0
    warmup_duration_s: float = 0.0

    baseline: Optional[PhaseMetrics] = None
    optimized: Optional[PhaseMetrics] = None

    # Deltas
    energy_savings_pct: float = 0.0
    throughput_change_pct: Optional[float] = None
    cost_savings_usd_per_hour: float = 0.0
    aem: float = 0.0  # Aluminati Efficiency Multiplier

    # Statistical significance
    energy_significant: bool = False
    throughput_significant: bool = False

    # Carbon
    carbon_zone: str = ""
    carbon_intensity_gco2e: float = 0.0
    baseline_co2_g: float = 0.0
    optimized_co2_g: float = 0.0
    co2_savings_g: float = 0.0

    recommendation: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Power Sampler (background thread)
# ═══════════════════════════════════════════════════════════════════════════════


class _PowerSampler:
    """Background thread that samples GPU power at ~10Hz."""

    def __init__(self, gpu_index: int, interval: float = 0.1):
        self._gpu_index = gpu_index
        self._interval = interval
        self._samples: list[tuple[float, float, int, int]] = []  # (time, power, util, temp)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, float, int, int]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return list(self._samples)

    def _run(self) -> None:
        if pynvml is None:
            return
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        while not self._stop.is_set():
            try:
                t = time.monotonic()
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                self._samples.append((t, power, util.gpu, temp))
            except Exception:
                pass
            self._stop.wait(timeout=self._interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  Throughput Parser
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns to detect throughput in command output
_THROUGHPUT_PATTERNS = [
    re.compile(r"([\d.]+)\s*(?:tok(?:en)?s?/s)", re.IGNORECASE),
    re.compile(r"([\d.]+)\s*(?:it/s)", re.IGNORECASE),
    re.compile(r"([\d.]+)\s*(?:img/s|images?/s)", re.IGNORECASE),
    re.compile(r"([\d.]+)\s*(?:samples?/s)", re.IGNORECASE),
    re.compile(r"([\d.]+)\s*(?:steps?/s)", re.IGNORECASE),
]


def _parse_throughput(output: str) -> Optional[float]:
    """Try to extract a throughput value from command output."""
    for pattern in _THROUGHPUT_PATTERNS:
        match = pattern.search(output)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  A/B Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════════


class ABExperimentRunner:
    """
    Runs structured A/B energy experiments with confidence intervals.
    """

    # Default electricity cost for savings estimates
    ELECTRICITY_COST_KWH = 0.12  # $/kWh average US

    def __init__(self, gpu_index: int = 0, sample_interval: float = 0.1):
        self._gpu_index = gpu_index
        self._sample_interval = sample_interval
        self._gpu_name = ""
        self._arch: Optional[ArchSpec] = None

        if pynvml is not None:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            name = pynvml.nvmlDeviceGetName(handle)
            self._gpu_name = name.decode() if isinstance(name, bytes) else name
            self._arch = resolve_arch(self._gpu_name)

    def run(
        self,
        baseline_cmd: str,
        optimized_cmd: str,
        duration_s: int = 120,
        iterations: int = 3,
        warmup_s: int = 30,
        cooldown_s: int = 10,
    ) -> ABResult:
        """Run an A/B experiment comparing two commands."""
        result = ABResult(
            gpu_name=self._gpu_name,
            gpu_index=self._gpu_index,
            arch_spec=self._arch.name if self._arch else None,
            warmup_duration_s=warmup_s,
        )

        # Phase 0: Idle baseline
        print(f"  [Phase 0] Measuring idle baseline ({5}s)...")
        idle_samples = self._sample_idle(5)
        result.idle_baseline_w = sum(s[1] for s in idle_samples) / max(len(idle_samples), 1)

        # Phase 1: Warmup (discarded)
        if warmup_s > 0:
            print(f"  [Phase 1] Warmup with baseline command ({warmup_s}s, discarded)...")
            self._run_command_timed(baseline_cmd, warmup_s)

        # Phase 2: Baseline measurements
        print(f"  [Phase 2] Baseline: {iterations} iterations x {duration_s}s...")
        result.baseline = self._run_command_phase("Baseline", baseline_cmd, duration_s, iterations)

        # Cooldown
        if cooldown_s > 0:
            print(f"  [Cooldown] {cooldown_s}s...")
            time.sleep(cooldown_s)

        # Phase 3: Optimized measurements
        print(f"  [Phase 3] Optimized: {iterations} iterations x {duration_s}s...")
        result.optimized = self._run_command_phase("Optimized", optimized_cmd, duration_s, iterations)

        # Phase 4: Compute deltas
        self._compute_deltas(result)
        return result

    def run_powercap(
        self,
        workload_cmd: str,
        baseline_watts: int,
        optimized_watts: int,
        duration_s: int = 60,
        iterations: int = 3,
        warmup_s: int = 30,
    ) -> ABResult:
        """Run an A/B experiment with power capping."""
        result = ABResult(
            gpu_name=self._gpu_name,
            gpu_index=self._gpu_index,
            arch_spec=self._arch.name if self._arch else None,
            warmup_duration_s=warmup_s,
        )

        # Phase 0: Idle baseline
        print(f"  [Phase 0] Measuring idle baseline...")
        idle_samples = self._sample_idle(5)
        result.idle_baseline_w = sum(s[1] for s in idle_samples) / max(len(idle_samples), 1)

        # Phase 1: Warmup at baseline power
        set_power_limit(self._gpu_index, baseline_watts)
        if warmup_s > 0:
            print(f"  [Phase 1] Warmup at {baseline_watts}W ({warmup_s}s, discarded)...")
            self._run_command_timed(workload_cmd, warmup_s)

        # Phase 2: Baseline at original power limit
        print(f"  [Phase 2] Baseline at {baseline_watts}W: {iterations} x {duration_s}s...")
        set_power_limit(self._gpu_index, baseline_watts)
        result.baseline = self._run_command_phase(
            f"Baseline ({baseline_watts}W)", workload_cmd, duration_s, iterations,
        )

        # Phase 3: Optimized at reduced power limit
        print(f"  [Phase 3] Optimized at {optimized_watts}W: {iterations} x {duration_s}s...")
        set_power_limit(self._gpu_index, optimized_watts)
        result.optimized = self._run_command_phase(
            f"Optimized ({optimized_watts}W)", workload_cmd, duration_s, iterations,
        )

        # Restore original power limit
        set_power_limit(self._gpu_index, baseline_watts)

        # Compute deltas
        self._compute_deltas(result)
        return result

    # ── Internal methods ──────────────────────────────────────────────────

    def _sample_idle(self, duration: float) -> list[tuple[float, float, int, int]]:
        """Sample idle GPU power for baseline measurement."""
        sampler = _PowerSampler(self._gpu_index, self._sample_interval)
        sampler.start()
        time.sleep(duration)
        return sampler.stop()

    def _run_command_timed(self, cmd: str, duration_s: int) -> None:
        """Run a command for a fixed duration, then kill it."""
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            proc.wait(timeout=duration_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _run_command_phase(
        self,
        label: str,
        cmd: str,
        duration_s: int,
        iterations: int,
    ) -> PhaseMetrics:
        """Run a command for K iterations, collecting power samples."""
        iter_energies: list[float] = []
        iter_powers: list[float] = []
        iter_durations: list[float] = []
        iter_throughputs: list[float] = []
        all_peak = 0.0
        all_util: list[float] = []
        all_temp: list[float] = []
        total_samples = 0

        for i in range(iterations):
            print(f"    Iteration {i+1}/{iterations}...", end=" ", flush=True)

            sampler = _PowerSampler(self._gpu_index, self._sample_interval)
            sampler.start()

            t0 = time.monotonic()
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                stdout_bytes, _ = proc.communicate(timeout=duration_s)
                stdout_text = stdout_bytes.decode(errors="replace")
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_text = ""
            elapsed = time.monotonic() - t0

            raw_samples = sampler.stop()

            if raw_samples:
                timestamps = [s[0] for s in raw_samples]
                powers = [s[1] for s in raw_samples]
                energy = trapezoidal_energy(timestamps, powers)
                avg_pwr = sum(powers) / len(powers)
                peak = max(powers)
                avg_u = sum(s[2] for s in raw_samples) / len(raw_samples)
                avg_t = sum(s[3] for s in raw_samples) / len(raw_samples)

                iter_energies.append(energy)
                iter_powers.append(avg_pwr)
                iter_durations.append(elapsed)
                all_peak = max(all_peak, peak)
                all_util.append(avg_u)
                all_temp.append(avg_t)
                total_samples += len(raw_samples)

                # Throughput detection
                tp = _parse_throughput(stdout_text)
                if tp is not None:
                    iter_throughputs.append(tp)

                print(f"{energy:.0f}J, {avg_pwr:.0f}W avg, {elapsed:.1f}s")
            else:
                print("(no samples)")

        # Build CIs
        throughput_ci = compute_ci(iter_throughputs) if iter_throughputs else None

        return PhaseMetrics(
            label=label,
            energy_j=compute_ci(iter_energies),
            mean_power_w=compute_ci(iter_powers),
            duration_s=compute_ci(iter_durations),
            throughput=throughput_ci,
            peak_power_w=all_peak,
            avg_util_pct=round(sum(all_util) / max(len(all_util), 1), 1),
            avg_temp_c=round(sum(all_temp) / max(len(all_temp), 1), 1),
            sample_count=total_samples,
        )

    def _compute_deltas(self, result: ABResult) -> None:
        """Compute savings, AEM, significance, and recommendation."""
        if not result.baseline or not result.optimized:
            return

        b = result.baseline
        o = result.optimized

        # Energy savings
        if b.energy_j.mean > 0:
            result.energy_savings_pct = round(
                (1 - o.energy_j.mean / b.energy_j.mean) * 100, 1
            )

        # Throughput change
        if b.throughput and o.throughput and b.throughput.mean > 0:
            result.throughput_change_pct = round(
                (o.throughput.mean / b.throughput.mean - 1) * 100, 1
            )

        # Cost savings ($/hr)
        energy_delta_j = b.energy_j.mean - o.energy_j.mean
        if b.duration_s.mean > 0:
            energy_delta_kwh_per_hr = (energy_delta_j / b.duration_s.mean) * 3600 / 3_600_000
            result.cost_savings_usd_per_hour = round(
                energy_delta_kwh_per_hr * self.ELECTRICITY_COST_KWH, 4
            )

        # AEM: % energy saved / % performance lost
        throughput_loss = 0.0
        if result.throughput_change_pct is not None and result.throughput_change_pct < 0:
            throughput_loss = abs(result.throughput_change_pct)

        if throughput_loss > 0 and result.energy_savings_pct > 0:
            result.aem = round(result.energy_savings_pct / throughput_loss, 1)
        elif result.energy_savings_pct > 0:
            result.aem = float("inf")

        # Statistical significance (non-overlapping CIs)
        result.energy_significant = check_significance(b.energy_j, o.energy_j)
        if b.throughput and o.throughput:
            result.throughput_significant = check_significance(b.throughput, o.throughput)

        # Carbon estimation
        try:
            carbon_client = ElectricityMapsClient()
            current = carbon_client.get_current()
            result.carbon_zone = current.zone
            result.carbon_intensity_gco2e = current.carbon_intensity_gco2e
            # Convert J to kWh (1 kWh = 3,600,000 J)
            baseline_kwh = b.energy_j.mean / 3_600_000
            optimized_kwh = o.energy_j.mean / 3_600_000
            result.baseline_co2_g = round(baseline_kwh * current.carbon_intensity_gco2e, 2)
            result.optimized_co2_g = round(optimized_kwh * current.carbon_intensity_gco2e, 2)
            result.co2_savings_g = round(result.baseline_co2_g - result.optimized_co2_g, 2)
        except Exception:
            pass  # Carbon is best-effort

        # Recommendation text
        if result.energy_savings_pct > 5 and result.energy_significant:
            if result.throughput_change_pct is not None and result.throughput_change_pct < -5:
                result.recommendation = (
                    f"The optimized config saves {result.energy_savings_pct:.1f}% energy but "
                    f"loses {abs(result.throughput_change_pct):.1f}% throughput. "
                    f"AEM = {result.aem}x (>1 means the energy gain outweighs the perf loss)."
                )
            else:
                result.recommendation = (
                    f"The optimized config saves {result.energy_savings_pct:.1f}% energy "
                    f"with minimal throughput impact. Recommended."
                )
        elif not result.energy_significant:
            result.recommendation = (
                "Energy difference is not statistically significant (95% CIs overlap). "
                "Consider running more iterations or longer durations."
            )
        else:
            result.recommendation = "No meaningful energy savings detected."


def check_significance(a: ConfidenceInterval, b: ConfidenceInterval) -> bool:
    """Check if two CIs are statistically significantly different (non-overlapping)."""
    return a.ci_upper < b.ci_lower or b.ci_upper < a.ci_lower


# ═══════════════════════════════════════════════════════════════════════════════
#  Output Rendering
# ═══════════════════════════════════════════════════════════════════════════════


def _ci_str(ci: ConfidenceInterval) -> str:
    """Format a CI for display."""
    return f"{ci.mean:.1f} [{ci.ci_lower:.1f}, {ci.ci_upper:.1f}]"


def _print_rich(result: ABResult) -> None:
    console = Console()
    console.print()
    console.print("[bold cyan]A/B Energy Experiment Report[/bold cyan]")
    console.print(f"  GPU: {result.gpu_name}" + (f" ({result.arch_spec})" if result.arch_spec else ""))
    console.print(f"  Idle baseline: {result.idle_baseline_w:.1f}W")
    console.print()

    if not result.baseline or not result.optimized:
        console.print("[red]Experiment incomplete.[/red]")
        return

    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column(result.baseline.label, justify="right")
    table.add_column(result.optimized.label, justify="right")
    table.add_column("Delta", justify="right")

    # Energy
    e_delta = f"{result.energy_savings_pct:+.1f}% savings"
    sig = " *" if result.energy_significant else ""
    table.add_row(
        "Energy (J)",
        _ci_str(result.baseline.energy_j),
        _ci_str(result.optimized.energy_j),
        Text(e_delta + sig, style="green" if result.energy_savings_pct > 0 else "red"),
    )

    # Power
    table.add_row(
        "Avg Power (W)",
        _ci_str(result.baseline.mean_power_w),
        _ci_str(result.optimized.mean_power_w),
        "",
    )

    # Duration
    table.add_row(
        "Duration (s)",
        _ci_str(result.baseline.duration_s),
        _ci_str(result.optimized.duration_s),
        "",
    )

    # Throughput
    if result.baseline.throughput and result.optimized.throughput:
        tp_delta = f"{result.throughput_change_pct:+.1f}%" if result.throughput_change_pct is not None else ""
        table.add_row(
            "Throughput",
            _ci_str(result.baseline.throughput),
            _ci_str(result.optimized.throughput),
            Text(tp_delta, style="green" if (result.throughput_change_pct or 0) >= 0 else "yellow"),
        )

    table.add_row("Peak Power (W)", f"{result.baseline.peak_power_w:.0f}",
                   f"{result.optimized.peak_power_w:.0f}", "")
    table.add_row("Avg Util (%)", f"{result.baseline.avg_util_pct:.0f}",
                   f"{result.optimized.avg_util_pct:.0f}", "")
    table.add_row("Avg Temp (C)", f"{result.baseline.avg_temp_c:.0f}",
                   f"{result.optimized.avg_temp_c:.0f}", "")

    console.print(table)
    console.print()

    # AEM
    if result.aem > 0 and result.aem != float("inf"):
        console.print(f"  [bold]Aluminati Efficiency Multiplier (AEM):[/bold] {result.aem}x")
    elif result.aem == float("inf"):
        console.print(f"  [bold]Aluminati Efficiency Multiplier (AEM):[/bold] inf (no throughput loss)")
    console.print(f"  [bold]Cost savings:[/bold] ${result.cost_savings_usd_per_hour:.4f}/hr")

    if result.carbon_zone:
        console.print(f"  [bold]Carbon zone:[/bold] {result.carbon_zone} ({result.carbon_intensity_gco2e:.0f} gCO2e/kWh)")
        console.print(f"  [bold]CO2:[/bold] Baseline {result.baseline_co2_g:.2f}g → Optimized {result.optimized_co2_g:.2f}g (saved {result.co2_savings_g:.2f}g)")
    console.print()

    # Recommendation
    style = "green" if "Recommended" in result.recommendation else "yellow"
    console.print(f"  [bold]{result.recommendation}[/bold]", style=style)
    if not result.energy_significant:
        console.print("  * = statistically significant (95% CI)", style="dim")
    console.print()


def _print_plain(result: ABResult) -> None:
    print()
    print("A/B Energy Experiment Report")
    print(f"  GPU: {result.gpu_name}" + (f" ({result.arch_spec})" if result.arch_spec else ""))
    print(f"  Idle baseline: {result.idle_baseline_w:.1f}W")
    print()

    if not result.baseline or not result.optimized:
        print("  Experiment incomplete.")
        return

    print(f"  {'Metric':<20} {'Baseline':>30} {'Optimized':>30} {'Delta':>20}")
    print(f"  {'-'*20} {'-'*30} {'-'*30} {'-'*20}")

    sig = " *" if result.energy_significant else ""
    print(f"  {'Energy (J)':<20} {_ci_str(result.baseline.energy_j):>30} "
          f"{_ci_str(result.optimized.energy_j):>30} {result.energy_savings_pct:>+.1f}% savings{sig}")
    print(f"  {'Avg Power (W)':<20} {_ci_str(result.baseline.mean_power_w):>30} "
          f"{_ci_str(result.optimized.mean_power_w):>30}")
    print(f"  {'Duration (s)':<20} {_ci_str(result.baseline.duration_s):>30} "
          f"{_ci_str(result.optimized.duration_s):>30}")

    if result.baseline.throughput and result.optimized.throughput:
        tp = f"{result.throughput_change_pct:+.1f}%" if result.throughput_change_pct is not None else ""
        print(f"  {'Throughput':<20} {_ci_str(result.baseline.throughput):>30} "
              f"{_ci_str(result.optimized.throughput):>30} {tp:>20}")

    print()
    if result.aem > 0:
        aem_str = f"{result.aem}x" if result.aem != float("inf") else "inf"
        print(f"  AEM: {aem_str}")
    print(f"  Cost savings: ${result.cost_savings_usd_per_hour:.4f}/hr")

    if result.carbon_zone:
        print(f"  Carbon zone: {result.carbon_zone} ({result.carbon_intensity_gco2e:.0f} gCO2e/kWh)")
        print(f"  CO2: Baseline {result.baseline_co2_g:.2f}g -> Optimized {result.optimized_co2_g:.2f}g (saved {result.co2_savings_g:.2f}g)")
    print()
    print(f"  {result.recommendation}")
    print()


def _print_json(result: ABResult) -> None:
    d = asdict(result)
    # Handle inf in AEM
    if d.get("aem") == float("inf"):
        d["aem"] = "inf"
    print(json.dumps(d, indent=2))


def _export_csv(result: ABResult, path: str) -> None:
    """Export per-phase iteration data to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "energy_j_mean", "energy_j_ci_lower", "energy_j_ci_upper",
            "power_w_mean", "duration_s_mean", "peak_power_w",
            "avg_util_pct", "avg_temp_c", "sample_count",
        ])
        for phase in [result.baseline, result.optimized]:
            if phase:
                writer.writerow([
                    phase.label,
                    f"{phase.energy_j.mean:.2f}",
                    f"{phase.energy_j.ci_lower:.2f}",
                    f"{phase.energy_j.ci_upper:.2f}",
                    f"{phase.mean_power_w.mean:.2f}",
                    f"{phase.duration_s.mean:.2f}",
                    f"{phase.peak_power_w:.2f}",
                    f"{phase.avg_util_pct:.1f}",
                    f"{phase.avg_temp_c:.1f}",
                    phase.sample_count,
                ])
    print(f"  CSV exported to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aluminatiai ab",
        description="A/B energy efficiency experiment runner.",
    )

    # Command mode
    p.add_argument("--baseline", type=str, metavar="CMD",
                   help="Baseline command to run")
    p.add_argument("--optimized", type=str, metavar="CMD",
                   help="Optimized command to run")

    # Power-cap mode
    p.add_argument("--powercap", action="store_true",
                   help="Enable power-cap mode (same workload, two power limits)")
    p.add_argument("--baseline-watts", type=int, metavar="W",
                   help="Power limit for baseline phase (power-cap mode)")
    p.add_argument("--optimized-watts", type=int, metavar="W",
                   help="Power limit for optimized phase (power-cap mode)")
    p.add_argument("--workload", type=str, metavar="CMD",
                   help="Workload command (power-cap mode)")

    # Shared
    p.add_argument("--gpu", type=int, default=0, metavar="N",
                   help="GPU index (default: 0)")
    p.add_argument("--duration", type=int, default=120, metavar="S",
                   help="Duration per iteration in seconds (default: 120)")
    p.add_argument("--iterations", type=int, default=3, metavar="N",
                   help="Measurement iterations (default: 3, minimum: 3)")
    p.add_argument("--warmup", type=int, default=30, metavar="S",
                   help="Warmup duration in seconds (default: 30)")
    p.add_argument("--cooldown", type=int, default=10, metavar="S",
                   help="Cooldown between phases in seconds (default: 10)")

    # Output
    p.add_argument("--json", action="store_true", dest="json_output",
                   help="Output results as JSON")
    p.add_argument("--csv", type=str, metavar="PATH", dest="csv_path",
                   help="Export results to CSV file")

    return p


def run_ab(args: argparse.Namespace) -> int:
    """Entry point for the ab subcommand."""
    iterations = max(3, args.iterations)

    if args.powercap:
        if not args.workload or not args.baseline_watts or not args.optimized_watts:
            print("ERROR: --powercap requires --workload, --baseline-watts, and --optimized-watts")
            return 1
        print(f"Running power-cap A/B experiment on GPU {args.gpu}")
        print(f"  Workload: {args.workload}")
        print(f"  Baseline: {args.baseline_watts}W | Optimized: {args.optimized_watts}W")
        print(f"  {iterations} iterations x {args.duration}s")
        print()

        runner = ABExperimentRunner(gpu_index=args.gpu)
        result = runner.run_powercap(
            workload_cmd=args.workload,
            baseline_watts=args.baseline_watts,
            optimized_watts=args.optimized_watts,
            duration_s=args.duration,
            iterations=iterations,
            warmup_s=args.warmup,
        )
    else:
        if not args.baseline or not args.optimized:
            print("ERROR: Provide --baseline and --optimized commands, or use --powercap mode")
            return 1
        print(f"Running A/B experiment on GPU {args.gpu}")
        print(f"  Baseline:  {args.baseline}")
        print(f"  Optimized: {args.optimized}")
        print(f"  {iterations} iterations x {args.duration}s")
        print()

        runner = ABExperimentRunner(gpu_index=args.gpu)
        result = runner.run(
            baseline_cmd=args.baseline,
            optimized_cmd=args.optimized,
            duration_s=args.duration,
            iterations=iterations,
            warmup_s=args.warmup,
            cooldown_s=args.cooldown,
        )

    # Output
    if args.json_output:
        _print_json(result)
    elif _rich:
        _print_rich(result)
    else:
        _print_plain(result)

    if args.csv_path:
        _export_csv(result, args.csv_path)

    return 0
