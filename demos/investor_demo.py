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
aluminatiai demo — Investor demo suite.

Runs 5 compelling GPU energy efficiency demos that prove real savings
with real numbers. Each demo completes in under 5 minutes.

Usage:
    aluminatiai demo --demo all --quick
    aluminatiai demo --demo 1 --gpu 0
    aluminatiai demo --demo 3 --full
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    _rich = True
except ImportError:
    _rich = False

# Resolve paths
_DEMOS_DIR = Path(__file__).parent
_WORKLOADS_DIR = _DEMOS_DIR / "workloads"

# ── Lazy imports (avoid loading NVML at import time) ─────────────────────────

def _get_runner(gpu_index: int):
    from ab import ABExperimentRunner
    return ABExperimentRunner(gpu_index=gpu_index)


def _get_analyzer():
    from optimize import WorkloadAnalyzer, _collect_samples
    return WorkloadAnalyzer, _collect_samples


def _get_carbon_client():
    from efficiency.carbon import ElectricityMapsClient
    return ElectricityMapsClient()


def _get_power_control():
    from efficiency.power_control import get_default_power_limit, set_power_limit
    return get_default_power_limit, set_power_limit


# ── Cloud cost lookup ────────────────────────────────────────────────────────

# Approximate spot/on-demand rates ($/hr) for common cloud GPUs
CLOUD_RATES_PER_HOUR: dict[str, float] = {
    "H100": 3.99, "H200": 5.49, "A100": 1.89,
    "RTX 4090": 0.59, "RTX 3090": 0.44, "L40S": 1.14,
    "L40": 0.89, "A10G": 0.50, "T4": 0.20, "V100": 0.80,
}


def _lookup_cloud_rate(gpu_name: str) -> float | None:
    """Match GPU name to cloud hourly rate via substring."""
    for key, rate in CLOUD_RATES_PER_HOUR.items():
        if key in gpu_name:
            return rate
    return None


# ── Console helpers ──────────────────────────────────────────────────────────

def _banner(num: int, total: int, title: str) -> None:
    if _rich:
        console = Console()
        console.print()
        console.print(Rule(style="cyan"))
        console.print(Panel(
            f"[bold white]Demo {num}/{total}[/bold white]\n[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        ))
    else:
        print()
        print("=" * 60)
        print(f"  Demo {num}/{total}: {title}")
        print("=" * 60)
        print()


def _success(msg: str) -> None:
    if _rich:
        Console().print(f"  [bold green]OK[/bold green] {msg}")
    else:
        print(f"  [OK] {msg}")


def _bold(msg: str) -> None:
    if _rich:
        Console().print(f"  [bold yellow]{msg}[/bold yellow]")
    else:
        print(f"  >>> {msg}")


# ── Demo 1: Power Cap A/B ───────────────────────────────────────────────────

def demo_1_powercap(gpu: int, duration: int, iterations: int, warmup: int) -> dict | None:
    """'Free Money' power cap test — same workload, lower TDP."""
    _banner(1, 5, '"Free Money" Power Cap A/B')

    get_default, set_limit = _get_power_control()
    try:
        default_w = get_default(gpu)
    except Exception:
        default_w = 450  # RTX 4090 fallback

    # Pre-flight: check if power capping is available
    can_cap = set_limit(gpu, default_w, quiet=True)
    if not can_cap:
        if _rich:
            Console().print(Panel(
                "[bold yellow]Power capping not available on this platform[/bold yellow]\n"
                "Cloud containers typically block GPU power limit changes.\n"
                "This demo requires bare-metal or on-prem GPU access.\n"
                "Skipping Demo 1.",
                border_style="yellow",
            ))
        else:
            print("  Power capping not available on this platform (cloud container).")
            print("  Skipping Demo 1.")
        return None

    capped_w = int(default_w * 0.70)

    print(f"  Default TDP: {default_w}W")
    print(f"  Capped TDP:  {capped_w}W (70%)")
    print()

    workload_cmd = (
        f"{sys.executable} {_WORKLOADS_DIR / 'matmul_workload.py'} "
        f"--duration {duration} --dtype bf16 --gpu {gpu}"
    )

    runner = _get_runner(gpu)
    result = runner.run_powercap(
        workload_cmd=workload_cmd,
        baseline_watts=default_w,
        optimized_watts=capped_w,
        duration_s=duration,
        iterations=iterations,
        warmup_s=warmup,
    )

    # Print result using ab.py's renderer
    from ab import _print_rich, _print_plain
    if _rich:
        _print_rich(result)
    else:
        _print_plain(result)

    _success(f"Energy savings: {result.energy_savings_pct:.1f}%")
    return asdict(result)


# ── Demo 2: FP32 vs BF16 ────────────────────────────────────────────────────

def demo_2_precision(gpu: int, duration: int, iterations: int, warmup: int) -> dict | None:
    """FP32 vs BF16 head-to-head — same model, different precision."""
    _banner(2, 5, "FP32 vs BF16 Head-to-Head")

    baseline_cmd = (
        f"{sys.executable} {_WORKLOADS_DIR / 'resnet_workload.py'} "
        f"--duration {duration} --dtype fp32 --batch-size 128 --gpu {gpu}"
    )
    optimized_cmd = (
        f"{sys.executable} {_WORKLOADS_DIR / 'resnet_workload.py'} "
        f"--duration {duration} --dtype bf16 --batch-size 128 --gpu {gpu}"
    )

    runner = _get_runner(gpu)
    result = runner.run(
        baseline_cmd=baseline_cmd,
        optimized_cmd=optimized_cmd,
        duration_s=duration,
        iterations=iterations,
        warmup_s=warmup,
        cooldown_s=5,
    )

    from ab import _print_rich, _print_plain
    if _rich:
        _print_rich(result)
    else:
        _print_plain(result)

    _success(f"Energy savings: {result.energy_savings_pct:.1f}%")
    if result.throughput_change_pct is not None:
        _success(f"Throughput change: {result.throughput_change_pct:+.1f}%")
    return asdict(result)


# ── Demo 3: Idle GPU Waste ───────────────────────────────────────────────────

def demo_3_idle_waste(gpu: int, sample_duration: int = 15) -> dict | None:
    """Show how much money an idle GPU burns."""
    _banner(3, 5, "Idle GPU Waste Detection")

    # Launch idle loader in background
    idle_cmd = (
        f"{sys.executable} {_WORKLOADS_DIR / 'idle_loader.py'} "
        f"--duration 120 --gpu {gpu}"
    )
    print("  Launching idle workload (model loaded, no work)...")
    proc = subprocess.Popen(
        idle_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(5)  # Let model load

    # Collect samples using optimize's collector
    WorkloadAnalyzer, collect_samples = _get_analyzer()

    print(f"  Sampling GPU for {sample_duration}s...")
    try:
        gpu_name, samples = collect_samples(gpu, sample_duration)
    except Exception as e:
        print(f"  WARNING: Could not collect samples: {e}")
        proc.kill()
        proc.wait()
        return None

    # Kill idle loader
    proc.kill()
    proc.wait()

    if not samples:
        print("  No samples collected.")
        return None

    # Analyze
    from efficiency.gpu_specs import resolve_arch
    arch = resolve_arch(gpu_name)
    analyzer = WorkloadAnalyzer(arch_spec=arch)
    result = analyzer.analyze(samples, gpu_name, gpu, float(sample_duration))

    # Calculate waste
    avg_power = result.avg_power_w
    idle_fraction = result.idle_fraction
    monthly_waste_kwh = (avg_power / 1000.0) * 720  # kWh per month
    electricity_waste_usd = monthly_waste_kwh * 0.12

    # Cloud instance cost (the real number)
    cloud_rate = _lookup_cloud_rate(gpu_name)
    cloud_monthly = cloud_rate * 720 if cloud_rate else None

    from optimize import _print_rich as opt_rich, _print_plain as opt_plain
    if _rich:
        opt_rich(result)
        console = Console()
        console.print()

        waste_lines = []
        if cloud_monthly:
            waste_lines.append(
                f"[bold red]This GPU is burning ${cloud_monthly:,.0f}/month in cloud costs doing nothing[/bold red]"
            )
            waste_lines.append(
                f"Cloud instance: ${cloud_rate:.2f}/hr = [bold]${cloud_monthly:,.0f}/month[/bold] | "
                f"Electricity: ${electricity_waste_usd:.2f}/month"
            )
        else:
            waste_lines.append(
                f"[bold red]This GPU is burning ${electricity_waste_usd:.2f}/month doing nothing[/bold red]"
            )
        waste_lines.append(
            f"Idle fraction: {idle_fraction*100:.0f}% | "
            f"Avg power: {avg_power:.0f}W | "
            f"Monthly waste: {monthly_waste_kwh:.0f} kWh"
        )
        console.print(Panel("\n".join(waste_lines), border_style="red"))
    else:
        opt_plain(result)
        if cloud_monthly:
            print(f"\n  >>> This GPU is burning ${cloud_monthly:,.0f}/month in cloud costs doing nothing")
            print(f"  Cloud: ${cloud_rate:.2f}/hr = ${cloud_monthly:,.0f}/mo | Electricity: ${electricity_waste_usd:.2f}/mo")
        else:
            print(f"\n  >>> This GPU is burning ${electricity_waste_usd:.2f}/month doing nothing")
        print(f"  Idle: {idle_fraction*100:.0f}% | Power: {avg_power:.0f}W | Waste: {monthly_waste_kwh:.0f} kWh/mo")

    return asdict(result)


# ── Demo 4: Carbon Cost ─────────────────────────────────────────────────────

def demo_4_carbon(gpu: int, duration: int, iterations: int, warmup: int) -> dict | None:
    """Show CO2 impact of inefficient vs optimized training."""
    _banner(4, 5, "Carbon Cost of Training")

    baseline_cmd = (
        f"{sys.executable} {_WORKLOADS_DIR / 'resnet_workload.py'} "
        f"--duration {duration} --dtype fp32 --batch-size 128 --gpu {gpu}"
    )
    optimized_cmd = (
        f"{sys.executable} {_WORKLOADS_DIR / 'resnet_workload.py'} "
        f"--duration {duration} --dtype bf16 --batch-size 128 --gpu {gpu}"
    )

    print("  Inefficient:  FP32, batch=128")
    print("  Optimized:    BF16, batch=128")
    print()

    runner = _get_runner(gpu)
    result = runner.run(
        baseline_cmd=baseline_cmd,
        optimized_cmd=optimized_cmd,
        duration_s=duration,
        iterations=iterations,
        warmup_s=warmup,
        cooldown_s=5,
    )

    from ab import _print_rich, _print_plain
    if _rich:
        _print_rich(result)
    else:
        _print_plain(result)

    if result.co2_savings_g > 0:
        _bold(f"CO2 saved per run: {result.co2_savings_g:.2f}g")
    _success(f"Energy savings: {result.energy_savings_pct:.1f}%")
    return asdict(result)


# ── Demo 5: Scaling Projections ──────────────────────────────────────────────

def demo_5_scaling(results: list[dict]) -> None:
    """Fleet-scale projections from previous demo results."""
    _banner(5, 5, "Fleet-Scale Savings Projections")

    from demos.scaling_report import run_from_dict

    demo_labels = {0: "Power Cap", 1: "FP32 vs BF16", 2: "Carbon (Inefficient vs Optimized)"}

    for i, result in enumerate(results):
        if result is None:
            continue
        label = demo_labels.get(i, f"Demo {i+1}")
        if _rich:
            Console().print(f"\n  [bold]Scaling: {label}[/bold]")
        else:
            print(f"\n  --- Scaling: {label} ---")
        run_from_dict(result)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_demo(args: argparse.Namespace) -> int:
    """Main entry point for the demo subcommand."""
    gpu = args.gpu

    if args.full:
        duration = 120
        iterations = 3
        warmup = 30
    else:
        # Quick mode (default)
        duration = 30
        iterations = 1
        warmup = 10

    mode = "FULL" if args.full else "QUICK"

    if _rich:
        console = Console()
        console.print()
        console.print(Panel(
            f"[bold white]AluminatAI Investor Demo Suite[/bold white]\n"
            f"Mode: [bold cyan]{mode}[/bold cyan] | GPU: {gpu} | "
            f"Duration: {duration}s | Iterations: {iterations}",
            border_style="bold cyan",
            padding=(1, 4),
        ))
    else:
        print()
        print("=" * 60)
        print(f"  AluminatAI Investor Demo Suite")
        print(f"  Mode: {mode} | GPU: {gpu} | {duration}s x {iterations} iterations")
        print("=" * 60)

    demo_choice = args.demo
    ab_results: list[dict] = []

    try:
        if demo_choice in ("all", "1"):
            r = demo_1_powercap(gpu, duration, iterations, warmup)
            ab_results.append(r)

        if demo_choice in ("all", "2"):
            r = demo_2_precision(gpu, duration, iterations, warmup)
            ab_results.append(r)

        if demo_choice in ("all", "3"):
            demo_3_idle_waste(gpu)

        if demo_choice in ("all", "4"):
            r = demo_4_carbon(gpu, duration, iterations, warmup)
            ab_results.append(r)

        if demo_choice in ("all", "5"):
            # Load from file if provided
            if hasattr(args, "result_file") and args.result_file:
                try:
                    with open(args.result_file) as f:
                        ab_results = [json.load(f)]
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"  ERROR: Could not load result file: {e}")
                    return 1

            if not ab_results:
                print("  Demo 5 requires results from Demos 1, 2, or 4.")
                print("  Run --demo all first, or provide --result-file PATH.")
            else:
                demo_5_scaling(ab_results)

    except KeyboardInterrupt:
        print("\n  Demo interrupted.")
        return 1

    if _rich:
        Console().print(Rule("Demo Complete", style="green"))
    else:
        print("\n" + "=" * 60)
        print("  Demo Complete")
        print("=" * 60)

    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aluminatiai demo",
        description="Investor demo suite — 5 GPU energy efficiency demos.",
    )
    p.add_argument("--demo", type=str, default="all",
                   choices=["all", "1", "2", "3", "4", "5"],
                   help="Which demo to run (default: all)")
    p.add_argument("--gpu", type=int, default=0, metavar="N",
                   help="GPU index (default: 0)")
    p.add_argument("--quick", action="store_true", default=True,
                   help="Quick mode: 30s, 1 iteration (default)")
    p.add_argument("--full", action="store_true", default=False,
                   help="Full mode: 120s, 3 iterations")
    p.add_argument("--result-file", type=str, metavar="PATH",
                   help="Path to ABResult JSON file (for Demo 5 standalone)")
    return p
