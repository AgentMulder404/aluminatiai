"""
aluminatiai recommend — find the best GPU for your workload.

Ranks GPUs by energy efficiency (J/TFLOP), cost, and CO2 impact for a
given workload archetype or benchmark profile.

Usage:
    aluminatiai recommend --workload llm-inference
    aluminatiai recommend --workload llm-training --gpu "RTX 4090"
    aluminatiai recommend --workload llm-inference --budget 2.0
    aluminatiai recommend --from-benchmark results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional

from efficiency.gpu_specs import (
    GPU_ARCHITECTURES,
    MODEL_PROFILES,
    WORKLOAD_ARCHETYPES,
    ModelProfile,
    resolve_arch,
)
from efficiency.hardware_match import HardwareMatchScorer
from efficiency.cloud_detect import GPU_HOURLY_RATES

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _rich = True
except ImportError:
    _rich = False

HOURS_PER_MONTH = 730
US_AVG_CO2_G_PER_KWH = 394.0


@dataclass
class GPURecommendation:
    rank: int
    gpu: str
    family: str
    score: float
    joules_per_tflop: float
    cost_per_hr: float
    cost_per_month: float
    kwh_per_month: float
    co2_per_month_kg: float
    vs_current_energy_pct: float
    vs_current_cost_pct: float
    recommendation: str


def recommend_gpus(
    workload: str,
    current_gpu: Optional[str] = None,
    top_n: int = 10,
    max_cost_per_hr: Optional[float] = None,
) -> list[GPURecommendation]:
    profile = WORKLOAD_ARCHETYPES.get(workload) or MODEL_PROFILES.get(workload)
    if not profile:
        return []

    if profile.tag not in MODEL_PROFILES:
        MODEL_PROFILES[profile.tag] = profile

    scorer = HardwareMatchScorer()
    rankings = scorer.rank_gpus_for_model(profile.tag)
    if not rankings:
        return []

    current_jpt: Optional[float] = None
    current_cost: Optional[float] = None
    if current_gpu:
        for r in rankings:
            if r["arch_name"] == current_gpu or current_gpu in r["arch_name"]:
                current_jpt = r["joules_per_tflop"]
                current_cost = GPU_HOURLY_RATES.get(r["arch_name"])
                break

    results: list[GPURecommendation] = []
    for r in rankings:
        arch_name = r["arch_name"]
        spec = GPU_ARCHITECTURES.get(arch_name)
        if not spec:
            continue

        hourly = GPU_HOURLY_RATES.get(arch_name)
        if hourly is None:
            continue

        if max_cost_per_hr is not None and hourly > max_cost_per_hr:
            continue

        util_frac = profile.typical_util_mid
        power_est = spec.tdp_w * (0.3 + 0.7 * util_frac)
        kwh_month = (power_est * HOURS_PER_MONTH) / 1000
        co2_month = (kwh_month * US_AVG_CO2_G_PER_KWH) / 1000
        cost_month = hourly * HOURS_PER_MONTH

        vs_energy = 0.0
        if current_jpt and current_jpt > 0:
            vs_energy = round((1.0 - r["joules_per_tflop"] / current_jpt) * 100, 1)

        vs_cost = 0.0
        if current_cost and current_cost > 0:
            vs_cost = round((1.0 - hourly / current_cost) * 100, 1)

        results.append(GPURecommendation(
            rank=0,
            gpu=arch_name,
            family=r.get("family", spec.family),
            score=r["score"],
            joules_per_tflop=round(r["joules_per_tflop"], 4),
            cost_per_hr=hourly,
            cost_per_month=round(cost_month, 2),
            kwh_per_month=round(kwh_month, 1),
            co2_per_month_kg=round(co2_month, 2),
            vs_current_energy_pct=vs_energy,
            vs_current_cost_pct=vs_cost,
            recommendation=r.get("recommendation", ""),
        ))

    results.sort(key=lambda x: x.joules_per_tflop)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results[:top_n]


def _print_rich(
    results: list[GPURecommendation],
    workload: str,
    current_gpu: Optional[str],
) -> None:
    console = Console()

    title = f"GPU Recommendations for '{workload}'"
    if current_gpu:
        title += f" (vs {current_gpu})"

    table = Table(title=title, show_lines=False, border_style="dim")
    table.add_column("#", style="dim", width=3)
    table.add_column("GPU", style="bold white")
    table.add_column("Family", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("J/TFLOP", justify="right", style="cyan")
    table.add_column("$/hr", justify="right")
    table.add_column("$/mo", justify="right")
    table.add_column("kWh/mo", justify="right", style="green")
    table.add_column("CO2/mo", justify="right")
    if current_gpu:
        table.add_column("Energy", justify="right")
        table.add_column("Cost", justify="right")

    for r in results:
        score_style = "green" if r.score >= 80 else "yellow" if r.score >= 50 else "red"
        energy_txt = ""
        cost_txt = ""
        if current_gpu:
            if r.vs_current_energy_pct > 0:
                energy_txt = f"[green]-{r.vs_current_energy_pct:.0f}%[/]"
            elif r.vs_current_energy_pct < 0:
                energy_txt = f"[red]+{abs(r.vs_current_energy_pct):.0f}%[/]"
            else:
                energy_txt = "same"
            if r.vs_current_cost_pct > 0:
                cost_txt = f"[green]-{r.vs_current_cost_pct:.0f}%[/]"
            elif r.vs_current_cost_pct < 0:
                cost_txt = f"[red]+{abs(r.vs_current_cost_pct):.0f}%[/]"
            else:
                cost_txt = "same"

        row = [
            str(r.rank),
            r.gpu,
            r.family,
            f"[{score_style}]{r.score:.0f}[/]",
            f"{r.joules_per_tflop:.3f}",
            f"${r.cost_per_hr:.2f}",
            f"${r.cost_per_month:,.0f}",
            f"{r.kwh_per_month:.0f}",
            f"{r.co2_per_month_kg:.1f}kg",
        ]
        if current_gpu:
            row.extend([energy_txt, cost_txt])
        table.add_row(*row)

    console.print()
    console.print(table)

    if results:
        best = results[0]
        console.print(
            f"\n  [bold green]Best pick:[/] {best.gpu} — "
            f"{best.joules_per_tflop:.3f} J/TFLOP, "
            f"${best.cost_per_hr:.2f}/hr, "
            f"{best.co2_per_month_kg:.1f} kgCO2/mo"
        )
        if current_gpu and best.vs_current_energy_pct > 0:
            console.print(
                f"  [bold]Switching saves:[/] "
                f"{best.vs_current_energy_pct:.0f}% energy, "
                f"${abs(best.cost_per_month - (results[-1].cost_per_month if len(results) > 1 else best.cost_per_month)):,.0f}/mo"
            )
    console.print()


def _print_plain(
    results: list[GPURecommendation],
    workload: str,
    current_gpu: Optional[str],
) -> None:
    header = f"GPU Recommendations for '{workload}'"
    if current_gpu:
        header += f" (vs {current_gpu})"
    print(f"\n{header}")
    print("-" * len(header))

    fmt = "{:<3} {:<26} {:<14} {:>5} {:>8} {:>6} {:>8} {:>7} {:>8}"
    print(fmt.format("#", "GPU", "Family", "Score", "J/TFLOP", "$/hr", "$/mo", "kWh/mo", "CO2/mo"))
    print(fmt.format("-" * 3, "-" * 26, "-" * 14, "-" * 5, "-" * 8, "-" * 6, "-" * 8, "-" * 7, "-" * 8))

    for r in results:
        print(fmt.format(
            r.rank, r.gpu, r.family,
            f"{r.score:.0f}", f"{r.joules_per_tflop:.3f}",
            f"${r.cost_per_hr:.2f}", f"${r.cost_per_month:,.0f}",
            f"{r.kwh_per_month:.0f}", f"{r.co2_per_month_kg:.1f}kg",
        ))

    if results:
        best = results[0]
        print(f"\nBest: {best.gpu} — {best.joules_per_tflop:.3f} J/TFLOP, ${best.cost_per_hr:.2f}/hr")
    print()


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aluminatiai recommend",
        description="Find the best GPU for your workload by energy efficiency and cost.",
    )
    p.add_argument(
        "--workload", "-w",
        choices=list(WORKLOAD_ARCHETYPES.keys()) + list(MODEL_PROFILES.keys()),
        help="Workload archetype or model profile name.",
    )
    p.add_argument(
        "--gpu", "-g",
        default=None,
        help="Current GPU name to compare against (e.g. 'RTX 4090').",
    )
    p.add_argument(
        "--top", "-n",
        type=int, default=10,
        help="Number of recommendations to show (default: 10).",
    )
    p.add_argument(
        "--budget",
        type=float, default=None,
        help="Max GPU cost per hour in USD (filters results).",
    )
    p.add_argument(
        "--json",
        action="store_true", dest="json_output",
        help="Output as JSON instead of table.",
    )
    p.add_argument(
        "--from-benchmark",
        default=None, metavar="FILE",
        help="Infer workload profile from benchmark results JSON.",
    )
    return p


def _profile_from_benchmark(path: str) -> Optional[ModelProfile]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading benchmark file: {e}", file=sys.stderr)
        return None

    avg_util = data.get("avg_util_pct", 50)
    avg_power = data.get("avg_power_w", 200)
    gpu_name = data.get("gpu_name", "unknown")

    spec = resolve_arch(gpu_name)
    if spec and spec.fp16_tflops > 0 and spec.memory_bw_gbps > 0:
        achieved_tflops = spec.fp16_tflops * (avg_util / 100)
        intensity = achieved_tflops * 1e12 / (spec.memory_bw_gbps * 1e9) if spec.memory_bw_gbps > 0 else 50
    else:
        intensity = 50

    is_mem_bound = intensity < 50
    precision = "bf16" if not is_mem_bound else "fp16"

    return ModelProfile(
        tag=f"benchmark-{gpu_name}",
        family="Benchmark",
        math_intensity=round(intensity, 1),
        precision=precision,
        is_memory_bound=is_mem_bound,
        typical_util_min=max(5, int(avg_util - 15)),
        typical_util_max=min(100, int(avg_util + 15)),
    )


def run_recommend(args: argparse.Namespace) -> int:
    if args.from_benchmark:
        profile = _profile_from_benchmark(args.from_benchmark)
        if not profile:
            return 1
        WORKLOAD_ARCHETYPES[profile.tag] = profile
        MODEL_PROFILES[profile.tag] = profile
        workload = profile.tag
    elif args.workload:
        workload = args.workload
    else:
        print("Error: provide --workload or --from-benchmark", file=sys.stderr)
        print(f"\nAvailable workloads: {', '.join(WORKLOAD_ARCHETYPES.keys())}", file=sys.stderr)
        print(f"Available models:    {', '.join(MODEL_PROFILES.keys())}", file=sys.stderr)
        return 1

    results = recommend_gpus(
        workload=workload,
        current_gpu=args.gpu,
        top_n=args.top,
        max_cost_per_hr=args.budget,
    )

    if not results:
        print("No recommendations found. Check workload name and budget filter.", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps([asdict(r) for r in results], indent=2))
    elif _rich:
        _print_rich(results, workload, args.gpu)
    else:
        _print_plain(results, workload, args.gpu)

    return 0
