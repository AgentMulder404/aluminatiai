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
Fleet-scale savings projections from A/B experiment results.

Takes an ABResult JSON (from `aluminatiai ab --json`) and extrapolates
savings to fleet sizes of 100, 1,000, and 10,000 GPUs.

Usage:
    python scaling_report.py result.json
    python scaling_report.py result.json --fleet 100,500,2000 --electricity-rate 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    _rich = True
except ImportError:
    _rich = False


# ── Data ─────────────────────────────────────────────────────────────────────

HOURS_PER_MONTH = 720  # 24 * 30
HOURS_PER_YEAR = 8760
KG_CO2_PER_TREE_PER_YEAR = 22.0
DEFAULT_ELECTRICITY_RATE = 0.12  # $/kWh


@dataclass
class FleetProjection:
    """Savings projection for a given fleet size."""
    fleet_size: int
    monthly_kwh_saved: float
    monthly_dollars_saved: float
    yearly_dollars_saved: float
    monthly_co2_saved_kg: float
    yearly_trees_equivalent: float


# ── Core Logic ───────────────────────────────────────────────────────────────

def compute_projections(
    ab_result: dict,
    fleet_sizes: list[int],
    electricity_rate: float = DEFAULT_ELECTRICITY_RATE,
) -> list[FleetProjection]:
    """
    Compute fleet-scale savings from an ABResult dict.

    Extracts per-GPU hourly savings from the AB result and scales
    to monthly/yearly values for each fleet size.
    """
    baseline = ab_result.get("baseline", {})
    optimized = ab_result.get("optimized", {})

    # Power delta (watts)
    baseline_power = _nested_mean(baseline, "mean_power_w")
    optimized_power = _nested_mean(optimized, "mean_power_w")
    power_delta_w = baseline_power - optimized_power

    if power_delta_w <= 0:
        power_delta_w = 0.0

    # Per-GPU hourly savings
    kwh_saved_per_gpu_hr = power_delta_w / 1000.0

    # CO2 per GPU per hour (grams)
    co2_savings_g_per_gpu_hr = 0.0
    carbon_intensity = ab_result.get("carbon_intensity_gco2e", 0.0)
    if carbon_intensity > 0:
        co2_savings_g_per_gpu_hr = kwh_saved_per_gpu_hr * carbon_intensity
    else:
        # Fallback: US average 394 gCO2e/kWh
        co2_savings_g_per_gpu_hr = kwh_saved_per_gpu_hr * 394.0

    projections = []
    for fleet in fleet_sizes:
        monthly_kwh = kwh_saved_per_gpu_hr * HOURS_PER_MONTH * fleet
        monthly_dollars = monthly_kwh * electricity_rate
        yearly_dollars = monthly_dollars * 12
        monthly_co2_kg = (co2_savings_g_per_gpu_hr * HOURS_PER_MONTH * fleet) / 1000.0
        yearly_trees = (monthly_co2_kg * 12) / KG_CO2_PER_TREE_PER_YEAR

        projections.append(FleetProjection(
            fleet_size=fleet,
            monthly_kwh_saved=round(monthly_kwh, 1),
            monthly_dollars_saved=round(monthly_dollars, 2),
            yearly_dollars_saved=round(yearly_dollars, 2),
            monthly_co2_saved_kg=round(monthly_co2_kg, 1),
            yearly_trees_equivalent=round(yearly_trees, 0),
        ))

    return projections


def _nested_mean(phase: dict, key: str) -> float:
    """Extract mean from a nested CI dict (e.g. phase['mean_power_w']['mean'])."""
    val = phase.get(key, {})
    if isinstance(val, dict):
        return val.get("mean", 0.0)
    return float(val) if val else 0.0


# ── Output ───────────────────────────────────────────────────────────────────

def print_report(
    ab_result: dict,
    projections: list[FleetProjection],
    electricity_rate: float,
) -> None:
    """Print the scaling report."""
    gpu_name = ab_result.get("gpu_name", "Unknown GPU")
    energy_savings_pct = ab_result.get("energy_savings_pct", 0.0)
    throughput_change = ab_result.get("throughput_change_pct")
    aem = ab_result.get("aem", 0.0)
    carbon_zone = ab_result.get("carbon_zone", "")

    if _rich:
        _print_rich(gpu_name, energy_savings_pct, throughput_change, aem,
                    carbon_zone, projections, electricity_rate)
    else:
        _print_plain(gpu_name, energy_savings_pct, throughput_change, aem,
                     carbon_zone, projections, electricity_rate)


def _print_rich(
    gpu_name: str,
    energy_pct: float,
    throughput_pct: float | None,
    aem: float,
    carbon_zone: str,
    projections: list[FleetProjection],
    rate: float,
) -> None:
    console = Console()
    console.print()
    console.print(Panel(
        f"[bold white]AluminatAI Fleet Savings Projection[/bold white]\n"
        f"GPU: {gpu_name} | Energy savings: {energy_pct:.1f}% | "
        f"Throughput impact: {_tp_str(throughput_pct)} | "
        f"AEM: {_aem_str(aem)}",
        border_style="cyan",
    ))

    table = Table(title="Monthly & Yearly Projections", show_lines=True)
    table.add_column("Fleet Size", justify="right", style="bold")
    table.add_column("kWh Saved/mo", justify="right")
    table.add_column("$ Saved/mo", justify="right", style="green")
    table.add_column("$ Saved/yr", justify="right", style="bold green")
    table.add_column("CO2 Saved/mo (kg)", justify="right")
    table.add_column("Trees/yr", justify="right", style="cyan")

    for p in projections:
        table.add_row(
            f"{p.fleet_size:,}",
            f"{p.monthly_kwh_saved:,.0f}",
            f"${p.monthly_dollars_saved:,.0f}",
            f"${p.yearly_dollars_saved:,.0f}",
            f"{p.monthly_co2_saved_kg:,.0f}",
            f"{p.yearly_trees_equivalent:,.0f}",
        )

    console.print(table)
    console.print()

    # Big summary line
    biggest = projections[-1]
    console.print(
        f"  [bold yellow]At {biggest.fleet_size:,} GPUs:[/bold yellow] "
        f"[bold green]${biggest.yearly_dollars_saved:,.0f}/year saved[/bold green] "
        f"= equivalent to planting [bold cyan]{biggest.yearly_trees_equivalent:,.0f} trees[/bold cyan]",
    )
    console.print(f"  Electricity rate: ${rate}/kWh", style="dim")
    if carbon_zone:
        console.print(f"  Carbon zone: {carbon_zone}", style="dim")
    console.print()


def _print_plain(
    gpu_name: str,
    energy_pct: float,
    throughput_pct: float | None,
    aem: float,
    carbon_zone: str,
    projections: list[FleetProjection],
    rate: float,
) -> None:
    print()
    print("=" * 70)
    print("  AluminatAI Fleet Savings Projection")
    print(f"  GPU: {gpu_name} | Energy savings: {energy_pct:.1f}% | "
          f"Throughput: {_tp_str(throughput_pct)} | AEM: {_aem_str(aem)}")
    print("=" * 70)
    print()
    print(f"  {'Fleet':>8} {'kWh/mo':>12} {'$/mo':>12} {'$/yr':>14} {'CO2 kg/mo':>12} {'Trees/yr':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*12} {'-'*10}")

    for p in projections:
        print(f"  {p.fleet_size:>8,} {p.monthly_kwh_saved:>12,.0f} "
              f"${p.monthly_dollars_saved:>11,.0f} ${p.yearly_dollars_saved:>13,.0f} "
              f"{p.monthly_co2_saved_kg:>12,.0f} {p.yearly_trees_equivalent:>10,.0f}")

    print()
    biggest = projections[-1]
    print(f"  >>> At {biggest.fleet_size:,} GPUs: ${biggest.yearly_dollars_saved:,.0f}/year saved "
          f"= {biggest.yearly_trees_equivalent:,.0f} trees planted")
    print()


def _tp_str(tp: float | None) -> str:
    if tp is None:
        return "N/A"
    return f"{tp:+.1f}%"


def _aem_str(aem: float) -> str:
    if aem == float("inf") or (isinstance(aem, str) and aem == "inf"):
        return "inf (no loss)"
    if aem > 0:
        return f"{aem}x"
    return "N/A"


# ── CLI ──────────────────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scaling_report",
        description="Fleet-scale savings projections from A/B results.",
    )
    p.add_argument("result_file", nargs="?", help="Path to ABResult JSON file")
    p.add_argument("--fleet", type=str, default="100,1000,10000",
                   help="Comma-separated fleet sizes (default: 100,1000,10000)")
    p.add_argument("--electricity-rate", type=float, default=DEFAULT_ELECTRICITY_RATE,
                   help=f"Electricity cost $/kWh (default: {DEFAULT_ELECTRICITY_RATE})")
    return p


def run_scaling_report(args: argparse.Namespace) -> int:
    """Entry point for standalone CLI usage."""
    if not args.result_file:
        print("ERROR: Provide a path to an ABResult JSON file", file=sys.stderr)
        return 1

    try:
        with open(args.result_file) as f:
            ab_result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fleet_sizes = [int(x.strip()) for x in args.fleet.split(",")]
    projections = compute_projections(ab_result, fleet_sizes, args.electricity_rate)
    print_report(ab_result, projections, args.electricity_rate)
    return 0


def run_from_dict(
    ab_result: dict,
    fleet_sizes: list[int] | None = None,
    electricity_rate: float = DEFAULT_ELECTRICITY_RATE,
) -> list[FleetProjection]:
    """Programmatic entry point for the demo orchestrator."""
    if fleet_sizes is None:
        fleet_sizes = [100, 1_000, 10_000]
    projections = compute_projections(ab_result, fleet_sizes, electricity_rate)
    print_report(ab_result, projections, electricity_rate)
    return projections


if __name__ == "__main__":
    sys.exit(run_scaling_report(make_parser().parse_args()))
