"""
Chargeback report generator — team-level GPU energy cost allocation.

Reads the local CSV manifest or API data grouped by team_id/model_tag
over a date range, computes costs and optional CO2 emissions.

CLI: aluminatiai report --from 2026-04-01 --to 2026-04-30 --format html --rate 0.12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TeamCost:
    team_id: str
    total_energy_kwh: float = 0.0
    total_cost_usd: float = 0.0
    total_co2_grams: float = 0.0
    gpu_hours: float = 0.0
    sample_count: int = 0
    models: dict = field(default_factory=lambda: defaultdict(float))


def generate_report(
    csv_path: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    rate_per_kwh: float = 0.12,
    carbon_intensity: float = 0.0,
    output_format: str = "json",
) -> str:
    """Generate a chargeback report from a CSV manifest.

    Returns the report as a string (JSON, CSV, or HTML).
    """
    path = Path(csv_path)
    if not path.exists():
        return json.dumps({"error": f"CSV file not found: {csv_path}"})

    from_dt = datetime.fromisoformat(from_date) if from_date else None
    to_dt = datetime.fromisoformat(to_date) if to_date else None

    teams: dict[str, TeamCost] = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if from_dt and ts < from_dt:
                        continue
                    if to_dt and ts > to_dt:
                        continue
                except ValueError:
                    continue

            team = row.get("team_id", "unresolved")
            model = row.get("model_tag", "untagged")
            energy_j = float(row.get("energy_delta_j", 0) or 0)
            energy_kwh = energy_j / 3_600_000

            if team not in teams:
                teams[team] = TeamCost(team_id=team)

            tc = teams[team]
            tc.total_energy_kwh += energy_kwh
            tc.total_cost_usd += energy_kwh * rate_per_kwh
            tc.sample_count += 1
            tc.models[model] += energy_kwh

            if carbon_intensity > 0:
                tc.total_co2_grams += energy_kwh * carbon_intensity

    sorted_teams = sorted(teams.values(), key=lambda t: t.total_cost_usd, reverse=True)

    if output_format == "json":
        return _render_json(sorted_teams, rate_per_kwh, from_date, to_date)
    elif output_format == "csv":
        return _render_csv(sorted_teams)
    elif output_format == "html":
        return _render_html(sorted_teams, rate_per_kwh, from_date, to_date)
    return _render_json(sorted_teams, rate_per_kwh, from_date, to_date)


def _render_json(teams: list[TeamCost], rate: float, from_d, to_d) -> str:
    data = {
        "report": "GPU Energy Chargeback",
        "period": {"from": from_d, "to": to_d},
        "rate_per_kwh_usd": rate,
        "total_cost_usd": round(sum(t.total_cost_usd for t in teams), 4),
        "total_energy_kwh": round(sum(t.total_energy_kwh for t in teams), 6),
        "teams": [
            {
                "team_id": t.team_id,
                "energy_kwh": round(t.total_energy_kwh, 6),
                "cost_usd": round(t.total_cost_usd, 4),
                "co2_grams": round(t.total_co2_grams, 2) if t.total_co2_grams else None,
                "samples": t.sample_count,
                "models": dict(t.models),
            }
            for t in teams
        ],
    }
    return json.dumps(data, indent=2)


def _render_csv(teams: list[TeamCost]) -> str:
    lines = ["team_id,energy_kwh,cost_usd,co2_grams,samples"]
    for t in teams:
        lines.append(
            f"{t.team_id},{t.total_energy_kwh:.6f},{t.total_cost_usd:.4f},"
            f"{t.total_co2_grams:.2f},{t.sample_count}"
        )
    return "\n".join(lines)


def _render_html(teams: list[TeamCost], rate: float, from_d, to_d) -> str:
    total_cost = sum(t.total_cost_usd for t in teams)
    total_energy = sum(t.total_energy_kwh for t in teams)

    rows = ""
    for t in teams:
        pct = (t.total_cost_usd / total_cost * 100) if total_cost > 0 else 0
        rows += f"""
        <tr>
            <td>{t.team_id}</td>
            <td>{t.total_energy_kwh:.4f}</td>
            <td>${t.total_cost_usd:.2f}</td>
            <td>{pct:.1f}%</td>
            <td>{t.sample_count:,}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>GPU Energy Chargeback Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #1a1a1a; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #1a1a1a; color: white; }}
        tr:hover {{ background: #f9f9f9; }}
        .footer {{ color: #888; font-size: 12px; margin-top: 40px; }}
    </style>
</head>
<body>
    <h1>GPU Energy Chargeback Report</h1>
    <div class="summary">
        <p><strong>Period:</strong> {from_d or 'all time'} to {to_d or 'present'}</p>
        <p><strong>Rate:</strong> ${rate:.2f}/kWh</p>
        <p><strong>Total Energy:</strong> {total_energy:.4f} kWh</p>
        <p><strong>Total Cost:</strong> ${total_cost:.2f}</p>
    </div>
    <table>
        <thead>
            <tr><th>Team</th><th>Energy (kWh)</th><th>Cost</th><th>Share</th><th>Samples</th></tr>
        </thead>
        <tbody>{rows}
        </tbody>
    </table>
    <p class="footer">Generated by AluminatAI GPU Agent</p>
</body>
</html>"""


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU energy chargeback report")
    parser.add_argument("--csv", required=True, help="Path to CSV manifest file")
    parser.add_argument("--from", dest="from_date", help="Start date (ISO format)")
    parser.add_argument("--to", dest="to_date", help="End date (ISO format)")
    parser.add_argument("--rate", type=float, default=0.12, help="$/kWh rate (default: 0.12)")
    parser.add_argument("--carbon", type=float, default=0.0, help="gCO2e/kWh for CO2 calculation")
    parser.add_argument("--format", dest="fmt", choices=["json", "csv", "html"], default="json")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    return parser


def run_report(args: argparse.Namespace) -> int:
    result = generate_report(
        csv_path=args.csv,
        from_date=args.from_date,
        to_date=args.to_date,
        rate_per_kwh=args.rate,
        carbon_intensity=args.carbon,
        output_format=args.fmt,
    )
    if args.output:
        Path(args.output).write_text(result)
        print(f"Report written to {args.output}")
    else:
        print(result)
    return 0
