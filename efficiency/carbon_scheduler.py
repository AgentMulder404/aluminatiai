"""
Carbon-aware job scheduling — recommend optimal start windows.

Given a job's estimated duration and the 24h carbon forecast, recommends
the time window with the lowest average carbon intensity.

CLI: aluminatiai carbon-schedule --duration 4h --zone US-CAL-CISO

Advisory-only in daemon mode (logs recommendation, does NOT auto-defer).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from efficiency.carbon import ElectricityMapsClient, CarbonForecast
except ImportError:
    ElectricityMapsClient = None  # type: ignore


@dataclass
class ScheduleRecommendation:
    recommended_start: datetime
    avg_intensity_gco2e: float
    current_intensity_gco2e: float
    savings_pct: float
    zone: str
    duration_hours: float


def find_optimal_window(
    zone: str,
    duration_hours: float,
    api_key: str = "",
) -> Optional[ScheduleRecommendation]:
    """Find the lowest-carbon window in the next 24h for a job of given duration.

    Returns None if forecast data is unavailable.
    """
    if ElectricityMapsClient is None:
        return None

    client = ElectricityMapsClient(zone=zone, api_key=api_key)

    # Get current intensity
    current = client.get_current()
    current_intensity = current.carbon_intensity_gco2e

    # Get 24h forecast
    forecast = client.get_forecast()
    if not forecast.hourly_intensities:
        return None

    # Sliding window to find lowest average intensity
    hours = forecast.hourly_intensities
    window_size = max(1, int(duration_hours))

    if len(hours) < window_size:
        return None

    best_start_idx = 0
    best_avg = float("inf")

    for i in range(len(hours) - window_size + 1):
        window = hours[i:i + window_size]
        avg = sum(h.carbon_intensity_gco2e for h in window) / len(window)
        if avg < best_avg:
            best_avg = avg
            best_start_idx = i

    best_start = hours[best_start_idx].datetime if hasattr(hours[best_start_idx], 'datetime') else None
    if best_start is None:
        best_start = datetime.now(timezone.utc) + timedelta(hours=best_start_idx)
    elif isinstance(best_start, str):
        try:
            best_start = datetime.fromisoformat(best_start.replace("Z", "+00:00"))
        except ValueError:
            best_start = datetime.now(timezone.utc) + timedelta(hours=best_start_idx)

    savings_pct = ((current_intensity - best_avg) / current_intensity * 100) if current_intensity > 0 else 0

    return ScheduleRecommendation(
        recommended_start=best_start,
        avg_intensity_gco2e=round(best_avg, 1),
        current_intensity_gco2e=round(current_intensity, 1),
        savings_pct=round(max(0, savings_pct), 1),
        zone=zone,
        duration_hours=duration_hours,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the lowest-carbon time window for a GPU job",
    )
    parser.add_argument("--duration", required=True,
                        help="Job duration (e.g., '4h', '30m', '1.5h')")
    parser.add_argument("--zone", default="",
                        help="Electricity Maps zone (default: ALUMINATAI_GRID_ZONE)")
    parser.add_argument("--api-key", default="",
                        help="Electricity Maps API key")
    return parser


def _parse_duration(s: str) -> float:
    """Parse duration string like '4h', '30m', '1.5h' into hours."""
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1])
    elif s.endswith("m"):
        return float(s[:-1]) / 60
    elif s.endswith("s"):
        return float(s[:-1]) / 3600
    return float(s)


def run_carbon_schedule(args: argparse.Namespace) -> int:
    import os
    zone = args.zone or os.getenv("ALUMINATAI_GRID_ZONE", "")
    if not zone:
        print("ERROR: --zone or ALUMINATAI_GRID_ZONE required")
        return 1

    try:
        duration_h = _parse_duration(args.duration)
    except ValueError:
        print(f"ERROR: invalid duration format: {args.duration}")
        return 1

    rec = find_optimal_window(zone=zone, duration_hours=duration_h, api_key=args.api_key)
    if rec is None:
        print(f"No forecast data available for zone {zone}")
        return 1

    print(f"Zone:              {rec.zone}")
    print(f"Job duration:      {rec.duration_hours:.1f}h")
    print(f"Current intensity: {rec.current_intensity_gco2e} gCO2e/kWh")
    print(f"Recommended start: {rec.recommended_start.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Window avg:        {rec.avg_intensity_gco2e} gCO2e/kWh")
    if rec.savings_pct > 0:
        print(f"CO2 savings:       ~{rec.savings_pct}% less carbon vs running now")
    else:
        print("Running now is already optimal (or very close).")
    return 0
