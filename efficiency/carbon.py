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
Carbon intensity tracking via the Electricity Maps API.

Provides real-time and forecasted grid carbon intensity so the agent
and CLI tools can:
  - Estimate CO2e per job
  - Recommend scheduling batch jobs during green hours
  - Compare carbon cost across regions

Requires ALUMINATAI_GRID_ZONE (e.g. "US-CAL-CISO") in config.
Optionally uses ELECTRICITY_MAPS_API_KEY for authenticated access
(higher rate limits + forecast endpoint).

API docs: https://static.electricitymaps.com/api/docs/index.html
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

# Electricity Maps free tier base URL (no auth required for /latest)
_BASE_URL = "https://api.electricitymaps.com/v3"

# US grid average fallback (EPA eGRID 2024)
DEFAULT_CARBON_INTENSITY = 394.0  # gCO2e/kWh

# Cache TTL — avoid hammering the API on every call
_CACHE_TTL_S = 300  # 5 minutes


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class CarbonIntensity:
    """Current grid carbon intensity for a zone."""
    zone: str
    carbon_intensity_gco2e: float  # gCO2e per kWh
    fossil_fuel_pct: float  # percentage of generation from fossil fuels
    renewable_pct: float  # percentage from renewables (wind + solar + hydro)
    timestamp: str  # ISO 8601 from the API
    is_cached: bool = False
    is_fallback: bool = False  # True if using default instead of live data


@dataclass
class ForecastWindow:
    """A single forecast time slot."""
    datetime: str  # ISO 8601
    carbon_intensity_gco2e: float


@dataclass
class CarbonForecast:
    """Carbon intensity forecast for the next 24h."""
    zone: str
    windows: list[ForecastWindow] = field(default_factory=list)
    best_window: Optional[ForecastWindow] = None  # lowest carbon slot
    worst_window: Optional[ForecastWindow] = None  # highest carbon slot
    current_vs_best_pct: float = 0.0  # how much greener the best window is


@dataclass
class CO2Estimate:
    """CO2 emissions estimate for a workload."""
    energy_kwh: float
    carbon_intensity_gco2e: float
    co2_grams: float
    co2_kg: float
    zone: str
    trees_equivalent: float  # kg CO2 absorbed by one tree per year ≈ 22kg


# ── Client ────────────────────────────────────────────────────────────────────

class ElectricityMapsClient:
    """
    Client for the Electricity Maps API.

    Usage:
        client = ElectricityMapsClient(zone="US-CAL-CISO")
        intensity = client.get_current()
        print(f"Grid: {intensity.carbon_intensity_gco2e} gCO2e/kWh")

        forecast = client.get_forecast()
        print(f"Best time to run: {forecast.best_window.datetime}")

        co2 = client.estimate_co2(energy_kwh=1.5)
        print(f"Job emitted {co2.co2_kg:.3f} kg CO2")
    """

    def __init__(self, zone: Optional[str] = None, api_key: Optional[str] = None):
        self.zone = zone or os.getenv("ALUMINATAI_GRID_ZONE", "")
        self.api_key = api_key or os.getenv("ELECTRICITY_MAPS_API_KEY", "")
        self._cache: Optional[CarbonIntensity] = None
        self._cache_time: float = 0.0

    def get_current(self) -> CarbonIntensity:
        """
        Fetch current carbon intensity for the configured zone.

        Returns cached data if within TTL. Falls back to default if
        the zone is not set or the API is unreachable.
        """
        # Check cache
        if self._cache and (time.time() - self._cache_time) < _CACHE_TTL_S:
            return CarbonIntensity(
                zone=self._cache.zone,
                carbon_intensity_gco2e=self._cache.carbon_intensity_gco2e,
                fossil_fuel_pct=self._cache.fossil_fuel_pct,
                renewable_pct=self._cache.renewable_pct,
                timestamp=self._cache.timestamp,
                is_cached=True,
            )

        if not self.zone:
            return self._fallback("No ALUMINATAI_GRID_ZONE configured")

        try:
            data = self._request(f"/carbon-intensity/latest?zone={self.zone}")
            intensity = CarbonIntensity(
                zone=self.zone,
                carbon_intensity_gco2e=data.get("carbonIntensity", DEFAULT_CARBON_INTENSITY),
                fossil_fuel_pct=data.get("fossilFuelPercentage", 0.0),
                renewable_pct=data.get("renewablePercentage", 0.0),
                timestamp=data.get("datetime", ""),
            )
            self._cache = intensity
            self._cache_time = time.time()
            return intensity
        except Exception:
            return self._fallback("API request failed")

    def get_forecast(self) -> CarbonForecast:
        """
        Fetch 24h carbon intensity forecast.

        Requires an API key (authenticated endpoint).
        Returns an empty forecast if unavailable.
        """
        if not self.zone:
            return CarbonForecast(zone="unknown")

        if not self.api_key:
            # Free tier doesn't support forecast — build from current only
            current = self.get_current()
            return CarbonForecast(
                zone=self.zone,
                windows=[ForecastWindow(
                    datetime=current.timestamp,
                    carbon_intensity_gco2e=current.carbon_intensity_gco2e,
                )],
            )

        try:
            data = self._request(f"/carbon-intensity/forecast?zone={self.zone}")
            forecast_data = data.get("forecast", [])

            windows = [
                ForecastWindow(
                    datetime=w.get("datetime", ""),
                    carbon_intensity_gco2e=w.get("carbonIntensity", DEFAULT_CARBON_INTENSITY),
                )
                for w in forecast_data
            ]

            if not windows:
                return CarbonForecast(zone=self.zone)

            best = min(windows, key=lambda w: w.carbon_intensity_gco2e)
            worst = max(windows, key=lambda w: w.carbon_intensity_gco2e)

            current = self.get_current()
            current_vs_best = 0.0
            if current.carbon_intensity_gco2e > 0:
                current_vs_best = round(
                    (1 - best.carbon_intensity_gco2e / current.carbon_intensity_gco2e) * 100, 1
                )

            return CarbonForecast(
                zone=self.zone,
                windows=windows,
                best_window=best,
                worst_window=worst,
                current_vs_best_pct=current_vs_best,
            )
        except Exception:
            return CarbonForecast(zone=self.zone)

    def estimate_co2(
        self,
        energy_kwh: float,
        carbon_intensity: Optional[float] = None,
    ) -> CO2Estimate:
        """
        Estimate CO2 emissions for a given energy consumption.

        Args:
            energy_kwh: Energy consumed in kWh.
            carbon_intensity: Override gCO2e/kWh (uses live data if omitted).
        """
        if carbon_intensity is None:
            current = self.get_current()
            carbon_intensity = current.carbon_intensity_gco2e
            zone = current.zone
        else:
            zone = self.zone or "manual"

        co2_g = energy_kwh * carbon_intensity
        co2_kg = co2_g / 1000.0

        return CO2Estimate(
            energy_kwh=round(energy_kwh, 6),
            carbon_intensity_gco2e=carbon_intensity,
            co2_grams=round(co2_g, 2),
            co2_kg=round(co2_kg, 4),
            zone=zone,
            trees_equivalent=round(co2_kg / 22.0, 4),  # ~22 kg CO2/tree/year
        )

    def is_green_hour(self, threshold_pct: float = 50.0) -> bool:
        """
        Check if the current grid has >= threshold_pct renewable energy.

        Default threshold: 50% renewables = "green hour".
        """
        current = self.get_current()
        return current.renewable_pct >= threshold_pct

    # ── Internal ──────────────────────────────────────────────────────────

    def _request(self, path: str) -> dict:
        """Make an HTTP GET to the Electricity Maps API."""
        url = f"{_BASE_URL}{path}"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["auth-token"] = self.api_key

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def _fallback(self, reason: str = "") -> CarbonIntensity:
        """Return a fallback CarbonIntensity using the US grid average."""
        return CarbonIntensity(
            zone=self.zone or "US-AVG",
            carbon_intensity_gco2e=DEFAULT_CARBON_INTENSITY,
            fossil_fuel_pct=60.0,  # US average
            renewable_pct=21.0,  # US average
            timestamp="",
            is_fallback=True,
        )
