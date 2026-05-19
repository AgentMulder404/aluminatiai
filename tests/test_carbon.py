# Copyright 2026 Kevin (AluminatiAI)
"""Tests for the carbon intensity tracking module."""
import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from efficiency.carbon import (
    ElectricityMapsClient,
    CarbonIntensity,
    CarbonForecast,
    ForecastWindow,
    CO2Estimate,
    DEFAULT_CARBON_INTENSITY,
    _CACHE_TTL_S,
)


class TestFallback:
    def test_no_zone_returns_fallback(self):
        client = ElectricityMapsClient(zone="")
        result = client.get_current()
        assert result.is_fallback is True
        assert result.carbon_intensity_gco2e == DEFAULT_CARBON_INTENSITY
        assert result.zone == "US-AVG"

    def test_fallback_values(self):
        client = ElectricityMapsClient(zone="")
        result = client.get_current()
        assert result.fossil_fuel_pct == 60.0
        assert result.renewable_pct == 21.0
        assert result.timestamp == ""


class TestCO2Estimation:
    def test_basic_estimate(self):
        client = ElectricityMapsClient(zone="")
        # Force fallback so we get known carbon intensity
        co2 = client.estimate_co2(energy_kwh=1.0)
        assert co2.co2_grams == DEFAULT_CARBON_INTENSITY  # 1 kWh * 394 gCO2e/kWh
        assert co2.co2_kg == round(DEFAULT_CARBON_INTENSITY / 1000.0, 4)
        assert co2.zone == "US-AVG"

    def test_manual_override(self):
        client = ElectricityMapsClient(zone="US-CAL-CISO")
        co2 = client.estimate_co2(energy_kwh=2.0, carbon_intensity=200.0)
        assert co2.co2_grams == 400.0
        assert co2.co2_kg == 0.4
        assert co2.zone == "US-CAL-CISO"

    def test_zero_energy(self):
        client = ElectricityMapsClient(zone="")
        co2 = client.estimate_co2(energy_kwh=0.0)
        assert co2.co2_grams == 0.0
        assert co2.co2_kg == 0.0

    def test_trees_equivalent(self):
        client = ElectricityMapsClient(zone="")
        co2 = client.estimate_co2(energy_kwh=10.0, carbon_intensity=100.0)
        # 10 kWh * 100 gCO2e = 1000g = 1kg; 1kg / 22kg per tree = 0.0455
        assert co2.trees_equivalent == round(1.0 / 22.0, 4)


class TestGreenHour:
    def test_green_when_high_renewables(self):
        client = ElectricityMapsClient(zone="")
        # Mock get_current to return high renewables
        mock_intensity = CarbonIntensity(
            zone="DK-DK1", carbon_intensity_gco2e=50.0,
            fossil_fuel_pct=20.0, renewable_pct=75.0,
            timestamp="2026-04-05T12:00:00Z",
        )
        with patch.object(client, "get_current", return_value=mock_intensity):
            assert client.is_green_hour() is True

    def test_not_green_when_low_renewables(self):
        client = ElectricityMapsClient(zone="")
        mock_intensity = CarbonIntensity(
            zone="US-MIDA-PJM", carbon_intensity_gco2e=450.0,
            fossil_fuel_pct=70.0, renewable_pct=15.0,
            timestamp="2026-04-05T12:00:00Z",
        )
        with patch.object(client, "get_current", return_value=mock_intensity):
            assert client.is_green_hour() is False

    def test_custom_threshold(self):
        client = ElectricityMapsClient(zone="")
        mock_intensity = CarbonIntensity(
            zone="FR", carbon_intensity_gco2e=80.0,
            fossil_fuel_pct=10.0, renewable_pct=35.0,
            timestamp="2026-04-05T12:00:00Z",
        )
        with patch.object(client, "get_current", return_value=mock_intensity):
            assert client.is_green_hour(threshold_pct=30.0) is True
            assert client.is_green_hour(threshold_pct=40.0) is False


class TestCache:
    def test_returns_cached_within_ttl(self):
        client = ElectricityMapsClient(zone="US-CAL-CISO")
        cached = CarbonIntensity(
            zone="US-CAL-CISO", carbon_intensity_gco2e=200.0,
            fossil_fuel_pct=30.0, renewable_pct=55.0,
            timestamp="2026-04-05T12:00:00Z",
        )
        client._cache = cached
        client._cache_time = time.time()  # Just cached

        result = client.get_current()
        assert result.is_cached is True
        assert result.carbon_intensity_gco2e == 200.0

    def test_stale_cache_triggers_fetch(self):
        client = ElectricityMapsClient(zone="US-CAL-CISO")
        cached = CarbonIntensity(
            zone="US-CAL-CISO", carbon_intensity_gco2e=200.0,
            fossil_fuel_pct=30.0, renewable_pct=55.0,
            timestamp="2026-04-05T12:00:00Z",
        )
        client._cache = cached
        client._cache_time = time.time() - _CACHE_TTL_S - 10  # Expired

        # API will fail → fallback
        result = client.get_current()
        # Should not be cached (either fresh fetch or fallback)
        assert result.is_cached is False


class TestForecast:
    def test_no_zone_returns_empty(self):
        client = ElectricityMapsClient(zone="")
        forecast = client.get_forecast()
        assert forecast.zone == "unknown"
        assert len(forecast.windows) == 0

    def test_no_api_key_uses_current(self):
        client = ElectricityMapsClient(zone="US-CAL-CISO", api_key="")
        mock_intensity = CarbonIntensity(
            zone="US-CAL-CISO", carbon_intensity_gco2e=250.0,
            fossil_fuel_pct=40.0, renewable_pct=45.0,
            timestamp="2026-04-05T12:00:00Z",
        )
        with patch.object(client, "get_current", return_value=mock_intensity):
            forecast = client.get_forecast()
            assert forecast.zone == "US-CAL-CISO"
            assert len(forecast.windows) == 1
            assert forecast.windows[0].carbon_intensity_gco2e == 250.0

    def test_forecast_with_api_key(self):
        client = ElectricityMapsClient(zone="DE", api_key="test-key")
        mock_response = {
            "forecast": [
                {"datetime": "2026-04-05T12:00:00Z", "carbonIntensity": 300.0},
                {"datetime": "2026-04-05T13:00:00Z", "carbonIntensity": 150.0},
                {"datetime": "2026-04-05T14:00:00Z", "carbonIntensity": 400.0},
            ]
        }
        mock_current = CarbonIntensity(
            zone="DE", carbon_intensity_gco2e=300.0,
            fossil_fuel_pct=50.0, renewable_pct=40.0,
            timestamp="2026-04-05T12:00:00Z",
        )
        with patch.object(client, "_request", return_value=mock_response), \
             patch.object(client, "get_current", return_value=mock_current):
            forecast = client.get_forecast()
            assert len(forecast.windows) == 3
            assert forecast.best_window.carbon_intensity_gco2e == 150.0
            assert forecast.worst_window.carbon_intensity_gco2e == 400.0
            assert forecast.current_vs_best_pct == 50.0  # (1 - 150/300) * 100


class TestAPIRequest:
    def test_request_adds_auth_header(self):
        client = ElectricityMapsClient(zone="US-CAL-CISO", api_key="my-secret-key")
        mock_data = {"carbonIntensity": 250.0}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("efficiency.carbon.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = client._request("/test")
            req = mock_open.call_args[0][0]
            assert req.get_header("Auth-token") == "my-secret-key"
            assert result == mock_data


class TestDataclasses:
    def test_carbon_intensity_fields(self):
        ci = CarbonIntensity(
            zone="FR", carbon_intensity_gco2e=58.0,
            fossil_fuel_pct=5.0, renewable_pct=25.0,
            timestamp="2026-04-05T12:00:00Z",
        )
        assert ci.zone == "FR"
        assert ci.is_cached is False
        assert ci.is_fallback is False

    def test_forecast_window(self):
        fw = ForecastWindow(datetime="2026-04-05T12:00:00Z", carbon_intensity_gco2e=100.0)
        assert fw.carbon_intensity_gco2e == 100.0

    def test_co2_estimate_fields(self):
        est = CO2Estimate(
            energy_kwh=1.5, carbon_intensity_gco2e=200.0,
            co2_grams=300.0, co2_kg=0.3, zone="DE",
            trees_equivalent=round(0.3 / 22.0, 4),
        )
        assert est.co2_grams == 300.0
        assert est.trees_equivalent == round(0.3 / 22.0, 4)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
