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
Energy efficiency analysis for GPU workloads.

Provides:
- GPU architecture specs and roofline model calculations
- Efficiency curve building from observed fleet metrics
- Hardware Match Score computation for workload-GPU pairing
- Scientific energy profiling with confidence intervals and optimization plans
"""

from .gpu_specs import ArchSpec, ModelProfile, GPU_ARCHITECTURES, MODEL_PROFILES
from .curve_builder import EfficiencyCurveBuilder
from .hardware_match import HardwareMatchScorer, MatchResult
from .profiler import (
    ScientificEnergyProfiler,
    ProfileResult,
    ConfidenceInterval,
    PowerSample,
    IterationResult,
)
from .stats import compute_ci, t_critical_95, trapezoidal_energy
from .power_control import set_power_limit, get_power_limit, get_default_power_limit
from .carbon import (
    ElectricityMapsClient,
    CarbonIntensity,
    CarbonForecast,
    ForecastWindow,
    CO2Estimate,
)

__all__ = [
    'ArchSpec',
    'ModelProfile',
    'GPU_ARCHITECTURES',
    'MODEL_PROFILES',
    'EfficiencyCurveBuilder',
    'HardwareMatchScorer',
    'MatchResult',
    'ScientificEnergyProfiler',
    'ProfileResult',
    'ConfidenceInterval',
    'PowerSample',
    'IterationResult',
    'compute_ci',
    't_critical_95',
    'trapezoidal_energy',
    'set_power_limit',
    'get_power_limit',
    'get_default_power_limit',
    'ElectricityMapsClient',
    'CarbonIntensity',
    'CarbonForecast',
    'ForecastWindow',
    'CO2Estimate',
]
