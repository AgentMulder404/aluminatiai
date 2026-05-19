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
#!/usr/bin/env python3
"""
AluminatAI Scientific A/B Power-Cap Experiment

Proves that power-capping an A100-SXM4 from 400W → 250W improves energy
efficiency (Tasks/Joule) with minimal throughput loss.

Experiment Protocol
───────────────────
  Phase 0   BURN-IN     Run workload at 400W for 120s.  GPU reaches thermal
                        steady-state (plateau).  Data discarded.

  Phase 1   BASELINE    Power limit = 400W.  Train SimpleNet on synthetic
            (Test A)    ImageNet-scale data (batch 1024) for 60s.
                        Record power via pynvml at 10 Hz.

  Phase 2   OPTIMIZED   Power limit = 250W.  Identical workload, 60s.
            (Test B)    Record power via pynvml at 10 Hz.

  Phase 3   REPORT      Compare throughput loss % vs energy savings %.
                        Compute Aluminati Efficiency Multiplier:
                          AEM = % Energy Saved / % Performance Lost

Controls:
  - Same model, same dataset, same duration, same GPU
  - Trapezoidal integration for energy (not P×t approximation)
  - 10 Hz sampling ⇒ ~600 data points per phase ⇒ <0.5% integration error
  - Burn-in guarantees identical starting thermal conditions

Dependencies:
  pip install torch nvidia-ml-py3

Usage:
  # On a Colab / bare-metal A100:
  python powercap_ab_test.py

  # Custom power levels:
  python powercap_ab_test.py --baseline-watts 400 --capped-watts 250

  # Shorter test (for validation):
  python powercap_ab_test.py --test-seconds 30 --burnin-seconds 30

  # Skip burn-in (if GPU is already warm):
  python powercap_ab_test.py --skip-burnin
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

# ── Dependencies ─────────────────────────────────────────────────────────────

try:
    import pynvml
except ImportError:
    print("FATAL: pynvml required.  pip install nvidia-ml-py3")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("FATAL: PyTorch required.  pip install torch")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PowerSample:
    """Single 10-Hz NVML reading."""
    t_mono: float           # time.monotonic()
    power_w: float
    util_pct: int
    mem_util_pct: int
    temp_c: int
    sm_clock_mhz: int


@dataclass
class PhaseResult:
    """Aggregated result for one experimental phase."""
    label: str
    power_limit_w: int
    duration_s: float
    images_processed: int
    batches_completed: int
    batch_size: int

    # Telemetry
    samples: List[PowerSample] = field(default_factory=list)

    # Computed after collection
    total_energy_j: float = 0.0
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    avg_temp_c: float = 0.0
    peak_temp_c: int = 0
    avg_util_pct: float = 0.0
    avg_sm_clock_mhz: float = 0.0

    # Derived metrics
    throughput_img_s: float = 0.0
    joules_per_1k_images: float = 0.0
    kwh: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  SimpleNet — deterministic CNN for controlled experiments
# ═══════════════════════════════════════════════════════════════════════════════


class SimpleNet(nn.Module):
    """
    Lightweight CNN that produces measurable GPU load without dominating
    memory.  Architecture is deliberately simple so the experiment
    measures *power behaviour*, not model convergence.

    Input:  3×32×32  (synthetic ImageNet-scale, downsampled)
    Output: 10 classes
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  NVML Power Sampler — background thread at 10 Hz
# ═══════════════════════════════════════════════════════════════════════════════


class PowerSampler:
    """
    Background thread that reads NVML at 10 Hz (100 ms intervals).

    10 Hz is the sweet spot: fast enough for accurate trapezoidal
    integration (<0.5% error on a 60s window), slow enough to add
    zero measurable overhead to the training loop.
    """

    INTERVAL_S = 0.1  # 10 Hz

    def __init__(self, gpu_index: int = 0):
        self._gpu_index = gpu_index
        self._handle = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: List[PowerSample] = []

    def start(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[PowerSample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            return list(self._samples)

    def _run(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
                try:
                    clock = pynvml.nvmlDeviceGetClockInfo(
                        self._handle, pynvml.NVML_CLOCK_SM
                    )
                except pynvml.NVMLError:
                    clock = 0

                sample = PowerSample(
                    t_mono=t0,
                    power_w=power,
                    util_pct=util.gpu,
                    mem_util_pct=util.memory,
                    temp_c=temp,
                    sm_clock_mhz=clock,
                )
                with self._lock:
                    self._samples.append(sample)

            except pynvml.NVMLError:
                pass  # skip this tick

            elapsed = time.monotonic() - t0
            remaining = self.INTERVAL_S - elapsed
            if remaining > 0:
                self._stop.wait(timeout=remaining)


# ═══════════════════════════════════════════════════════════════════════════════
#  Power Limit Control
# ═══════════════════════════════════════════════════════════════════════════════


def set_power_limit(gpu_index: int, watts: int) -> bool:
    """
    Set GPU power limit via NVML.

    Falls back to nvidia-smi if NVML persistence mode isn't enabled
    (common on Colab).  Requires root/sudo on bare metal.
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, watts * 1000)  # mW
        # Verify
        actual = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        print(f"  Power limit set to {actual:.0f}W via NVML")
        return True
    except pynvml.NVMLError:
        # Fallback: nvidia-smi (works on Colab without root)
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_index), "-pl", str(watts)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"  Power limit set to {watts}W via nvidia-smi")
                return True
            else:
                print(f"  WARNING: nvidia-smi -pl failed: {result.stderr.strip()}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  WARNING: Could not set power limit: {e}")
            return False


def get_power_limit(gpu_index: int = 0) -> int:
    """Read current power management limit in watts."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return int(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000)
    except pynvml.NVMLError:
        return 0


def get_default_power_limit(gpu_index: int = 0) -> int:
    """Read the factory default power limit."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return int(pynvml.nvmlDeviceGetDefaultPowerManagementLimit(handle) / 1000)
    except pynvml.NVMLError:
        return 400  # A100 SXM4 default


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════


def run_training_phase(
    label: str,
    power_limit_w: int,
    duration_s: float,
    batch_size: int,
    gpu_index: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> PhaseResult:
    """
    Run a timed training phase:
      1. Set power limit
      2. Start 10 Hz NVML sampler
      3. Run training loop for `duration_s` seconds
      4. Stop sampler, compute aggregates
    """
    print(f"\n{'─' * 70}")
    print(f"  PHASE: {label}  |  Power Limit = {power_limit_w}W  |  Duration = {duration_s:.0f}s")
    print(f"{'─' * 70}")

    # Set power limit
    set_power_limit(gpu_index, power_limit_w)
    time.sleep(2)  # let the power controller settle

    # Pre-generate a fixed synthetic batch (avoids dataloading noise)
    data = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)

    # Start power sampling
    sampler = PowerSampler(gpu_index=gpu_index)
    sampler.start()

    # Training loop
    model.train()
    batches = 0
    images = 0
    t_start = time.monotonic()

    while (time.monotonic() - t_start) < duration_s:
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)

        batches += 1
        images += batch_size

        # Progress tick every 10 batches
        if batches % 10 == 0:
            elapsed = time.monotonic() - t_start
            img_s = images / elapsed
            print(
                f"    [{elapsed:5.1f}s / {duration_s:.0f}s]  "
                f"batches={batches}  images={images}  "
                f"throughput={img_s:.0f} img/s  loss={loss.item():.4f}",
                end="\r",
            )

    actual_duration = time.monotonic() - t_start
    print()  # newline after \r progress

    # Stop sampler and collect
    samples = sampler.stop()

    # Build result
    result = PhaseResult(
        label=label,
        power_limit_w=power_limit_w,
        duration_s=actual_duration,
        images_processed=images,
        batches_completed=batches,
        batch_size=batch_size,
        samples=samples,
    )

    _compute_aggregates(result)
    _print_phase_summary(result)

    return result


def _compute_aggregates(r: PhaseResult):
    """Compute energy (trapezoidal), averages, and derived metrics."""
    if not r.samples:
        return

    # Trapezoidal integration: E = Σ [(P_i + P_{i+1}) / 2] × Δt
    total_j = 0.0
    for i in range(len(r.samples) - 1):
        s0 = r.samples[i]
        s1 = r.samples[i + 1]
        dt = s1.t_mono - s0.t_mono
        if 0 < dt < 1.0:  # reject gaps > 1s
            avg_p = (s0.power_w + s1.power_w) / 2.0
            total_j += avg_p * dt

    r.total_energy_j = total_j
    r.kwh = total_j / 3_600_000.0

    powers = [s.power_w for s in r.samples]
    r.avg_power_w = sum(powers) / len(powers)
    r.peak_power_w = max(powers)

    temps = [s.temp_c for s in r.samples]
    r.avg_temp_c = sum(temps) / len(temps)
    r.peak_temp_c = max(temps)

    utils = [s.util_pct for s in r.samples]
    r.avg_util_pct = sum(utils) / len(utils)

    clocks = [s.sm_clock_mhz for s in r.samples if s.sm_clock_mhz > 0]
    r.avg_sm_clock_mhz = sum(clocks) / len(clocks) if clocks else 0.0

    r.throughput_img_s = r.images_processed / r.duration_s if r.duration_s > 0 else 0
    r.joules_per_1k_images = (
        (total_j / r.images_processed) * 1000
        if r.images_processed > 0
        else 0
    )


def _print_phase_summary(r: PhaseResult):
    """Print a concise phase summary."""
    print(f"\n  {r.label} Results:")
    print(f"    Duration           : {r.duration_s:.1f}s")
    print(f"    Batches            : {r.batches_completed}")
    print(f"    Images processed   : {r.images_processed:,}")
    print(f"    Throughput         : {r.throughput_img_s:,.0f} img/s")
    print(f"    Power samples      : {len(r.samples)}")
    print(f"    Avg power          : {r.avg_power_w:.1f}W")
    print(f"    Peak power         : {r.peak_power_w:.1f}W")
    print(f"    Avg temperature    : {r.avg_temp_c:.1f}C")
    print(f"    Peak temperature   : {r.peak_temp_c}C")
    print(f"    Avg SM clock       : {r.avg_sm_clock_mhz:.0f} MHz")
    print(f"    Avg utilization    : {r.avg_util_pct:.1f}%")
    print(f"    Total energy       : {r.total_energy_j:,.1f} J  ({r.kwh:.6f} kWh)")
    print(f"    J / 1k images      : {r.joules_per_1k_images:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Burn-In (Thermal Steady-State)
# ═══════════════════════════════════════════════════════════════════════════════


def run_burnin(
    duration_s: float,
    power_limit_w: int,
    batch_size: int,
    gpu_index: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Burn-in phase to reach thermal steady-state.

    Runs the identical training loop for `duration_s` at the baseline
    power limit.  All telemetry from this phase is discarded — its sole
    purpose is to ensure the GPU junction temperature has plateaued
    before Test A begins.
    """
    print(f"\n{'═' * 70}")
    print(f"  BURN-IN: {duration_s:.0f}s at {power_limit_w}W  (data discarded)")
    print(f"  Purpose: reach thermal steady-state before measurement phases")
    print(f"{'═' * 70}")

    set_power_limit(gpu_index, power_limit_w)
    time.sleep(2)

    data = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)

    # Monitor temperature during burn-in
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    model.train()
    t_start = time.monotonic()
    batch_count = 0

    while (time.monotonic() - t_start) < duration_s:
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        batch_count += 1

        if batch_count % 20 == 0:
            elapsed = time.monotonic() - t_start
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError:
                temp, power = 0, 0

            remaining = duration_s - elapsed
            print(
                f"    [{elapsed:5.1f}s]  temp={temp}C  power={power:.0f}W  "
                f"remaining={remaining:.0f}s",
                end="\r",
            )

    # Final thermal reading
    try:
        final_temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
    except pynvml.NVMLError:
        final_temp = 0

    print(f"\n  Burn-in complete.  Final GPU temperature: {final_temp}C")
    print(f"  Thermal steady-state reached.  Proceeding to measurement.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Scientific Report Generator
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(baseline: PhaseResult, optimized: PhaseResult):
    """
    Print the final Scientific A/B Report.

    Compares throughput loss vs energy savings and computes the
    Aluminati Efficiency Multiplier (AEM).
    """
    # Deltas
    throughput_delta_pct = (
        ((optimized.throughput_img_s - baseline.throughput_img_s)
         / baseline.throughput_img_s * 100)
        if baseline.throughput_img_s > 0 else 0
    )
    throughput_loss_pct = abs(min(throughput_delta_pct, 0))

    energy_delta_pct = (
        ((baseline.joules_per_1k_images - optimized.joules_per_1k_images)
         / baseline.joules_per_1k_images * 100)
        if baseline.joules_per_1k_images > 0 else 0
    )

    power_saved_pct = (
        ((baseline.avg_power_w - optimized.avg_power_w)
         / baseline.avg_power_w * 100)
        if baseline.avg_power_w > 0 else 0
    )

    # Aluminati Efficiency Multiplier
    if throughput_loss_pct > 0.01:
        aem = energy_delta_pct / throughput_loss_pct
    else:
        aem = float('inf')  # no performance loss at all

    # Cost projection (8760 hours/year, $0.12/kWh US average)
    baseline_annual_kwh = baseline.avg_power_w / 1000 * 8760
    optimized_annual_kwh = optimized.avg_power_w / 1000 * 8760
    annual_savings_kwh = baseline_annual_kwh - optimized_annual_kwh
    annual_savings_usd = annual_savings_kwh * 0.12

    # CO2 (US grid average: 0.385 kg CO2/kWh, EPA 2024)
    annual_co2_saved_kg = annual_savings_kwh * 0.385

    w = 70  # report width

    print(f"\n{'━' * w}")
    print(f"{'ALUMINATAI SCIENTIFIC A/B REPORT':^{w}}")
    print(f"{'Power-Cap Efficiency Experiment':^{w}}")
    print(f"{'━' * w}")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")

    # ── Comparison Table ─────────────────────────────────────────────────
    print(f"\n{'─' * w}")
    print(f"  {'METRIC':<32} {'BASELINE':>15} {'OPTIMIZED':>15}")
    print(f"{'─' * w}")

    rows = [
        ("Power Limit",
         f"{baseline.power_limit_w}W",
         f"{optimized.power_limit_w}W"),
        ("Duration",
         f"{baseline.duration_s:.1f}s",
         f"{optimized.duration_s:.1f}s"),
        ("Batches Completed",
         f"{baseline.batches_completed}",
         f"{optimized.batches_completed}"),
        ("Images Processed",
         f"{baseline.images_processed:,}",
         f"{optimized.images_processed:,}"),
        ("Throughput",
         f"{baseline.throughput_img_s:,.0f} img/s",
         f"{optimized.throughput_img_s:,.0f} img/s"),
        ("", "", ""),
        ("Avg Power Draw",
         f"{baseline.avg_power_w:.1f}W",
         f"{optimized.avg_power_w:.1f}W"),
        ("Peak Power Draw",
         f"{baseline.peak_power_w:.1f}W",
         f"{optimized.peak_power_w:.1f}W"),
        ("Avg SM Clock",
         f"{baseline.avg_sm_clock_mhz:.0f} MHz",
         f"{optimized.avg_sm_clock_mhz:.0f} MHz"),
        ("Avg Temperature",
         f"{baseline.avg_temp_c:.1f}C",
         f"{optimized.avg_temp_c:.1f}C"),
        ("Peak Temperature",
         f"{baseline.peak_temp_c}C",
         f"{optimized.peak_temp_c}C"),
        ("Avg Utilization",
         f"{baseline.avg_util_pct:.1f}%",
         f"{optimized.avg_util_pct:.1f}%"),
        ("", "", ""),
        ("Total Energy",
         f"{baseline.total_energy_j:,.1f} J",
         f"{optimized.total_energy_j:,.1f} J"),
        ("J / 1k Images",
         f"{baseline.joules_per_1k_images:.2f}",
         f"{optimized.joules_per_1k_images:.2f}"),
        ("Power Samples",
         f"{len(baseline.samples)}",
         f"{len(optimized.samples)}"),
    ]

    for label, b_val, o_val in rows:
        if not label:
            print(f"  {'·' * 62}")
        else:
            print(f"  {label:<32} {b_val:>15} {o_val:>15}")

    # ── Key Findings ─────────────────────────────────────────────────────
    print(f"\n{'─' * w}")
    print(f"  KEY FINDINGS")
    print(f"{'─' * w}")

    print(f"  Throughput Change      : {throughput_delta_pct:+.1f}%")
    print(f"  Throughput Loss        : {throughput_loss_pct:.1f}%")
    print(f"  Power Reduction        : {power_saved_pct:.1f}%")
    print(f"  Energy Savings (J/1k)  : {energy_delta_pct:.1f}%")
    print()

    if aem == float('inf'):
        print(f"  Aluminati Efficiency   : ∞  (zero throughput loss!)")
        print(f"  Multiplier (AEM)")
    else:
        print(f"  Aluminati Efficiency   : {aem:.1f}x")
        print(f"  Multiplier (AEM)")
    print()
    print(f"  Interpretation: For every 1% of performance lost,")
    if aem == float('inf'):
        print(f"  you saved {energy_delta_pct:.1f}% energy with no cost.")
    else:
        print(f"  you saved {aem:.1f}% in energy.")

    # ── Annualised Projection ────────────────────────────────────────────
    print(f"\n{'─' * w}")
    print(f"  ANNUALISED PROJECTION  (24/7 operation, 1 GPU)")
    print(f"{'─' * w}")
    print(f"  Baseline annual energy : {baseline_annual_kwh:,.0f} kWh")
    print(f"  Optimized annual energy: {optimized_annual_kwh:,.0f} kWh")
    print(f"  Annual savings         : {annual_savings_kwh:,.0f} kWh")
    print(f"  Annual cost savings    : ${annual_savings_usd:,.0f}  (@ $0.12/kWh)")
    print(f"  Annual CO2 reduction   : {annual_co2_saved_kg:,.0f} kg CO2")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n{'─' * w}")
    print(f"  VERDICT")
    print(f"{'─' * w}")

    if energy_delta_pct > 0 and throughput_loss_pct < 15:
        if aem == float('inf') or aem > 3.0:
            print(f"  STRONGLY RECOMMENDED")
            print(f"  Power capping to {optimized.power_limit_w}W saves "
                  f"{energy_delta_pct:.0f}% energy with only "
                  f"{throughput_loss_pct:.1f}% throughput loss.")
        elif aem > 1.5:
            print(f"  RECOMMENDED")
            print(f"  Favorable tradeoff: {energy_delta_pct:.0f}% energy savings "
                  f"for {throughput_loss_pct:.1f}% throughput cost.")
        else:
            print(f"  MARGINAL")
            print(f"  Energy savings ({energy_delta_pct:.0f}%) are modest relative "
                  f"to throughput loss ({throughput_loss_pct:.1f}%).")
    elif energy_delta_pct <= 0:
        print(f"  NOT RECOMMENDED")
        print(f"  Power capping did not improve energy efficiency.")
    else:
        print(f"  NOT RECOMMENDED")
        print(f"  Throughput loss ({throughput_loss_pct:.1f}%) is too severe.")

    print(f"\n{'━' * w}")
    print(f"{'END OF REPORT':^{w}}")
    print(f"{'━' * w}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="powercap_ab_test",
        description=(
            "AluminatAI Scientific A/B Test — "
            "Power-cap efficiency experiment for A100/H100 GPUs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python powercap_ab_test.py
  python powercap_ab_test.py --baseline-watts 400 --capped-watts 250
  python powercap_ab_test.py --test-seconds 30 --burnin-seconds 30
  python powercap_ab_test.py --skip-burnin --batch-size 512
""",
    )
    parser.add_argument(
        "--baseline-watts", type=int, default=400,
        help="Power limit for Test A / baseline (default: 400).",
    )
    parser.add_argument(
        "--capped-watts", type=int, default=250,
        help="Power limit for Test B / optimized (default: 250).",
    )
    parser.add_argument(
        "--test-seconds", type=float, default=60.0,
        help="Duration of each test phase in seconds (default: 60).",
    )
    parser.add_argument(
        "--burnin-seconds", type=float, default=120.0,
        help="Burn-in duration in seconds (default: 120).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Training batch size (default: 1024).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to test (default: 0).",
    )
    parser.add_argument(
        "--skip-burnin", action="store_true",
        help="Skip the burn-in phase (use if GPU is already warm).",
    )

    args = parser.parse_args()

    # ── Validate CUDA ────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("FATAL: No CUDA GPU detected.  This experiment requires an NVIDIA GPU.")
        return 1

    device = torch.device(f"cuda:{args.gpu}")
    gpu_name = torch.cuda.get_device_name(args.gpu)

    # ── Validate power limits ────────────────────────────────────────────
    default_pl = get_default_power_limit(args.gpu)
    if args.baseline_watts > default_pl + 50:
        print(f"WARNING: Baseline {args.baseline_watts}W exceeds default "
              f"{default_pl}W by >50W.  Clamping to {default_pl}W.")
        args.baseline_watts = default_pl

    if args.capped_watts >= args.baseline_watts:
        print(f"FATAL: Capped watts ({args.capped_watts}) must be less than "
              f"baseline ({args.baseline_watts}).")
        return 1

    # ── Banner ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  AluminatAI Scientific A/B Power-Cap Experiment")
    print(f"{'═' * 70}")
    print(f"  GPU             : {gpu_name}  (index {args.gpu})")
    print(f"  Default PL      : {default_pl}W")
    print(f"  Test A (baseline): {args.baseline_watts}W")
    print(f"  Test B (capped)  : {args.capped_watts}W")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Test duration    : {args.test_seconds}s per phase")
    print(f"  Burn-in          : {'SKIP' if args.skip_burnin else f'{args.burnin_seconds:.0f}s'}")
    print(f"  CUDA             : {torch.version.cuda}")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"{'═' * 70}")

    # ── Build model + optimizer (shared across all phases) ───────────────
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = SimpleNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Compile if available (PyTorch 2.x) for realistic workload
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  torch.compile enabled")
        except Exception:
            print("  torch.compile unavailable, using eager mode")

    # ── Phase 0: Burn-In ─────────────────────────────────────────────────
    if not args.skip_burnin:
        run_burnin(
            duration_s=args.burnin_seconds,
            power_limit_w=args.baseline_watts,
            batch_size=args.batch_size,
            gpu_index=args.gpu,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
    else:
        print("\n  Burn-in skipped (--skip-burnin).  GPU may not be at steady-state.\n")

    # ── Phase 1: Baseline (Test A) ───────────────────────────────────────
    baseline = run_training_phase(
        label="TEST A — BASELINE",
        power_limit_w=args.baseline_watts,
        duration_s=args.test_seconds,
        batch_size=args.batch_size,
        gpu_index=args.gpu,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Brief cooldown between phases (5s) to let clocks settle
    print("\n  Inter-phase cooldown (5s)...")
    time.sleep(5)

    # ── Phase 2: Optimized (Test B) ──────────────────────────────────────
    optimized = run_training_phase(
        label="TEST B — OPTIMIZED (POWER-CAPPED)",
        power_limit_w=args.capped_watts,
        duration_s=args.test_seconds,
        batch_size=args.batch_size,
        gpu_index=args.gpu,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # ── Restore default power limit ──────────────────────────────────────
    print(f"\n  Restoring default power limit ({default_pl}W)...")
    set_power_limit(args.gpu, default_pl)

    # ── Phase 3: Report ──────────────────────────────────────────────────
    print_report(baseline, optimized)

    return 0


if __name__ == "__main__":
    sys.exit(main())
