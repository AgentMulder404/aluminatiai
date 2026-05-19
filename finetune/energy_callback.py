#!/usr/bin/env python3
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
"""HuggingFace TrainerCallback that tracks energy metrics per training step.

Logs power draw, Joules-per-token, cumulative energy, estimated cost, and
CO2 emissions. Writes a step-level metrics JSON for the GreenTune dashboard.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests as _requests
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

try:
    from .rocm_power import PowerSamplerThread
except ImportError:
    from rocm_power import PowerSamplerThread


# US average grid carbon intensity (gCO2/kWh) — conservative default
DEFAULT_CARBON_INTENSITY = 390.0

# US average commercial electricity rate ($/kWh)
DEFAULT_ENERGY_PRICE = 0.10


@dataclass
class StepMetrics:
    step: int
    timestamp: float
    loss: float
    learning_rate: float
    tokens_processed: int
    step_time_s: float
    avg_power_w: float
    peak_power_w: float
    step_joules: float
    joules_per_token: float
    cumulative_joules: float
    cumulative_kwh: float
    cumulative_cost_usd: float
    cumulative_co2_grams: float
    temperature_c: float
    tokens_per_second: float


class EnergyCallback(TrainerCallback):
    """Tracks real-time energy consumption during HF Trainer training."""

    def __init__(
        self,
        gpu_index: int = 0,
        sample_interval_s: float = 0.5,
        carbon_intensity_gco2_kwh: float = DEFAULT_CARBON_INTENSITY,
        energy_price_usd_kwh: float = DEFAULT_ENERGY_PRICE,
        output_dir: Optional[str] = None,
        tokens_per_sample: Optional[int] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.gpu_index = gpu_index
        self.sample_interval_s = sample_interval_s
        self.carbon_intensity = carbon_intensity_gco2_kwh
        self.energy_price = energy_price_usd_kwh
        self.output_dir = output_dir
        self.tokens_per_sample = tokens_per_sample
        if api_url:
            api_url = api_url.rstrip("/")
            suffix = "/api/admin/fine-tuning/ingest"
            if api_url.endswith(suffix):
                api_url = api_url[: -len(suffix)]
        self.api_url = api_url
        self.api_key = api_key
        self.run_name = run_name

        self._sampler: Optional[PowerSamplerThread] = None
        self._step_metrics: list[dict] = []
        self._step_start_time: float = 0.0
        self._step_start_joules: float = 0.0
        self._total_tokens: int = 0
        self._training_start: float = 0.0
        self._run_id: Optional[str] = None
        self._last_power_upload_idx: int = 0

    def _api_post(self, payload: dict):
        """Fire-and-forget POST to the dashboard ingest API."""
        if not self.api_url or not self.api_key:
            return
        def _send():
            try:
                _requests.post(
                    f"{self.api_url}/api/admin/fine-tuning/ingest",
                    json=payload,
                    headers={"X-API-Key": self.api_key},
                    timeout=5,
                )
            except Exception:
                pass
        threading.Thread(target=_send, daemon=True).start()

    def _upload_power_samples(self):
        """Upload any new power samples since the last upload."""
        if not self._sampler or not self._run_id:
            return
        samples = self._sampler.accumulator.samples
        new_samples = samples[self._last_power_upload_idx:]
        if not new_samples:
            return
        self._last_power_upload_idx = len(samples)
        self._api_post({
            "action": "power_samples",
            "run_id": self._run_id,
            "samples": [
                {
                    "t": round(s.timestamp - self._training_start, 3),
                    "w": round(s.power_w, 1),
                    "c": round(s.temperature_c, 1),
                }
                for s in new_samples
            ],
        })

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._sampler = PowerSamplerThread(
            gpu_index=self.gpu_index,
            interval_s=self.sample_interval_s,
        )
        self._sampler.start()
        self._training_start = time.time()
        self._step_start_time = time.time()
        self._step_start_joules = 0.0

        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Register run with dashboard API
        if self.api_url and self.api_key:
            try:
                resp = _requests.post(
                    f"{self.api_url}/api/admin/fine-tuning/ingest",
                    json={
                        "action": "run_start",
                        "name": self.run_name or "GreenTune Run",
                        "model": getattr(args, '_name_or_path', None),
                        "epochs": args.num_train_epochs,
                        "batch_size": args.per_device_train_batch_size,
                        "grad_accum": args.gradient_accumulation_steps,
                        "effective_batch_size": (
                            args.per_device_train_batch_size
                            * args.gradient_accumulation_steps
                        ),
                        "learning_rate": args.learning_rate,
                        "max_seq_length": getattr(args, 'max_seq_length', None),
                    },
                    headers={"X-API-Key": self.api_key},
                    timeout=10,
                )
                data = resp.json()
                self._run_id = data.get("run_id")
                print(f"  Dashboard: registered run {self._run_id}")
            except Exception as e:
                print(f"  Dashboard: failed to register run — {e}")

        print(f"\n{'='*60}")
        print(f"  GreenTune Energy Monitor — GPU {self.gpu_index}")
        print(f"  Sampling power every {self.sample_interval_s}s")
        print(f"  Carbon intensity: {self.carbon_intensity} gCO2/kWh")
        print(f"  Energy price: ${self.energy_price}/kWh")
        if self._run_id:
            print(f"  Live dashboard: {self.api_url}/admin/fine-tuning")
        print(f"{'='*60}\n")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._step_start_time = time.time()
        if self._sampler:
            self._step_start_joules = self._sampler.accumulator.total_joules

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        if not self._sampler or not logs:
            return

        now = time.time()
        step_time = now - self._step_start_time
        acc = self._sampler.accumulator
        step_joules = acc.total_joules - self._step_start_joules

        # Estimate tokens processed this logging interval
        batch_tokens = self._estimate_tokens(args, state)
        self._total_tokens += batch_tokens

        joules_per_token = step_joules / batch_tokens if batch_tokens > 0 else 0.0
        tokens_per_sec = batch_tokens / step_time if step_time > 0 else 0.0

        cumulative_kwh = acc.total_kwh
        cumulative_cost = cumulative_kwh * self.energy_price
        cumulative_co2 = cumulative_kwh * self.carbon_intensity

        # Latest temperature from last sample
        temp_c = acc.samples[-1].temperature_c if acc.samples else 0.0

        step_data = StepMetrics(
            step=state.global_step,
            timestamp=now,
            loss=logs.get("loss", 0.0),
            learning_rate=logs.get("learning_rate", 0.0),
            tokens_processed=batch_tokens,
            step_time_s=round(step_time, 3),
            avg_power_w=round(acc.avg_power_w, 1),
            peak_power_w=round(acc.peak_power_w, 1),
            step_joules=round(step_joules, 2),
            joules_per_token=round(joules_per_token, 4),
            cumulative_joules=round(acc.total_joules, 2),
            cumulative_kwh=round(cumulative_kwh, 8),
            cumulative_cost_usd=round(cumulative_cost, 6),
            cumulative_co2_grams=round(cumulative_co2, 4),
            temperature_c=round(temp_c, 1),
            tokens_per_second=round(tokens_per_sec, 1),
        )

        self._step_metrics.append(step_data.__dict__)

        # Upload step to dashboard
        if self._run_id:
            self._api_post({"action": "step", "run_id": self._run_id, **step_data.__dict__})
            self._upload_power_samples()

        # Live log line
        print(
            f"  step {state.global_step:>5} | "
            f"loss {logs.get('loss', 0):.4f} | "
            f"{acc.avg_power_w:.0f}W avg | "
            f"{step_joules:.1f}J | "
            f"{joules_per_token:.2f} J/tok | "
            f"{tokens_per_sec:.0f} tok/s | "
            f"{cumulative_co2:.1f}g CO2 | "
            f"${cumulative_cost:.4f}"
        )

        # Append to logs so TensorBoard/WandB picks them up
        logs["energy/avg_power_w"] = acc.avg_power_w
        logs["energy/peak_power_w"] = acc.peak_power_w
        logs["energy/step_joules"] = step_joules
        logs["energy/joules_per_token"] = joules_per_token
        logs["energy/cumulative_kwh"] = cumulative_kwh
        logs["energy/cumulative_cost_usd"] = cumulative_cost
        logs["energy/cumulative_co2_grams"] = cumulative_co2
        logs["energy/tokens_per_second"] = tokens_per_sec
        logs["energy/temperature_c"] = temp_c

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self._sampler:
            return

        acc = self._sampler.stop()
        duration = time.time() - self._training_start

        summary = {
            "training_duration_s": round(duration, 1),
            "total_steps": state.global_step,
            "total_tokens": self._total_tokens,
            "total_joules": round(acc.total_joules, 2),
            "total_kwh": round(acc.total_kwh, 8),
            "avg_power_w": round(acc.avg_power_w, 1),
            "peak_power_w": round(acc.peak_power_w, 1),
            "total_cost_usd": round(acc.total_kwh * self.energy_price, 6),
            "total_co2_grams": round(acc.total_kwh * self.carbon_intensity, 4),
            "avg_joules_per_token": (
                round(acc.total_joules / self._total_tokens, 4)
                if self._total_tokens > 0
                else 0.0
            ),
            "avg_tokens_per_second": (
                round(self._total_tokens / duration, 1) if duration > 0 else 0.0
            ),
            "power_samples": len(acc.samples),
        }

        output = {
            "summary": summary,
            "steps": self._step_metrics,
        }

        # Save to disk
        if self.output_dir:
            metrics_path = Path(self.output_dir) / "energy_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(output, f, indent=2)

            # Also save raw power samples for detailed analysis
            samples_path = Path(self.output_dir) / "power_samples.json"
            with open(samples_path, "w") as f:
                json.dump(
                    [
                        {
                            "t": round(s.timestamp - self._training_start, 3),
                            "w": round(s.power_w, 1),
                            "c": round(s.temperature_c, 1),
                        }
                        for s in acc.samples
                    ],
                    f,
                )

        # Upload final power samples and run completion
        if self._run_id:
            self._upload_power_samples()
            self._api_post({
                "action": "run_complete",
                "run_id": self._run_id,
                **summary,
                "train_loss": state.log_history[-1].get("train_loss", 0) if state.log_history else 0,
            })

        # Print summary
        print(f"\n{'='*60}")
        print(f"  GreenTune Energy Report")
        print(f"{'='*60}")
        print(f"  Duration:          {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Total steps:       {state.global_step}")
        print(f"  Total tokens:      {self._total_tokens:,}")
        print(f"  Avg power:         {summary['avg_power_w']} W")
        print(f"  Peak power:        {summary['peak_power_w']} W")
        print(f"  Total energy:      {summary['total_joules']:.1f} J ({summary['total_kwh']:.6f} kWh)")
        print(f"  Avg J/token:       {summary['avg_joules_per_token']:.4f}")
        print(f"  Avg tokens/sec:    {summary['avg_tokens_per_second']:.1f}")
        print(f"  Energy cost:       ${summary['total_cost_usd']:.4f}")
        print(f"  CO2 emissions:     {summary['total_co2_grams']:.2f} g")
        if self.output_dir:
            print(f"  Metrics saved:     {Path(self.output_dir) / 'energy_metrics.json'}")
        print(f"{'='*60}\n")

    def _estimate_tokens(
        self, args: TrainingArguments, state: TrainerState
    ) -> int:
        if self.tokens_per_sample:
            steps_since_last = max(1, args.logging_steps)
            samples_per_step = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
            return self.tokens_per_sample * samples_per_step * steps_since_last

        # Fallback: use state.num_input_tokens_seen if available (HF ≥ 4.40)
        if hasattr(state, "num_input_tokens_seen") and state.num_input_tokens_seen:
            return state.num_input_tokens_seen - getattr(
                self, "_last_tokens_seen", 0
            )

        # Conservative fallback: assume 512 tokens per sample
        steps_since_last = max(1, args.logging_steps)
        samples_per_step = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        return 512 * samples_per_step * steps_since_last

    def get_metrics(self) -> list[dict]:
        return self._step_metrics
