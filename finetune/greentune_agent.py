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
# AluminatiAI — https://github.com/AgentMulder404/AluminatiAI-GreenTune
"""GreenTune Agent — Gemini-powered Energy Intelligence for LLM fine-tuning.

An autonomous agent that:
  1. Accepts natural language requests for fine-tuning jobs
  2. Analyzes historical energy data to recommend optimal configs
  3. Enforces energy governance policies (Lobster Trap)
  4. Launches and monitors training with real-time energy tracking
  5. Provides live explanations and alerts via Gemini

Usage:
    python greentune_agent.py --interactive
    python greentune_agent.py --request "Fine-tune Qwen-7B with lowest J/token"
    python greentune_agent.py --request "Compare batch sizes 1 vs 2 vs 4"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types


# ── Energy Governance Policies (Lobster Trap) ──

@dataclass
class EnergyPolicy:
    name: str
    description: str
    max_kwh: Optional[float] = None
    max_co2_grams: Optional[float] = None
    max_cost_usd: Optional[float] = None
    max_joules_per_token: Optional[float] = None
    max_duration_hours: Optional[float] = None
    require_efficiency_gain: bool = False


DEFAULT_POLICIES: list[EnergyPolicy] = [
    EnergyPolicy(
        name="carbon_budget",
        description="Reject jobs projected to exceed 50g CO2",
        max_co2_grams=50.0,
    ),
    EnergyPolicy(
        name="energy_cap",
        description="No single run may exceed 1 kWh",
        max_kwh=1.0,
    ),
    EnergyPolicy(
        name="efficiency_floor",
        description="J/token must not exceed 2x the best known config",
        max_joules_per_token=0.8,
    ),
    EnergyPolicy(
        name="cost_guard",
        description="Training energy cost must stay under $1",
        max_cost_usd=1.0,
    ),
]


@dataclass
class EnergyProjection:
    estimated_duration_s: float
    estimated_joules: float
    estimated_kwh: float
    estimated_co2_grams: float
    estimated_cost_usd: float
    estimated_jpt: float
    policy_violations: list[str] = field(default_factory=list)

    @property
    def passes_policy(self) -> bool:
        return len(self.policy_violations) == 0


# ── Historical Data Loader ──

def load_historical_runs(data_dir: str = "output") -> list[dict]:
    """Load all previous energy_metrics.json files from output directories."""
    runs = []
    base = Path(data_dir)
    if not base.exists():
        return runs

    for metrics_file in base.rglob("energy_metrics.json"):
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            config_file = metrics_file.parent / "run_config.json"
            config = {}
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
            runs.append({
                "path": str(metrics_file),
                "metrics": data,
                "config": config,
            })
        except Exception:
            continue

    return runs


def format_historical_context(runs: list[dict]) -> str:
    """Format historical runs into a context string for Gemini."""
    if not runs:
        return "No historical runs found. This will be the first run."

    lines = ["Historical energy data from previous fine-tuning runs:\n"]
    for i, run in enumerate(runs):
        summary = run["metrics"].get("summary", {})
        config = run.get("config", {})
        lines.append(f"Run {i+1}: {config.get('model', 'unknown')}")
        lines.append(f"  Batch size: {config.get('batch_size', '?')}, "
                      f"Grad accum: {config.get('grad_accum', '?')}, "
                      f"Effective batch: {config.get('effective_batch_size', '?')}")
        lines.append(f"  Duration: {summary.get('training_duration_s', 0):.1f}s")
        lines.append(f"  Total energy: {summary.get('total_joules', 0):.0f} J "
                      f"({summary.get('total_kwh', 0):.6f} kWh)")
        lines.append(f"  Avg power: {summary.get('avg_power_w', 0):.0f}W, "
                      f"Peak: {summary.get('peak_power_w', 0):.0f}W")
        lines.append(f"  J/token: {summary.get('avg_joules_per_token', 0):.4f}")
        lines.append(f"  CO2: {summary.get('total_co2_grams', 0):.2f}g")
        lines.append(f"  Cost: ${summary.get('total_cost_usd', 0):.4f}")
        lines.append("")

    return "\n".join(lines)


# ── Energy Projection Engine ──

def project_energy(
    config: dict,
    historical_runs: list[dict],
    policies: list[EnergyPolicy],
    avg_power_w: float = 680.0,
) -> EnergyProjection:
    """Project energy consumption for a proposed config based on historical data."""
    batch_size = config.get("batch_size", 2)
    grad_accum = config.get("grad_accum", 4)
    eff_batch = batch_size * grad_accum
    samples = config.get("total_samples", 500)
    epochs = config.get("epochs", 1)
    total_steps = (samples * epochs) // eff_batch

    # Estimate step time from historical data or default
    step_time = 2.5  # seconds per step default
    for run in historical_runs:
        run_config = run.get("config", {})
        if run_config.get("batch_size") == batch_size:
            run_summary = run["metrics"].get("summary", {})
            if run_summary.get("total_steps", 0) > 0:
                step_time = (
                    run_summary["training_duration_s"] / run_summary["total_steps"]
                )
                avg_power_w = run_summary.get("avg_power_w", avg_power_w)
                break

    duration = total_steps * step_time
    joules = avg_power_w * duration
    kwh = joules / 3_600_000
    co2 = kwh * 390  # US average gCO2/kWh
    cost = kwh * 0.10  # $/kWh
    tokens = samples * epochs * 512  # ~512 tokens/sample average
    jpt = joules / tokens if tokens > 0 else 0

    projection = EnergyProjection(
        estimated_duration_s=round(duration, 1),
        estimated_joules=round(joules, 1),
        estimated_kwh=round(kwh, 6),
        estimated_co2_grams=round(co2, 2),
        estimated_cost_usd=round(cost, 4),
        estimated_jpt=round(jpt, 4),
    )

    # Check policies
    for p in policies:
        if p.max_kwh and kwh > p.max_kwh:
            projection.policy_violations.append(
                f"[{p.name}] Projected {kwh:.4f} kWh exceeds limit of {p.max_kwh} kWh"
            )
        if p.max_co2_grams and co2 > p.max_co2_grams:
            projection.policy_violations.append(
                f"[{p.name}] Projected {co2:.1f}g CO2 exceeds limit of {p.max_co2_grams}g"
            )
        if p.max_cost_usd and cost > p.max_cost_usd:
            projection.policy_violations.append(
                f"[{p.name}] Projected ${cost:.4f} exceeds limit of ${p.max_cost_usd}"
            )
        if p.max_joules_per_token and jpt > p.max_joules_per_token:
            projection.policy_violations.append(
                f"[{p.name}] Projected {jpt:.4f} J/tok exceeds limit of {p.max_joules_per_token} J/tok"
            )

    return projection


# ── Gemini Agent ──

SYSTEM_PROMPT = """You are GreenTune Agent, an Energy Intelligence AI for LLM fine-tuning.

You help enterprise teams fine-tune language models efficiently by:
1. Analyzing energy data from previous training runs on AMD MI300X GPUs
2. Recommending optimal hyperparameter configurations to minimize Joules-per-token
3. Enforcing energy governance policies before launching jobs
4. Explaining energy tradeoffs in plain language

Key facts you know:
- AMD MI300X draws ~750W regardless of batch size (power saturates)
- Smaller batch sizes = same power × more time = MORE total energy wasted
- Larger batch sizes are almost always more energy-efficient on MI300X
- QLoRA with 4-bit NF4 quantization loads 7B models in ~5GB VRAM
- Energy is measured via amdsmi hardware telemetry, not estimated

When recommending configs, always output a structured JSON block with:
{
  "model": "model name",
  "batch_size": N,
  "grad_accum": N,
  "epochs": N,
  "lora_rank": N,
  "learning_rate": float,
  "hermes_max": N,
  "reasoning": "why this config"
}

When the user asks to start training, confirm the energy projection and policy check first.
Be concise, data-driven, and always mention the energy impact."""


class GreenTuneAgent:
    """Gemini-powered agent for energy-intelligent fine-tuning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        policies: Optional[list[EnergyPolicy]] = None,
        data_dir: str = "output",
        dashboard_url: Optional[str] = None,
        dashboard_api_key: Optional[str] = None,
    ):
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("Set GOOGLE_API_KEY or pass api_key")

        self.client = genai.Client(api_key=key)
        self.model_name = model_name
        self.policies = policies or DEFAULT_POLICIES
        self.data_dir = data_dir
        self.dashboard_url = dashboard_url
        self.dashboard_api_key = dashboard_api_key
        self._history: list[types.Content] = []

        # Load historical data
        self.historical_runs = load_historical_runs(data_dir)
        self.history_context = format_historical_context(self.historical_runs)

        # Send historical context as first message
        if self.historical_runs:
            self.ask(
                f"Here is the historical energy data from previous runs. "
                f"Use this to inform your recommendations:\n\n{self.history_context}"
            )

    def ask(self, request: str) -> str:
        """Send a natural language request to the agent."""
        self._history.append(types.Content(role="user", parts=[types.Part(text=request)]))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=self._history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )

        text = response.text or ""
        self._history.append(types.Content(role="model", parts=[types.Part(text=text)]))
        return text

    def recommend_config(self, request: str) -> dict:
        """Get a structured config recommendation."""
        prompt = (
            f"Based on the historical energy data, recommend an optimal config for: "
            f"{request}\n\n"
            f"Active energy policies:\n"
            + "\n".join(f"  - {p.name}: {p.description}" for p in self.policies)
            + "\n\nRespond with your reasoning and a JSON config block."
        )
        response = self.ask(prompt)
        print(f"\n{response}\n")

        # Try to extract JSON config from response
        config = self._extract_json(response)
        return config

    def check_and_launch(self, config: dict, dry_run: bool = False) -> Optional[str]:
        """Project energy, check policies, and optionally launch training."""
        projection = project_energy(config, self.historical_runs, self.policies)

        print(f"\n{'='*60}")
        print(f"  Energy Projection")
        print(f"{'='*60}")
        print(f"  Duration:    {projection.estimated_duration_s:.0f}s")
        print(f"  Energy:      {projection.estimated_joules:.0f} J ({projection.estimated_kwh:.6f} kWh)")
        print(f"  J/token:     {projection.estimated_jpt:.4f}")
        print(f"  CO2:         {projection.estimated_co2_grams:.2f}g")
        print(f"  Cost:        ${projection.estimated_cost_usd:.4f}")

        if projection.policy_violations:
            print(f"\n  POLICY VIOLATIONS:")
            for v in projection.policy_violations:
                print(f"    {v}")
            print(f"{'='*60}\n")

            # Ask Gemini to explain and suggest alternatives
            explanation = self.ask(
                f"The proposed config violates these energy policies:\n"
                + "\n".join(projection.policy_violations)
                + "\n\nExplain why and suggest a compliant alternative."
            )
            print(explanation)
            return None

        print(f"\n  All policies passed.")
        print(f"{'='*60}\n")

        if dry_run:
            print("  [DRY RUN] Would launch training with this config.")
            return None

        return self._launch_training(config)

    def analyze_completed_run(self, metrics_path: str) -> str:
        """Have Gemini analyze a completed training run."""
        with open(metrics_path) as f:
            metrics = json.load(f)

        prompt = (
            f"Analyze this completed fine-tuning run and provide insights:\n\n"
            f"{json.dumps(metrics['summary'], indent=2)}\n\n"
            f"Step-level data (first and last 3 steps):\n"
            f"{json.dumps(metrics['steps'][:3] + metrics['steps'][-3:], indent=2)}\n\n"
            f"Provide:\n"
            f"1. Energy efficiency assessment\n"
            f"2. Whether the power draw pattern is optimal for MI300X\n"
            f"3. Specific recommendations for the next run\n"
            f"4. Comparison to our historical best if available"
        )
        return self.ask(prompt)

    def _launch_training(self, config: dict) -> str:
        """Build and execute the greentune.py command."""
        cmd = [
            sys.executable, "greentune.py",
            "--model", config.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            "--batch-size", str(config.get("batch_size", 2)),
            "--grad-accum", str(config.get("grad_accum", 4)),
            "--epochs", str(config.get("epochs", 1)),
            "--lora-rank", str(config.get("lora_rank", 16)),
            "--lr", str(config.get("learning_rate", 0.0002)),
            "--hermes-only",
            "--hermes-max", str(config.get("hermes_max", 500)),
            "--logging-steps", "10",
        ]

        if self.dashboard_url and self.dashboard_api_key:
            cmd.extend([
                "--api-url", self.dashboard_url,
                "--api-key", self.dashboard_api_key,
                "--run-name", config.get("run_name", "GreenTune Agent Run"),
            ])

        print(f"  Launching: {' '.join(cmd)}\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            print(f"  {line}", end="")

        process.wait()

        if process.returncode == 0:
            # Analyze the completed run
            metrics_path = Path("output/greentune-run/energy_metrics.json")
            if metrics_path.exists():
                print("\n  Training complete. Analyzing results...\n")
                analysis = self.analyze_completed_run(str(metrics_path))
                print(analysis)
                return str(metrics_path)

        return ""

    def _extract_json(self, text: str) -> dict:
        """Extract a JSON block from Gemini's response."""
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding raw JSON
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {}


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="GreenTune Agent — Energy Intelligence for LLM Fine-Tuning")
    parser.add_argument("--request", type=str, help="Natural language request")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--analyze", type=str, help="Analyze a completed energy_metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually launch training")
    parser.add_argument("--api-key", type=str, help="Google API key (or set GOOGLE_API_KEY)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--data-dir", default="output", help="Directory with historical runs")
    parser.add_argument("--dashboard-url", type=str, help="Dashboard URL for live upload")
    parser.add_argument("--dashboard-api-key", type=str, help="Dashboard API key")
    parser.add_argument("--show-policies", action="store_true", help="Show active energy policies")
    args = parser.parse_args()

    if args.show_policies:
        print(f"\n{'='*60}")
        print(f"  Active Energy Governance Policies (Lobster Trap)")
        print(f"{'='*60}")
        for p in DEFAULT_POLICIES:
            print(f"\n  {p.name}:")
            print(f"    {p.description}")
            if p.max_kwh:
                print(f"    Limit: {p.max_kwh} kWh")
            if p.max_co2_grams:
                print(f"    Limit: {p.max_co2_grams}g CO2")
            if p.max_cost_usd:
                print(f"    Limit: ${p.max_cost_usd}")
            if p.max_joules_per_token:
                print(f"    Limit: {p.max_joules_per_token} J/token")
        print(f"\n{'='*60}\n")
        return

    agent = GreenTuneAgent(
        api_key=args.api_key,
        model_name=args.model,
        data_dir=args.data_dir,
        dashboard_url=args.dashboard_url,
        dashboard_api_key=args.dashboard_api_key,
    )

    if args.analyze:
        print(agent.analyze_completed_run(args.analyze))
        return

    if args.request:
        config = agent.recommend_config(args.request)
        if config:
            agent.check_and_launch(config, dry_run=args.dry_run)
        return

    if args.interactive:
        print(f"\n{'='*60}")
        print(f"  GreenTune Agent — Energy Intelligence")
        print(f"  Powered by Gemini | {len(agent.historical_runs)} historical runs loaded")
        print(f"  {len(agent.policies)} energy policies active")
        print(f"{'='*60}")
        print(f"\n  Commands:")
        print(f"    Type a request to get config recommendations")
        print(f"    'run <request>' — recommend AND launch training")
        print(f"    'policies' — show active energy policies")
        print(f"    'history' — show historical run data")
        print(f"    'quit' — exit\n")

        while True:
            try:
                user_input = input("greentune> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "policies":
                for p in agent.policies:
                    print(f"  {p.name}: {p.description}")
                continue
            if user_input.lower() == "history":
                print(agent.history_context)
                continue

            if user_input.lower().startswith("run "):
                request = user_input[4:]
                config = agent.recommend_config(request)
                if config:
                    confirm = input("\n  Launch training? [y/N] ").strip().lower()
                    if confirm == "y":
                        agent.check_and_launch(config, dry_run=args.dry_run)
            else:
                response = agent.ask(user_input)
                print(f"\n{response}\n")


if __name__ == "__main__":
    main()
