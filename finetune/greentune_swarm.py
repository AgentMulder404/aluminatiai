"""
GreenTune Agent Swarm — algorithmic energy optimizer
for LLM fine-tuning on AMD MI300X GPUs.

No LLM API keys required. Runs entirely offline.

Architecture:
  Orchestrator ──┬── Config Optimizer (grid search over hyperparameters)
                 ├── Policy Guardian (enforces Lobster Trap constraints)
                 └── Energy Analyst (projects & ranks by J/token)

  Goal → Scan History → Generate Search Space → Project Energy →
  Check Policies → Rank by Efficiency → Recommend Best
"""

import glob
import json
import math
import time
from dataclasses import dataclass
from typing import Optional

# ─── Energy constants ───────────────────────────────────────────

MI300X_TDP_W = 750
GRID_CO2_KG_PER_KWH = 0.39
ENERGY_COST_PER_KWH = 0.12
BASELINE_JPT = 0.355


@dataclass
class EnergyPolicy:
    name: str
    description: str
    limit: float
    unit: str


LOBSTER_TRAP = [
    EnergyPolicy("carbon_budget", "Max CO2 per run", 50.0, "g"),
    EnergyPolicy("energy_cap", "Max energy per run", 1.0, "kWh"),
    EnergyPolicy("efficiency_floor", "Max joules per token", 0.8, "J/tok"),
    EnergyPolicy("cost_guard", "Max energy cost per run", 1.00, "USD"),
]

# ─── Tool functions ───────────────────────────────────────────


def list_historical_runs() -> dict:
    """Scan output/ for completed training runs and return energy metrics."""
    runs = []
    for path in sorted(glob.glob("output/*/energy_metrics.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            s = data.get("energy_summary", data.get("summary", {}))
            c = data.get("config", {})
            runs.append({
                "path": path,
                "model": c.get("model_name", c.get("model", "unknown")),
                "batch_size": c.get("batch_size"),
                "grad_accum": c.get("gradient_accumulation_steps", c.get("grad_accum")),
                "epochs": c.get("num_epochs", c.get("epochs")),
                "lora_rank": c.get("lora_rank"),
                "total_joules": s.get("total_energy_joules", s.get("total_joules")),
                "joules_per_token": s.get("joules_per_token", s.get("avg_joules_per_token")),
                "duration_sec": s.get("total_duration_seconds", s.get("training_duration_s")),
                "co2_grams": s.get("co2_grams", s.get("total_co2_grams")),
                "cost_usd": s.get("cost_usd", s.get("total_cost_usd")),
                "avg_power_w": s.get("avg_power_watts", s.get("avg_power_w")),
            })
        except Exception:
            continue

    if not runs:
        runs = [
            {
                "name": "Baseline (bs=2, ga=4)",
                "batch_size": 2, "grad_accum": 4, "epochs": 1,
                "lora_rank": 16, "total_joules": 87300,
                "joules_per_token": 0.355, "duration_sec": 138.5,
                "co2_grams": 9.46, "cost_usd": 0.0024,
                "avg_power_w": 630,
            },
            {
                "name": "Small Batch (bs=1, ga=8)",
                "batch_size": 1, "grad_accum": 8, "epochs": 1,
                "lora_rank": 16, "total_joules": 113834,
                "joules_per_token": 0.463, "duration_sec": 178.9,
                "co2_grams": 12.33, "cost_usd": 0.0032,
                "avg_power_w": 636,
            },
        ]
    return {"runs": runs, "count": len(runs)}


def project_energy(
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    lora_rank: int = 16,
    max_samples: int = 500,
    model_name: str = "Qwen/Qwen2.5-7B",
) -> dict:
    """Project energy consumption for a proposed training config."""
    effective_batch = batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(max_samples / effective_batch)
    total_steps = steps_per_epoch * num_epochs

    time_per_step = 0.4 + (batch_size * 0.15) + (lora_rank / 64)
    total_seconds = total_steps * time_per_step

    avg_power = MI300X_TDP_W * 0.85
    total_joules = avg_power * total_seconds
    total_kwh = total_joules / 3_600_000

    tokens_per_sample = 512
    total_tokens = max_samples * tokens_per_sample * num_epochs
    jpt = total_joules / total_tokens if total_tokens > 0 else 0

    co2_grams = total_kwh * GRID_CO2_KG_PER_KWH * 1000
    cost_usd = total_kwh * ENERGY_COST_PER_KWH

    return {
        "config": {
            "model": model_name,
            "batch_size": batch_size,
            "grad_accum": gradient_accumulation_steps,
            "epochs": num_epochs,
            "lora_rank": lora_rank,
            "max_samples": max_samples,
        },
        "projection": {
            "total_steps": total_steps,
            "duration_seconds": round(total_seconds, 1),
            "duration_human": f"{total_seconds / 60:.1f} min",
            "avg_power_watts": round(avg_power, 1),
            "total_joules": round(total_joules, 1),
            "total_kwh": round(total_kwh, 4),
            "joules_per_token": round(jpt, 4),
            "total_tokens": total_tokens,
            "co2_grams": round(co2_grams, 2),
            "cost_usd": round(cost_usd, 4),
        },
    }


def check_policies(
    total_joules: float = 0,
    joules_per_token: float = 0,
    co2_grams: float = 0,
    cost_usd: float = 0,
) -> dict:
    """Check projected energy against Lobster Trap policies."""
    results = []
    all_pass = True
    for p in LOBSTER_TRAP:
        if p.name == "carbon_budget":
            value = co2_grams
        elif p.name == "energy_cap":
            value = total_joules / 3_600_000
        elif p.name == "efficiency_floor":
            value = joules_per_token
        elif p.name == "cost_guard":
            value = cost_usd
        else:
            continue
        passed = value <= p.limit
        headroom = ((p.limit - value) / p.limit * 100) if p.limit > 0 else 0
        if not passed:
            all_pass = False
        results.append({
            "policy": p.name, "limit": p.limit, "unit": p.unit,
            "actual": round(value, 4), "passed": passed,
            "headroom_pct": round(headroom, 1),
        })
    return {"all_passed": all_pass, "policies": results}


def compare_configs(configs: list) -> dict:
    """Compare projected configs and rank by energy efficiency."""
    if not configs:
        return {"error": "No configs to compare"}
    ranked = sorted(
        configs,
        key=lambda c: c.get("projection", {}).get("joules_per_token", float("inf")),
    )
    best = ranked[0]
    worst = ranked[-1]
    best_jpt = best.get("projection", {}).get("joules_per_token", 0)
    worst_jpt = worst.get("projection", {}).get("joules_per_token", 0)
    savings = ((worst_jpt - best_jpt) / worst_jpt * 100) if worst_jpt > 0 else 0
    return {
        "ranked": ranked,
        "best": best,
        "worst": worst,
        "energy_savings_pct": round(savings, 1),
    }


def get_hardware_info() -> dict:
    """AMD MI300X hardware specifications."""
    return {
        "gpu": "AMD Instinct MI300X",
        "architecture": "CDNA3 (gfx942)",
        "vram": "192 GB HBM3",
        "tdp_watts": 750,
        "memory_bandwidth": "5.3 TB/s",
        "compute": "1307.4 TFLOPS (FP16)",
        "monitoring": "amdsmi at 0.5s intervals",
    }


# ─── Search space ─────────────────────────────────────────────

DEFAULT_SEARCH_SPACE = {
    "batch_sizes": [1, 2, 4, 8, 16, 32],
    "grad_accum_steps": [1, 2, 4, 8],
    "lora_ranks": [8, 16, 32],
    "epochs": [1],
    "max_samples": 500,
    "model": "Qwen/Qwen2.5-7B",
}


# ─── Swarm orchestrator (no LLM) ─────────────────────────────


class GreenTuneSwarm:
    """Algorithmic swarm optimizer for energy-efficient fine-tuning.

    Runs a grid search over hyperparameters, projects energy for each,
    enforces Lobster Trap policies, and ranks by J/token efficiency.
    No API keys required — works completely offline.
    """

    def __init__(
        self,
        verbose: bool = True,
        on_event=None,
        search_space: Optional[dict] = None,
        # Legacy kwargs accepted but ignored (backward compat)
        api_key: Optional[str] = None,
    ):
        self.verbose = verbose
        self.on_event = on_event
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.trace: list = []

    def _emit(self, event_type: str, agent: str, data: str = ""):
        entry = {"type": event_type, "agent": agent, "data": data, "ts": time.time()}
        self.trace.append(entry)
        if self.on_event:
            self.on_event(entry)
        if self.verbose:
            try:
                from rich.console import Console
                c = Console()
                colors = {
                    "Config Optimizer": "green",
                    "Policy Guardian": "red",
                    "Energy Analyst": "yellow",
                    "Orchestrator": "blue",
                }
                c.print(f"[bold {colors.get(agent, 'white')}][{agent}][/] {event_type}")
                if data:
                    c.print(f"  {data[:300]}")
            except ImportError:
                print(f"[{agent}] {event_type}: {data[:200]}")

    def optimize(self, goal: str = "", max_iterations: int = 3) -> dict:
        """Run the optimization loop. max_iterations is accepted for API compat."""
        self._emit("swarm_start", "Orchestrator", goal or "Minimize J/token")

        # Phase 1 — Load historical data
        self._emit("phase", "Energy Analyst", "Analyzing historical training runs")
        history = list_historical_runs()
        hw = get_hardware_info()
        self._emit("analysis", "Energy Analyst",
                    f"{history['count']} historical runs, GPU: {hw['gpu']}")

        best_historical_jpt = BASELINE_JPT
        for run in history["runs"]:
            jpt = run.get("joules_per_token", float("inf"))
            if jpt and jpt < best_historical_jpt:
                best_historical_jpt = jpt
        self._emit("analysis", "Energy Analyst",
                    f"Best historical J/token: {best_historical_jpt:.4f}")

        # Phase 2 — Grid search over configs
        self._emit("phase", "Config Optimizer", "Generating search space")
        ss = self.search_space
        projections = []
        for bs in ss["batch_sizes"]:
            for ga in ss["grad_accum_steps"]:
                for rank in ss["lora_ranks"]:
                    for epochs in ss["epochs"]:
                        proj = project_energy(
                            batch_size=bs,
                            gradient_accumulation_steps=ga,
                            num_epochs=epochs,
                            lora_rank=rank,
                            max_samples=ss["max_samples"],
                            model_name=ss["model"],
                        )
                        projections.append(proj)

        total_configs = len(projections)
        self._emit("tool_result", "Config Optimizer",
                    f"Projected energy for {total_configs} configs")

        # Phase 3 — Policy checks
        self._emit("phase", "Policy Guardian", "Checking Lobster Trap compliance")
        passing = []
        failing = []
        for proj in projections:
            p = proj["projection"]
            check = check_policies(
                total_joules=p["total_joules"],
                joules_per_token=p["joules_per_token"],
                co2_grams=p["co2_grams"],
                cost_usd=p["cost_usd"],
            )
            entry = {"config": proj["config"], "projection": p, "policies": check}
            if check["all_passed"]:
                passing.append(entry)
            else:
                failing.append(entry)

        self._emit("tool_result", "Policy Guardian",
                    f"{len(passing)}/{total_configs} configs pass all policies, "
                    f"{len(failing)} rejected")

        # Phase 4 — Rank passing configs
        self._emit("phase", "Energy Analyst", "Ranking configs by efficiency")
        if not passing:
            self._emit("error", "Policy Guardian", "No configs pass all policies")
            return {
                "success": False,
                "recommendation": None,
                "iterations": 1,
                "trace": self.trace,
                "tool_calls": [],
                "final_text": "No configs passed Lobster Trap policies.",
                "all_projections": projections,
            }

        passing.sort(key=lambda x: x["projection"]["joules_per_token"])
        best = passing[0]
        worst = passing[-1]

        best_jpt = best["projection"]["joules_per_token"]
        worst_jpt = worst["projection"]["joules_per_token"]
        savings_vs_worst = ((worst_jpt - best_jpt) / worst_jpt * 100) if worst_jpt > 0 else 0
        savings_vs_baseline = ((best_historical_jpt - best_jpt) / best_historical_jpt * 100) if best_historical_jpt > 0 else 0

        self._emit("tool_result", "Energy Analyst",
                    f"Best: bs={best['config']['batch_size']} ga={best['config']['grad_accum']} "
                    f"rank={best['config']['lora_rank']} → {best_jpt:.4f} J/tok")

        # Build recommendation
        bc = best["config"]
        bp = best["projection"]
        recommendation = {
            "model": bc["model"],
            "batch_size": bc["batch_size"],
            "grad_accum": bc["grad_accum"],
            "epochs": bc["epochs"],
            "lora_rank": bc["lora_rank"],
            "max_samples": bc["max_samples"],
            "projected_jpt": bp["joules_per_token"],
            "projected_co2_g": bp["co2_grams"],
            "projected_cost_usd": bp["cost_usd"],
            "projected_duration": bp["duration_human"],
            "projected_total_joules": bp["total_joules"],
            "savings_vs_baseline_pct": round(savings_vs_baseline, 1),
            "savings_vs_worst_pct": round(savings_vs_worst, 1),
            "configs_evaluated": total_configs,
            "configs_passed": len(passing),
        }

        # Top 5 for context
        top5 = []
        for entry in passing[:5]:
            c = entry["config"]
            p = entry["projection"]
            top5.append({
                "batch_size": c["batch_size"],
                "grad_accum": c["grad_accum"],
                "lora_rank": c["lora_rank"],
                "jpt": p["joules_per_token"],
                "co2_g": p["co2_grams"],
                "cost_usd": p["cost_usd"],
                "duration": p["duration_human"],
            })

        self._emit("recommendation", "Energy Analyst",
                    json.dumps(recommendation, indent=2))
        self._emit("swarm_complete", "Orchestrator", "Optimization complete")

        return {
            "success": True,
            "recommendation": recommendation,
            "top5": top5,
            "iterations": 1,
            "trace": self.trace,
            "tool_calls": [],
            "final_text": json.dumps(recommendation, indent=2),
            "configs_evaluated": total_configs,
            "configs_passed": len(passing),
        }


# ─── CLI entry point ───────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GreenTune Swarm — energy-efficient hyperparameter optimizer (no API key needed)")
    parser.add_argument(
        "--goal",
        default="Find the most energy-efficient QLoRA config for Qwen2.5-7B on 500 Hermes traces on AMD MI300X",
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated batch sizes to search")
    parser.add_argument("--grad-accum", type=str, default="1,2,4,8",
                        help="Comma-separated grad accum steps to search")
    parser.add_argument("--lora-ranks", type=str, default="8,16,32",
                        help="Comma-separated LoRA ranks to search")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    # Legacy flag — accepted but ignored
    parser.add_argument("--api-key", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, default=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    search_space = {
        "batch_sizes": [int(x) for x in args.batch_sizes.split(",")],
        "grad_accum_steps": [int(x) for x in args.grad_accum.split(",")],
        "lora_ranks": [int(x) for x in args.lora_ranks.split(",")],
        "epochs": [1],
        "max_samples": args.max_samples,
        "model": args.model,
    }

    total_combos = (
        len(search_space["batch_sizes"])
        * len(search_space["grad_accum_steps"])
        * len(search_space["lora_ranks"])
    )

    if not args.json:
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel(
                f"[bold]Goal:[/bold] {args.goal}\n"
                f"[bold]Search space:[/bold] {total_combos} configs\n"
                f"[bold]Agents:[/bold] Config Optimizer, Policy Guardian, Energy Analyst\n"
                f"[bold]Mode:[/bold] Offline (no API keys)",
                title="GreenTune Agent Swarm", border_style="green",
            ))
        except ImportError:
            print(f"GreenTune Agent Swarm")
            print(f"Goal: {args.goal}")
            print(f"Search space: {total_combos} configs")
            print(f"Mode: Offline (no API keys)\n")

    swarm = GreenTuneSwarm(
        verbose=not args.quiet and not args.json,
        search_space=search_space,
    )
    result = swarm.optimize(args.goal)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            console = Console()

            if result["success"]:
                rec = result["recommendation"]
                console.print(Panel(
                    json.dumps(rec, indent=2),
                    title="Recommended Config", border_style="green",
                ))

                if result.get("top5"):
                    table = Table(title="Top 5 Configs by J/token")
                    table.add_column("#", style="dim")
                    table.add_column("Batch Size", style="cyan")
                    table.add_column("Grad Accum", style="cyan")
                    table.add_column("LoRA Rank", style="cyan")
                    table.add_column("J/tok", style="green")
                    table.add_column("CO2 (g)", style="yellow")
                    table.add_column("Cost", style="yellow")
                    table.add_column("Duration", style="dim")
                    for i, cfg in enumerate(result["top5"]):
                        table.add_row(
                            str(i + 1),
                            str(cfg["batch_size"]),
                            str(cfg["grad_accum"]),
                            str(cfg["lora_rank"]),
                            f"{cfg['jpt']:.4f}",
                            f"{cfg['co2_g']:.2f}",
                            f"${cfg['cost_usd']:.4f}",
                            cfg["duration"],
                        )
                    console.print(table)

                console.print(
                    f"\n[dim]{result['configs_evaluated']} configs evaluated, "
                    f"{result['configs_passed']} passed policies, "
                    f"{len(result['trace'])} events[/dim]"
                )
            else:
                console.print(Panel(
                    "No configs passed all Lobster Trap policies.\n"
                    "Try relaxing the search space or increasing max_samples.",
                    title="No Valid Config Found", border_style="red",
                ))
        except ImportError:
            if result["success"]:
                print(json.dumps(result["recommendation"], indent=2))
            else:
                print("No configs passed all policies.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        if not args.json:
            print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
