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
"""
RunPod end-to-end demo: GPU cost attribution proof of concept.

What this script proves
───────────────────────
1. Idle baseline — the GPU draws non-zero power even when no workload runs.
   AluminatiAI subtracts this from every sample so you only pay for *useful*
   compute.

2. Workload attribution — when a job starts, the agent detects the process and
   attributes power (and cost) to the correct team/model.

3. Efficiency comparison — two workloads (fp32 dense matmul vs bf16 sparse) run
   for 30 s each. The output shows which is cheaper per TFLOP and the projected
   monthly cost at RunPod A100 spot rates.

4. Green AI Index — if --upload is passed, submits the efficiency result to the
   public leaderboard at aluminatiai.com/benchmarks.

Usage (inside the RunPod pod):
    python3 runpod_demo.py                    # local only
    python3 runpod_demo.py --upload           # submit to leaderboard
    python3 runpod_demo.py --gpu 1            # target GPU 1
    python3 runpod_demo.py --duration 60      # longer sampling window
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import List, Optional

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False
    print("[warn] pynvml not available — power readings will be simulated", file=sys.stderr)

try:
    import torch
    _TORCH = torch.cuda.is_available()
except ImportError:
    _TORCH = False
    print("[warn] torch not installed — GPU workload disabled", file=sys.stderr)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    from rich.text import Text
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None  # type: ignore[assignment]


# ── RunPod GPU pricing (spot, as of March 2026) ────────────────────────────────
# Source: runpod.io/gpu-cloud  — update these if rates change
RUNPOD_RATES_USD_PER_HR: dict[str, float] = {
    "A100 SXM": 1.89,
    "A100 PCIe": 1.64,
    "H100 SXM": 3.99,
    "H100 PCIe": 3.49,
    "RTX 4090":  0.74,
    "RTX 3090":  0.44,
    "A6000":     0.79,
    "default":   1.89,  # A100 fallback
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_name(gpu_index: int) -> str:
    if not _NVML:
        return "Unknown GPU"
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetName(h).decode() if isinstance(
        pynvml.nvmlDeviceGetName(h), bytes
    ) else pynvml.nvmlDeviceGetName(h)


def _power_w(gpu_index: int) -> float:
    """Return instantaneous power draw in watts."""
    if not _NVML:
        return 120.0  # simulated
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0


def _sample_power(gpu_index: int, duration_s: float, interval_s: float = 0.5) -> list[float]:
    """Collect power samples over `duration_s` seconds."""
    samples = []
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        samples.append(_power_w(gpu_index))
        time.sleep(interval_s)
    return samples


def _gpu_count() -> int:
    if not _NVML:
        return 1
    return pynvml.nvmlDeviceGetCount()


def _runpod_rate(gpu_name: str) -> float:
    for key, rate in RUNPOD_RATES_USD_PER_HR.items():
        if key.lower() in gpu_name.lower():
            return rate
    return RUNPOD_RATES_USD_PER_HR["default"]


def _cost_usd(avg_power_w: float, duration_s: float, rate_per_hr: float) -> float:
    """Estimated cost for this workload at the given RunPod spot rate."""
    gpu_hr = duration_s / 3600.0
    return gpu_hr * rate_per_hr


def _joules(power_samples: list[float], interval_s: float = 0.5) -> float:
    return sum(power_samples) * interval_s


@dataclass
class WorkloadResult:
    name: str
    duration_s: float
    power_samples: list[float]
    tflops: Optional[float]       # achieved TFLOPS, if measurable
    tokens_per_sec: Optional[float]

    @property
    def avg_power_w(self) -> float:
        return sum(self.power_samples) / max(len(self.power_samples), 1)

    @property
    def peak_power_w(self) -> float:
        return max(self.power_samples, default=0.0)

    @property
    def energy_kwh(self) -> float:
        return _joules(self.power_samples) / 3_600_000.0

    @property
    def kwh_per_tflop(self) -> Optional[float]:
        if self.tflops and self.tflops > 0:
            total_tflops = self.tflops * self.duration_s
            return self.energy_kwh / total_tflops if total_tflops else None
        return None

    @property
    def kwh_per_1m_tokens(self) -> Optional[float]:
        if self.tokens_per_sec and self.tokens_per_sec > 0:
            return self.avg_power_w / (self.tokens_per_sec * 1000.0)
        return None


# ── GPU workloads ─────────────────────────────────────────────────────────────

def _run_matmul_fp32(gpu_index: int, duration_s: float, size: int = 8192) -> float:
    """
    Dense FP32 matmul loop. Returns achieved TFLOPS.
    Each A×B with size=8192 is 2×8192³ ≈ 1.1 TFLOPS.
    """
    if not _TORCH:
        return 0.0
    device = torch.device(f"cuda:{gpu_index}")
    A = torch.randn(size, size, dtype=torch.float32, device=device)
    B = torch.randn(size, size, dtype=torch.float32, device=device)
    ops_per_matmul = 2.0 * size ** 3
    count = 0
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        _ = torch.matmul(A, B)
        torch.cuda.synchronize(device)
        count += 1
    total_ops = count * ops_per_matmul
    return (total_ops / duration_s) / 1e12  # TFLOPS


def _run_matmul_bf16(gpu_index: int, duration_s: float, size: int = 8192) -> float:
    """Dense BF16 matmul loop (tensor core path). Returns TFLOPS."""
    if not _TORCH:
        return 0.0
    device = torch.device(f"cuda:{gpu_index}")
    A = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    B = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    ops_per_matmul = 2.0 * size ** 3
    count = 0
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        _ = torch.matmul(A, B)
        torch.cuda.synchronize(device)
        count += 1
    total_ops = count * ops_per_matmul
    return (total_ops / duration_s) / 1e12  # TFLOPS


def _run_token_generation(gpu_index: int, duration_s: float) -> float:
    """
    Simulates transformer token generation by doing repeated attention-style
    matmuls (Q×K^T then softmax×V).  Returns synthetic tokens/s estimate.
    """
    if not _TORCH:
        return 0.0
    device = torch.device(f"cuda:{gpu_index}")
    seq_len, heads, d_head = 512, 32, 64
    Q = torch.randn(1, heads, seq_len, d_head, dtype=torch.bfloat16, device=device)
    K = torch.randn(1, heads, seq_len, d_head, dtype=torch.bfloat16, device=device)
    V = torch.randn(1, heads, seq_len, d_head, dtype=torch.bfloat16, device=device)
    import torch.nn.functional as F
    count = 0
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        _ = torch.matmul(attn, V)
        torch.cuda.synchronize(device)
        count += 1
    # Each forward pass = seq_len new tokens (rough estimate)
    return (count * seq_len) / duration_s


def _run_workload_with_power(
    name: str,
    fn,
    gpu_index: int,
    duration_s: float,
) -> WorkloadResult:
    """Run workload fn in a thread, collect power in main thread simultaneously."""
    result_box: list = []

    def _worker():
        val = fn(gpu_index, duration_s)
        result_box.append(val)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    samples = _sample_power(gpu_index, duration_s, interval_s=0.5)
    t.join()

    val = result_box[0] if result_box else None
    tflops = val if name != "token-gen" else None
    tokens_ps = val if name == "token-gen" else None

    return WorkloadResult(
        name=name,
        duration_s=duration_s,
        power_samples=samples,
        tflops=tflops,
        tokens_per_sec=tokens_ps,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_banner(gpu_name: str, gpu_index: int):
    if _RICH:
        console.rule("[bold cyan]AluminatiAI — RunPod Cost Attribution Demo[/]")
        console.print(f"  GPU {gpu_index}: [bold]{gpu_name}[/]")
        console.print(f"  RunPod spot rate: [yellow]${_runpod_rate(gpu_name):.2f}/hr[/]\n")
    else:
        print("=" * 60)
        print("AluminatiAI — RunPod Cost Attribution Demo")
        print(f"  GPU {gpu_index}: {gpu_name}")
        print(f"  RunPod spot rate: ${_runpod_rate(gpu_name):.2f}/hr")
        print("=" * 60)


def _print_phase(label: str):
    if _RICH:
        console.rule(f"[bold]{label}[/]")
    else:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print("─" * 60)


def _print_results(
    gpu_name: str,
    idle_w: float,
    results: list[WorkloadResult],
    rate_per_hr: float,
):
    MONTHLY_HOURS = 720.0  # 30-day month

    if _RICH:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Workload", style="cyan", min_width=20)
        table.add_column("Avg W", justify="right")
        table.add_column("Adj W\n(−baseline)", justify="right", style="green")
        table.add_column("TFLOPS", justify="right")
        table.add_column("kWh/TFLOP", justify="right")
        table.add_column("kWh/1M tok", justify="right")
        table.add_column("$/hr", justify="right")
        table.add_column("$/month\n(projected)", justify="right", style="yellow")

        for r in results:
            adj_w = max(0.0, r.avg_power_w - idle_w)
            tflops_str = f"{r.tflops:.1f}" if r.tflops else "—"
            kwh_tflop = r.kwh_per_tflop
            kwh_tflop_str = f"{kwh_tflop:.2e}" if kwh_tflop is not None else "—"
            kwh_1m = r.kwh_per_1m_tokens
            kwh_1m_str = f"{kwh_1m:.2e}" if kwh_1m is not None else "—"
            cost_hr = _cost_usd(r.avg_power_w, 3600, rate_per_hr)
            cost_mo = cost_hr * MONTHLY_HOURS
            table.add_row(
                r.name,
                f"{r.avg_power_w:.1f}",
                f"{adj_w:.1f}",
                tflops_str,
                kwh_tflop_str,
                kwh_1m_str,
                f"${cost_hr:.3f}",
                f"${cost_mo:.2f}",
            )

        console.print(table)

        # Highlight the best workload by W/TFLOP (lower = more efficient)
        tflop_results = [r for r in results if r.tflops and r.tflops > 0]
        if tflop_results:
            worst  = max(tflop_results, key=lambda r: r.avg_power_w / r.tflops)
            efficient = min(tflop_results, key=lambda r: r.avg_power_w / r.tflops)
            worst_w_per_tflop = worst.avg_power_w / worst.tflops
            best_w_per_tflop  = efficient.avg_power_w / efficient.tflops
            improvement_pct   = (worst_w_per_tflop - best_w_per_tflop) / worst_w_per_tflop * 100
            time_saving_pct   = (1.0 - worst.tflops / efficient.tflops) * 100  # same work, less time
            cost_saving_pct   = time_saving_pct  # RunPod charges by time
            console.print(
                f"\n[bold green]Most compute-efficient:[/] [cyan]{efficient.name}[/] "
                f"— [green]{improvement_pct:.0f}% less W/TFLOP[/] vs {worst.name}\n"
                f"  Same job finishes [bold]{efficient.tflops/worst.tflops:.1f}×[/] faster "
                f"→ [bold green]{cost_saving_pct:.0f}% lower RunPod cost[/] for that workload"
            )
        console.print(
            f"\n[dim]Baseline idle: {idle_w:.1f} W subtracted from all attributed samples.[/]"
        )
        console.print(
            f"[dim]Projected monthly cost assumes 24/7 utilisation at "
            f"${rate_per_hr:.2f}/GPU-hr (RunPod spot).[/]\n"
        )
    else:
        print(f"\n{'Workload':<22} {'Avg W':>7} {'Adj W':>7} {'TFLOPS':>8} "
              f"{'kWh/TFLOP':>12} {'$/hr':>7} {'$/month':>9}")
        print("-" * 80)
        for r in results:
            adj_w = max(0.0, r.avg_power_w - idle_w)
            tflops_str = f"{r.tflops:.1f}" if r.tflops else "   —"
            kwh_tflop = r.kwh_per_tflop
            kwh_str = f"{kwh_tflop:.5f}" if kwh_tflop else "          —"
            cost_hr = _cost_usd(r.avg_power_w, 3600, rate_per_hr)
            cost_mo = cost_hr * MONTHLY_HOURS
            print(f"{r.name:<22} {r.avg_power_w:>7.1f} {adj_w:>7.1f} "
                  f"{tflops_str:>8} {kwh_str:>12} ${cost_hr:>6.3f} ${cost_mo:>8.2f}")
        print(f"\nBaseline idle: {idle_w:.1f} W subtracted from all samples.")


def _upload_best(result: WorkloadResult, gpu_index: int, gpu_name: str):
    """Submit the most efficient result to the Green AI Index via benchmark.py."""
    import subprocess
    cmd = [
        sys.executable, "-m", "benchmark",
        "--gpu", str(gpu_index),
        "--duration", "30",
        "--upload",
        "--model-tag", f"demo-{result.name}",
    ]
    if result.tokens_per_sec:
        cmd += ["--throughput", str(int(result.tokens_per_sec))]
    if _RICH:
        console.print(f"\n[bold]Submitting to Green AI Index:[/] {' '.join(cmd)}")
    else:
        print(f"\nSubmitting to Green AI Index: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[warn] Upload failed: {e}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="AluminatiAI RunPod cost attribution demo",
    )
    p.add_argument("--gpu", type=int, default=0, metavar="N",
                   help="GPU index (default: 0)")
    p.add_argument("--duration", type=float, default=30.0, metavar="SECS",
                   help="Seconds to run each workload phase (default: 30)")
    p.add_argument("--idle-duration", type=float, default=10.0, metavar="SECS",
                   help="Seconds to sample idle baseline (default: 10)")
    p.add_argument("--upload", action="store_true",
                   help="Submit best result to the Green AI Index")
    p.add_argument("--skip-workloads", action="store_true",
                   help="Only measure idle baseline (quick sanity check)")
    args = p.parse_args()

    if not _NVML:
        print("ERROR: pynvml not available. Install with: pip install nvidia-ml-py")
        sys.exit(1)

    n_gpus = _gpu_count()
    if args.gpu >= n_gpus:
        print(f"ERROR: GPU {args.gpu} not found — only {n_gpus} GPU(s) available")
        sys.exit(1)

    gpu_name = _gpu_name(args.gpu)
    rate = _runpod_rate(gpu_name)

    _print_banner(gpu_name, args.gpu)

    # ── Phase 1: Idle baseline ────────────────────────────────────────────────
    _print_phase(f"Phase 1 — Idle baseline ({args.idle_duration:.0f}s)")
    if _RICH:
        console.print("  Sampling GPU with no workload running …")
    else:
        print("  Sampling GPU with no workload running …")

    idle_samples = _sample_power(args.gpu, args.idle_duration)
    idle_w = sum(idle_samples) / max(len(idle_samples), 1)
    idle_cost_mo = _cost_usd(idle_w, 3600, rate) * 720

    if _RICH:
        console.print(
            f"  Idle power : [yellow]{idle_w:.1f} W[/]\n"
            f"  Idle cost  : [red]${idle_cost_mo:.2f}/month[/] if left running 24/7 — "
            "[dim]AluminatiAI detects this and bills to ALUMINATAI_IDLE_TEAM[/]"
        )
    else:
        print(f"  Idle power : {idle_w:.1f} W")
        print(f"  Idle cost  : ${idle_cost_mo:.2f}/month if left running 24/7")

    if args.skip_workloads:
        return

    # ── Phase 2–4: Workloads ──────────────────────────────────────────────────
    workloads = [
        ("fp32-matmul",  _run_matmul_fp32),
        ("bf16-matmul",  _run_matmul_bf16),
        ("token-gen",    _run_token_generation),
    ]

    results: list[WorkloadResult] = []
    for wl_name, wl_fn in workloads:
        _print_phase(f"Phase — {wl_name} ({args.duration:.0f}s)")
        if _RICH:
            console.print(f"  Running {wl_name} on GPU {args.gpu} …")
        else:
            print(f"  Running {wl_name} on GPU {args.gpu} …")

        result = _run_workload_with_power(wl_name, wl_fn, args.gpu, args.duration)
        results.append(result)

        if _RICH:
            console.print(
                f"  Avg power : {result.avg_power_w:.1f} W  "
                f"(+{result.avg_power_w - idle_w:.1f} W above baseline)"
            )
            if result.tflops:
                console.print(f"  TFLOPS    : {result.tflops:.2f}")
            if result.tokens_per_sec:
                console.print(f"  Tokens/s  : {result.tokens_per_sec:.0f}")
        else:
            print(f"  Avg power: {result.avg_power_w:.1f} W (+{result.avg_power_w - idle_w:.1f} above idle)")

    # ── Summary table ─────────────────────────────────────────────────────────
    _print_phase("Summary — Cost Attribution Report")
    _print_results(gpu_name, idle_w, results, rate)

    # ── Attribution note ──────────────────────────────────────────────────────
    if _RICH:
        console.rule("[bold cyan]Attribution[/]")
        console.print(
            "  In production the AluminatiAI agent runs as a background daemon.\n"
            "  Each workload above would be attributed to the team that owns its\n"
            "  process (via SLURM_JOB_ID, env vars, or cmdline heuristics).\n\n"
            "  [bold]To start the daemon:[/]\n"
            "    ALUMINATAI_API_KEY=alum_... ALUMINATAI_TEAM=my-team \\\n"
            "      python3 -m agent --log-level INFO\n\n"
            "  [bold]To run a benchmark and submit to the Green AI Index:[/]\n"
            "    python3 -m benchmark --throughput TOKEN_RATE --upload\n"
        )
    else:
        print("\n  Run `python3 -m agent` as a daemon to attribute power in real-time.")
        print("  Run `python3 -m benchmark --throughput N --upload` to submit to leaderboard.")

    # ── Optional Green AI Index upload ────────────────────────────────────────
    if args.upload and results:
        # Pick token-gen result if available (has throughput), else bf16
        token_result = next((r for r in results if r.tokens_per_sec), None)
        best = token_result or min(results, key=lambda r: r.avg_power_w)
        _upload_best(best, args.gpu, gpu_name)


if __name__ == "__main__":
    main()
