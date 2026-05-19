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
aluminatiai benchmark — GPU energy baseline measurement.

Samples power draw + utilization for --duration seconds, then prints a
rich terminal report.  With --upload, posts results to the AluminatiAI
benchmarks API using the configured API key.

Usage:
    aluminatiai benchmark [--gpu N] [--duration SECONDS] [--upload]
                          [--throughput TOKENS_PER_SEC] [--framework FRAMEWORK]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

try:
    import pynvml  # type: ignore[import-untyped]
except ImportError:
    pynvml = None  # type: ignore[assignment]

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _rich = True
except ImportError:
    _rich = False


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aluminatiai benchmark",
        description="Measure GPU energy baseline and (optionally) upload to AluminatiAI.",
    )
    p.add_argument(
        "--gpu", type=int, default=0, metavar="N",
        help="GPU index to benchmark (default: 0)",
    )
    p.add_argument(
        "--duration", type=int, default=60, metavar="SECONDS",
        help="Sampling duration in seconds (default: 60)",
    )
    p.add_argument(
        "--upload", action="store_true",
        help="Upload results to AluminatiAI benchmarks API",
    )
    p.add_argument(
        "--model-tag", default="", metavar="TAG",
        help="Model tag label (e.g. llama-3-8b) for the upload",
    )
    p.add_argument(
        "--throughput", type=float, default=None, metavar="TOKENS_PER_SEC",
        help="Measured tokens/s (or samples/s) during the window. Enables kWh/1M tokens.",
    )
    p.add_argument(
        "--framework", default="unknown", metavar="FRAMEWORK",
        help="Inference framework: pytorch, jax, vllm, ollama, triton, etc.",
    )
    return p


def _resolve_arch(gpu_name: str) -> Optional[object]:
    """Import resolve_arch from the efficiency module without crashing if absent."""
    try:
        from efficiency.gpu_specs import resolve_arch  # type: ignore[import]
        return resolve_arch(gpu_name)
    except Exception:
        return None


def _print_plain(report: dict) -> None:
    print("\n=== AluminatiAI GPU Benchmark ===")
    print(f"  GPU           : {report['gpu_name']} (index {report['gpu_index']})")
    print(f"  Duration      : {report['duration_s']} s")
    print(f"  Avg power     : {report['avg_power_w']:.1f} W")
    print(f"  Avg util      : {report['avg_util_pct']:.1f} %")
    print(f"  J / GPU-hr    : {report['j_per_gpu_hour']:,.0f}")
    print(f"  kWh / GPU-hr  : {report['kwh_per_gpu_hour']:.4f}")
    if report.get("j_per_tflop") is not None:
        print(f"  J / TFLOP     : {report['j_per_tflop']:.4f}")
    if report.get("kwh_per_1m_tokens") is not None:
        print(f"  kWh / 1M tok  : {report['kwh_per_1m_tokens']:.6f}")
    if report.get("framework") and report["framework"] != "unknown":
        print(f"  Framework     : {report['framework']}")
    print()


def _print_rich(report: dict) -> None:
    console = Console()
    console.print("\n[bold]AluminatiAI GPU Benchmark[/bold]", style="green")

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Field", style="dim")
    t.add_column("Value")
    t.add_row("GPU", f"{report['gpu_name']} (index {report['gpu_index']})")
    t.add_row("Duration", f"{report['duration_s']} s")
    t.add_row("Avg power", f"{report['avg_power_w']:.1f} W")
    t.add_row("Avg util", f"{report['avg_util_pct']:.1f} %")
    t.add_row(
        "J / GPU-hr",
        Text(f"{report['j_per_gpu_hour']:,.0f}", style="bold green"),
    )
    t.add_row("kWh / GPU-hr", f"{report['kwh_per_gpu_hour']:.4f}")
    if report.get("j_per_tflop") is not None:
        t.add_row("J / TFLOP", f"{report['j_per_tflop']:.4f}")
    if report.get("kwh_per_1m_tokens") is not None:
        t.add_row(
            "kWh / 1M tokens",
            Text(f"{report['kwh_per_1m_tokens']:.6f}", style="bold cyan"),
        )
    if report.get("framework") and report["framework"] != "unknown":
        t.add_row("Framework", report["framework"])
    console.print(t)
    console.print()


def run_benchmark(args: argparse.Namespace) -> int:
    if pynvml is None:
        print(
            "Error: pynvml not installed. Install with: pip install nvidia-ml-py",
            file=sys.stderr,
        )
        return 1

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"Error: NVML init failed — {e}", file=sys.stderr)
        return 1

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if args.gpu >= device_count:
            print(
                f"Error: GPU index {args.gpu} not found (system has {device_count} GPU(s)).",
                file=sys.stderr,
            )
            return 1

        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        gpu_name: str = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()

        print(
            f"Sampling GPU {args.gpu} ({gpu_name}) for {args.duration} s …",
            flush=True,
        )

        power_samples: list[float] = []
        util_samples: list[float] = []
        duration = args.duration

        for _ in range(duration):
            try:
                # Power in milliwatts → convert to watts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_samples.append(power_mw / 1000.0)
            except pynvml.NVMLError:
                pass

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_samples.append(float(util.gpu))
            except pynvml.NVMLError:
                pass

            time.sleep(1.0)

        if not power_samples:
            print("Error: No power readings collected.", file=sys.stderr)
            return 1

        avg_power_w = sum(power_samples) / len(power_samples)
        avg_util_pct = (
            sum(util_samples) / len(util_samples) if util_samples else 0.0
        )

        # J/GPU-hr = avg_power_W × 3600 s/hr
        j_per_gpu_hour = avg_power_w * 3600.0
        kwh_per_gpu_hour = j_per_gpu_hour / 3_600_000.0

        # J/TFLOP estimate (only if GPU wasn't idle during the run)
        j_per_tflop: Optional[float] = None
        if avg_util_pct > 5:
            arch = _resolve_arch(gpu_name)
            if arch is not None:
                util_frac = avg_util_pct / 100.0
                effective_tflops = arch.fp16_tflops * util_frac  # type: ignore[attr-defined]
                if effective_tflops > 0:
                    j_per_tflop = avg_power_w / effective_tflops

        # kWh per 1M tokens (inference efficiency):
        # kwh_per_1m_tokens = avg_power_w / (tokens_per_second × 1000)
        # e.g. 300W GPU, 1500 tok/s → 300 / 1_500_000 = 0.000200 kWh/1M tokens
        kwh_per_1m_tokens: Optional[float] = None
        if args.throughput and args.throughput > 0:
            kwh_per_1m_tokens = round(avg_power_w / (args.throughput * 1_000.0), 6)

        report = {
            "gpu_index": args.gpu,
            "gpu_name": gpu_name,
            "duration_s": duration,
            "avg_power_w": round(avg_power_w, 2),
            "avg_util_pct": round(avg_util_pct, 1),
            "j_per_gpu_hour": round(j_per_gpu_hour, 2),
            "kwh_per_gpu_hour": round(kwh_per_gpu_hour, 6),
            "j_per_tflop": round(j_per_tflop, 4) if j_per_tflop is not None else None,
            "throughput_tokens_s": args.throughput,
            "framework": args.framework,
            "kwh_per_1m_tokens": kwh_per_1m_tokens,
        }

        if _rich:
            _print_rich(report)
        else:
            _print_plain(report)

        if not args.upload:
            print(
                "To submit to the Green AI Index:\n"
                f"  aluminatiai benchmark --throughput <TOKENS/S>"
                f" --model-tag <MODEL> --upload"
            )

        if args.upload:
            _upload(report, args.model_tag)

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return 0


def _upload(report: dict, model_tag: str) -> None:
    """POST benchmark result to /api/benchmarks/hardware."""
    import os

    api_key = os.getenv("ALUMINATAI_API_KEY", "")
    endpoint = os.getenv(
        "ALUMINATAI_API_ENDPOINT",
        "https://aluminatiai.com/v1/metrics/ingest",
    )
    # Derive base URL from ingest endpoint
    base = endpoint.rstrip("/")
    if base.endswith("/v1/metrics/ingest"):
        base = base[: -len("/v1/metrics/ingest")]

    hardware_url = f"{base}/api/benchmarks/hardware"

    payload = {
        "gpu_arch": report["gpu_name"],
        "model_tag": model_tag or "unknown",
        "avg_power_w": report["avg_power_w"],
        "energy_j_per_gpu_hour": report["j_per_gpu_hour"],
        "duration_seconds": report["duration_s"],
        "gpu_count": 1,
        "precision_tag": "unknown",
        "tokens_per_second": report.get("throughput_tokens_s"),
        "framework_tag": report.get("framework", "unknown"),
        "kwh_per_1m_tokens": report.get("kwh_per_1m_tokens"),
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        hardware_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
            print(f"Uploaded: {body}")
    except urllib.error.HTTPError as e:
        print(f"Upload failed: HTTP {e.code}", file=sys.stderr)
    except Exception as e:
        print(f"Upload failed: {e}", file=sys.stderr)
