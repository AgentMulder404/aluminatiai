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
Dense matrix multiplication workload for A/B energy testing.

Runs configurable FP32 or BF16 matmul on GPU and prints throughput
in a format parseable by `aluminatiai ab` (XXX.X it/s).

Usage:
    python matmul_workload.py --duration 30 --dtype bf16
    python matmul_workload.py --duration 60 --dtype fp32 --size 4096 --gpu 1
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


def run(duration: int, dtype_str: str, size: int, gpu: int) -> None:
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32
    label = dtype_str.upper()

    print(f"[matmul-{dtype_str}] {size}x{size} on GPU {gpu} for {duration}s", flush=True)

    # Allocate matrices
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)

    # Warmup
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()

    total_iters = 0
    start = time.monotonic()
    last_report = start

    while True:
        torch.mm(a, b)
        total_iters += 1

        now = time.monotonic()
        elapsed = now - start

        if elapsed >= duration:
            break

        # Progress every 5 seconds
        if now - last_report >= 5.0:
            rate = total_iters / elapsed
            print(f"[{elapsed:.0f}s] {rate:.1f} it/s", flush=True)
            last_report = now

    torch.cuda.synchronize()
    total_elapsed = time.monotonic() - start
    final_rate = total_iters / total_elapsed

    # Final line — this is what ab.py parses
    print(f"{final_rate:.1f} it/s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Dense matmul workload")
    p.add_argument("--duration", type=int, default=30, help="Duration in seconds (default: 30)")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16", help="Data type (default: bf16)")
    p.add_argument("--size", type=int, default=8192, help="Matrix size NxN (default: 8192)")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    args = p.parse_args()

    run(args.duration, args.dtype, args.size, args.gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
