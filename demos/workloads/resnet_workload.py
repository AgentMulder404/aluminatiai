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
ResNet-18 training workload for A/B energy testing.

Trains ResNet-18 on synthetic ImageNet-scale data and prints throughput
in a format parseable by `aluminatiai ab` (XXX.X samples/s).

Usage:
    python resnet_workload.py --duration 30 --dtype fp32 --batch-size 128
    python resnet_workload.py --duration 60 --dtype bf16 --batch-size 256 --gpu 1
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

try:
    from torchvision.models import resnet18
except ImportError:
    print("ERROR: torchvision is required. Install with: pip install torchvision", file=sys.stderr)
    sys.exit(1)


def run(duration: int, dtype_str: str, batch_size: int, gpu: int) -> None:
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    use_amp = dtype_str in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    print(f"[resnet18-{dtype_str}] batch={batch_size} on GPU {gpu} for {duration}s", flush=True)

    # Model + optimizer
    model = resnet18(weights=None, num_classes=1000).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Synthetic data (pinned in GPU memory for speed)
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)

    # Warmup
    for _ in range(3):
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(images)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    total_samples = 0
    start = time.monotonic()
    last_report = start

    while True:
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(images)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_samples += batch_size
        now = time.monotonic()
        elapsed = now - start

        if elapsed >= duration:
            break

        # Progress every 5 seconds
        if now - last_report >= 5.0:
            rate = total_samples / elapsed
            print(f"[{elapsed:.0f}s] {rate:.1f} samples/s | loss={loss.item():.3f}", flush=True)
            last_report = now

    torch.cuda.synchronize()
    total_elapsed = time.monotonic() - start
    final_rate = total_samples / total_elapsed

    # Final line — this is what ab.py parses
    print(f"{final_rate:.1f} samples/s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="ResNet-18 training workload")
    p.add_argument("--duration", type=int, default=30, help="Duration in seconds (default: 30)")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32", help="Precision (default: fp32)")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    args = p.parse_args()

    run(args.duration, args.dtype, args.batch_size, args.gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
