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
Idle GPU loader — simulates wasted GPU resources.

Loads a model and dummy tensors into GPU memory, then sits idle.
Used by Demo 3 to show the cost of idle GPUs.

Usage:
    python idle_loader.py --duration 120 --gpu 0
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required.", file=sys.stderr)
    sys.exit(1)

try:
    from torchvision.models import resnet50
except ImportError:
    resnet50 = None


def run(duration: int, gpu: int) -> None:
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    # Load a real model (~100MB)
    if resnet50 is not None:
        model = resnet50(weights=None, num_classes=1000).to(device)
        model.eval()
        model_name = "ResNet-50"
    else:
        model_name = "(no torchvision)"

    # Allocate ~4GB of dummy tensors to simulate a loaded LLM
    dummy_tensors = []
    for _ in range(4):
        dummy_tensors.append(torch.randn(256, 1024, 1024, device=device))  # ~1GB each

    allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    print(f"Model loaded: {model_name}", flush=True)
    print(f"GPU memory allocated: {allocated_gb:.1f} GB", flush=True)
    print(f"Sitting idle for {duration}s — this GPU is doing nothing...", flush=True)

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        pass

    print("Idle period complete.", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Idle GPU loader (simulates waste)")
    p.add_argument("--duration", type=int, default=120, help="Idle duration in seconds (default: 120)")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    args = p.parse_args()

    run(args.duration, args.gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
