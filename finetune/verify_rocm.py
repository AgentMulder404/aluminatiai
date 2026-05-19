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
"""Verify ROCm + PyTorch + QLoRA stack is functional on MI300X."""

import shlex
import subprocess
import sys
import time


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_cmd(cmd: str) -> str:
    try:
        return subprocess.check_output(shlex.split(cmd), text=True).strip()
    except Exception as e:
        return f"FAILED: {e}"


def run_pipe(cmd1: str, cmd2: str) -> str:
    try:
        p1 = subprocess.Popen(shlex.split(cmd1), stdout=subprocess.PIPE, text=True)
        p2 = subprocess.Popen(shlex.split(cmd2), stdin=p1.stdout, stdout=subprocess.PIPE, text=True)
        p1.stdout.close()
        out, _ = p2.communicate()
        return out.strip()
    except Exception as e:
        return f"FAILED: {e}"


def main() -> int:
    errors = 0

    # ── 1. ROCm driver ──
    section("1. ROCm Driver & GPU Detection")
    print(f"rocm-smi:\n{run_cmd('rocm-smi --showproductname')}")
    gpu_count_str = run_pipe("rocm-smi --showid", "grep -c GPU")
    print(f"GPU count: {gpu_count_str}")

    # ── 2. PyTorch + HIP ──
    section("2. PyTorch + ROCm/HIP")
    import torch

    print(f"PyTorch version:  {torch.__version__}")
    print(f"ROCm available:   {torch.cuda.is_available()}")
    print(f"HIP version:      {torch.version.hip or 'N/A'}")
    print(f"GPU count:        {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("\nFATAL: No ROCm GPU available to PyTorch.")
        print("Check: --device=/dev/kfd --device=/dev/dri flags in Docker")
        print("Check: pip install torch --index-url .../rocm6.4")
        return 1

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total memory:    {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multi-processor: {props.multi_processor_count}")
        print(f"  GCN arch:        {props.gcnArchName}")

    # ── 3. Compute test ──
    section("3. Compute Sanity Check (bf16 matmul)")
    device = torch.device("cuda:0")
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tflops = (2 * 4096**3 * 10) / elapsed / 1e12
    print(f"bf16 matmul: {tflops:.1f} TFLOPS ({elapsed * 1000:.1f} ms for 10 iters)")

    # ── 4. Flash Attention ──
    section("4. Flash Attention (SDPA)")
    q = torch.randn(2, 32, 512, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(2, 32, 512, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 32, 512, 128, dtype=torch.bfloat16, device=device)

    try:
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            print(f"SDPA flash output shape: {out.shape}  OK")
    except Exception as e:
        print(f"SDPA flash FAILED: {e}")
        errors += 1

    try:
        from flash_attn import flash_attn_func

        out2 = flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        print(f"flash_attn_func output:  {out2.shape}  OK")
    except ImportError:
        print("flash_attn package not installed (SDPA is sufficient)")
    except Exception as e:
        print(f"flash_attn_func failed: {e}")

    # ── 5. bitsandbytes ──
    section("5. bitsandbytes (QLoRA)")
    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes version: {bnb.__version__}")
        linear_4bit = bnb.nn.Linear4bit(256, 128, bias=False, quant_type="nf4")
        linear_4bit = linear_4bit.to(device)
        x = torch.randn(1, 256, dtype=torch.float16, device=device)
        y = linear_4bit(x)
        print(f"4-bit linear output shape: {y.shape}  OK")
    except ImportError:
        print("bitsandbytes NOT installed — QLoRA will not work")
        errors += 1
    except Exception as e:
        print(f"bitsandbytes test FAILED: {e}")
        print("Try: cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx942")
        errors += 1

    # ── 6. HuggingFace stack ──
    section("6. HuggingFace Libraries")
    try:
        import transformers
        import peft
        import trl
        import accelerate
        import datasets

        print(f"transformers: {transformers.__version__}")
        print(f"peft:         {peft.__version__}")
        print(f"trl:          {trl.__version__}")
        print(f"accelerate:   {accelerate.__version__}")
        print(f"datasets:     {datasets.__version__}")
    except ImportError as e:
        print(f"Missing library: {e}")
        errors += 1

    # ── 7. Power monitoring ──
    section("7. Power Monitoring")
    power_out = run_cmd("rocm-smi --showpower")
    print(f"rocm-smi --showpower:\n{power_out}")

    try:
        from amdsmi import (
            amdsmi_init,
            amdsmi_shut_down,
            amdsmi_get_processor_handles,
            amdsmi_get_power_info,
        )

        amdsmi_init()
        devices = amdsmi_get_processor_handles()
        for i, dev in enumerate(devices):
            pwr = amdsmi_get_power_info(dev)
            print(f"\namdsmi GPU {i}:")
            print(f"  Current power:  {pwr['current_socket_power']} W")
            print(f"  Average power:  {pwr['average_socket_power']} W")
            print(f"  Power limit:    {pwr['power_limit']} W")
        amdsmi_shut_down()
        print("amdsmi Python bindings: OK")
    except ImportError:
        print("\namdsmi not available — will use rocm-smi CLI fallback")
    except Exception as e:
        print(f"\namdsmi error: {e}")

    # ── Summary ──
    section("SUMMARY")
    if errors > 0:
        print(f"WARNINGS: {errors} check(s) failed. Review above.")
        return 1

    props = torch.cuda.get_device_properties(0)
    print("All checks passed. Ready for GreenTune fine-tuning.")
    print(f"GPU:     {props.name} ({props.total_memory / 1024**3:.0f} GB)")
    print(f"ROCm:    {torch.version.hip}")
    print(f"PyTorch: {torch.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
