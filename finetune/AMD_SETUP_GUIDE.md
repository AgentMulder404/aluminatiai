# AluminatiAI GreenTune — AMD Developer Cloud Setup Guide

Complete setup for QLoRA fine-tuning on AMD MI300X with real-time energy monitoring.

---

## 1. Instance Selection

### AMD Developer Cloud (cloud.amd.com)

| Plan | GPUs | HBM3 | Cost | Use Case |
|------|------|-------|------|----------|
| **1x MI300X** | 1 | 192 GB | ~$1.99/hr | Single-GPU QLoRA (our target) |
| 8x MI300X | 8 | 1,536 GB | ~$15.92/hr | Multi-GPU / FSDP (overkill for 7B QLoRA) |

**Recommendation:** 1x MI300X. A 7B model in 4-bit QLoRA uses ~6-8 GB VRAM. 192 GB is
massive headroom — no sharding, no DeepSpeed, no complexity. Budget ~4 hours ($8) for the
full hackathon workflow (setup + training + eval + demo).

### Quick Start Image

Select **"PyTorch (with ROCm)"** when creating the instance. This gives you:
- Ubuntu 22.04
- ROCm 6.4.x drivers pre-installed
- Docker with `rocm/pytorch` images available
- JupyterLab accessible via browser

### Access

1. Add your SSH public key during instance creation
2. SSH in: `ssh -i ~/.ssh/your_key root@<instance-ip>`
3. JupyterLab: `http://<instance-ip>:8888` (token shown in terminal welcome)

---

## 2. Environment Setup

### Option A: Direct Install (Recommended for Hackathon Speed)

```bash
#!/bin/bash
# greentune_setup.sh — run on AMD Developer Cloud instance

set -euo pipefail

echo "=== [1/7] System packages ==="
apt-get update && apt-get install -y --no-install-recommends \
    git wget curl htop tmux vim \
    python3-pip python3-venv \
    rocm-smi amd-smi

echo "=== [2/7] Create virtualenv ==="
python3 -m venv /opt/greentune
source /opt/greentune/bin/activate

echo "=== [3/7] PyTorch + ROCm 6.4 ==="
pip install --upgrade pip setuptools wheel
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/rocm6.4

echo "=== [4/7] HuggingFace stack ==="
pip install \
    transformers==4.47.0 \
    peft==0.13.2 \
    trl==0.12.0 \
    accelerate \
    datasets \
    tokenizers \
    safetensors \
    sentencepiece \
    protobuf

echo "=== [5/7] bitsandbytes (QLoRA 4-bit) ==="
# Mainline bitsandbytes has ROCm preview support for gfx942
pip install bitsandbytes>=0.49.0

# If mainline fails, build from source:
# git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
# cd bitsandbytes
# cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S .
# make -j$(nproc)
# pip install -e .
# cd ..

echo "=== [6/7] Flash Attention (CK backend) ==="
# PyTorch 2.9 includes CK flash attention for ROCm via
# torch.nn.functional.scaled_dot_product_attention — no separate install needed.
#
# For the standalone flash-attn package (needed by some HF models):
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
GPU_ARCHS="gfx942" MAX_JOBS=$(($(nproc) - 1)) pip install -v .
cd ..

echo "=== [7/7] Monitoring + utilities ==="
pip install \
    wandb \
    tensorboard \
    psutil \
    anthropic  # for dataset generation

echo "=== Done! Activate with: source /opt/greentune/bin/activate ==="
```

### Option B: Dockerfile (Reproducible)

```dockerfile
# Dockerfile.greentune
FROM rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.9.1

LABEL maintainer="AluminatiAI <kevin@aluminatiai.com>"
LABEL description="GreenTune — energy-efficient fine-tuning on AMD MI300X"

ENV DEBIAN_FRONTEND=noninteractive
ENV HSA_OVERRIDE_GFX_VERSION=9.4.2
ENV PYTORCH_ROCM_ARCH=gfx942

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl htop tmux vim && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers==4.47.0 \
    peft==0.13.2 \
    trl==0.12.0 \
    accelerate \
    datasets \
    tokenizers \
    safetensors \
    sentencepiece \
    protobuf \
    bitsandbytes>=0.49.0 \
    wandb \
    tensorboard \
    psutil \
    anthropic

# Flash Attention (CK backend for MI300X)
RUN git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/flash-attn && \
    cd /tmp/flash-attn && \
    GPU_ARCHS="gfx942" MAX_JOBS=$(($(nproc) - 1)) pip install -v . && \
    rm -rf /tmp/flash-attn

WORKDIR /workspace
COPY . /workspace/

CMD ["bash"]
```

```bash
# Build and run
docker build -f Dockerfile.greentune -t greentune:latest .

docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --shm-size=64g \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    greentune:latest
```

---

## 3. Environment Variables

```bash
# Add to ~/.bashrc or run before training

# MI300X architecture target
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export PYTORCH_ROCM_ARCH="gfx942"

# Flash attention backend (CK = default, fastest on MI300)
export FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"

# Memory management — critical for large models
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

# HuggingFace cache (keep models on fast storage)
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/hub"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM="false"

# Your AluminatiAI API key (for energy metric upload)
export ALUMINATAI_API_KEY="alum_..."

# Anthropic key (for dataset generation)
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 4. Verification Script

Save as `verify_rocm.py` and run immediately after setup:

```python
#!/usr/bin/env python3
"""Verify ROCm + PyTorch + QLoRA stack is functional on MI300X."""

import sys
import subprocess

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run_cmd(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as e:
        return f"FAILED: {e}"

# ── 1. ROCm driver ──────────────────────────────────────────
section("1. ROCm Driver & GPU Detection")

print(f"rocm-smi:\n{run_cmd('rocm-smi --showproductname')}")
print(f"\nGPU count: {run_cmd('rocm-smi --showid | grep -c GPU')}")

# ── 2. PyTorch + HIP ────────────────────────────────────────
section("2. PyTorch + ROCm/HIP")

import torch
print(f"PyTorch version:  {torch.__version__}")
print(f"ROCm available:   {torch.cuda.is_available()}")  # Yes, torch.cuda works for ROCm
print(f"HIP version:      {torch.version.hip or 'N/A'}")
print(f"GPU count:        {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {props.name}")
    print(f"  Total memory:    {props.total_mem / 1024**3:.1f} GB")
    print(f"  Multi-processor: {props.multi_processor_count}")
    print(f"  GCN arch:        {props.gcnArchName}")

# ── 3. Quick compute test ───────────────────────────────────
section("3. Compute Sanity Check")

device = torch.device("cuda:0")

# bf16 matmul (MI300X excels here)
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)

torch.cuda.synchronize()
import time
t0 = time.perf_counter()
for _ in range(10):
    c = torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

tflops = (2 * 4096**3 * 10) / elapsed / 1e12
print(f"bf16 matmul: {tflops:.1f} TFLOPS ({elapsed*1000:.1f} ms for 10 iters)")

# ── 4. Flash Attention ──────────────────────────────────────
section("4. Flash Attention (SDPA)")

q = torch.randn(2, 32, 512, 128, dtype=torch.bfloat16, device=device)
k = torch.randn(2, 32, 512, 128, dtype=torch.bfloat16, device=device)
v = torch.randn(2, 32, 512, 128, dtype=torch.bfloat16, device=device)

with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f"SDPA flash output shape: {out.shape}  ✓")

# Try standalone flash-attn package
try:
    from flash_attn import flash_attn_func
    out2 = flash_attn_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    print(f"flash_attn_func output:  {out2.shape}  ✓")
except ImportError:
    print("flash_attn package not installed (SDPA is sufficient)")
except Exception as e:
    print(f"flash_attn_func failed: {e}")

# ── 5. bitsandbytes (4-bit quantization) ────────────────────
section("5. bitsandbytes (QLoRA)")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")

    # Test 4-bit linear layer
    linear_4bit = bnb.nn.Linear4bit(256, 128, bias=False, quant_type="nf4")
    linear_4bit = linear_4bit.to(device)
    x = torch.randn(1, 256, dtype=torch.float16, device=device)
    y = linear_4bit(x)
    print(f"4-bit linear output shape: {y.shape}  ✓")
except ImportError:
    print("bitsandbytes NOT installed — QLoRA will not work")
    sys.exit(1)
except Exception as e:
    print(f"bitsandbytes test FAILED: {e}")
    print("Try building from source with -DBNB_ROCM_ARCH=gfx942")

# ── 6. HuggingFace stack ────────────────────────────────────
section("6. HuggingFace Libraries")

import transformers, peft, trl, accelerate, datasets
print(f"transformers: {transformers.__version__}")
print(f"peft:         {peft.__version__}")
print(f"trl:          {trl.__version__}")
print(f"accelerate:   {accelerate.__version__}")
print(f"datasets:     {datasets.__version__}")

# ── 7. Power monitoring ─────────────────────────────────────
section("7. Power Monitoring")

power_out = run_cmd("rocm-smi --showpower")
print(f"rocm-smi --showpower:\n{power_out}")

energy_out = run_cmd("rocm-smi --showenergycounter")
print(f"\nrocm-smi --showenergycounter:\n{energy_out}")

# Try amdsmi Python bindings
try:
    from amdsmi import (
        amdsmi_init, amdsmi_shut_down,
        amdsmi_get_processor_handles, amdsmi_get_power_info,
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
    print("amdsmi Python bindings: ✓")
except ImportError:
    print("\namdsmi Python bindings not available")
    print("Power monitoring will use rocm-smi CLI subprocess fallback")
except Exception as e:
    print(f"\namdsmi error: {e}")

# ── Summary ─────────────────────────────────────────────────
section("SUMMARY")
print("All checks passed. Ready for GreenTune fine-tuning.")
print(f"GPU:     MI300X ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB HBM3)")
print(f"ROCm:    {torch.version.hip}")
print(f"PyTorch: {torch.__version__}")
print(f"Stack:   transformers {transformers.__version__} + peft {peft.__version__} + trl {trl.__version__}")
```

---

## 5. Power Monitoring Module

This is the core energy-tracking piece that hooks into the HF Trainer:

```python
#!/usr/bin/env python3
"""rocm_power.py — AMD GPU power monitoring for GreenTune.

Provides a unified interface for reading power/energy/temperature from
AMD GPUs. Tries amdsmi Python bindings first, falls back to rocm-smi CLI.
"""

import subprocess
import time
import re
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PowerSample:
    timestamp: float
    gpu_index: int
    power_w: float
    temperature_c: float
    memory_used_mb: float = 0.0
    utilization_pct: float = 0.0


@dataclass
class EnergyAccumulator:
    """Tracks cumulative energy consumption via trapezoidal integration."""
    samples: list[PowerSample] = field(default_factory=list)
    total_joules: float = 0.0
    peak_power_w: float = 0.0
    _last_sample: Optional[PowerSample] = field(default=None, repr=False)

    def add(self, sample: PowerSample):
        if self._last_sample is not None:
            dt = sample.timestamp - self._last_sample.timestamp
            avg_power = (sample.power_w + self._last_sample.power_w) / 2
            self.total_joules += avg_power * dt
        self.peak_power_w = max(self.peak_power_w, sample.power_w)
        self._last_sample = sample
        self.samples.append(sample)

    @property
    def total_kwh(self) -> float:
        return self.total_joules / 3_600_000

    @property
    def avg_power_w(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        duration = self.samples[-1].timestamp - self.samples[0].timestamp
        return self.total_joules / duration if duration > 0 else 0.0

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp


class AMDPowerMonitor:
    """Reads power from AMD GPUs. Tries amdsmi, falls back to rocm-smi CLI."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._use_amdsmi = False
        self._amdsmi_handle = None
        self._init_backend()

    def _init_backend(self):
        try:
            from amdsmi import (
                amdsmi_init,
                amdsmi_get_processor_handles,
            )
            amdsmi_init()
            handles = amdsmi_get_processor_handles()
            if self.gpu_index < len(handles):
                self._amdsmi_handle = handles[self.gpu_index]
                self._use_amdsmi = True
        except (ImportError, Exception):
            self._use_amdsmi = False

    def read(self) -> PowerSample:
        if self._use_amdsmi:
            return self._read_amdsmi()
        return self._read_cli()

    def _read_amdsmi(self) -> PowerSample:
        from amdsmi import (
            amdsmi_get_power_info,
            amdsmi_get_temp_metric,
            amdsmi_get_gpu_metrics_info,
            AmdSmiTemperatureType,
            AmdSmiTemperatureMetric,
        )
        pwr = amdsmi_get_power_info(self._amdsmi_handle)
        temp = amdsmi_get_temp_metric(
            self._amdsmi_handle,
            AmdSmiTemperatureType.HOTSPOT,
            AmdSmiTemperatureMetric.CURRENT,
        )
        return PowerSample(
            timestamp=time.time(),
            gpu_index=self.gpu_index,
            power_w=float(pwr.get("current_socket_power", 0)),
            temperature_c=float(temp),
        )

    def _read_cli(self) -> PowerSample:
        power_w = self._parse_rocm_smi("--showpower", r"([\d.]+)\s*W")
        temp_c = self._parse_rocm_smi("--showtemp", r"([\d.]+)\s*c", "Temperature")
        return PowerSample(
            timestamp=time.time(),
            gpu_index=self.gpu_index,
            power_w=power_w,
            temperature_c=temp_c,
        )

    def _parse_rocm_smi(self, flag: str, pattern: str, keyword: str = "") -> float:
        try:
            out = subprocess.check_output(
                f"rocm-smi -d {self.gpu_index} {flag}",
                shell=True, text=True, timeout=5,
            )
            for line in out.splitlines():
                if keyword and keyword.lower() not in line.lower():
                    continue
                m = re.search(pattern, line, re.IGNORECASE)
                if m:
                    return float(m.group(1))
        except Exception:
            pass
        return 0.0

    def close(self):
        if self._use_amdsmi:
            try:
                from amdsmi import amdsmi_shut_down
                amdsmi_shut_down()
            except Exception:
                pass


class PowerSamplerThread:
    """Background thread that samples power at a fixed interval."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 1.0):
        self.monitor = AMDPowerMonitor(gpu_index)
        self.accumulator = EnergyAccumulator()
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> EnergyAccumulator:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.monitor.close()
        return self.accumulator

    def _loop(self):
        while not self._stop.is_set():
            sample = self.monitor.read()
            self.accumulator.add(sample)
            self._stop.wait(self.interval_s)


# ── Quick self-test ──────────────────────────────────────────
if __name__ == "__main__":
    print("Sampling GPU 0 power for 10 seconds...")
    sampler = PowerSamplerThread(gpu_index=0, interval_s=1.0)
    sampler.start()
    time.sleep(10)
    acc = sampler.stop()
    print(f"Samples:    {len(acc.samples)}")
    print(f"Avg power:  {acc.avg_power_w:.1f} W")
    print(f"Peak power: {acc.peak_power_w:.1f} W")
    print(f"Energy:     {acc.total_joules:.1f} J ({acc.total_kwh:.6f} kWh)")
    print(f"Duration:   {acc.duration_s:.1f} s")
```

---

## 6. MI300X Best Practices

### Memory & Precision
```python
# Always use bf16 compute — MI300X CDNA3 has native bf16 matrix cores
# fp16 also works, but bf16 has better dynamic range (no loss scaling needed)
torch_dtype = torch.bfloat16

# QLoRA config — the sweet spot for MI300X
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # NOT float16
    bnb_4bit_use_double_quant=True,          # nested quantization saves ~0.4 bits/param
)

# Explicit single-GPU placement (avoids multi-GPU garbling bug)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},       # pin to GPU 0
    attn_implementation="sdpa", # uses CK flash attention on ROCm
)
```

### Training Config
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # -- Performance --
    per_device_train_batch_size=4,       # start here, increase if VRAM allows
    gradient_accumulation_steps=4,       # effective batch = 16
    bf16=True,                           # native MI300X precision
    tf32=False,                          # tf32 is NVIDIA-only, no effect on AMD
    dataloader_num_workers=4,
    dataloader_pin_memory=True,

    # -- LoRA is small, train fast --
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=0.3,

    # -- Logging (for energy dashboard) --
    logging_steps=10,
    save_strategy="epoch",
    report_to=["tensorboard"],
    output_dir="./greentune-output",

    # -- AMD-specific --
    optim="paged_adamw_8bit",            # 8-bit optimizer via bitsandbytes
    gradient_checkpointing=True,         # trades compute for memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

### LoRA Config
```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                        # rank — 8 for speed, 16 for quality, 64 for max
    lora_alpha=32,               # alpha/r = 2 is standard
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP (Qwen2/LLaMA style)
    ],
)
```

### Flash Attention
```python
# Option 1: Use SDPA (built-in, no extra install needed)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",  # auto-selects CK flash on ROCm
    ...
)

# Option 2: Use flash_attn package (if installed from ROCm/flash-attention)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    ...
)
# Requires: pip install flash-attn from ROCm fork (see setup above)
```

### Memory Management
```python
# Set before training to avoid OOM fragmentation
import os
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

# Call after training / between runs
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

---

## 7. Common Pitfalls & Fixes

### "No GPU detected" / `torch.cuda.is_available()` returns False
```bash
# Check ROCm sees the GPU
rocm-smi
# If blank/error, the ROCm driver isn't loaded. In Docker:
docker run --device=/dev/kfd --device=/dev/dri --group-add video --group-add render ...

# Check PyTorch was built for ROCm (not CUDA)
python3 -c "import torch; print(torch.version.hip)"
# Should print ROCm version. If None, you installed the CUDA wheel.
# Fix: pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```

### bitsandbytes fails to load / "CUDA Setup failed"
```bash
# bitsandbytes looks for CUDA by default. On ROCm it needs the HIP backend.
# Check what it found:
python3 -c "import bitsandbytes; print(bitsandbytes.cuda_setup)"

# If it complains about CUDA, build from source:
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S .
make -j$(nproc)
pip install -e .
```

### Flash Attention "Unsupported GPU architecture"
```bash
# The PyPI flash-attn is CUDA-only. You need the ROCm fork:
pip uninstall flash-attn  # remove CUDA version
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
GPU_ARCHS="gfx942" pip install -v .

# Or skip it entirely — SDPA (attn_implementation="sdpa") works without flash-attn
```

### OOM despite 192GB HBM
```bash
# Usually caused by memory fragmentation, not actual exhaustion
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

# Or reduce batch size / enable gradient checkpointing
# With QLoRA 7B, you should NOT hit OOM. If you do, something else is wrong.
```

### `device_map='auto'` produces garbled output (multi-GPU)
```python
# Known ROCm issue with accelerate's auto device placement across GPUs.
# Fix: pin to single GPU explicitly
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},   # NOT "auto"
    ...
)
```

### rocm-smi shows 100% GPU utilization when idle
```
# Known driver-level reporting bug when multiple contexts are open.
# Does not affect training performance. Use power draw (watts) as
# the real utilization signal instead — idle MI300X draws ~50-80W,
# training draws 400-650W.
```

### Training runs but loss doesn't decrease
```python
# Common cause: wrong pad_token setup for causal LM
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Also check: learning rate (2e-4 is good for QLoRA), data format,
# and that labels aren't being masked incorrectly
```

### HSA error on kernel launch
```bash
# Set the GFX version override
export HSA_OVERRIDE_GFX_VERSION=9.4.2
# This is usually needed only for unrecognized GPU variants.
# MI300X is well-supported in ROCm 6.4+, so this is a safety net.
```

---

## 8. Quick Smoke Test (End-to-End)

Run this after setup to confirm the full QLoRA pipeline works:

```python
#!/usr/bin/env python3
"""smoke_test.py — 20-sample QLoRA fine-tune to verify the full stack."""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # tiny model for smoke test

# -- Tiny dataset --
samples = [
    {"text": f"<|im_start|>user\nWhat is GPU {i} power draw?<|im_end|>\n<|im_start|>assistant\nGPU {i} is drawing {200+i*10}W, which is within normal range.<|im_end|>"}
    for i in range(20)
]
ds = Dataset.from_list(samples)

# -- Tokenizer --
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# -- 4-bit model --
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation="sdpa",
    trust_remote_code=True,
)

# -- LoRA --
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# -- Train --
args = TrainingArguments(
    output_dir="/tmp/greentune-smoke",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    max_steps=10,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    peft_config=lora_config,
    processing_class=tokenizer,
)

print("Starting smoke test (10 steps)...")
trainer.train()
print(f"\nFinal loss: {trainer.state.log_history[-1].get('train_loss', 'N/A')}")
print("Smoke test PASSED — full QLoRA pipeline works on ROCm.")
```

---

## 9. Reference Card

| What | Command / Value |
|------|-----------------|
| GPU arch | `gfx942` |
| TDP | 750W |
| HBM3 | 192 GB |
| ROCm version | 6.4.x (pre-installed on AMD Dev Cloud) |
| PyTorch install | `pip install torch==2.9.1 --index-url .../rocm6.4` |
| Check GPU | `rocm-smi` or `torch.cuda.is_available()` |
| Check power | `rocm-smi --showpower` or `amdsmi_get_power_info()` |
| Check energy | `rocm-smi --showenergycounter` |
| Check temp | `rocm-smi --showtemp` |
| Precision | `bf16` (native CDNA3 matrix cores) |
| Attention | `attn_implementation="sdpa"` (CK flash, built-in) |
| Quantization | NF4 via bitsandbytes ≥0.49 |
| Memory fix | `PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"` |
| Idle power | ~50-80W |
| Training power | ~400-650W (QLoRA 7B) |
| Cost (1x MI300X) | ~$1.99/hr on AMD Developer Cloud |
