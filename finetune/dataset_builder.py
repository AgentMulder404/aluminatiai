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
"""GreenTune dataset builder — generates synthetic instruction pairs via Claude.

Produces an Alpaca-format JSON dataset for fine-tuning an LLM on GPU
infrastructure operations, energy efficiency, and cost attribution.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python dataset_builder.py                        # 5000 samples (default)
    python dataset_builder.py --samples 1000         # fewer for speed
    python dataset_builder.py --output data/my.json  # custom output path
    python dataset_builder.py --resume               # continue from partial output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("pip install anthropic")
    sys.exit(1)

CATEGORIES = [
    {
        "name": "gpu_power_diagnosis",
        "weight": 15,
        "system": "You are an expert GPU infrastructure engineer specializing in power management and energy efficiency.",
        "prompt": """Generate {batch} unique instruction/response pairs about GPU power diagnosis.

Topics to cover (vary across pairs):
- Abnormal power draw patterns (idle too high, load too low, spikes)
- TDP analysis for different GPU models (MI300X, A100, H100, RTX 4090)
- Power capping and its effects on throughput
- PCIe power delivery issues
- Multi-GPU power balancing
- Distinguishing power issues from thermal throttling

Each response should include specific wattage numbers, diagnostic steps, and actionable fixes.""",
    },
    {
        "name": "energy_efficiency",
        "weight": 20,
        "system": "You are a Green AI researcher focused on measuring and reducing the energy footprint of ML workloads.",
        "prompt": """Generate {batch} unique instruction/response pairs about ML energy efficiency.

Topics to cover (vary across pairs):
- Joules-per-token as a training metric
- Energy cost of fine-tuning vs inference vs prompt engineering
- Mixed precision (bf16 vs fp16 vs fp32) energy impact
- Batch size effects on energy efficiency
- Gradient checkpointing energy tradeoffs
- LoRA/QLoRA energy savings vs full fine-tuning
- Carbon footprint calculation from power draw
- Grid carbon intensity by region
- Energy-aware hyperparameter tuning
- Comparing energy efficiency across GPU architectures

Each response should include specific numbers (watts, joules, kWh, grams CO2) and practical recommendations.""",
    },
    {
        "name": "rocm_operations",
        "weight": 15,
        "system": "You are an AMD ROCm platform engineer with deep expertise in the ROCm software stack and MI300 hardware.",
        "prompt": """Generate {batch} unique instruction/response pairs about AMD ROCm operations.

Topics to cover (vary across pairs):
- rocm-smi and amd-smi command usage
- ROCm driver installation and troubleshooting
- MI300X architecture (CDNA3, HBM3, 192GB, 750W TDP)
- HIP vs CUDA translation
- PyTorch on ROCm setup and gotchas
- Flash attention on ROCm (CK backend vs Triton)
- torch.compile with HIP backend
- Multi-GPU topology and NUMA awareness on MI300
- ROCm Docker container setup (--device=/dev/kfd --device=/dev/dri)
- Performance tuning: PYTORCH_HIP_ALLOC_CONF, HSA_OVERRIDE_GFX_VERSION
- bitsandbytes on ROCm (building from source for gfx942)

Each response should include exact commands, flag names, and expected outputs.""",
    },
    {
        "name": "cost_attribution",
        "weight": 15,
        "system": "You are a GPU cloud cost analyst specializing in multi-tenant GPU cost attribution and chargeback.",
        "prompt": """Generate {batch} unique instruction/response pairs about GPU cost attribution.

Topics to cover (vary across pairs):
- Splitting GPU costs across multiple users/teams
- GPU-hours vs energy-based billing
- Idle cost attribution strategies
- Spot vs on-demand pricing optimization
- Cost modeling for fine-tuning jobs (time × power × rate)
- Chargeback rate calculation
- Cost comparison across cloud providers (AMD Developer Cloud, Lambda, RunPod)
- ROI of fine-tuning vs larger model + prompt engineering
- Budget alerts and cost cap enforcement
- Multi-GPU job cost tracking

Each response should include concrete dollar amounts, formulas, and comparison tables where appropriate.""",
    },
    {
        "name": "finetuning_ops",
        "weight": 20,
        "system": "You are an ML engineer specializing in efficient fine-tuning of large language models.",
        "prompt": """Generate {batch} unique instruction/response pairs about LLM fine-tuning operations.

Topics to cover (vary across pairs):
- LoRA vs QLoRA vs full fine-tuning: when to use each
- LoRA rank selection (r=8 vs 16 vs 64) and its impact
- QLoRA NF4 quantization mechanics
- Dataset preparation for instruction tuning (Alpaca format, ChatML, ShareGPT)
- Learning rate scheduling for LoRA (cosine, linear warmup)
- Gradient accumulation for effective batch size
- Evaluation strategies (loss, perplexity, domain-specific metrics)
- Catastrophic forgetting mitigation
- Training on single GPU vs distributed (FSDP, DeepSpeed)
- HuggingFace Trainer + PEFT + TRL workflow
- Common training failures (NaN loss, divergence, overfitting)
- Hyperparameter search for fine-tuning
- Model merging (LoRA adapter + base model)

Each response should be practical, include code snippets or config examples, and reference specific library versions.""",
    },
    {
        "name": "workload_scheduling",
        "weight": 10,
        "system": "You are a GPU cluster operations engineer managing multi-tenant GPU infrastructure.",
        "prompt": """Generate {batch} unique instruction/response pairs about GPU workload scheduling.

Topics to cover (vary across pairs):
- Job scheduling strategies (FIFO, priority, preemption)
- GPU utilization monitoring and alerting
- Detecting and handling GPU memory leaks
- Process attribution (which user/job owns which GPU)
- Slurm prologue/epilogue for GPU monitoring
- Kubernetes GPU scheduling (device plugins, time-slicing)
- Power-aware scheduling (shift jobs to off-peak hours)
- Thermal management and throttle detection
- Multi-tenant isolation and security
- Agent-based monitoring architecture (heartbeats, WAL, backoff)

Each response should include practical examples and operational procedures.""",
    },
    {
        "name": "aluminatai_product",
        "weight": 5,
        "system": "You are a technical support engineer for AluminatiAI, a GPU monitoring and energy attribution platform.",
        "prompt": """Generate {batch} unique instruction/response pairs about using AluminatiAI.

Topics to cover (vary across pairs):
- Installing and configuring the aluminatai-agent CLI
- API key setup and rotation
- Reading the energy dashboard
- Understanding the Green AI Index / benchmark leaderboard
- Interpreting Joules-per-GPU-hour metrics
- Setting up budget alerts
- Chargeback report generation
- Agent heartbeat and connectivity troubleshooting
- Integrating with MLflow, W&B, OpenTelemetry
- Benchmark CLI usage (aluminatiai benchmark --gpu 0 --duration 60 --upload)

Each response should be helpful and product-aware without being salesy.""",
    },
]

BATCH_TEMPLATE = """Generate exactly {batch} instruction/response pairs as a JSON array.

Each element must have exactly these fields:
- "instruction": The user's question or request (1-3 sentences)
- "input": Optional additional context (empty string if none)
- "output": A detailed, helpful response (3-8 sentences with specific numbers/commands)

Rules:
- Every pair must be unique — no duplicate questions
- Vary difficulty: mix beginner, intermediate, and expert questions
- Include specific numbers: wattages, costs, temperatures, percentages
- Reference real tools and commands where applicable
- Responses should be authoritative but concise
- Do NOT include any text outside the JSON array

Return ONLY the JSON array, starting with [ and ending with ]."""


def build_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def generate_batch(
    client: anthropic.Anthropic,
    category: dict,
    batch_size: int,
) -> list[dict]:
    prompt = BATCH_TEMPLATE.format(batch=batch_size)
    category_prompt = category["prompt"].format(batch=batch_size)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=category["system"],
        messages=[
            {"role": "user", "content": f"{category_prompt}\n\n{prompt}"},
        ],
    )

    text = response.content[0].text.strip()

    # Extract JSON array from response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  Warning: no JSON array found, skipping batch")
        return []

    try:
        pairs = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error: {e}, skipping batch")
        return []

    valid = []
    for p in pairs:
        if isinstance(p, dict) and "instruction" in p and "output" in p:
            valid.append({
                "instruction": p["instruction"],
                "input": p.get("input", ""),
                "output": p["output"],
                "category": category["name"],
            })
    return valid


def format_as_chatml(sample: dict) -> str:
    parts = ["<|im_start|>user"]
    if sample["input"]:
        parts.append(f"{sample['instruction']}\n\nContext: {sample['input']}")
    else:
        parts.append(sample["instruction"])
    parts.append("<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{sample['output']}<|im_end|>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="GreenTune dataset builder")
    parser.add_argument(
        "--samples", type=int, default=5000, help="Total samples to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Samples per API call"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/greentune_dataset.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing partial output"
    )
    parser.add_argument(
        "--chatml", action="store_true", help="Also output ChatML-formatted text file"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing file
    all_samples: list[dict] = []
    if args.resume and output_path.exists():
        with open(output_path) as f:
            all_samples = json.load(f)
        print(f"Resumed with {len(all_samples)} existing samples")

    remaining = args.samples - len(all_samples)
    if remaining <= 0:
        print(f"Already have {len(all_samples)} samples, nothing to do")
        return

    # Calculate samples per category based on weights
    total_weight = sum(c["weight"] for c in CATEGORIES)
    category_targets = {}
    for cat in CATEGORIES:
        category_targets[cat["name"]] = max(
            1, int(remaining * cat["weight"] / total_weight)
        )

    # Subtract already-generated counts
    existing_counts = {}
    for s in all_samples:
        cat = s.get("category", "unknown")
        existing_counts[cat] = existing_counts.get(cat, 0) + 1

    for cat_name in category_targets:
        category_targets[cat_name] = max(
            0, category_targets[cat_name] - existing_counts.get(cat_name, 0)
        )

    client = build_client()

    print(f"Generating {remaining} samples across {len(CATEGORIES)} categories...\n")
    for cat in CATEGORIES:
        target = category_targets[cat["name"]]
        if target <= 0:
            continue

        generated = 0
        batch_num = 0
        print(f"[{cat['name']}] target={target}")

        while generated < target:
            batch_size = min(args.batch_size, target - generated)
            batch_num += 1

            try:
                pairs = generate_batch(client, cat, batch_size)
                all_samples.extend(pairs)
                generated += len(pairs)
                print(f"  batch {batch_num}: +{len(pairs)} (total {generated}/{target})")

                # Save after each batch (crash-safe)
                with open(output_path, "w") as f:
                    json.dump(all_samples, f, indent=2)

                time.sleep(0.5)

            except anthropic.RateLimitError:
                print("  Rate limited, waiting 60s...")
                time.sleep(60)
            except Exception as e:
                print(f"  Error: {e}, retrying in 10s...")
                time.sleep(10)

        print()

    # Deduplicate by instruction text
    seen = set()
    deduped = []
    for s in all_samples:
        key = s["instruction"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    removed = len(all_samples) - len(deduped)
    all_samples = deduped

    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"{'='*60}")
    print(f"Dataset saved: {output_path}")
    print(f"Total samples: {len(all_samples)} ({removed} duplicates removed)")
    print(f"Categories:")
    cat_counts: dict[str, int] = {}
    for s in all_samples:
        c = s.get("category", "unknown")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for name, count in sorted(cat_counts.items()):
        print(f"  {name}: {count}")

    # ChatML output for direct SFTTrainer consumption
    if args.chatml:
        chatml_path = output_path.with_suffix(".chatml.jsonl")
        with open(chatml_path, "w") as f:
            for s in all_samples:
                line = json.dumps({"text": format_as_chatml(s)})
                f.write(line + "\n")
        print(f"ChatML JSONL: {chatml_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
