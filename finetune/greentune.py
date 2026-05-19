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
"""GreenTune — energy-efficient QLoRA fine-tuning on AMD MI300X.

Fine-tunes a base LLM on a blended dataset:
  1. AdversaLLC/hermes-agent-reasoning-traces (agent reasoning + tool use)
  2. Synthetic GreenTune domain data (GPU ops, energy efficiency, ROCm)

Tracks real-time power consumption, Joules-per-token, and CO2.

Usage:
    python greentune.py                                    # both datasets merged
    python greentune.py --no-hermes                        # domain data only
    python greentune.py --hermes-only                      # hermes data only
    python greentune.py --hermes-config glm-5.1            # specific hermes config
    python greentune.py --hermes-max 3000                  # cap hermes samples
    python greentune.py --dataset data/greentune_dataset.json --epochs 3
    python greentune.py --lora-rank 8 --batch-size 8       # faster, less quality
    python greentune.py --eval                              # run eval after training
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import random

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

try:
    from .energy_callback import EnergyCallback
except ImportError:
    from energy_callback import EnergyCallback

# ── Defaults ──────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = "data/greentune_dataset.json"
DEFAULT_OUTPUT = "output/greentune-run"
MAX_SEQ_LENGTH = 1024

HERMES_DATASET = "AdversaLLC/hermes-agent-reasoning-traces"
HERMES_CONFIG = "glm-5.1"

# ShareGPT role mapping → ChatML roles
ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "tool": "tool",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GreenTune fine-tuning")

    # Model
    p.add_argument("--model", default=DEFAULT_MODEL, help="Base model ID")
    p.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)

    # Dataset
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to domain dataset JSON")
    p.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")
    p.add_argument("--no-hermes", action="store_true", help="Skip Hermes agent traces, domain data only")
    p.add_argument("--hermes-only", action="store_true", help="Hermes data only, skip domain data")
    p.add_argument("--hermes-config", default=HERMES_CONFIG, help="Hermes dataset config")
    p.add_argument("--hermes-max", type=int, default=0, help="Cap Hermes samples (0 = use all)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    # QLoRA
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (r)")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--load-in-8bit", action="store_true", default=False)

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-strategy", default="epoch")
    p.add_argument("--max-steps", type=int, default=-1, help="Override epochs if > 0")

    # Energy monitoring
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--power-sample-interval", type=float, default=0.5)
    p.add_argument("--carbon-intensity", type=float, default=390.0, help="gCO2/kWh")
    p.add_argument("--energy-price", type=float, default=0.10, help="$/kWh")

    # Live dashboard upload
    p.add_argument("--api-url", default=None, help="Dashboard base URL for live upload (e.g. https://www.aluminatiai.com)")
    p.add_argument("--api-key", default=None, help="AluminatiAI API key for live upload")
    p.add_argument("--run-name", default=None, help="Name for this run on the dashboard")

    # Output
    p.add_argument("--output", default=DEFAULT_OUTPUT)

    # Eval
    p.add_argument("--eval", action="store_true", help="Run domain eval after training")
    p.add_argument("--eval-prompts", default=None, help="JSON file with eval prompts")

    return p.parse_args()


def load_and_prepare_dataset(
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset | None]:
    """Load and merge Hermes + domain datasets, then optionally split."""

    datasets_to_merge: list[Dataset] = []

    # ── 1. Hermes agent reasoning traces ──
    if not args.no_hermes:
        print(f"\nLoading Hermes: {HERMES_DATASET} [{args.hermes_config}]...")
        hermes_raw = load_dataset(
            HERMES_DATASET, args.hermes_config, split="train"
        )

        if args.hermes_max > 0 and len(hermes_raw) > args.hermes_max:
            hermes_raw = hermes_raw.shuffle(seed=args.seed).select(
                range(args.hermes_max)
            )

        hermes_formatted = []
        skipped = 0
        for sample in hermes_raw:
            text = format_sharegpt_to_chatml(sample["conversations"])
            if text:
                hermes_formatted.append({
                    "text": text,
                    "source": "hermes",
                })
            else:
                skipped += 1

        hermes_ds = Dataset.from_list(hermes_formatted)
        print(f"  Hermes: {len(hermes_ds)} samples ({skipped} skipped)")
        datasets_to_merge.append(hermes_ds)

    # ── 2. Domain dataset (Alpaca JSON from dataset_builder.py) ──
    if not args.hermes_only:
        data_path = Path(args.dataset)
        if data_path.exists():
            domain_ds = load_domain_dataset(data_path)
            print(f"  Domain: {len(domain_ds)} samples from {data_path}")
            datasets_to_merge.append(domain_ds)
        elif not args.no_hermes:
            print(f"  Domain: {data_path} not found, using Hermes only")
        else:
            print(f"Error: no datasets available. {data_path} not found and --no-hermes set.")
            sys.exit(1)

    # ── Merge and shuffle ──
    if len(datasets_to_merge) == 1:
        merged = datasets_to_merge[0]
    else:
        merged = concatenate_datasets(datasets_to_merge)

    merged = merged.shuffle(seed=args.seed)

    # Count sources
    source_counts: dict[str, int] = {}
    for s in merged:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nMerged dataset: {len(merged)} total samples")
    for src, count in sorted(source_counts.items()):
        pct = 100 * count / len(merged)
        print(f"  {src}: {count} ({pct:.1f}%)")

    # Drop the source column before training (SFTTrainer only needs "text")
    if "source" in merged.column_names:
        merged = merged.remove_columns("source")

    # ── Split ──
    if args.val_split > 0 and len(merged) > 100:
        split = merged.train_test_split(test_size=args.val_split, seed=args.seed)
        print(f"  Train: {len(split['train'])}, Val: {len(split['test'])}")
        return split["train"], split["test"]

    return merged, None


def load_domain_dataset(data_path: Path) -> Dataset:
    """Load Alpaca-format JSON or ChatML JSONL from dataset_builder.py."""

    if data_path.suffix == ".jsonl":
        ds = load_dataset("json", data_files=str(data_path), split="train")
        return ds

    with open(data_path) as f:
        raw = json.load(f)

    formatted = []
    for sample in raw:
        text = format_alpaca_to_chatml(sample)
        formatted.append({"text": text, "source": "domain"})

    return Dataset.from_list(formatted)


def format_sharegpt_to_chatml(conversations: list[dict]) -> str | None:
    """Convert ShareGPT-format turns to ChatML string.

    Hermes format: [{"from": "system"|"human"|"gpt"|"tool", "value": "..."}]
    Output: <|im_start|>role\nvalue<|im_end|>\n per turn
    """
    if not conversations:
        return None

    parts = []
    for turn in conversations:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        value = turn["value"]
        if not value:
            continue
        parts.append(f"<|im_start|>{role}\n{value}<|im_end|>")

    if not parts:
        return None

    return "\n".join(parts)


def format_alpaca_to_chatml(sample: dict) -> str:
    """Convert Alpaca-format sample to ChatML for Qwen2.5."""
    instruction = sample["instruction"]
    context = sample.get("input", "")
    output = sample["output"]

    user_msg = instruction
    if context:
        user_msg += f"\n\nContext: {context}"

    return (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load base model with QLoRA quantization config."""

    print(f"\nLoading {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        quant_label = "8-bit"
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        quant_label = "4-bit NF4"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": args.gpu_index},
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    param_count = sum(p.numel() for p in model.parameters())
    mem_gb = torch.cuda.memory_allocated(args.gpu_index) / 1024**3
    print(f"  Parameters:  {param_count / 1e9:.1f}B")
    print(f"  Quantization: {quant_label}")
    print(f"  VRAM used:   {mem_gb:.1f} GB")

    return model, tokenizer


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print(f"\nLoRA config:")
    print(f"  Rank:        {config.r}")
    print(f"  Alpha:       {config.lora_alpha}")
    print(f"  Targets:     {config.target_modules}")

    return config


def build_training_args(args: argparse.Namespace) -> SFTConfig:
    return SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        report_to=["tensorboard"],
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        max_length=args.max_seq_length,
    )


def run_eval(
    model,
    tokenizer,
    args: argparse.Namespace,
    eval_prompts: list[str] | None = None,
):
    """Run domain evaluation prompts through the fine-tuned model."""

    if eval_prompts is None:
        eval_prompts = [
            "My MI300X is drawing 680W during a QLoRA fine-tune of a 7B model. Is this normal?",
            "How do I calculate Joules-per-token for a training run?",
            "Compare the energy cost of LoRA r=8 vs r=64 for fine-tuning a 7B model.",
            "What rocm-smi command shows current GPU power consumption?",
            "We have 3 teams sharing 4 MI300X GPUs. How should we split the energy bill?",
            "Our training loss stopped decreasing after epoch 2. What should I try?",
            "How do I set up the AluminatiAI agent for monitoring on an AMD GPU?",
            "What is the carbon footprint of fine-tuning a 7B model for 1 hour on MI300X?",
        ]

    print(f"\n{'='*60}")
    print(f"  Domain Evaluation — {len(eval_prompts)} prompts")
    print(f"{'='*60}\n")

    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    results = []

    for i, prompt in enumerate(eval_prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        results.append({"prompt": prompt, "response": response})
        print(f"[{i+1}/{len(eval_prompts)}] {prompt[:80]}...")
        print(f"  → {response[:200]}...\n")

    # Save eval results
    eval_path = Path(args.output) / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Eval results saved to {eval_path}")

    return results


def main():
    args = parse_args()

    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"{'='*60}")
    print(f"  AluminatiAI GreenTune")
    print(f"  Energy-Efficient Fine-Tuning on AMD ROCm")
    print(f"{'='*60}")

    # GPU info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(args.gpu_index)
        print(f"\nGPU {args.gpu_index}: {props.name}")
        print(f"  VRAM:    {props.total_memory / 1024**3:.0f} GB")
        print(f"  ROCm:    {torch.version.hip or 'N/A'}")
        print(f"  PyTorch: {torch.__version__}")
    else:
        print("WARNING: No GPU detected. Training will be extremely slow.")

    # Load data
    train_ds, val_ds = load_and_prepare_dataset(args)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args)

    # LoRA
    lora_config = build_lora_config(args)

    # Training args
    training_args = build_training_args(args)

    # Energy callback
    energy_cb = EnergyCallback(
        gpu_index=args.gpu_index,
        sample_interval_s=args.power_sample_interval,
        carbon_intensity_gco2_kwh=args.carbon_intensity,
        energy_price_usd_kwh=args.energy_price,
        output_dir=args.output,
        api_url=args.api_url,
        api_key=args.api_key,
        run_name=args.run_name or f"GreenTune {args.model.split('/')[-1]} bs={args.batch_size}",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[energy_cb],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"\nStarting training...\n")

    # Train
    result = trainer.train()

    # Save adapter
    adapter_path = Path(args.output) / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved to {adapter_path}")

    # Save training summary
    dataset_sources = []
    if not args.no_hermes:
        dataset_sources.append(f"hermes:{args.hermes_config}")
    if not args.hermes_only:
        dataset_sources.append(f"domain:{args.dataset}")

    summary = {
        "model": args.model,
        "datasets": dataset_sources,
        "hermes_config": args.hermes_config if not args.no_hermes else None,
        "hermes_max": args.hermes_max if not args.no_hermes else None,
        "domain_dataset": args.dataset if not args.hermes_only else None,
        "total_train_samples": len(train_ds),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "quantization": "4-bit NF4" if args.load_in_4bit else "8-bit",
        "training_runtime_s": result.metrics["train_runtime"],
        "train_loss": result.metrics.get("train_loss"),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
    }
    with open(Path(args.output) / "run_config.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Eval
    if args.eval:
        eval_prompts = None
        if args.eval_prompts:
            with open(args.eval_prompts) as f:
                eval_prompts = json.load(f)
        run_eval(model, tokenizer, args, eval_prompts)

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nGreenTune complete. Output: {args.output}/")


if __name__ == "__main__":
    main()
