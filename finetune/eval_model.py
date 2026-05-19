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
"""Standalone eval — load a fine-tuned adapter and run domain prompts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "output/greentune-run/adapter"

PROMPTS = [
    "My MI300X is drawing 680W during a QLoRA fine-tune of a 7B model. Is this normal?",
    "How do I calculate Joules-per-token for a training run?",
    "What rocm-smi command shows current GPU power consumption?",
    "Compare the energy cost of LoRA r=8 vs r=64 for fine-tuning a 7B model.",
    "We have 3 teams sharing 4 MI300X GPUs. How should we split the energy bill?",
    "Our training loss stopped decreasing after epoch 2. What should I try?",
    "How do I set up the AluminatiAI agent for monitoring on an AMD GPU?",
    "What is the carbon footprint of fine-tuning a 7B model for 1 hour on MI300X?",
]


def main():
    print("Loading base model in bf16 (192GB VRAM — no quantization needed for eval)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    print(f"Loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, torch_dtype=torch.bfloat16)
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    print(f"\nRunning {len(PROMPTS)} eval prompts...\n")
    results = []

    for i, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
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
        print(f"[{i+1}/{len(PROMPTS)}] {prompt}")
        print(f"  → {response[:300]}")
        print()

    out_path = Path("output/greentune-run/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
