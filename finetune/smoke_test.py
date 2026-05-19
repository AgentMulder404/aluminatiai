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
"""Smoke test — 10-step QLoRA fine-tune to verify the full stack on ROCm."""

from __future__ import annotations

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    print(f"Loading {MODEL_ID} in 4-bit QLoRA mode...")

    # Tiny synthetic dataset
    samples = [
        {
            "text": (
                f"<|im_start|>user\n"
                f"What is GPU {i} power draw?<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"GPU {i} is drawing {200 + i * 10}W, which is within normal "
                f"operating range for this workload.<|im_end|>"
            )
        }
        for i in range(20)
    ]
    ds = Dataset.from_list(samples)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

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

    print("Starting smoke test (10 training steps)...")
    result = trainer.train()

    final_loss = trainer.state.log_history[-1].get("train_loss", "N/A")
    print(f"\nTraining time:  {result.metrics['train_runtime']:.1f}s")
    print(f"Final loss:     {final_loss}")
    print(f"Samples/sec:    {result.metrics['train_samples_per_second']:.1f}")
    print("\nSmoke test PASSED — full QLoRA pipeline works on ROCm.")


if __name__ == "__main__":
    main()
