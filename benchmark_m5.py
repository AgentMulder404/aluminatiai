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

#!/usr/bin/env python3
"""
AluminatiAI Energy Dashboard Benchmark — Apple Silicon (M5) Edition

Runs four jobs with distinct power profiles to populate the 'Energy by Job'
section of the AluminatiAI dashboard.

Jobs:
  1. Light-Chat          — Short prompt, low token count (chatbot simulation)
  2. Deep-Analysis       — 2000-word essay, sustained power draw, 25W+ prefill spike
  3. Stress-Test         — Max batch + parallel decode, push GPU to thermal limit
  4. MLX-LoRA-Finetune   — LoRA fine-tuning via Apple MLX, training-style sustained power

Requirements:
  - llama.cpp with Metal support  (brew install llama.cpp)
  - A GGUF model file             (e.g. Llama-3.2-3B-Instruct-Q4_K_M.gguf)
  - mlx-lm                        (pip install mlx-lm)
  - sudo access for powermetrics  (power & thermal monitoring)
  - Python 3.10+, requests        (pip install requests)

Usage:
  python benchmark_m5.py --model ~/models/model.gguf
  python benchmark_m5.py --model ./model.gguf --api-key alum_xxxxx
  python benchmark_m5.py --model ./model.gguf --jobs Light-Chat Deep-Analysis
  python benchmark_m5.py --model ./model.gguf --mlx-model mlx-community/Llama-3.2-3B-Instruct-4bit
  python benchmark_m5.py --model ./model.gguf --thermal-limit 95 --save-json results.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THERMAL_LIMIT_C = 105  # Pause threshold (Apple Silicon throttles ~110 C)
THERMAL_HYSTERESIS_C = 5  # Resume when temp drops this far below limit
SAMPLE_INTERVAL_MS = 1000  # powermetrics sampling interval
GPU_UUID = "apple-m5-integrated-gpu"
GPU_NAME = "Apple M5 GPU"
GPU_INDEX = 0
POWER_LIMIT_W = 45.0  # Approximate M5 Pro GPU power envelope
DEFAULT_API_ENDPOINT = "https://aluminatiai.com/api/metrics/ingest"

# ---------------------------------------------------------------------------
# Job Definitions
# ---------------------------------------------------------------------------

JOBS = {
    "Light-Chat": {
        "prompt": (
            "You are a helpful assistant. The user asks: "
            "What are three practical benefits of renewable energy for homeowners? "
            "Give a concise answer."
        ),
        "n_predict": 128,
        "ctx_size": 512,
        "batch_size": 512,
        "parallel": 1,
        "description": "Short chatbot exchange — low sustained power, quick prefill",
    },
    "Deep-Analysis": {
        "prompt": (
            "Write a detailed 2000-word analytical essay on the economic, environmental, "
            "and geopolitical implications of transitioning the global energy grid from "
            "fossil fuels to 100%% renewable sources by 2050. Address the challenges of "
            "energy storage at scale, the role of nuclear power as a bridge technology, "
            "the impact on developing nations, stranded assets in the oil and gas sector, "
            "supply-chain dependencies for critical minerals, and workforce retraining. "
            "Include specific data points and cite historical precedents for large-scale "
            "infrastructure transitions such as rural electrification and the build-out "
            "of interstate highway systems. Conclude with a realistic phased timeline."
        ),
        "n_predict": 2048,
        "ctx_size": 4096,
        "batch_size": 512,
        "parallel": 1,
        "description": "Sustained generation — expect 25W+ prefill spike, long decode phase",
    },
    "Stress-Test": {
        "prompt": (
            "You are a world-class systems architect. Design a complete production-ready "
            "distributed computing platform. Include: "
            "(1) A microservices architecture with 12 services — each with full API "
            "contracts, data models, error handling, and SLA definitions. "
            "(2) A custom Raft-variant consensus protocol with formal leader election, "
            "log replication, and membership change procedures. "
            "(3) A cost-based query optimizer for a columnar database engine supporting "
            "predicate pushdown, join reordering, and parallel scan operators. "
            "(4) A real-time stream processing pipeline handling 1M events/second with "
            "exactly-once semantics via a two-phase commit with external offset storage. "
            "(5) A disaster recovery strategy with RPO < 1 minute across 3 regions. "
            "Provide complete pseudocode for every component and analyze Big-O complexity "
            "of each critical code path. Then repeat the full design a second time with "
            "alternative technology choices and compare trade-offs."
        ),
        "n_predict": 4096,
        "ctx_size": 8192,
        "batch_size": 2048,  # Max batch to saturate Metal compute units
        "parallel": 1,
        "description": "Maximum batch + context — push GPU to thermal limit",
    },
}


# ---------------------------------------------------------------------------
# MLX LoRA fine-tuning job
# ---------------------------------------------------------------------------

MLX_JOB_NAME = "MLX-LoRA-Finetune"
DEFAULT_MLX_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

MLX_JOB_CFG = {
    "description": "LoRA fine-tuning via Apple MLX — sustained training-style power, high memory bandwidth",
    "iters": 100,
    "batch_size": 4,
    "lora_layers": 8,
    "seq_length": 512,
    "learning_rate": 1e-4,
}

# Synthetic instruction dataset — 25 pairs is enough to fill 100 iters at batch 4
_SYNTHETIC_PAIRS = [
    ("Explain gradient descent in one paragraph.",
     "Gradient descent is an iterative optimization algorithm that minimizes a loss function by repeatedly stepping in the direction of the negative gradient. At each step the parameters are updated by subtracting the gradient multiplied by a learning rate, slowly moving toward a local minimum of the loss surface."),
    ("What is backpropagation?",
     "Backpropagation is the algorithm used to compute gradients in neural networks. It applies the chain rule of calculus layer by layer, propagating the error signal backwards from the output to each weight so that gradient descent can update them efficiently."),
    ("Describe the transformer architecture.",
     "The transformer uses stacked self-attention and feed-forward layers. An encoder maps input tokens to contextual embeddings; a decoder auto-regressively generates output tokens by attending to both its own past outputs and the encoder representations."),
    ("What is LoRA and why is it useful?",
     "LoRA (Low-Rank Adaptation) fine-tunes large models by injecting small trainable rank-decomposition matrices into existing weight matrices. Because only these low-rank updates are trained, parameter count and memory usage stay minimal while task-specific performance improves significantly."),
    ("How does mixed-precision training work?",
     "Mixed-precision training stores weights in FP32 but performs forward and backward passes in FP16 (or BF16). A loss scaling factor prevents underflow in gradients. The result is faster computation and lower memory usage with negligible accuracy loss."),
    ("What is the difference between batch and layer normalisation?",
     "Batch normalisation normalises across the batch dimension, which requires a sufficient batch size and behaves differently at train vs inference time. Layer normalisation normalises across the feature dimension for each sample independently, making it well-suited to sequence models and small batches."),
    ("Explain the attention mechanism.",
     "Attention computes a weighted sum of value vectors, where weights are determined by the compatibility (dot product) of query and key vectors, scaled by the square root of the key dimension and passed through a softmax. Multi-head attention runs this in parallel across several projected subspaces."),
    ("What is a learning rate scheduler?",
     "A learning rate scheduler adjusts the learning rate during training according to a predefined schedule. Common strategies include cosine annealing, linear warmup followed by decay, and reduce-on-plateau, each aiming to stabilise early training and refine convergence later."),
    ("Describe the RLHF training pipeline.",
     "RLHF (Reinforcement Learning from Human Feedback) has three stages: supervised fine-tuning on demonstrations, training a reward model from human preference rankings, and then optimising the language model against that reward model using PPO while a KL penalty keeps it close to the original policy."),
    ("What is weight decay and how does it regularise models?",
     "Weight decay adds an L2 penalty on parameter magnitudes to the loss, discouraging large weights and reducing overfitting. In AdamW this is implemented as a direct multiplicative decay applied to weights before the gradient update, cleanly separating regularisation from the adaptive moment estimates."),
    ("How does Flash Attention reduce memory usage?",
     "Flash Attention reorders the attention computation to tile the query, key, and value matrices and keep intermediate results in fast SRAM rather than writing the full attention matrix to HBM. This reduces memory reads and writes from O(N²) to O(N), enabling much longer sequences without OOM errors."),
    ("What is speculative decoding?",
     "Speculative decoding uses a small draft model to generate candidate tokens cheaply, then verifies several tokens in parallel with the larger target model. When tokens are accepted the large model skips those steps, yielding 2–4× throughput with identical output distribution."),
    ("Explain the role of the KV cache in inference.",
     "The KV cache stores the key and value tensors for all previously generated tokens so that each new decoding step only needs to compute attention over one new query instead of recomputing the full sequence. This reduces per-step cost from O(N²) to O(N) at the expense of memory proportional to sequence length."),
    ("What is quantisation and what are its trade-offs?",
     "Quantisation represents model weights and activations with fewer bits (e.g. INT8 or INT4) to reduce memory footprint and speed up computation on hardware with efficient integer units. The trade-off is a potential drop in accuracy, mitigated by techniques like GPTQ, AWQ, and QLoRA that quantise more carefully."),
    ("Describe the pre-training objective of a GPT-style model.",
     "GPT models are pre-trained with a causal language modelling objective: given a sequence of tokens, predict the next token at every position. The cross-entropy loss is averaged over all positions, and the model processes tokens left-to-right with a causal attention mask that prevents attending to future tokens."),
    ("What is tokenisation and why does it matter?",
     "Tokenisation converts raw text into a sequence of integer IDs from a fixed vocabulary. BPE and SentencePiece learn subword units that balance vocabulary size with sequence length. The choice of tokeniser affects training efficiency, multilingual coverage, and the model's ability to handle rare or novel words."),
    ("How does dropout prevent overfitting?",
     "Dropout randomly zeroes a fraction of activations during each training forward pass, forcing the network to learn redundant representations and preventing co-adaptation of neurons. At inference, dropout is disabled and activations are scaled to maintain expected values, acting as an ensemble of thinned networks."),
    ("What is the vanishing gradient problem?",
     "In deep networks, gradients are multiplied together through many layers during backpropagation. If weights are small or activations saturate (e.g. sigmoid), these products shrink exponentially, making early layer weights update negligibly. Solutions include ReLU activations, residual connections, and normalisation layers."),
    ("Explain positional encoding in transformers.",
     "Since attention is permutation-invariant, positional encodings inject information about token order. Sinusoidal encodings use fixed sine and cosine functions of different frequencies; learned positional embeddings are trained end-to-end. Rotary (RoPE) and ALiBi encodings enable better length generalisation."),
    ("What is the difference between encoder-only, decoder-only, and encoder-decoder models?",
     "Encoder-only models (BERT) use bidirectional attention and excel at classification and embedding tasks. Decoder-only models (GPT) use causal attention for autoregressive generation. Encoder-decoder models (T5, BART) combine a bidirectional encoder with a causal decoder, well-suited to sequence-to-sequence tasks like translation and summarisation."),
    ("How does AdamW differ from Adam?",
     "Adam applies L2 regularisation by adding the weight penalty to the gradient before the adaptive update, which conflates regularisation with the moment estimates. AdamW decouples weight decay by applying it directly to the weights after the gradient step, resulting in more consistent and effective regularisation across all parameter scales."),
    ("What is perplexity as a language model metric?",
     "Perplexity is the exponentiated average negative log-likelihood per token: exp(-(1/N) * sum(log P(token_i))). It measures how surprised the model is by a held-out corpus — lower perplexity means the model assigns higher probability to the actual next tokens on average."),
    ("Describe continual pre-training.",
     "Continual pre-training extends a model's training on new domain-specific text without full re-training from scratch. A low learning rate and careful data mixing prevent catastrophic forgetting of previously learned knowledge while still adapting the model's representations to the target domain."),
    ("What is model distillation?",
     "Knowledge distillation trains a smaller student model to mimic a larger teacher model's output distribution (soft labels) rather than just the hard ground-truth labels. The soft probabilities carry richer information about the teacher's learned representations, often yielding a more capable student than training on hard labels alone."),
    ("How does grouped query attention (GQA) work?",
     "GQA partitions the query heads into groups that share a single key-value head, sitting between multi-head attention (one KV per query head) and multi-query attention (one KV for all query heads). It reduces KV cache memory and inference bandwidth while retaining most of the quality of full multi-head attention."),
]


def generate_synthetic_dataset(data_dir: Path) -> None:
    """Write train.jsonl and valid.jsonl into data_dir for mlx_lm.lora."""
    template = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "{a}<|eot_id|>"
    )
    rows = [json.dumps({"text": template.format(q=q, a=a)}) for q, a in _SYNTHETIC_PAIRS]
    (data_dir / "train.jsonl").write_text("\n".join(rows))
    # Use last 3 as a tiny validation split
    (data_dir / "valid.jsonl").write_text("\n".join(rows[-3:]))


@dataclass
class MLXTrainResult:
    """Parsed training stats from mlx_lm.lora stderr/stdout."""
    iters_completed: int = 0
    final_loss: float = 0.0
    tokens_per_second: float = 0.0
    total_ms: float = 0.0
    raw_output: str = ""


def run_mlx_finetune(
    model_id: str,
    data_dir: Path,
    iters: int,
    batch_size: int,
    lora_layers: int,
    seq_length: int,
    learning_rate: float,
    adapter_dir: Path,
) -> MLXTrainResult:
    """
    Run mlx_lm.lora as a subprocess and parse training output.

    mlx_lm prints lines like:
      Iter 10: Train loss 2.345, It/sec 3.21, Tokens/sec 1643.2
    """
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_id,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--lora-layers", str(lora_layers),
        "--seq-length", str(seq_length),
        "--learning-rate", str(learning_rate),
        "--steps-per-report", "10",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed_ms = (time.time() - t0) * 1000.0

    output = result.stdout + result.stderr
    parsed = MLXTrainResult(raw_output=output, total_ms=elapsed_ms)

    # Parse "Iter N: Train loss X.XXX, It/sec Y.YY, Tokens/sec Z.ZZ"
    for m in re.finditer(
        r"Iter\s+(\d+).*?Train loss\s+([\d.]+).*?Tokens/sec\s+([\d.]+)",
        output,
    ):
        parsed.iters_completed = int(m.group(1))
        parsed.final_loss = float(m.group(2))
        parsed.tokens_per_second = float(m.group(3))

    return parsed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SoCReading:
    """Single SoC power / thermal sample from powermetrics."""

    timestamp: str = ""
    gpu_power_w: float = 0.0
    cpu_power_w: float = 0.0
    package_power_w: float = 0.0
    die_temp_c: float = 0.0
    gpu_freq_mhz: int = 0


@dataclass
class InferenceResult:
    """Parsed performance counters from llama.cpp output."""

    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    eval_ms: float = 0.0
    total_ms: float = 0.0
    raw_stderr: str = ""


# ---------------------------------------------------------------------------
# Power & Thermal Monitor
# ---------------------------------------------------------------------------


class PowerThermalMonitor:
    """
    Background monitor using macOS `powermetrics`.

    Parses the human-readable text output (more stable across macOS versions
    than plist). Requires sudo — the script prompts once at startup.
    """

    def __init__(self, interval_ms: int = SAMPLE_INTERVAL_MS):
        self._interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest = SoCReading()
        self._history: list[SoCReading] = []
        self._running = False
        self.available = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        # Wait for the first sample (or timeout)
        deadline = time.time() + (self._interval_ms / 1000.0) + 2.0
        while time.time() < deadline and not self.available:
            time.sleep(0.2)

    def stop(self) -> None:
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # -- properties ----------------------------------------------------------

    @property
    def latest(self) -> SoCReading:
        with self._lock:
            return self._latest

    def history_since(self, iso_start: str) -> list[SoCReading]:
        with self._lock:
            return [r for r in self._history if r.timestamp >= iso_start]

    # -- internals -----------------------------------------------------------

    def _reader_loop(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [
                    "sudo", "powermetrics",
                    "--samplers", "gpu_power,smc",
                    "-i", str(self._interval_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            print("  [WARN] powermetrics binary not found")
            return

        chunk: list[str] = []
        for line in self._proc.stdout:  # type: ignore[union-attr]
            if not self._running:
                break
            # powermetrics separates samples with a line of dashes or "*****"
            if line.startswith("*****") or line.startswith("Machine model"):
                if chunk:
                    self._parse_chunk(chunk)
                chunk = [line]
            else:
                chunk.append(line)
        # parse any trailing chunk
        if chunk:
            self._parse_chunk(chunk)

    def _parse_chunk(self, lines: list[str]) -> None:
        text = "".join(lines)
        reading = SoCReading(timestamp=datetime.now(timezone.utc).isoformat())

        # GPU power — e.g. "GPU Power: 12345 mW" or "GPU HW active residency: ..."
        gpu_power_match = re.search(r"GPU Power:\s*([\d.]+)\s*mW", text)
        if gpu_power_match:
            reading.gpu_power_w = float(gpu_power_match.group(1)) / 1000.0

        # CPU power
        cpu_power_match = re.search(r"CPU Power:\s*([\d.]+)\s*mW", text)
        if cpu_power_match:
            reading.cpu_power_w = float(cpu_power_match.group(1)) / 1000.0

        # Combined / package power
        pkg_match = re.search(r"Combined Power.*?:\s*([\d.]+)\s*mW", text)
        if pkg_match:
            reading.package_power_w = float(pkg_match.group(1)) / 1000.0

        # GPU frequency
        freq_match = re.search(r"GPU (?:HW active|requested) frequency:\s*([\d.]+)\s*MHz", text)
        if freq_match:
            reading.gpu_freq_mhz = int(float(freq_match.group(1)))

        # Die temperature — e.g. "Die temperature: 54.32 C"
        temp_match = re.search(r"die temperature:\s*([\d.]+)\s*C", text, re.IGNORECASE)
        if temp_match:
            reading.die_temp_c = float(temp_match.group(1))

        with self._lock:
            self._latest = reading
            self._history.append(reading)

        if not self.available:
            self.available = True


# ---------------------------------------------------------------------------
# Thermal Safety
# ---------------------------------------------------------------------------


def thermal_ok(monitor: PowerThermalMonitor, limit_c: float) -> bool:
    """Return True if temperature is within safe range."""
    if not monitor.available:
        return True
    temp = monitor.latest.die_temp_c
    return temp <= 0 or temp < limit_c


def thermal_cooldown(monitor: PowerThermalMonitor, limit_c: float) -> None:
    """Block until SoC cools below (limit - hysteresis)."""
    temp = monitor.latest.die_temp_c
    target = limit_c - THERMAL_HYSTERESIS_C
    print(f"\n  !! SoC die temperature is {temp:.0f} C (limit {limit_c:.0f} C)")
    print(f"     Pausing until temperature drops below {target:.0f} C ...")
    while True:
        time.sleep(5)
        temp = monitor.latest.die_temp_c
        if temp <= 0 or temp < target:
            print(f"     Cooled to {temp:.0f} C — resuming")
            return
        print(f"     {temp:.0f} C — still cooling ...")


# ---------------------------------------------------------------------------
# llama.cpp Runner
# ---------------------------------------------------------------------------


def find_llama_binary() -> Optional[str]:
    """Auto-detect llama-cli / llama.cpp main binary."""
    candidates = [
        "llama-cli",
        "/opt/homebrew/bin/llama-cli",
        "/usr/local/bin/llama-cli",
        "main",  # older build-from-source name
    ]
    for name in candidates:
        try:
            r = subprocess.run(
                [name, "--version"],
                capture_output=True,
                timeout=5,
            )
            if r.returncode == 0:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Also try `--help` since some builds lack `--version`
    for name in candidates[:2]:
        try:
            r = subprocess.run(
                [name, "--help"],
                capture_output=True,
                timeout=5,
            )
            if r.returncode == 0:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def run_llama_inference(
    binary: str,
    model: str,
    prompt: str,
    n_predict: int,
    ctx_size: int = 2048,
    batch_size: int = 512,
    n_gpu_layers: int = 99,
    extra_args: Optional[list[str]] = None,
) -> InferenceResult:
    """
    Invoke llama-cli and parse the timing stats from stderr.

    The -ngl 99 flag forces all transformer layers onto the Metal backend.
    """
    cmd = [
        binary,
        "-m", model,
        "-p", prompt,
        "-n", str(n_predict),
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size),
        "-b", str(batch_size),
        "--no-display-prompt",  # cleaner stdout
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900,  # 15 min hard timeout per job
    )

    parsed = InferenceResult(raw_stderr=result.stderr)

    stderr = result.stderr

    # llama.cpp timing lines look like:
    #   llama_print_timings: prompt eval time = 234.56 ms / 42 tokens ( 5.58 ms per token, 179.12 tokens per second)
    #   llama_print_timings:        eval time = 8901.23 ms / 128 tokens ( 69.54 ms per token,  14.38 tokens per second)
    #   llama_print_timings:       total time = 9135.79 ms / 170 tokens

    # Prompt eval
    m = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)",
        stderr,
    )
    if m:
        parsed.prompt_ms = float(m.group(1))
        parsed.prompt_tokens = int(m.group(2))

    # Generation eval
    m = re.search(
        r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)"
        r".*?([\d.]+)\s*tokens per second",
        stderr,
    )
    if m:
        parsed.eval_ms = float(m.group(1))
        parsed.tokens_generated = int(m.group(2))
        parsed.tokens_per_second = float(m.group(3))

    # Total time
    m = re.search(r"total time\s*=\s*([\d.]+)\s*ms", stderr)
    if m:
        parsed.total_ms = float(m.group(1))

    # Fallback: if llama.cpp returned successfully but we couldn't parse timing
    if parsed.tokens_generated == 0 and result.returncode == 0:
        # Count output tokens from stdout as rough estimate
        words = len(result.stdout.split())
        parsed.tokens_generated = max(words, 1)

    return parsed


# ---------------------------------------------------------------------------
# Metric payload builder (matches /api/metrics/ingest schema)
# ---------------------------------------------------------------------------


def build_metric_payload(
    reading: SoCReading,
    job_id: str,
    job_name: str,
    energy_delta_j: Optional[float] = None,
) -> dict:
    """Build one metric record matching the MetricPayload TypeScript interface."""
    # Estimate utilization from power draw relative to TDP
    util_pct = min(100, max(0, int(reading.gpu_power_w / POWER_LIMIT_W * 100)))

    return {
        "timestamp": reading.timestamp,
        "gpu_index": GPU_INDEX,
        "gpu_uuid": GPU_UUID,
        "gpu_name": GPU_NAME,
        "power_draw_w": round(reading.gpu_power_w, 2),
        "power_limit_w": POWER_LIMIT_W,
        "energy_delta_j": round(energy_delta_j, 4) if energy_delta_j is not None else None,
        "utilization_gpu_pct": util_pct,
        "utilization_memory_pct": 0,  # unified memory — not directly measurable per-GPU
        "temperature_c": int(reading.die_temp_c) if reading.die_temp_c > 0 else 40,
        "fan_speed_pct": 0,
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "sm_clock_mhz": reading.gpu_freq_mhz or None,
        "job_id": job_id,
    }


# ---------------------------------------------------------------------------
# Dashboard upload
# ---------------------------------------------------------------------------


def upload_metrics(
    metrics: list[dict],
    api_key: str,
    endpoint: str,
) -> int:
    """POST metrics to /api/metrics/ingest. Returns count uploaded or 0 on failure."""
    if not REQUESTS_AVAILABLE:
        print("  [WARN] `requests` not installed — skipping upload (pip install requests)")
        return 0
    if not api_key:
        return 0

    # API accepts max 1000 per request — batch if needed
    uploaded = 0
    for i in range(0, len(metrics), 1000):
        batch = metrics[i : i + 1000]
        try:
            resp = requests.post(
                endpoint,
                json=batch,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": api_key,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                uploaded += data.get("inserted", len(batch))
            else:
                print(f"  [WARN] Upload batch failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            print(f"  [WARN] Upload error: {e}")

    return uploaded


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


def run_job(
    job_name: str,
    job_cfg: dict,
    binary: str,
    model: str,
    monitor: PowerThermalMonitor,
    thermal_limit: float,
    api_key: str,
    api_endpoint: str,
) -> dict:
    """Execute one benchmark job, collect metrics, upload, and print summary."""

    job_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    start_iso = start_time.isoformat()

    print(f"\n{'=' * 70}")
    print(f"  JOB: {job_name}")
    print(f"  {job_cfg['description']}")
    print(f"  Tokens requested: {job_cfg['n_predict']}")
    print(f"  Batch size:       {job_cfg['batch_size']}")
    print(f"  Context size:     {job_cfg['ctx_size']}")
    print(f"  Job ID:           {job_id}")
    print(f"{'=' * 70}")

    # Thermal pre-check
    if not thermal_ok(monitor, thermal_limit):
        thermal_cooldown(monitor, thermal_limit)

    # Run inference
    wall_start = time.time()

    result = run_llama_inference(
        binary=binary,
        model=model,
        prompt=job_cfg["prompt"],
        n_predict=job_cfg["n_predict"],
        ctx_size=job_cfg["ctx_size"],
        batch_size=job_cfg["batch_size"],
        n_gpu_layers=99,
    )

    wall_elapsed = time.time() - wall_start
    end_time = datetime.now(timezone.utc)

    # Gather power/thermal samples recorded during this job
    samples = monitor.history_since(start_iso)

    # Build metric payloads with energy deltas
    payloads: list[dict] = []
    prev_ts: Optional[str] = None
    for s in samples:
        energy_delta = None
        if prev_ts is not None:
            try:
                t0 = datetime.fromisoformat(prev_ts)
                t1 = datetime.fromisoformat(s.timestamp)
                dt = (t1 - t0).total_seconds()
                energy_delta = s.gpu_power_w * dt  # E = P * dt (joules)
            except (ValueError, TypeError):
                pass  # malformed timestamp — skip energy delta
        prev_ts = s.timestamp

        payloads.append(build_metric_payload(
            reading=s,
            job_id=job_id,
            job_name=job_name,
            energy_delta_j=energy_delta,
        ))

    # Aggregated stats
    if samples and any(s.gpu_power_w > 0 for s in samples):
        powered = [s for s in samples if s.gpu_power_w > 0]
        avg_power = sum(s.gpu_power_w for s in powered) / len(powered)
        max_power = max(s.gpu_power_w for s in powered)
    else:
        avg_power = max_power = 0.0

    if samples and any(s.die_temp_c > 0 for s in samples):
        temped = [s for s in samples if s.die_temp_c > 0]
        avg_temp = sum(s.die_temp_c for s in temped) / len(temped)
        max_temp = max(s.die_temp_c for s in temped)
    else:
        avg_temp = max_temp = 0.0

    total_energy_j = avg_power * wall_elapsed
    total_energy_kwh = total_energy_j / 3_600_000.0

    # Upload
    uploaded = 0
    if api_key and payloads:
        uploaded = upload_metrics(payloads, api_key, api_endpoint)
        if uploaded:
            print(f"  Uploaded {uploaded} metric samples to dashboard")

    # Print summary
    print(f"\n  --- {job_name} Summary ---")
    print(f"  Wall time:             {wall_elapsed:>8.1f} s")
    print(f"  Prompt tokens:         {result.prompt_tokens:>8}")
    print(f"  Tokens generated:      {result.tokens_generated:>8}")
    print(f"  Generation speed:      {result.tokens_per_second:>8.1f} tok/s")
    if result.prompt_ms > 0:
        print(f"  Prefill time:          {result.prompt_ms:>8.0f} ms")
    if result.eval_ms > 0:
        print(f"  Decode time:           {result.eval_ms:>8.0f} ms")
    print(f"  Power samples:         {len(samples):>8}")

    if avg_power > 0:
        print(f"  Avg GPU power:         {avg_power:>8.1f} W")
        print(f"  Peak GPU power:        {max_power:>8.1f} W")
        print(f"  Avg die temperature:   {avg_temp:>8.0f} C")
        print(f"  Peak die temperature:  {max_temp:>8.0f} C")
        print(f"  Total energy:          {total_energy_j:>8.1f} J  ({total_energy_kwh:.6f} kWh)")
        if result.tokens_generated > 0:
            j_per_tok = total_energy_j / result.tokens_generated
            w_s_per_tok = avg_power / max(result.tokens_per_second, 0.01)
            print(f"  Energy / token:        {j_per_tok:>8.3f} J/tok  ({w_s_per_tok:.3f} W*s/tok)")
    else:
        print("  (powermetrics data unavailable — power/energy stats omitted)")

    # Thermal post-check
    if not thermal_ok(monitor, thermal_limit):
        print(f"  !! Post-job thermal warning: {monitor.latest.die_temp_c:.0f} C")

    return {
        "job_id": job_id,
        "job_name": job_name,
        "start_time": start_iso,
        "end_time": end_time.isoformat(),
        "duration_s": round(wall_elapsed, 2),
        "prompt_tokens": result.prompt_tokens,
        "tokens_generated": result.tokens_generated,
        "tokens_per_second": round(result.tokens_per_second, 2),
        "prompt_ms": round(result.prompt_ms, 1),
        "eval_ms": round(result.eval_ms, 1),
        "avg_power_w": round(avg_power, 2),
        "max_power_w": round(max_power, 2),
        "total_energy_j": round(total_energy_j, 2),
        "total_energy_kwh": round(total_energy_kwh, 8),
        "avg_temp_c": round(avg_temp, 1),
        "max_temp_c": round(max_temp, 1),
        "power_samples": len(samples),
        "metrics_uploaded": uploaded,
    }


# ---------------------------------------------------------------------------
# MLX job runner
# ---------------------------------------------------------------------------


def run_mlx_job(
    mlx_model: str,
    monitor: PowerThermalMonitor,
    thermal_limit: float,
    api_key: str,
    api_endpoint: str,
) -> dict:
    """Run the MLX LoRA fine-tuning job, collect power metrics, and upload."""
    cfg = MLX_JOB_CFG
    job_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    start_iso = start_time.isoformat()

    print(f"\n{'=' * 70}")
    print(f"  JOB: {MLX_JOB_NAME}")
    print(f"  {cfg['description']}")
    print(f"  MLX model:  {mlx_model}")
    print(f"  Iters:      {cfg['iters']}   Batch: {cfg['batch_size']}   LoRA layers: {cfg['lora_layers']}")
    print(f"  Job ID:     {job_id}")
    print(f"{'=' * 70}")

    if not thermal_ok(monitor, thermal_limit):
        thermal_cooldown(monitor, thermal_limit)

    # Create temp dirs for dataset and adapter weights
    with tempfile.TemporaryDirectory(prefix="alum_mlx_") as tmpdir:
        data_dir = Path(tmpdir) / "data"
        adapter_dir = Path(tmpdir) / "adapters"
        data_dir.mkdir()
        adapter_dir.mkdir()

        generate_synthetic_dataset(data_dir)
        print(f"  Synthetic dataset: {len(_SYNTHETIC_PAIRS)} instruction pairs → {data_dir / 'train.jsonl'}")
        print("  Starting MLX LoRA training ...\n")

        wall_start = time.time()
        mlx_result = run_mlx_finetune(
            model_id=mlx_model,
            data_dir=data_dir,
            iters=cfg["iters"],
            batch_size=cfg["batch_size"],
            lora_layers=cfg["lora_layers"],
            seq_length=cfg["seq_length"],
            learning_rate=cfg["learning_rate"],
            adapter_dir=adapter_dir,
        )
        wall_elapsed = time.time() - wall_start
        end_time = datetime.now(timezone.utc)

    samples = monitor.history_since(start_iso)

    # Build metric payloads
    payloads: list[dict] = []
    prev_ts: Optional[str] = None
    for s in samples:
        energy_delta = None
        if prev_ts is not None:
            try:
                t0 = datetime.fromisoformat(prev_ts)
                t1 = datetime.fromisoformat(s.timestamp)
                dt = (t1 - t0).total_seconds()
                energy_delta = s.gpu_power_w * dt
            except (ValueError, TypeError):
                pass  # malformed timestamp — skip energy delta
        prev_ts = s.timestamp
        payloads.append(build_metric_payload(
            reading=s,
            job_id=job_id,
            job_name=MLX_JOB_NAME,
            energy_delta_j=energy_delta,
        ))

    # Aggregate power/thermal
    powered = [s for s in samples if s.gpu_power_w > 0]
    avg_power = sum(s.gpu_power_w for s in powered) / len(powered) if powered else 0.0
    max_power = max((s.gpu_power_w for s in powered), default=0.0)
    temped = [s for s in samples if s.die_temp_c > 0]
    avg_temp = sum(s.die_temp_c for s in temped) / len(temped) if temped else 0.0
    max_temp = max((s.die_temp_c for s in temped), default=0.0)
    total_energy_j = avg_power * wall_elapsed
    total_energy_kwh = total_energy_j / 3_600_000.0

    # Upload
    uploaded = 0
    if api_key and payloads:
        uploaded = upload_metrics(payloads, api_key, api_endpoint)
        if uploaded:
            print(f"  Uploaded {uploaded} metric samples to dashboard")

    # Print summary
    print(f"\n  --- {MLX_JOB_NAME} Summary ---")
    print(f"  Wall time:             {wall_elapsed:>8.1f} s")
    if mlx_result.iters_completed:
        print(f"  Iters completed:       {mlx_result.iters_completed:>8}")
        print(f"  Final train loss:      {mlx_result.final_loss:>8.4f}")
        print(f"  Training tok/s:        {mlx_result.tokens_per_second:>8.1f}")
    else:
        print("  (Could not parse MLX training output — check mlx-lm is installed)")
        if mlx_result.raw_output:
            print(f"  Last output: {mlx_result.raw_output[-300:].strip()}")
    print(f"  Power samples:         {len(samples):>8}")
    if avg_power > 0:
        print(f"  Avg GPU power:         {avg_power:>8.1f} W")
        print(f"  Peak GPU power:        {max_power:>8.1f} W")
        print(f"  Avg die temperature:   {avg_temp:>8.0f} C")
        print(f"  Peak die temperature:  {max_temp:>8.0f} C")
        print(f"  Total energy:          {total_energy_j:>8.1f} J  ({total_energy_kwh:.6f} kWh)")

    if not thermal_ok(monitor, thermal_limit):
        print(f"  !! Post-job thermal warning: {monitor.latest.die_temp_c:.0f} C")

    return {
        "job_id": job_id,
        "job_name": MLX_JOB_NAME,
        "start_time": start_iso,
        "end_time": end_time.isoformat(),
        "duration_s": round(wall_elapsed, 2),
        "iters_completed": mlx_result.iters_completed,
        "final_loss": mlx_result.final_loss,
        "tokens_per_second": round(mlx_result.tokens_per_second, 2),
        "avg_power_w": round(avg_power, 2),
        "max_power_w": round(max_power, 2),
        "total_energy_j": round(total_energy_j, 2),
        "total_energy_kwh": round(total_energy_kwh, 8),
        "avg_temp_c": round(avg_temp, 1),
        "max_temp_c": round(max_temp, 1),
        "power_samples": len(samples),
        "metrics_uploaded": uploaded,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AluminatiAI M5 Energy Benchmark — GPU Power Profile Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model ~/models/llama-3.2-3b-q4_k_m.gguf
  %(prog)s --model ./model.gguf --api-key alum_xxxxx
  %(prog)s --model ./model.gguf --jobs Light-Chat Deep-Analysis
  %(prog)s --model ./model.gguf --thermal-limit 95 --save-json results.json
        """,
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--binary", default=None,
        help="Path to llama-cli binary (auto-detected if omitted)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ALUMINATAI_API_KEY", ""),
        help="AluminatiAI API key for dashboard upload (env: ALUMINATAI_API_KEY)",
    )
    parser.add_argument(
        "--api-endpoint",
        default=os.getenv("ALUMINATAI_API_ENDPOINT", DEFAULT_API_ENDPOINT),
        help="Metrics ingestion endpoint (env: ALUMINATAI_API_ENDPOINT)",
    )
    parser.add_argument(
        "--mlx-model",
        default=os.getenv("ALUMINATAI_MLX_MODEL", DEFAULT_MLX_MODEL),
        help=f"HuggingFace model ID for MLX fine-tuning job (default: {DEFAULT_MLX_MODEL})",
    )
    all_job_names = list(JOBS.keys()) + [MLX_JOB_NAME]
    parser.add_argument(
        "--jobs", nargs="+",
        choices=all_job_names,
        default=all_job_names,
        help="Which jobs to run (default: all four)",
    )
    parser.add_argument(
        "--thermal-limit", type=float, default=THERMAL_LIMIT_C,
        help=f"Pause if SoC die temp exceeds this value in Celsius (default: {THERMAL_LIMIT_C})",
    )
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="Skip the warm-up phase (model load cost goes to first job)",
    )
    parser.add_argument(
        "--save-json", default=None,
        help="Save detailed results to a JSON file",
    )

    args = parser.parse_args()

    # -- Validate model file -------------------------------------------------
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    # -- Find llama-cli ------------------------------------------------------
    binary = args.binary or find_llama_binary()
    if not binary:
        print("ERROR: llama-cli not found.")
        print("  Install:  brew install llama.cpp")
        print("  Or pass:  --binary /path/to/llama-cli")
        sys.exit(1)

    # -- Banner --------------------------------------------------------------
    llama_jobs = [j for j in args.jobs if j != MLX_JOB_NAME]
    mlx_jobs   = [j for j in args.jobs if j == MLX_JOB_NAME]

    print()
    print("  AluminatiAI M5 Energy Benchmark")
    print(f"  {'=' * 50}")
    print(f"  GGUF model:     {model_path.name}")
    print(f"  llama-cli:      {binary}")
    print(f"  GPU offload:    -ngl 99  (all layers on Metal)")
    if mlx_jobs:
        print(f"  MLX model:      {args.mlx_model}")
    print(f"  Jobs:           {', '.join(args.jobs)}")
    print(f"  Thermal limit:  {args.thermal_limit} C")
    upload_status = "Enabled" if args.api_key else "Disabled (no --api-key)"
    print(f"  Dashboard:      {upload_status}")
    print(f"  {'=' * 50}")

    # -- Start power/thermal monitor -----------------------------------------
    print("\n  Starting power & thermal monitor ...")
    print("  (If prompted, enter your sudo password for powermetrics)\n")
    monitor = PowerThermalMonitor(interval_ms=SAMPLE_INTERVAL_MS)
    monitor.start()

    if monitor.available:
        r = monitor.latest
        temp_str = f"{r.die_temp_c:.0f} C" if r.die_temp_c > 0 else "n/a"
        gpu_str = f"{r.gpu_power_w:.1f} W" if r.gpu_power_w > 0 else "idle"
        print(f"  powermetrics active — die temp: {temp_str}, GPU: {gpu_str}")
    else:
        print("  powermetrics unavailable — running without power/thermal data")
        print("  (Re-run with sudo access for full instrumentation)")

    # -- Warm-up phase -------------------------------------------------------
    if not args.skip_warmup:
        print("\n  --- WARM-UP PHASE ---")
        print("  Loading model into Metal GPU memory (excluded from job energy) ...")
        warmup = run_llama_inference(
            binary=binary,
            model=str(model_path),
            prompt="Hello",
            n_predict=1,
            ctx_size=512,
            batch_size=512,
            n_gpu_layers=99,
        )
        if warmup.prompt_ms > 0:
            print(f"  Model loaded — prefill: {warmup.prompt_ms:.0f} ms ({warmup.prompt_tokens} tokens)")
        else:
            print("  Model loaded.")
        # Let thermals settle after the load
        time.sleep(3)

    # -- Run jobs ------------------------------------------------------------
    all_results: list[dict] = []

    for i, job_name in enumerate(args.jobs):
        if job_name == MLX_JOB_NAME:
            result = run_mlx_job(
                mlx_model=args.mlx_model,
                monitor=monitor,
                thermal_limit=args.thermal_limit,
                api_key=args.api_key,
                api_endpoint=args.api_endpoint,
            )
        else:
            cfg = JOBS[job_name]
            result = run_job(
                job_name=job_name,
                job_cfg=cfg,
                binary=binary,
                model=str(model_path),
                monitor=monitor,
                thermal_limit=args.thermal_limit,
                api_key=args.api_key,
                api_endpoint=args.api_endpoint,
            )
        all_results.append(result)

        # Inter-job cooldown (skip after last job)
        if i < len(args.jobs) - 1:
            print("\n  Cooling pause between jobs (5s) ...")
            time.sleep(5)

    # -- Final summary -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'=' * 70}")
    header = (
        f"  {'Job':<18} {'Tokens':>7} {'tok/s':>7} "
        f"{'Avg W':>7} {'Peak W':>7} {'Energy J':>9} {'J/tok':>7}"
    )
    print(header)
    print(f"  {'-' * 64}")

    for r in all_results:
        if r["job_name"] == MLX_JOB_NAME:
            # Show iters + loss instead of tokens for the fine-tuning job
            toks_str = f"{r.get('iters_completed', 0)}it"
            tps_str  = f"{r['tokens_per_second']:>7.1f}"
            j_tok    = r["total_energy_j"] / max(r.get("iters_completed", 1), 1)
            print(
                f"  {r['job_name']:<18} "
                f"{toks_str:>7} "
                f"{tps_str} "
                f"{r['avg_power_w']:>7.1f} "
                f"{r['max_power_w']:>7.1f} "
                f"{r['total_energy_j']:>9.1f} "
                f"{'n/a':>7}"
            )
        else:
            toks = r.get("tokens_generated", 0)
            j_tok = r["total_energy_j"] / max(toks, 1)
            print(
                f"  {r['job_name']:<18} "
                f"{toks:>7} "
                f"{r['tokens_per_second']:>7.1f} "
                f"{r['avg_power_w']:>7.1f} "
                f"{r['max_power_w']:>7.1f} "
                f"{r['total_energy_j']:>9.1f} "
                f"{j_tok:>7.3f}"
            )

    total_energy = sum(r["total_energy_j"] for r in all_results)
    total_tokens = sum(r.get("tokens_generated", 0) for r in all_results)
    total_wall = sum(r["duration_s"] for r in all_results)
    total_j_tok = total_energy / max(total_tokens, 1)

    print(f"  {'-' * 64}")
    print(
        f"  {'TOTAL':<18} "
        f"{total_tokens:>7} "
        f"{'':>7} {'':>7} {'':>7} "
        f"{total_energy:>9.1f} "
        f"{total_j_tok:>7.3f}"
    )
    print()
    print(f"  Total wall time:   {total_wall:.1f} s")
    print(f"  Total energy:      {total_energy:.1f} J  ({total_energy / 3_600_000:.6f} kWh)")

    if args.api_key:
        total_uploaded = sum(r["metrics_uploaded"] for r in all_results)
        print(f"  Metrics uploaded:  {total_uploaded}")
        print(f"  Dashboard:         Check 'Energy by Job' to see {len(all_results)} distinct profiles")

    print()

    # -- Save JSON -----------------------------------------------------------
    if args.save_json:
        out = Path(args.save_json)
        out.write_text(json.dumps(all_results, indent=2))
        print(f"  Results saved to {out}")

    # -- Cleanup -------------------------------------------------------------
    monitor.stop()


if __name__ == "__main__":
    main()
