# AluminatiAI GreenTune — AMD Developer Hackathon Submission

**Track:** Fine-Tuning on AMD  
**Team:** AluminatiAI (Kevin Mello)  
**GitHub:** [github.com/AgentMulder404/AluminatAI](https://github.com/AgentMulder404/AluminatAI)  
**Live Dashboard:** [aluminatai.com/admin/fine-tuning](https://aluminatai.com/admin/fine-tuning)

---

## Problem

Every organization fine-tuning LLMs today is flying blind on energy. Teams track loss curves, learning rates, and throughput — but nobody measures the energy cost per token produced. This matters because:

- **GPU energy is invisible.** A training run on MI300X draws 750W at peak, but teams never see that number or its cost implications.
- **Optimization is guesswork.** Without per-token energy metrics, engineers can't compare training configurations on efficiency — they optimize for speed alone.
- **Carbon compliance is coming.** The EU AI Act and SEC climate disclosures will require energy reporting for AI workloads. No tooling exists to generate this data during training.

## Solution: GreenTune

GreenTune is an energy-aware fine-tuning pipeline built on AMD ROCm that treats **Joules-per-token as a first-class training metric** alongside loss and throughput.

### What it does

1. **Real-time GPU power monitoring** via `amdsmi` Python bindings with `rocm-smi` CLI fallback — samples power, temperature, utilization, and VRAM every 0.5s during training
2. **Per-step energy accounting** — computes J/token, cumulative kWh, energy cost ($), and CO2 emissions at every logging step using trapezoidal integration
3. **Energy-efficient QLoRA fine-tuning** — 4-bit NF4 quantization, LoRA rank 16, gradient checkpointing on Qwen2.5-7B-Instruct
4. **Live admin dashboard** — interactive charts comparing training runs on energy efficiency, with auto-generated optimization insights
5. **Backend-agnostic GPU agent** — auto-detects AMD (amdsmi/rocm-smi) or NVIDIA (NVML) GPUs, reports identical `GPUMetrics` to the platform

### Architecture

```
┌──────────────────────────────────────────────────┐
│  AluminatiAI Platform (Next.js + Supabase)       │
│  ┌────────────────────────────────────────────┐   │
│  │  GreenTune Dashboard (/admin/fine-tuning)  │   │
│  │  • Run Monitor — power curves, loss charts │   │
│  │  • Leaderboard — side-by-side comparison   │   │
│  │  • ROI Calculator — cost projections       │   │
│  │  • Playground — model evaluation           │   │
│  └────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────┘
                       │ metrics JSON
┌──────────────────────┴───────────────────────────┐
│  GreenTune Training Pipeline (Python / ROCm)     │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ greentune.py │  │ energy_callback.py       │  │
│  │ QLoRA + SFT  │──│ PowerSamplerThread       │  │
│  │ Qwen2.5-7B   │  │ EnergyAccumulator        │  │
│  └──────────────┘  │ J/token per step         │  │
│                     └──────────┬───────────────┘  │
│  ┌──────────────────────────┐  │                  │
│  │ amd_collector.py         │  │ amdsmi / CLI     │
│  │ AMDGPUCollector          │──┘                  │
│  │ (drop-in for NVIDIA)     │                     │
│  └──────────────────────────┘                     │
│                     │                             │
│              AMD MI300X (ROCm 7.0)                │
│              192GB HBM3 · 750W TDP · gfx942       │
└───────────────────────────────────────────────────┘
```

## AMD Hardware Advantage

GreenTune was built and tested on **AMD Instinct MI300X** via the AMD Developer Cloud:

| Spec | Value |
|------|-------|
| GPU | AMD Instinct MI300X |
| VRAM | 192 GB HBM3 |
| TDP | 750W |
| Architecture | CDNA3 (gfx942) |
| ROCm | 7.0.51831 |
| PyTorch | 2.9.0+rocm7.0.0 |

**Why MI300X for fine-tuning:**
- **192GB VRAM** — loads Qwen2.5-7B in 4-bit NF4 with only 5.2GB, leaving massive headroom for batch size scaling and longer sequences
- **No model sharding needed** — single-GPU fine-tuning eliminates multi-GPU communication overhead, simplifying energy measurement
- **`amdsmi` power telemetry** — hardware-level power monitoring via Python bindings with sub-second granularity, no sudo required
- **ROCm ecosystem** — native PyTorch support, `bitsandbytes` for quantization, `peft` + `trl` for QLoRA/SFT

## Results

### Training Runs on MI300X

We ran identical QLoRA fine-tuning jobs on Qwen2.5-7B-Instruct (500 Hermes agent reasoning traces, 1 epoch, LoRA rank 16, NF4 quantization) with two different batch configurations:

| Metric | Baseline (bs=2, ga=4) | Small Batch (bs=1, ga=8) | Delta |
|--------|----------------------|--------------------------|-------|
| Effective Batch Size | 8 | 8 | Same |
| Training Duration | 138.5s (2.3 min) | 178.9s (3.0 min) | +29% |
| Avg Power Draw | 630.5 W | 637.2 W | +1% |
| Peak Power Draw | 752 W | 749 W | ~Same |
| Total Energy | 87,300 J (0.024 kWh) | 113,834 J (0.032 kWh) | **+30%** |
| J/Token | 0.355 | 0.463 | **+30%** |
| Tokens/sec | 1,774 | 1,374 | -23% |
| Energy Cost | $0.0024 | $0.0032 | +33% |
| CO2 Emissions | 9.46 g | 12.33 g | +30% |
| Final Loss | 1.054 | 2.114 | — |

### Key Finding

**Smaller batch sizes do NOT reduce power draw on MI300X.** The GPU saturates at ~750W regardless of batch configuration. Reducing batch size from 2 to 1 (while maintaining the same effective batch via gradient accumulation) resulted in:

- **Same power draw** — the MI300X runs at full power either way
- **29% longer training time** — fewer tokens processed per step
- **30% more total energy wasted** — same watts × more time = more Joules
- **30% worse J/token efficiency** — the metric that matters

**This insight is invisible without per-token energy measurement.** A team looking at only loss and throughput might assume smaller batches are "lighter" on the GPU. AluminatiAI's energy monitoring proves the opposite and provides data-driven guidance: **maximize batch size on MI300X to optimize energy efficiency.**

### Cost Context

| What | Value |
|------|-------|
| Total GPU compute cost (baseline run) | $0.41 (12.3 min × $1.99/hr) |
| Energy cost of that run | $0.0024 |
| Energy as % of compute | 0.6% |
| CO2 equivalent | 9.46g (≈ driving 77 feet) |
| Hackathon total GPU spend | ~$15 |

Energy cost is <1% of compute cost today — but at datacenter scale (1000 GPUs × 24/7), that's $87,600/year in electricity alone. Knowing which configurations waste 30% of that is worth measuring.

## Technical Implementation

### Files Created/Modified for This Hackathon

**New files:**
- `agent/amd_collector.py` — AMD GPU metrics collector (amdsmi + rocm-smi CLI fallback), drop-in replacement for NVIDIA collector
- `agent/finetune/greentune.py` — QLoRA fine-tuning pipeline with integrated energy monitoring
- `agent/finetune/energy_callback.py` — HuggingFace Trainer callback that samples GPU power and computes per-step energy metrics
- `agent/finetune/rocm_power.py` — AMD-specific power monitor (PowerSamplerThread, EnergyAccumulator)
- `agent/finetune/dataset_builder.py` — Synthetic GPU/energy domain dataset generator
- `agent/finetune/eval_model.py` — Post-training evaluation harness
- `app/admin/fine-tuning/page.tsx` — 4-tab admin dashboard with embedded real metrics
- `app/api/admin/fine-tuning/route.ts` — API route for Claude-powered run analysis

**Modified files:**
- `agent/agent.py` — Added AMD GPU auto-detection: tries NVIDIA (NVML) first, falls back to AMD (amdsmi/rocm-smi)
- `app/admin/layout.tsx` — Added Fine-Tuning nav link to admin sidebar

### Energy Measurement Method

Power is sampled via `amdsmi.amdsmi_get_power_info()` every 0.5 seconds in a background thread. Energy per interval is computed using trapezoidal integration:

```
energy_delta = (power_current + power_previous) / 2 × dt
```

Per-step metrics aggregate these samples between training steps to produce:
- `step_joules` — total energy for the logging window
- `joules_per_token` — step_joules / tokens_in_window
- `cumulative_kwh` — running total for the entire run
- `cumulative_co2_grams` — kWh × carbon intensity (390 gCO2/kWh US average)

### GPU Agent Auto-Detection

The AluminatiAI agent automatically detects the GPU vendor at startup:

```python
# Try NVIDIA first
collector = GPUCollector(collect_clocks=False)  # uses pynvml

# Fall back to AMD
if collector is None:
    collector = AMDGPUCollector(collect_clocks=False)  # uses amdsmi
```

Both collectors return identical `GPUMetrics` dataclass instances, making the entire agent loop backend-agnostic. The same dashboard, API, and alerting infrastructure works on NVIDIA A100/H100 and AMD MI300X without code changes.

## Demo

The live dashboard at `/admin/fine-tuning` shows:

1. **Run Monitor** — Select between training runs. View 4 stat cards (energy, power, cost, CO2), GPU power curve over time with temperature overlay, training loss alongside J/token efficiency, run configuration details, and step-level metrics table.

2. **Efficiency Leaderboard** — Side-by-side comparison table highlighting the winner per metric (green = better). Auto-generated insight banner explains the 30% efficiency finding. Bar chart comparing J/token across runs.

3. **ROI Calculator** — Input GPU rate, training time, and power draw to compute cost breakdowns. Compares QLoRA vs full fine-tuning vs API-only approaches with energy and carbon metrics.

4. **Model Playground** — Test prompts against Claude with GreenTune domain context. View evaluation results from training runs.

## What's Next

- **Cross-GPU benchmarking** — Run the same workload on A100, H100, and MI300X to build a J/token comparison database (Green AI Index)
- **Automated config recommendations** — Use energy data to suggest optimal batch size, LoRA rank, and sequence length for a given GPU
- **Carbon compliance reports** — Generate audit-ready energy and emissions reports for EU AI Act / SEC disclosure requirements
- **Fleet-level optimization** — For multi-GPU clusters, identify which workloads should run on which hardware based on J/token efficiency

## Reproducibility

```bash
# SSH into AMD Developer Cloud instance
ssh -i ~/.ssh/amd_devcloud root@<ip>

# Inside the ROCm container
cd /workspace/AluminatAI/agent

# Terminal 1: Start the GPU agent
python3 agent.py --interval 2 --dry-run --duration 300

# Terminal 2: Run fine-tuning
cd finetune
python3 greentune.py \
  --hermes-only --hermes-max 500 \
  --epochs 1 --batch-size 2 --grad-accum 4 \
  --logging-steps 10

# Output: output/greentune-run/
#   energy_metrics.json — summary + per-step energy data
#   power_samples.json  — raw 0.5s power readings
#   run_config.json     — training configuration
#   adapter/            — LoRA adapter weights
```

---

*Built with AMD Instinct MI300X on AMD Developer Cloud. All energy measurements are from real training runs, not simulations.*
