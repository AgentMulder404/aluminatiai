<p align="center">
  <strong>AluminatAI</strong><br>
  GPU Energy Monitoring &amp; Energy-Efficient LLM Fine-Tuning
</p>

<p align="center">
  <a href="https://pypi.org/project/aluminatiai/"><img src="https://img.shields.io/pypi/v/aluminatiai" alt="PyPI"></a>
  <a href="https://pypi.org/project/aluminatiai/"><img src="https://img.shields.io/pypi/pyversions/aluminatiai" alt="Python"></a>
  <a href="https://github.com/AgentMulder404/aluminatiai/blob/main/LICENSE"><img src="https://img.shields.io/github/license/AgentMulder404/aluminatiai" alt="License"></a>
</p>

---

Open-source Python agent that monitors GPU power consumption, attributes energy costs to individual jobs, and optimizes LLM fine-tuning for minimum Joules-per-token.

Works on **NVIDIA**, **AMD (ROCm)**, **Intel Gaudi**, **Intel Arc**, **Apple Silicon**, and **CPU-only (RAPL)** machines.

## Install

```bash
pip install aluminatiai                # GPU monitoring agent
pip install aluminatiai[finetune]      # + QLoRA training with energy tracking
pip install aluminatiai[greentune]     # everything
```

## What It Does

| Capability | Description |
|---|---|
| **GPU Monitoring** | Power, temperature, utilization sampled every 5s, attributed to jobs, streamed to dashboard |
| **Cost Attribution** | Per-job energy costs across multi-tenant GPU clusters (Slurm, K8s, Run:ai) |
| **GreenTune** | Energy-efficient QLoRA fine-tuning with real AMD MI300X telemetry |
| **Swarm Optimizer** | Offline hyperparameter search that minimizes J/token — no API keys needed |
| **Lobster Trap** | Energy governance: carbon budget, efficiency floor, cost guard per training run |
| **Prometheus** | `/metrics` endpoint with GPU power, energy, attribution, and upload health gauges |

---

## GreenTune — Energy-Efficient Fine-Tuning

GreenTune tracks real-time power consumption during LLM fine-tuning and optimizes hyperparameters to minimize energy waste. Built for AMD MI300X (192GB HBM3, 750W TDP) with ROCm, also works on NVIDIA GPUs.

### Swarm Optimizer (no API key needed)

```bash
aluminatiai swarm --max-samples 500
```

Runs an exhaustive grid search over batch size, gradient accumulation, and LoRA rank. Projects energy for each config, enforces Lobster Trap policies, and ranks by J/token efficiency.

```
┏━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ # ┃ Batch Size┃ Grad Accum┃ LoRA Rank┃ J/tok  ┃ CO2 (g) ┃ Cost    ┃ Duration ┃
┡━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ 1 │ 32        │ 8         │ 8        │ 0.0265 │ 0.74    │ $0.0002 │ 0.2 min  │
│ 2 │ 32        │ 8         │ 16       │ 0.0271 │ 0.75    │ $0.0002 │ 0.2 min  │
│ 3 │ 32        │ 8         │ 32       │ 0.0284 │ 0.79    │ $0.0002 │ 0.2 min  │
│ 4 │ 16        │ 8         │ 8        │ 0.0291 │ 0.81    │ $0.0002 │ 0.2 min  │
│ 5 │ 16        │ 8         │ 16       │ 0.0304 │ 0.84    │ $0.0003 │ 0.2 min  │
└───┴───────────┴───────────┴──────────┴────────┴─────────┴─────────┴──────────┘
```

### EnergyCallback — Drop Into Any HuggingFace Trainer

```python
from aluminatiai.finetune import EnergyCallback

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[EnergyCallback(gpu_index=0)],
)
trainer.train()
```

Tracks per-step power draw, Joules-per-token, cumulative energy, CO2 emissions, and cost. Outputs a full energy report at the end of training.

### Train with Live Dashboard Upload

```bash
aluminatiai train \
  --hermes-only --hermes-max 500 \
  --batch-size 4 --grad-accum 4 \
  --lora-rank 16 --epochs 1 \
  --api-url https://www.aluminatiai.com \
  --api-key alum_your_key_here \
  --run-name "My Training Run"
```

### Lobster Trap — Energy Governance

Every training config is checked against four policies before it runs:

| Policy | Limit | What it enforces |
|---|---|---|
| `carbon_budget` | 50g CO2 | Max carbon emissions per run |
| `energy_cap` | 1 kWh | Max total energy per run |
| `efficiency_floor` | 0.8 J/tok | Max joules per token |
| `cost_guard` | $1.00 | Max energy cost per run |

### Python API

```python
from aluminatiai.finetune import GreenTuneSwarm

swarm = GreenTuneSwarm()
result = swarm.optimize("Minimize J/token for Qwen2.5-7B")

print(result["recommendation"])
# {'batch_size': 32, 'grad_accum': 8, 'lora_rank': 8, 'projected_jpt': 0.0265, ...}
```

---

## GPU Monitoring Agent

### Quick Start

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
aluminatiai
```

Get your API key at [aluminatiai.com/dashboard](https://aluminatiai.com/dashboard). The agent detects your GPU, starts sampling, and uploads metrics. That's it.

### Supported Hardware

| Backend | GPUs | Primary SDK | Fallback |
|---|---|---|---|
| **NVIDIA** | A100, H100, H200, L40S, RTX 4090, T4, V100 | `nvidia-ml-py` (NVML) | — |
| **AMD** | MI300X, MI300A, MI325X, MI250X, MI210, MI100 | `amdsmi` | `rocm-smi` |
| **Intel Gaudi** | Gaudi, Gaudi2, Gaudi3 | `pyhlml` (SynapseAI) | `hl-smi` |
| **Intel Arc** | A770, A750, B580, Flex 170, Max 1550 | `xpu-smi` (oneAPI) | hwmon sysfs |
| **Apple Silicon** | M1–M5 Pro/Max/Ultra | `powermetrics` (sudo) | `ioreg` |
| **CPU-only** | Any x86 (Intel/AMD) | RAPL sysfs | — |

Auto-detected at startup. No configuration needed.

### Product Tiers

| Tier | Mode | What it does |
|---|---|---|
| **Monitor** | Default | Read-only metrics, cost attribution, Prometheus, carbon tracking |
| **Advisor** | Opt-in | Recommendations with approval workflows: "GPU 3 is 40% idle — cap to 200W?" |
| **Swarm** | Opt-in | Autonomous fleet-wide optimization: power capping, thermal balancing, carbon-aware scheduling |

```bash
aluminatiai                                                        # Monitor
AUTO_TUNE_ENABLED=1 COMMAND_POLL_ENABLED=1 aluminatiai             # Advisor
SWARM_ENABLED=1 COMMAND_POLL_ENABLED=1 AUTO_TUNE_ENABLED=1 aluminatiai  # Swarm
```

---

## CLI Reference

| Command | Description |
|---|---|
| `aluminatiai run` | Main daemon — collect, attribute, upload (default) |
| `aluminatiai train` | GreenTune QLoRA fine-tuning with energy tracking |
| `aluminatiai swarm` | Hyperparameter optimizer (offline, no API keys) |
| `aluminatiai benchmark` | GPU power baseline and efficiency measurement |
| `aluminatiai optimize` | Real-time efficiency analysis with recommendations |
| `aluminatiai ab` | A/B test energy efficiency between configs |
| `aluminatiai carbon-schedule` | Find lowest-carbon window for a job |
| `aluminatiai report` | Generate chargeback reports (CSV/HTML/JSON) |
| `aluminatiai query` | Query local SQLite time-series store |
| `aluminatiai recommend` | GPU recommender — rank GPUs by efficiency and cost |

### aluminatiai run

```bash
aluminatiai                            # run forever (default)
aluminatiai --interval 2               # sample every 2 seconds
aluminatiai --duration 3600            # run for 1 hour then exit
aluminatiai --dry-run                  # collect + attribute, skip uploads
aluminatiai --prometheus-only          # local Prometheus only, no cloud
```

### aluminatiai train

```bash
aluminatiai train --hermes-only --hermes-max 500 --batch-size 4
aluminatiai train --model Qwen/Qwen2.5-7B-Instruct --epochs 3
aluminatiai train --lora-rank 8 --batch-size 8       # faster, less quality
aluminatiai train --eval                              # run eval after training
```

### aluminatiai swarm

```bash
aluminatiai swarm                                     # default search space
aluminatiai swarm --max-samples 500 --model Qwen/Qwen2.5-7B
aluminatiai swarm --batch-sizes 1,2,4,8,16,32         # custom search
aluminatiai swarm --lora-ranks 8,16,32,64             # custom LoRA ranks
aluminatiai swarm --json                              # JSON output for automation
aluminatiai swarm --output results.json               # save to file
```

### aluminatiai benchmark

```bash
aluminatiai benchmark                              # 60s power baseline
aluminatiai benchmark --gpu 0 --duration 120       # specific GPU, 2 min
aluminatiai benchmark --upload                     # submit to Green AI Index
```

---

## Job Attribution

The agent attributes GPU power to individual jobs using a 7-step resolution pipeline:

| Priority | Method | Confidence | Source |
|---|---|---|---|
| 1 | `ALUMINATAI_TEAM` env var | 1.00 | Explicit user tag |
| 2 | Scheduler env vars | 0.90 | `SLURM_JOB_ID`, `RUNAI_JOB_NAME`, K8s pod UID |
| 3 | Scheduler poll | 0.75 | `gpu_to_job()` query |
| 4 | Custom rules file | 0.60 | JSON regex patterns |
| 5 | Cmdline heuristics | 0.40 | Built-in patterns (jupyter, vllm, torchserve, ollama) |
| 6 | Memory split | 0.20 | Power split by GPU memory usage |
| 7 | Idle attribution | 0.30 | `ALUMINATAI_IDLE_TEAM` fallback |

```bash
# Tag your workload
ALUMINATAI_TEAM=nlp-team ALUMINATAI_MODEL=llama3-finetune python train.py
```

---

## ML Framework Integrations

### MLflow

```python
from aluminatiai.integrations.mlflow_callback import AluminatiMLflowCallback
trainer.add_callback(AluminatiMLflowCallback())
```

### Weights & Biases

```python
from aluminatiai.integrations.wandb_callback import AluminatiWandbCallback
trainer.add_callback(AluminatiWandbCallback())
```

### OpenTelemetry

```python
from aluminatiai.integrations.otel_exporter import AluminatiOtelExporter
exporter = AluminatiOtelExporter()
```

---

## Prometheus Metrics

Default port 9100. Key metrics:

| Metric | Type | Description |
|---|---|---|
| `aluminatai_gpu_power_watts` | Gauge | Current power per GPU |
| `aluminatai_gpu_energy_joules_total` | Counter | Cumulative energy per GPU |
| `aluminatai_gpu_utilization_pct` | Gauge | Compute utilization |
| `aluminatai_gpu_temperature_c` | Gauge | Temperature |
| `aluminatai_upload_success_total` | Counter | Successful uploads |
| `aluminatai_attribution_confidence` | Gauge | Attribution confidence (0–1) |

```yaml
scrape_configs:
  - job_name: aluminatiai
    static_configs:
      - targets: ['gpu-host:9100']
```

---

## Deployment

### One-line install (Linux + systemd)

```bash
curl -sSL https://get.aluminatiai.com | bash
```

### Docker (NVIDIA)

```bash
docker run --rm --runtime=nvidia --pid=host \
  -e ALUMINATAI_API_KEY=alum_your_key_here \
  ghcr.io/agentmulder404/aluminatai-agent:latest
```

### Kubernetes DaemonSet

```bash
kubectl apply -f deploy/k8s/daemonset.yaml
```

---

## Configuration

Settings are read in priority order: **env vars** > **config file** > **defaults**.

```bash
aluminatiai --config /etc/aluminatai.json
```

<details>
<summary><strong>Full configuration reference</strong></summary>

### API & Upload

| Env var | Default | Description |
|---|---|---|
| `ALUMINATAI_API_KEY` | *(required)* | Your API key |
| `ALUMINATAI_API_ENDPOINT` | `https://…/v1/metrics/ingest` | Ingest endpoint |
| `UPLOAD_INTERVAL` | `60` | Seconds between flushes |
| `UPLOAD_BATCH_SIZE` | `100` | Metrics per request |

### Sampling

| Env var | Default | Description |
|---|---|---|
| `SAMPLE_INTERVAL` | `5.0` | Seconds between GPU samples |

### Advisor Tier

| Env var | Default | Description |
|---|---|---|
| `AUTO_TUNE_ENABLED` | `false` | Enable optimization recommendations |
| `COMMAND_POLL_ENABLED` | `false` | Enable polling for approved commands |

### Swarm Tier

| Env var | Default | Description |
|---|---|---|
| `SWARM_ENABLED` | `false` | Enable fleet-wide optimization |
| `SWARM_EVAL_INTERVAL` | `300` | Seconds between fleet evaluations |

Built-in fleet policies: `idle_gpu_power_cap`, `thermal_balancing`, `carbon_aware_fleet_cap`, `fleet_gpu_rightsizing`.

Safety: max 25% fleet blast radius, canary ramp-up, leader election, adaptive polling.

### Prometheus

| Env var | Default | Description |
|---|---|---|
| `METRICS_PORT` | `9100` | Scrape port (0 = disabled) |
| `METRICS_BASIC_AUTH` | *(none)* | `user:pass` for HTTP Basic Auth |

### Security

| Env var | Default | Description |
|---|---|---|
| `OFFLINE_MODE` | `false` | WAL only, no HTTP uploads |
| `ALUMINATAI_CA_BUNDLE` | *(none)* | Custom CA PEM path |
| `ALUMINATAI_CLIENT_CERT` | *(none)* | mTLS client cert |

</details>

---

## Package Structure

```
aluminatiai/
├── agent.py              # Main daemon
├── cli.py                # CLI router (run, train, swarm, benchmark, ...)
├── collector.py          # NVIDIA GPU collector (NVML)
├── amd_collector.py      # AMD GPU collector (amdsmi / rocm-smi)
├── gaudi_collector.py    # Intel Gaudi collector
├── intel_arc_collector.py# Intel Arc collector
├── apple_collector.py    # Apple Silicon collector
├── rapl_collector.py     # CPU-only RAPL collector
├── uploader.py           # HTTPS upload + WAL + backoff
├── metrics_server.py     # Prometheus /metrics endpoint
├── attribution/          # 7-step job attribution engine
├── schedulers/           # Slurm, K8s, Run:ai adapters
├── integrations/         # MLflow, W&B, OpenTelemetry callbacks
├── efficiency/           # Energy analysis, carbon scheduling, roofline
├── swarm/                # Fleet-wide optimization (leader election, policies)
├── finetune/             # GreenTune — energy-efficient fine-tuning
│   ├── greentune.py      # QLoRA training with energy tracking
│   ├── greentune_swarm.py# Offline hyperparameter optimizer
│   ├── energy_callback.py# HuggingFace TrainerCallback for energy metrics
│   ├── rocm_power.py     # AMD GPU power monitoring (amdsmi / rocm-smi)
│   └── dataset_builder.py# Synthetic dataset generation via Claude
└── tests/
```

## Development

```bash
git clone https://github.com/AgentMulder404/aluminatiai.git
cd aluminatiai
pip install -e ".[all]"
python -m pytest tests/ -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
