# Getting Started

## Prerequisites

- **Any GPU**: NVIDIA (driver 450.80.02+), AMD (ROCm 6+), Intel Gaudi (SynapseAI), Intel Arc (oneAPI), or Apple Silicon (macOS)
- **Or CPU-only**: Linux with Intel/AMD RAPL sysfs access
- Python 3.8+
- AluminatAI API key — get one at [aluminatiai.com/dashboard](https://aluminatiai.com/dashboard)

## Install

```bash
pip install aluminatiai
```

## Start monitoring

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
aluminatiai
```

The agent auto-detects your hardware. You'll see output like:

```
[aluminatai] Detected 8x NVIDIA H100-SXM5-80GB — backend: NVIDIA (NVML)
[aluminatai] Sampling every 5.0s, uploading every 60s
```

Or on other hardware:
```
[aluminatai] Detected 2x AMD Instinct MI300X — backend: AMD (amdsmi)
[aluminatai] Detected 1x Intel Gaudi2 — backend: Intel Gaudi (hl-smi)
[aluminatai] Detected 1x Intel Arc A770 — backend: Intel Arc (xpu-smi)
[aluminatai] Detected 1x Apple M5 Max GPU — backend: Apple Silicon (ioreg)
[aluminatai] Detected 2x CPU sockets — backend: CPU (RAPL)
```

## Tag workloads for per-job attribution

Set env vars before launching your training job:

```bash
ALUMINATAI_TEAM=nlp-team \
ALUMINATAI_MODEL=llama3-finetune \
python train.py
```

The agent automatically picks up these tags from the process environment and attributes GPU energy to your team.

For Slurm, Kubernetes, or Run:ai — attribution is automatic via scheduler integration.

## MLflow / W&B integration

```python
# MLflow
from aluminatiai.integrations.mlflow_callback import AluminatiMLflowCallback
with mlflow.start_run():
    trainer.add_callback(AluminatiMLflowCallback())
    trainer.train()

# Weights & Biases
from aluminatiai.integrations.wandb_callback import AluminatiWandbCallback
wandb.init(project="my-project")
trainer.add_callback(AluminatiWandbCallback())
trainer.train()
```

Energy metrics (`energy_kwh`, `cost_usd`, `co2_kg`) are logged automatically at run end.

## Benchmark your GPU

```bash
aluminatiai benchmark                         # 60s power baseline
aluminatiai benchmark --duration 120 --upload  # 2 min, submit to Green AI Index
```

## Analyze efficiency

```bash
aluminatiai optimize              # real-time roofline analysis
aluminatiai optimize --json       # machine-readable output
```

## Production deployment

### systemd (recommended for Linux)

```bash
curl -sSL https://get.aluminatiai.com | bash
```

Or manually:

```bash
sudo cp deploy/aluminatai-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aluminatai-agent
```

### Kubernetes DaemonSet

```bash
kubectl apply -f deploy/k8s/daemonset.yaml
```

### Docker (NVIDIA)

```bash
docker run --rm --runtime=nvidia --pid=host \
  -e ALUMINATAI_API_KEY=alum_your_key_here \
  ghcr.io/agentmulder404/aluminatai-agent:latest
```

## Prometheus

The agent exposes GPU metrics on port 9100 by default:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: aluminatiai
    static_configs:
      - targets: ['gpu-host:9100']
```

Disable with `METRICS_PORT=0`.

## Hardware-specific setup

### AMD

```bash
pip install amdsmi    # or ensure rocm-smi is in PATH
```

### Intel Gaudi

```bash
# pyhlml ships with SynapseAI — no extra install needed
# or set custom hl-smi path:
export HL_SMI_PATH=/opt/habanalabs/bin/hl-smi
```

### Intel Arc

```bash
# xpu-smi ships with oneAPI Base Toolkit
# or set custom path:
export XPU_SMI_PATH=/opt/intel/oneapi/xpu-smi/bin/xpu-smi
```

### Apple Silicon

For accurate power readings, enable passwordless powermetrics:

```
# /etc/sudoers (via visudo)
your_username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

Without this, the agent falls back to ioreg (estimates power from utilization).

### CPU-only

```bash
export CPU_ONLY_MODE=1
aluminatiai
```

## Enable the Advisor (recommendations + one-click apply)

The Advisor tier surfaces GPU optimization recommendations in your dashboard and lets you apply them with one click.

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
export AUTO_TUNE_ENABLED=1          # enable roofline analysis
export COMMAND_POLL_ENABLED=1       # enable command execution
aluminatiai
```

The agent analyzes your GPU workloads and uploads recommendations like:
- "GPU 3 is 40% idle — cap power to 200W (save ~25%)"
- "GPU 0 running FP32 — switch to BF16 (save ~40% energy)"
- "Defer job to 2am for 15% less carbon"

View and act on recommendations at `/dashboard/advisor`:
- **Apply** — sends a power cap command to the agent
- **Dismiss** — hides the recommendation
- **Rollback** — reverts an applied change

The dashboard auto-refreshes every 15 seconds.

## Enable the Swarm (fleet-wide autonomous optimization)

The Swarm tier coordinates optimization across all your GPU nodes. One agent becomes the fleet leader and evaluates policies across every machine.

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
export AUTO_TUNE_ENABLED=1
export COMMAND_POLL_ENABLED=1
export SWARM_ENABLED=1              # participate in leader election
export ALUMINATAI_CLUSTER_TAG=prod  # group by cluster (optional)
aluminatiai
```

**How it works:**
1. All agents with `SWARM_ENABLED=1` are leader candidates
2. One agent per cluster wins a 10-minute lease (auto-renews)
3. The leader fetches fleet-wide GPU metrics from the cloud
4. Four built-in policies evaluate across all nodes:
   - **Idle GPU power cap** — caps idle GPUs fleet-wide
   - **Thermal balancing** — reduces power on overheating GPUs
   - **Carbon-aware fleet cap** — throttles during high-carbon periods
   - **Fleet GPU right-sizing** — flags underutilized GPUs
5. Recommendations appear in the dashboard with one-click approve

**Safety at scale (1000+ nodes):**
- Blast radius limited to 25% of fleet per evaluation
- New policies ramp from 10% of fleet, doubling each cycle
- Only one leader per cluster (prevents duplicate commands)
- Command polling backs off when idle (60s → 300s)

If the leader goes down, another agent automatically takes over within 10 minutes.

## Common issues

### "No collector available"

The agent couldn't find any supported GPU or CPU energy interface. Check:

```bash
nvidia-smi          # NVIDIA
rocm-smi            # AMD
hl-smi              # Intel Gaudi
xpu-smi discovery   # Intel Arc
ls /sys/class/powercap/intel-rapl:0/energy_uj   # CPU RAPL
```

### "Failed to initialize NVML"

```bash
sudo usermod -a -G video $USER   # add to video group, then re-login
```

### "RAPL unavailable: permission denied"

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
```

### Agent exits immediately

```bash
LOG_LEVEL=DEBUG aluminatiai   # see detailed startup diagnostics
```

## Next steps

- [Product tiers (Monitor → Advisor → Swarm)](README.md#product-tiers)
- [Advisor configuration](README.md#advisor-tier-recommendations--commands)
- [Swarm configuration](README.md#swarm-tier-fleet-wide-optimization)
- [Full configuration reference](README.md#configuration-reference)
- [Attribution pipeline details](README.md#attribution)
- [Prometheus metrics list](README.md#prometheus-metrics)
- [Security & hardening](README.md#security)
