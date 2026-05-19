# AluminatAI Agent

The open-source GPU energy monitoring agent for [AluminatAI](https://aluminatiai.com).

Runs on any GPU machine — NVIDIA, AMD, Intel Gaudi, Intel Arc, Apple Silicon, or CPU-only — samples power every 5 seconds, attributes energy to individual jobs, and streams dollar costs to your dashboard.

## Supported Hardware

| Backend | GPUs | Primary SDK | CLI Fallback |
|---------|------|-------------|--------------|
| **NVIDIA** | A100, H100, H200, L40S, RTX 4090, T4, V100, … | `nvidia-ml-py` (NVML) | — |
| **AMD** | MI300X, MI300A, MI325X, MI250X, MI210, MI100, … | `amdsmi` | `rocm-smi` |
| **Intel Gaudi** | Gaudi, Gaudi2, Gaudi3 | `pyhlml` (SynapseAI) | `hl-smi` |
| **Intel Arc** | A770, A750, A580, B580, Flex 170/140, Max 1550/1100 | `xpu-smi` (oneAPI) | hwmon sysfs + `intel_gpu_top` |
| **Apple Silicon** | M1–M5, Pro/Max/Ultra | `powermetrics` (sudo) | `ioreg` |
| **CPU-only** | Any x86 (Intel/AMD) | RAPL sysfs | — |

The agent auto-detects your hardware at startup. No configuration needed — just install and run.

**Detection cascade:** NVIDIA → AMD → Gaudi → Intel Arc → Apple Silicon → RAPL (CPU-only)

## Product Tiers

The agent supports three operating modes, each building on the previous:

| Tier | Mode | What it does | Key features |
|------|------|-------------|--------------|
| **Monitor** | Default | Read-only metrics + attribution | Power tracking, cost dashboards, Prometheus, carbon tracking |
| **Advisor** | Opt-in | Recommendations + approval workflows | "GPU 3 is 40% idle — cap to 200W?" with one-click apply/rollback |
| **Swarm** | Opt-in | Autonomous multi-agent optimization | Fleet-wide power capping, thermal balancing, carbon-aware scheduling, leader election |

All tiers share the same agent binary. Enable higher tiers via environment variables:

```bash
# Monitor (default — no extra config)
aluminatiai

# Advisor — agent uploads recommendations, polls for approved commands
AUTO_TUNE_ENABLED=1 COMMAND_POLL_ENABLED=1 aluminatiai

# Swarm — one agent becomes fleet leader, optimizes across all nodes
SWARM_ENABLED=1 COMMAND_POLL_ENABLED=1 AUTO_TUNE_ENABLED=1 aluminatiai
```

## Install

```bash
pip install aluminatiai
```

Optional extras:

```bash
pip install 'aluminatiai[prometheus]'     # Prometheus /metrics endpoint
pip install 'aluminatiai[secure]'         # Encrypted WAL (AES-128 Fernet)
pip install 'aluminatiai[observability]'  # YAML config + OTEL exporter
pip install 'aluminatiai[benchmark]'      # Benchmark CLI dependencies
pip install 'aluminatiai[dcgm]'           # DCGM phase decomposition
pip install 'aluminatiai[all]'            # Everything
```

## Quick Start

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
aluminatiai
```

Get your API key at [aluminatiai.com/dashboard](https://aluminatiai.com/dashboard).

The agent will detect your GPU, start sampling, and upload metrics to your dashboard. That's it.

## CLI Commands

The `aluminatiai` command includes 8 subcommands:

### `aluminatiai` / `aluminatiai run`

Main daemon. Collects GPU metrics, attributes energy to jobs, uploads to the cloud.

```bash
aluminatiai                            # run forever (default)
aluminatiai --interval 2               # sample every 2 seconds
aluminatiai --duration 3600            # run for 1 hour then exit
aluminatiai --output /data/metrics.csv # also write a local CSV manifest
aluminatiai --dry-run                  # collect + attribute, skip uploads
aluminatiai --prometheus-only          # local Prometheus only, no cloud
```

### `aluminatiai benchmark`

Measure GPU power baseline and energy efficiency.

```bash
aluminatiai benchmark                              # 60s power baseline
aluminatiai benchmark --gpu 0 --duration 120       # specific GPU, 2 min
aluminatiai benchmark --upload                     # submit to Green AI Index
aluminatiai benchmark --model-tag llama-3-70b      # tag with model profile
```

Output includes average power (W), J/GPU-hr, kWh/GPU-hr, and roofline efficiency rating.

### `aluminatiai optimize`

Real-time efficiency analysis with actionable recommendations.

```bash
aluminatiai optimize                    # analyze all GPUs, 60s window
aluminatiai optimize --gpu 0 --json     # JSON output for automation
aluminatiai optimize --duration 300     # 5 minute analysis window
```

Detects compute precision, classifies memory-bound vs. compute-bound workloads, and ranks recommendations (P1/P2/P3) for power caps, precision switches, and GPU right-sizing.

### `aluminatiai ab`

A/B testing framework for comparing GPU energy efficiency between configurations.

```bash
aluminatiai ab --baseline "power_limit=300" --variant "power_limit=250" --duration 120
```

Produces statistical comparison with confidence intervals, energy savings, and throughput impact (AEM — Adjusted Energy Metric).

### `aluminatiai carbon-schedule`

Recommends the optimal time to start a job based on grid carbon intensity forecasts.

```bash
aluminatiai carbon-schedule --duration 4h --zone US-CAL-CISO
```

Uses the [Electricity Maps](https://electricitymaps.com) API to find the lowest-carbon window in the next 24 hours.

### `aluminatiai report`

Generate chargeback reports for cost attribution.

```bash
aluminatiai report --format csv --output chargeback.csv
aluminatiai report --format html --from 2026-05-01 --to 2026-05-07
aluminatiai report --format json --with-carbon
```

### `aluminatiai query`

Query the local SQLite time-series database.

```bash
aluminatiai query --metric power --gpu 0 --from 2026-05-08 --to 2026-05-09
```

### `aluminatiai replay`

Export and optionally clear the offline WAL.

```bash
aluminatiai replay --output metrics.csv
aluminatiai replay --output metrics.csv --clear
```

## Configuration

Settings are read in priority order (highest wins):

1. **Environment variables** (`ALUMINATAI_*`, `SAMPLE_INTERVAL`, etc.)
2. **Config file** — JSON or YAML (via `--config` flag or `ALUMINATAI_CONFIG` env var)
3. **Built-in defaults**

### Config file

```bash
aluminatiai --config /etc/aluminatai.json
# or
ALUMINATAI_CONFIG=/etc/aluminatai.yaml aluminatiai
```

Default search order when `ALUMINATAI_CONFIG` is unset:
- `./aluminatai.json`
- `./aluminatai.yaml`
- `~/.config/aluminatai/config.json`

Example `aluminatai.json`:
```json
{
  "api_key": "alum_your_key_here",
  "sample_interval": 2.0,
  "upload_interval": 30,
  "metrics_port": 9100,
  "log_format": "json"
}
```

YAML config requires `pip install 'aluminatiai[observability]'`.

### Configuration Reference

#### API & Upload

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `ALUMINATAI_API_KEY` | `api_key` | *(required)* | Your API key |
| `ALUMINATAI_API_ENDPOINT` | `api_endpoint` | `https://…/v1/metrics/ingest` | Ingest endpoint URL |
| `UPLOAD_INTERVAL` | `upload_interval` | `60` | Seconds between metric flushes |
| `UPLOAD_BATCH_SIZE` | `upload_batch_size` | `100` | Metrics per HTTP request |
| `UPLOAD_MAX_RETRIES` | `upload_max_retries` | `5` | Max retry attempts (exponential backoff) |
| `UPLOAD_MAX_RETRY_DELAY` | `upload_max_retry_delay` | `60` | Backoff cap in seconds |
| `UPLOAD_TIMEOUT` | — | `30` | HTTP request timeout in seconds |

#### Sampling

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `SAMPLE_INTERVAL` | `sample_interval` | `5.0` | Seconds between GPU samples |
| `NVML_TIMEOUT` | — | `2.0` | Per-GPU collection timeout |

#### Write-Ahead Log (WAL)

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `DATA_DIR` | `data_dir` | `./data` | Base data directory |
| `WAL_MAX_MB` | `wal_max_mb` | `512` | WAL size cap |
| `WAL_MAX_AGE_HOURS` | `wal_max_age_hours` | `24` | WAL retention period |

#### Hardware Backends

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `CPU_ONLY_MODE` | — | `false` | Skip GPU detection, use RAPL only |
| `RAPL_ENABLED` | — | `true` | Enable CPU RAPL energy counters |
| `RAPL_CPU_MODEL_OVERRIDE` | — | *(auto)* | Override CPU model name |
| `GAUDI_ENABLED` | — | `true` | Enable Intel Gaudi collector |
| `HL_SMI_PATH` | — | `hl-smi` | Custom path to hl-smi binary |
| `INTEL_ARC_ENABLED` | — | `true` | Enable Intel Arc collector |
| `XPU_SMI_PATH` | — | `xpu-smi` | Custom path to xpu-smi binary |
| `APPLE_POWERMETRICS_ENABLED` | — | `true` | Enable powermetrics (requires sudo NOPASSWD) |
| `APPLE_POWERMETRICS_INTERVAL_MS` | — | `1000` | powermetrics sampling interval (ms) |
| `APPLE_CHIP_TDP_OVERRIDE` | — | *(auto)* | Override Apple GPU TDP estimate (watts) |
| `DCGM_ENABLED` | `dcgm_enabled` | `true` | Enable DCGM phase decomposition (NVIDIA) |

#### Prometheus Metrics Server

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `METRICS_PORT` | `metrics_port` | `9100` | Scrape port (`0` = disabled) |
| `METRICS_BIND_HOST` | `metrics_bind_host` | *(all)* | Bind address |
| `METRICS_BASIC_AUTH` | `metrics_basic_auth` | *(none)* | `user:pass` for HTTP Basic Auth |

#### Attribution

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `ALUMINATAI_ATTRIBUTION_CONFIG` | `attribution_config` | *(auto-search)* | Path to `attribution_rules.json` |
| `ALUMINATAI_TRUSTED_UIDS` | `trusted_uids` | *(all)* | Comma-separated UIDs for spoofing protection |
| `PID_SMOOTH_WINDOW` | `pid_smooth_window` | `30` | PID stability window (seconds) |
| `PID_STABLE_THRESHOLD` | `pid_stable_threshold` | `0.60` | Fraction of window a PID must appear in |

#### Auto-Tuning & Power Budget

| Env var | Default | Description |
|---------|---------|-------------|
| `AUTO_TUNE_ENABLED` | `false` | Enable periodic roofline analysis |
| `AUTO_TUNE_INTERVAL` | `300` | Analysis interval (seconds) |
| `AUTO_TUNE_MIN_SAVINGS_PCT` | `10` | Min savings to recommend a power cap |
| `POWER_BUDGET_ENABLED` | `false` | Enable per-GPU power cap enforcement |
| `POWER_BUDGET_WATTS` | `0` | Per-GPU power cap (watts, 0 = disabled) |

#### Advisor Tier (Recommendations + Commands)

| Env var | Default | Description |
|---------|---------|-------------|
| `COMMAND_POLL_ENABLED` | `false` | Enable polling for approved commands |
| `COMMAND_POLL_INTERVAL` | `60` | Base poll interval (seconds); adapts up to 5 min when idle |

When `AUTO_TUNE_ENABLED=1` and `COMMAND_POLL_ENABLED=1`, the agent:
1. Runs roofline analysis every `AUTO_TUNE_INTERVAL` seconds
2. Uploads optimization recommendations to the cloud dashboard
3. Polls for user-approved commands (power caps, rollbacks)
4. Executes approved commands with safety validation (100–1200W range)

The dashboard shows recommendations at `/dashboard/advisor` with one-click approve, dismiss, and rollback.

#### Swarm Tier (Fleet-Wide Optimization)

| Env var | Default | Description |
|---------|---------|-------------|
| `SWARM_ENABLED` | `false` | Enable swarm leader candidacy |
| `SWARM_EVAL_INTERVAL` | `300` | Seconds between fleet policy evaluations |
| `SWARM_MAX_RECS` | `20` | Max recommendations per eval cycle |

When enabled, the agent participates in **leader election** — one agent per (user, cluster) becomes the swarm leader. The leader:

1. Acquires a 10-minute lease via `POST /api/agent/swarm/lease`
2. Fetches fleet-wide GPU state via `GET /api/agent/fleet-state`
3. Evaluates 4 built-in policies across all nodes
4. Uploads cross-node recommendations with blast radius limiting
5. Other agents receive and execute approved commands

**Built-in policies:**

| Policy | What it detects | Action |
|--------|----------------|--------|
| `idle_gpu_power_cap` | GPUs with <10% utilization | Cap to 40% of TDP |
| `thermal_balancing` | Single GPU overheating while others are cool | Reduce power 15% on hot GPU |
| `carbon_aware_fleet_cap` | Grid carbon >400 gCO2e/kWh | Cap non-critical GPUs to 65% |
| `fleet_gpu_rightsizing` | GPUs consistently underutilized | Flag for consolidation |

**Safety guardrails:**

- **Blast radius**: max 25% of fleet affected per eval (configurable)
- **Canary ramp-up**: new policies start at 10% of fleet, double each successful eval
- **Leader election**: only one leader per cluster — prevents duplicate commands
- **Adaptive polling**: command polling backs off 60s → 300s when idle, resets on command
- **Priority sorting**: P1 thermal/safety recs get through before P2/P3

#### Fleet Aggregation

| Env var | Default | Description |
|---------|---------|-------------|
| `FLEET_AGGREGATOR_ENABLED` | `false` | Enable fleet aggregation endpoint |
| `FLEET_AGGREGATOR_PORT` | `9101` | Aggregator HTTP port |
| `FLEET_AGGREGATOR_PEERS` | *(none)* | Comma-separated peer URLs |

#### Multi-Agent High-Frequency Sampling

| Env var | Default | Description |
|---------|---------|-------------|
| `MULTI_AGENT_ENABLED` | `false` | Enable high-frequency ring buffer sampling |
| `FAST_SAMPLE_INTERVAL` | `0.2` | Fast sample interval (seconds) |
| `FAST_SAMPLE_BUFFER_SIZE` | `100` | Ring buffer size per GPU |

#### Idle Calibration

| Env var | Default | Description |
|---------|---------|-------------|
| `IDLE_BASELINE_WINDOW` | `30` | Seconds to calibrate idle power at startup |
| `WARMUP_DISCARD_SECONDS` | `45` | Discard samples in this startup window |

#### Cluster Identity

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `ALUMINATAI_CLUSTER_TAG` | `cluster_tag` | *(none)* | Cluster identifier (e.g., `aws-us-west-2`) |
| `ALUMINATAI_LOCATION_HINT` | `location_hint` | *(none)* | Free-text location (shown in UI) |
| `ALUMINATAI_GRID_ZONE` | `grid_zone` | *(none)* | Electricity Maps zone (e.g., `US-CAL-CISO`) |
| `HEARTBEAT_INTERVAL` | `heartbeat_interval` | `300` | Heartbeat interval (seconds) |

#### TLS & Proxy

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `HTTPS_PROXY` | `https_proxy` | *(none)* | HTTPS proxy URL |
| `ALUMINATAI_CA_BUNDLE` | `ca_bundle` | *(none)* | Path to custom CA PEM |
| `ALUMINATAI_CLIENT_CERT` | `client_cert` | *(none)* | mTLS client cert path |
| `ALUMINATAI_CLIENT_KEY` | `client_key` | *(none)* | mTLS client key path |

#### Run Modes

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `OFFLINE_MODE` | `offline_mode` | `false` | WAL only, no HTTP uploads |
| `DRY_RUN` | `dry_run` | `false` | Collect + attribute, skip uploads and WAL |
| `PROMETHEUS_ONLY` | `prometheus_only` | `false` | Local Prometheus only |

#### Logging

| Env var | Config key | Default | Description |
|---------|-----------|---------|-------------|
| `LOG_LEVEL` | `log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FORMAT` | `log_format` | `text` | `text` or `json` (newline-delimited JSON for ELK/Loki) |
| `LOG_DIR` | — | `./logs` | Log file directory |

## Deployment

### One-line installer (Linux + systemd)

```bash
curl -sSL https://get.aluminatiai.com | bash
```

| Flag | Effect |
|------|--------|
| `--local` | Install from local source (dev / air-gapped) |
| `--no-service` | Package only — skip systemd setup |
| `--unattended` / `-y` | Non-interactive; requires `ALUMINATAI_API_KEY` env var |

```bash
# CI / non-interactive
ALUMINATAI_API_KEY=alum_xxx curl -sSL https://get.aluminatiai.com | bash -s -- --unattended

# Check service health
sudo systemctl status aluminatai-agent
sudo journalctl -u aluminatai-agent -f
```

### Manual systemd setup

```bash
pip install aluminatiai

# Create system user and directories
sudo useradd --system --no-create-home --shell /usr/sbin/nologin aluminatai
sudo install -d -m 0700 -o aluminatai -g aluminatai /var/lib/aluminatai
sudo install -d -m 0755 -o aluminatai -g aluminatai /var/log/aluminatai
sudo install -d -m 0750 /etc/aluminatai

# Write the env file (mode 600 — contains your API key)
sudo tee /etc/aluminatai/agent.env > /dev/null <<'EOF'
ALUMINATAI_API_KEY=alum_your_key_here
SAMPLE_INTERVAL=5.0
UPLOAD_INTERVAL=60
METRICS_PORT=9100
LOG_LEVEL=INFO
EOF
sudo chmod 600 /etc/aluminatai/agent.env

# Install the unit file
sudo cp deploy/aluminatai-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aluminatai-agent
```

The service unit includes systemd security hardening: `NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp`, `MemoryMax=256M`, and system call filtering.

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

The DaemonSet includes RBAC for pod metadata queries (used by the K8s attribution adapter).

### Slurm

Add to your job prolog/epilog scripts:

```bash
# /etc/slurm/prolog.d/aluminatiai.sh
source /etc/aluminatai/agent.env
aluminatiai &

# /etc/slurm/epilog.d/aluminatiai-stop.sh
pkill -f aluminatiai
```

## Attribution

The agent attributes GPU power to jobs using a multi-step resolution pipeline. The first step that matches wins.

### Resolution pipeline

| Priority | Method | Confidence | How it works |
|----------|--------|------------|--------------|
| 1 | `ALUMINATAI_TEAM` env var | 1.00 | Explicit user tag — most trustworthy |
| 1.5 | `/api/v1/tag` REST registration | 0.95 | Background polling every 30s |
| 2 | Scheduler env vars | 0.90 | `SLURM_JOB_ID`, `RUNAI_JOB_NAME`, K8s pod UID |
| 3 | Scheduler poll | 0.75 | `gpu_to_job()` fallback query |
| 4 | Custom rules file | 0.60 | JSON regex patterns (see below) |
| 5 | Cmdline heuristics | 0.40 | Built-in patterns (jupyter, vllm, torchserve, ollama, …) |
| 6 | Memory split | 0.20 | Unresolved power split by GPU memory usage |
| 7 | Idle attribution | 0.30 | `ALUMINATAI_IDLE_TEAM` env var fallback |

### Tagging workloads

```bash
# Simplest: set env vars before launching your job
ALUMINATAI_TEAM=nlp-team \
ALUMINATAI_MODEL=llama3-finetune \
python train.py
```

### Custom attribution rules

Create an `attribution_rules.json` file to map command-line patterns to teams:

```json
{
  "rules": [
    { "pattern": "python.*gpt4_train", "team": "llm-infra", "model": "gpt4",     "priority": 10 },
    { "pattern": "vllm.*llama",        "team": "inference",  "model": "llama",    "priority": 5  },
    { "pattern": "jupyter",            "team": "research",   "model": "notebook", "priority": 1  }
  ]
}
```

Search order for the rules file:
1. `ALUMINATAI_ATTRIBUTION_CONFIG` env var (explicit path)
2. `./attribution_rules.json`
3. `~/.config/aluminatai/attribution_rules.json`

### Supported schedulers

| Scheduler | Detection | Job metadata source |
|-----------|-----------|-------------------|
| **Slurm** | `SLURM_JOB_ID` env var | `scontrol show job` |
| **Kubernetes** | Pod cgroup UID | K8s API (requires RBAC) |
| **Run:ai** | `RUNAI_JOB_NAME` env var | Run:ai API |

### Spoofing protection

On multi-user hosts, restrict which UIDs can self-tag:

```bash
export ALUMINATAI_TRUSTED_UIDS=0,1000   # only root and UID 1000 may use ALUMINATAI_TEAM
```

When unset, all UIDs are trusted (backward compatible).

## Prometheus Metrics

The agent exposes a `/metrics` endpoint (default port 9100) with these gauges and counters:

### GPU metrics

| Metric | Type | Description |
|--------|------|-------------|
| `aluminatai_gpu_power_watts` | Gauge | Current power draw per GPU |
| `aluminatai_gpu_energy_joules_total` | Counter | Cumulative energy per GPU |
| `aluminatai_gpu_utilization_pct` | Gauge | GPU compute utilization |
| `aluminatai_gpu_temperature_c` | Gauge | GPU temperature |

### Phase decomposition (DCGM)

| Metric | Type | Description |
|--------|------|-------------|
| `aluminatai_gpu_tensor_power_watts` | Gauge | Tensor core power |
| `aluminatai_gpu_fp16_power_watts` | Gauge | FP16 path power |
| `aluminatai_gpu_memory_power_watts` | Gauge | Memory subsystem power |
| `aluminatai_gpu_idle_power_watts` | Gauge | Baseline idle power |

### Upload health

| Metric | Type | Description |
|--------|------|-------------|
| `aluminatai_upload_success_total` | Counter | Successful uploads |
| `aluminatai_upload_failure_total` | Counter | Failed uploads |
| `aluminatai_buffer_size` | Gauge | In-memory buffer entries pending |
| `aluminatai_wal_size_bytes` | Gauge | WAL file size |

### Attribution

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `aluminatai_attribution_confidence` | Gauge | gpu_index, job_id, method | Confidence score (0–1) |
| `aluminatai_attribution_uncertainty_pct` | Gauge | gpu_index, job_id | ± % uncertainty |
| `aluminatai_attribution_unresolved_total` | Counter | — | Unattributed power |

### Agent health

| Metric | Type | Description |
|--------|------|-------------|
| `aluminatai_agent_uptime_seconds` | Gauge | Agent uptime |
| `aluminatai_agent_info` | Gauge | Version, hostname, mode metadata |

### Prometheus scrape config

```yaml
scrape_configs:
  - job_name: aluminatiai
    static_configs:
      - targets: ['gpu-host:9100']
```

## ML Framework Integrations

### MLflow

```python
from aluminatiai.integrations.mlflow_callback import AluminatiMLflowCallback

with mlflow.start_run():
    trainer.add_callback(AluminatiMLflowCallback())
    trainer.train()
    # energy_kwh, cost_usd, co2_kg logged automatically at run end
```

### Weights & Biases

```python
from aluminatiai.integrations.wandb_callback import AluminatiWandbCallback

wandb.init(project="my-project")
trainer.add_callback(AluminatiWandbCallback())
trainer.train()
# energy metrics logged to wandb.run.summary
```

### OpenTelemetry

```python
from aluminatiai.integrations.otel_exporter import AluminatiOtelExporter

exporter = AluminatiOtelExporter()
# GPU metrics exported as OTEL span attributes to Jaeger/Datadog/etc.
```

## Hardware-Specific Notes

### NVIDIA

Standard NVML-based collection. Works out of the box with any NVIDIA datacenter or consumer GPU with driver 450.80.02+.

```bash
pip install nvidia-ml-py   # or nvidia-ml-py3
```

### AMD

Requires either `amdsmi` (Python bindings from ROCm 6+) or `rocm-smi` CLI in PATH.

```bash
pip install amdsmi   # preferred
# or ensure rocm-smi is in PATH
```

### Intel Gaudi

Requires either `pyhlml` (ships with SynapseAI driver) or `hl-smi` CLI in PATH.

```bash
# pyhlml is installed with the Habana SynapseAI SDK
# or set HL_SMI_PATH if hl-smi is not in PATH
export HL_SMI_PATH=/opt/habanalabs/bin/hl-smi
```

### Intel Arc

Requires `xpu-smi` (ships with Intel oneAPI Base Toolkit) or the xe/i915 kernel driver with hwmon sysfs.

```bash
# xpu-smi is installed with the oneAPI toolkit
# or set XPU_SMI_PATH if not in PATH
export XPU_SMI_PATH=/opt/intel/oneapi/xpu-smi/bin/xpu-smi
```

### Apple Silicon

Uses `powermetrics` for accurate power reading (requires passwordless sudo) or falls back to `ioreg` (utilization only, estimates power from TDP).

For powermetrics access, add to `/etc/sudoers`:
```
your_username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

Without sudo access, the agent uses ioreg (less accurate but no privileges needed).

### CPU-Only (RAPL)

For machines with no discrete GPU. Monitors CPU package power via Intel/AMD RAPL sysfs counters.

```bash
export CPU_ONLY_MODE=1   # skip GPU detection entirely
aluminatiai
```

Requires read access to `/sys/class/powercap/intel-rapl:*` or `/sys/class/powercap/amd_rapl:*`.

## Security

### Environment variable privacy

The agent reads `/proc/<pid>/environ` to attribute jobs. Only a small allowlist of env var keys is retained:

```
SLURM_JOB_ID, RUNAI_JOB_NAME, KUBERNETES_SERVICE_HOST,
ALUMINATAI_TEAM, ALUMINATAI_MODEL, ALUMINATAI_* (any prefix)
```

All other env vars (credentials, tokens, database URLs) are dropped immediately.

### WAL encryption

The write-ahead log is encrypted automatically when `ALUMINATAI_API_KEY` is set and the `cryptography` package is installed:

```bash
pip install 'aluminatiai[secure]'
```

Encryption key = SHA-256(API_KEY), using AES-128 Fernet. Without the package, the agent falls back to plaintext WAL with a one-time warning.

### Prometheus endpoint hardening

```bash
# Bind to localhost only
export METRICS_BIND_HOST=127.0.0.1

# Require HTTP Basic Auth
export METRICS_BASIC_AUTH=scrape_user:strong_password
```

Use a TLS-terminating reverse proxy (nginx, Caddy) in front of the metrics endpoint in production.

### Offline / air-gapped clusters

```bash
# No outbound HTTP — all metrics go to WAL
OFFLINE_MODE=1 aluminatiai

# Later, on a machine with network access
aluminatiai replay --output metrics.csv --clear
```

### Directory permissions

Data, WAL, and log directories are created with mode `0o700` (owner-only access).

## Self-Hosting

Point the agent at your own ingest endpoint:

```bash
ALUMINATAI_API_ENDPOINT=https://your-api.internal/v1/metrics/ingest \
ALUMINATAI_API_KEY=your_key \
aluminatiai
```

## Package Structure

```
agent/
├── agent.py              # Daemon entry point, signal handling, main loop
├── cli.py                # CLI router (subcommand dispatch)
├── config.py             # Config file + env var loader with validation
├── collector.py          # NVIDIA GPU collector (NVML)
├── amd_collector.py      # AMD GPU collector (amdsmi / rocm-smi)
├── gaudi_collector.py    # Intel Gaudi collector (pyhlml / hl-smi)
├── intel_arc_collector.py# Intel Arc collector (xpu-smi / hwmon)
├── apple_collector.py    # Apple Silicon collector (powermetrics / ioreg)
├── rapl_collector.py     # CPU-only collector (RAPL sysfs)
├── uploader.py           # HTTPS upload + WAL + exponential backoff
├── metrics_server.py     # Prometheus /metrics endpoint
├── fleet_aggregator.py   # Multi-node fleet metric rollups
├── benchmark.py          # GPU power baseline CLI
├── attribution/          # Job attribution engine
│   ├── engine.py         # 7-step resolution pipeline
│   ├── pid_resolver.py   # PID → team resolver
│   ├── pid_smoother.py   # Transient PID filtering (30s window)
│   ├── process_probe.py  # /proc reader (environ, cmdline, cgroup)
│   └── rules.py          # Custom JSON attribution rules
├── schedulers/           # Scheduler adapters
│   ├── slurm.py          # Slurm (scontrol)
│   ├── kubernetes.py     # Kubernetes (pod UID → K8s API)
│   └── runai.py          # Run:ai
├── integrations/         # ML framework callbacks
│   ├── mlflow_callback.py
│   ├── wandb_callback.py
│   └── otel_exporter.py
├── recommendation_reporter.py  # Uploads optimization recs to cloud (Advisor)
├── command_receiver.py         # Polls + executes approved commands (Advisor)
├── swarm/                      # Fleet-wide optimization (Swarm)
│   ├── policy_engine.py        # Leader election, blast radius, ramp-up
│   ├── fleet_state.py          # Fleet snapshot data model
│   └── policies.py             # 4 built-in fleet policies
├── efficiency/           # Energy analysis
│   ├── gpu_specs.py      # 45 GPU architecture specs + roofline model
│   ├── rapl.py           # Multi-socket RAPL reader
│   ├── auto_tuner.py     # Periodic power cap recommendations
│   ├── optimize.py       # Real-time efficiency analyzer
│   ├── carbon.py         # Electricity Maps carbon intensity
│   ├── carbon_scheduler.py # Carbon-aware job scheduling
│   ├── curve_builder.py  # Fleet efficiency curves
│   ├── hardware_match.py # Roofline hardware match scorer
│   └── power_control.py  # NVML power limit enforcement
├── storage/
│   └── tsdb.py           # Local SQLite time-series store
├── deploy/               # Production deployment files
│   ├── aluminatai-agent.service  # systemd unit (hardened)
│   ├── k8s/              # K8s DaemonSet + RBAC
│   └── install.sh        # One-line installer
└── tests/                # 16 test files, 300+ tests
```

## Development

```bash
git clone https://github.com/AgentMulder404/AluminatAI.git
cd AluminatAI/agent
pip install -e ".[all]"
python -m pytest tests/ --ignore=tests/powercap_ab_test.py -v
```

## License

Apache 2.0 — see [LICENSE](../LICENSE).
