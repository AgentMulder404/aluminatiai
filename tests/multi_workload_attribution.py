#!/usr/bin/env python3
"""
Multi-Workload GPU Power Attribution Test
==========================================
Launches 3 concurrent GPU jobs (inference / training / memory-stress),
each tagged with a different ALUMINATAI_TEAM env var.

Monitors NVML per-process memory every 10s for 5 minutes, attributes
total GPU power proportionally, then prints a summary table.

Usage:
    python multi_workload_attribution.py

Requires the AluminatAI agent running in another terminal:
    ALUMINATAI_API_KEY=<key> ALUMINATAI_API_ENDPOINT=https://www.aluminatiai.com/v1/metrics/ingest aluminatiai
"""
import os, sys, time, signal, subprocess, textwrap, pathlib
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────
GPU_INDEX       = 0
RUN_SECONDS     = 300
SAMPLE_INTERVAL = 10
STRESS_GB       = 10
PROMETHEUS_PORT = 9100

# ── Worker scripts ────────────────────────────────────────────────

WORKER_INFERENCE = textwrap.dedent("""\
    import os, sys
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from transformers import pipeline

    print('[inference] loading TinyLlama...', flush=True)
    pipe = pipeline(
        'text-generation',
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        torch_dtype=torch.float16,
        device=0,
    )
    prompts = [
        'Explain GPU energy attribution in one sentence:',
        'What is the idle power problem in ML clusters?',
        'Why does a GPU draw power even at 0% utilization?',
    ]
    i = 0
    print('[inference] running', flush=True)
    while True:
        pipe(prompts[i % len(prompts)], max_new_tokens=64, do_sample=False)
        i += 1
        if i % 5 == 0:
            print(f'[inference] step {i}', flush=True)
""")

WORKER_TRAINING = textwrap.dedent("""\
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset

    # Pre-allocate so NVML sees this process before model loads
    _warmup = torch.zeros(512, 512, device='cuda')
    del _warmup

    print('[training] loading bert-tiny...', flush=True)
    tok = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model = AutoModelForSequenceClassification.from_pretrained(
        'prajjwal1/bert-tiny', num_labels=2).cuda()

    ds = load_dataset('glue', 'sst2', split='train[:200]')
    enc = tok(ds['sentence'], padding=True, truncation=True,
              max_length=64, return_tensors='pt')
    input_ids      = enc['input_ids'].cuda()
    attention_mask = enc['attention_mask'].cuda()
    labels         = torch.tensor(ds['label']).cuda()

    # Hold 1GB so NVML reports a meaningful memory footprint for this process
    _hold = torch.zeros(256, 1024, 1024, device='cuda', dtype=torch.float32)

    opt = AdamW(model.parameters(), lr=2e-5)
    step = 0
    print('[training] running', flush=True)
    while True:
        opt.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        opt.step()
        step += 1
        if step % 20 == 0:
            print(f'[training] step {step} loss={out.loss.item():.4f}', flush=True)
""")

WORKER_STRESS = textwrap.dedent(f"""\
    import os, torch

    GB = {STRESS_GB}
    print(f'[stress] allocating {{GB}}GB on GPU...', flush=True)
    # hold a large allocation to inflate memory footprint
    _hold = torch.zeros(int(GB * 1024**3 / 4), dtype=torch.float32, device='cuda')
    a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
    b = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
    i = 0
    print('[stress] running', flush=True)
    while True:
        a = torch.nn.functional.relu(a @ b)
        b = torch.nn.functional.relu(b @ a)
        a = a / (a.norm() + 1e-8)
        b = b / (b.norm() + 1e-8)
        i += 1
        if i % 50 == 0:
            print(f'[stress] step {{i}}', flush=True)
""")

# ── Helpers ───────────────────────────────────────────────────────

def write_and_launch(name, code, team):
    path = pathlib.Path(f'/tmp/worker_{name}.py')
    path.write_text(code)
    p = subprocess.Popen(
        [sys.executable, str(path)],
        # Pass team in launch env so /proc/<pid>/environ contains it
        env={**os.environ, 'ALUMINATAI_TEAM': team},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"  launched {name:<12} team={team:<12} PID={p.pid}")
    return p

pid_team_cache = {}
_any_pid_to_team = {}   # populated at startup: all pid variants -> team

def _read_environ_team(pid):
    try:
        with open(f'/proc/{pid}/environ', 'rb') as f:
            env = f.read().decode('utf-8', errors='replace')
        for kv in env.split('\x00'):
            if kv.startswith('ALUMINATAI_TEAM='):
                return kv.split('=', 1)[1]
    except (PermissionError, FileNotFoundError):
        pass
    return None

def _nspids(pid):
    """Return all NSpid values from /proc/<pid>/status (handles PID namespaces)."""
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('NSpid:'):
                    return [int(x) for x in line.split()[1:]]
    except (FileNotFoundError, ValueError):
        pass
    return [pid]

def _ppid(pid):
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('PPid:'):
                    return int(line.split()[1])
    except (FileNotFoundError, ValueError):
        pass
    return None

def build_team_map():
    """Scan /proc for all processes with ALUMINATAI_TEAM, map all their PID variants."""
    global _any_pid_to_team
    result = {}
    try:
        for entry in os.scandir('/proc'):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            team = _read_environ_team(pid)
            if team:
                for ns_pid in _nspids(pid):
                    result[ns_pid] = team
                result[pid] = team
    except (PermissionError, OSError):
        pass
    _any_pid_to_team = result
    return result

def get_team(pid):
    """Look up team by checking PID map (handles namespace gaps) then tree walk."""
    if pid in pid_team_cache:
        return pid_team_cache[pid]
    # Direct hit from pre-built map
    if pid in _any_pid_to_team:
        pid_team_cache[pid] = _any_pid_to_team[pid]
        return _any_pid_to_team[pid]
    # Walk parent tree (for child CUDA processes)
    current, visited = pid, set()
    while current and current not in visited:
        visited.add(current)
        team = _read_environ_team(current)
        if team:
            pid_team_cache[pid] = team
            return team
        if current in _any_pid_to_team:
            pid_team_cache[pid] = _any_pid_to_team[current]
            return _any_pid_to_team[current]
        parent = _ppid(current)
        if not parent or parent == current:
            break
        current = parent
    pid_team_cache[pid] = f'pid:{pid}'
    return pid_team_cache[pid]

# ── Main ──────────────────────────────────────────────────────────

def main():
    try:
        import pynvml
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'nvidia-ml-py'])
        import pynvml

    print("\n══════════════════════════════════════════════════")
    print("  Multi-Workload GPU Power Attribution")
    print("══════════════════════════════════════════════════\n")

    # Launch workers
    print("Launching workers...")
    procs = {
        'inference': write_and_launch('inference', WORKER_INFERENCE, 'inference'),
        'training':  write_and_launch('training',  WORKER_TRAINING,  'training'),
        'stress':    write_and_launch('stress',     WORKER_STRESS,    'stress'),
    }

    print(f"\nWaiting 90s for models to load (TinyLlama + BERT-tiny)...")
    import threading, io

    def drain(name, proc):
        for line in io.TextIOWrapper(proc.stdout, errors='replace'):
            print(f"  [{name}] {line.rstrip()}", flush=True)

    for name, p in procs.items():
        threading.Thread(target=drain, args=(name, p), daemon=True).start()

    time.sleep(90)

    # Build PID->team map (handles PID namespace gaps between /proc and NVML)
    print("Building PID→team map...")
    found = build_team_map()
    print(f"  found {len(found)} PID entries: {set(found.values())}")

    # Init NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)

    samples = defaultdict(list)  # team -> [(attributed_w, mem_mb, fraction)]
    total_power_samples = []

    print(f"\nMonitoring {RUN_SECONDS}s (samples every {SAMPLE_INTERVAL}s)...\n")
    print(f"  {'Job':<14} {'Mem MB':>8} {'Fraction':>9} {'Power W':>9}")
    print("  " + "─" * 44)

    start = time.time()
    next_print = start + 30

    while time.time() - start < RUN_SECONDS:
        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        total_power_samples.append(power_w)

        gpu_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        total_mem = sum(p.usedGpuMemory for p in gpu_procs) or 1

        for p in gpu_procs:
            team = get_team(p.pid)
            frac = p.usedGpuMemory / total_mem
            mem_mb = p.usedGpuMemory / 1024**2
            samples[team].append((power_w * frac, mem_mb, frac))

        if time.time() >= next_print:
            build_team_map()  # refresh to pick up any new child pids
            elapsed = int(time.time() - start)
            print(f"\n  [{elapsed}s] total GPU: {power_w:.1f}W")
            for p in gpu_procs:
                team = get_team(p.pid)
                frac = p.usedGpuMemory / total_mem
                mem_mb = p.usedGpuMemory / 1024**2
                print(f"  {team:<14} {mem_mb:>8.0f} {frac*100:>8.1f}% {power_w*frac:>8.1f}W")
            next_print += 30

        time.sleep(SAMPLE_INTERVAL)

    pynvml.nvmlShutdown()

    # ── Results table ─────────────────────────────────────────────
    avg_total = sum(total_power_samples) / len(total_power_samples)

    print(f"\n\n══════════════════════════════════════════════════════════════════════")
    print(f"  RESULTS — {RUN_SECONDS//60}-minute attribution  (avg GPU total: {avg_total:.1f}W)")
    print(f"══════════════════════════════════════════════════════════════════════")
    print(f"  {'Job':<14} {'Avg Mem MB':>10} {'Mem %':>7} {'Avg Power W':>12} {'kWh':>10}")
    print(f"  {'─'*64}")

    rows = []
    for team, data in samples.items():
        avg_power = sum(d[0] for d in data) / len(data)
        avg_mem   = sum(d[1] for d in data) / len(data)
        avg_frac  = sum(d[2] for d in data) / len(data)
        kwh       = avg_power * (RUN_SECONDS / 3600) / 1000
        rows.append((team, avg_mem, avg_frac, avg_power, kwh))

    rows.sort(key=lambda r: -r[3])
    for team, avg_mem, avg_frac, avg_power, kwh in rows:
        bar = '█' * int(avg_frac * 24)
        print(f"  {team:<14} {avg_mem:>10.0f} {avg_frac*100:>6.1f}% {avg_power:>12.1f} {kwh:>10.6f}")
        print(f"  {'':14} {bar}")

    total_kwh = sum(r[4] for r in rows)
    print(f"  {'─'*64}")
    print(f"  {'TOTAL':<14} {'':>10} {'100%':>7} {avg_total:>12.1f} {total_kwh:>10.6f}")
    print(f"══════════════════════════════════════════════════════════════════════\n")

    # ── Prometheus snapshot ───────────────────────────────────────
    try:
        import urllib.request, re
        with urllib.request.urlopen(f'http://localhost:{PROMETHEUS_PORT}/metrics', timeout=3) as r:
            txt = r.read().decode()

        def pval(m):
            hit = re.search(rf'^{re.escape(m)}(?:\{{[^}}]*\}})? (\S+)', txt, re.M)
            return float(hit.group(1)) if hit else None

        pw = pval('aluminatai_gpu_power_watts')
        up = pval('aluminatai_agent_uptime_seconds')
        ok = pval('aluminatai_upload_success_total')
        fail = pval('aluminatai_upload_failure_total')
        print(f"  Prometheus:  power={pw:.1f}W  uptime={int((up or 0)//60)}m  "
              f"uploads={int(ok or 0)} ok / {int(fail or 0)} failed")
    except Exception:
        print("  (Prometheus not reachable — is the agent running?)")

    # ── Cleanup ───────────────────────────────────────────────────
    print("\nStopping workers...")
    for name, p in procs.items():
        try:
            p.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for name, p in procs.items():
        if p.poll() is None:
            p.kill()
    print("Done. Check dashboard → https://www.aluminatiai.com/dashboard\n")

if __name__ == '__main__':
    main()
