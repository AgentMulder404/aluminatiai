#!/usr/bin/env bash
# AluminatAI GPU Agent — installer
# Usage (recommended):
#   curl -sSL https://get.aluminatiai.com | bash
# Usage (local dev):
#   bash install.sh --local
#
# What this script does:
#   1. Checks prerequisites (NVIDIA drivers, Python 3.9+, pip, systemd)
#   2. Installs the aluminatiai Python package from PyPI (or locally with --local)
#   3. Creates a dedicated system user and data/log directories
#   4. Writes /etc/aluminatai/agent.env with your API key
#   5. Installs and starts the systemd service
#
# Supported: Ubuntu 20.04+, Debian 11+, RHEL/Rocky 8+, Amazon Linux 2023
# Requires:  root or sudo

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}  OK${NC}  $*"; }
warn() { echo -e "${YELLOW} WARN${NC}  $*"; }
die()  { echo -e "${RED}ERROR${NC}  $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}==> $*${NC}"; }

# ── Parse flags ───────────────────────────────────────────────────────────────

LOCAL_INSTALL=0
SKIP_SERVICE=0
UNATTENDED=0
PACKAGE_SOURCE="pypi"  # "pypi" | "local"

for arg in "$@"; do
    case "$arg" in
        --local)     LOCAL_INSTALL=1; PACKAGE_SOURCE="local" ;;
        --no-service) SKIP_SERVICE=1 ;;
        --unattended|-y) UNATTENDED=1 ;;
        --help|-h)
            cat <<'EOF'
AluminatAI Agent installer

Options:
  --local        Install from local source directory instead of PyPI
                 (for development or air-gapped installs with a downloaded wheel)
  --no-service   Install the Python package only; skip systemd setup
  --unattended   Non-interactive; requires ALUMINATAI_API_KEY env var to be set
  -y             Alias for --unattended

Examples:
  # Standard install from PyPI
  curl -sSL https://get.aluminatiai.com | bash

  # Air-gapped: install from local wheel
  pip download aluminatiai -d /tmp/pkg
  bash install.sh --local

  # Package only (manage service yourself)
  bash install.sh --no-service

  # CI / non-interactive
  ALUMINATAI_API_KEY=alum_xxx bash install.sh --unattended
EOF
            exit 0
            ;;
        *)
            die "Unknown option: $arg  (run with --help for usage)"
            ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────

echo
echo -e "${BOLD}AluminatAI GPU Agent Installer${NC}"
echo "  Docs:      https://aluminatiai.com/docs/agent"
echo "  Dashboard: https://aluminatiai.com/dashboard"
echo

# ── Check: root ───────────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
    # Re-execute with sudo if available
    if command -v sudo &>/dev/null; then
        exec sudo bash "$0" "$@"
    fi
    die "This script must be run as root.  Try: sudo bash install.sh"
fi

# ── Check: NVIDIA drivers ─────────────────────────────────────────────────────

step "Checking prerequisites"

if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found.  Install NVIDIA drivers first.\n       See: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
ok "NVIDIA drivers detected  (${GPU_NAME})"

# ── Check: Python 3.9+ ────────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    die "python3 not found.  Install Python 3.9+ first."
fi

PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJ=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MIN=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [[ "$PYTHON_MAJ" -lt 3 || ( "$PYTHON_MAJ" -eq 3 && "$PYTHON_MIN" -lt 9 ) ]]; then
    die "Python 3.9+ required (found $PYTHON_VER).  Upgrade Python first."
fi
ok "Python $PYTHON_VER"

# ── Check: pip ────────────────────────────────────────────────────────────────

if ! python3 -m pip --version &>/dev/null; then
    die "pip not found.  Install with: python3 -m ensurepip --upgrade"
fi
ok "pip available"

# ── Check: systemd (unless --no-service) ─────────────────────────────────────

if [[ $SKIP_SERVICE -eq 0 ]]; then
    if ! command -v systemctl &>/dev/null; then
        warn "systemd not found — installing package only (no service setup)."
        warn "Start manually:  aluminatiai"
        SKIP_SERVICE=1
    else
        ok "systemd available"
    fi
fi

# ── Install package ───────────────────────────────────────────────────────────

step "Installing aluminatiai package"

if [[ $LOCAL_INSTALL -eq 1 ]]; then
    # Find the closest pyproject.toml to locate the source tree.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
    if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
        SRC_DIR="$SCRIPT_DIR"
    elif [[ -f "$SCRIPT_DIR/../agent/pyproject.toml" ]]; then
        SRC_DIR="$(cd "$SCRIPT_DIR/../agent" && pwd)"
    else
        die "Cannot find pyproject.toml.  Run from the agent/ directory or use PyPI install."
    fi
    echo "  Source: $SRC_DIR"
    python3 -m pip install "$SRC_DIR" --quiet
else
    python3 -m pip install --upgrade aluminatiai --quiet
fi

INSTALLED_VER=$(python3 -c 'import importlib.metadata; print(importlib.metadata.version("aluminatiai"))' 2>/dev/null || echo "unknown")
ok "aluminatiai $INSTALLED_VER installed"

# Find the installed binary
AGENT_BIN=$(command -v aluminatiai 2>/dev/null || python3 -m site --user-base 2>/dev/null | xargs -I{} find {}/bin -name aluminatiai 2>/dev/null | head -1 || echo "")
if [[ -z "$AGENT_BIN" ]]; then
    # Try common pip install locations
    for candidate in /usr/local/bin/aluminatiai /usr/bin/aluminatiai ~/.local/bin/aluminatiai; do
        [[ -x "$candidate" ]] && AGENT_BIN="$candidate" && break
    done
fi
[[ -z "$AGENT_BIN" ]] && die "aluminatiai binary not found after install.  Check PATH."
ok "Binary: $AGENT_BIN"

if [[ $SKIP_SERVICE -eq 1 ]]; then
    echo
    echo -e "${GREEN}Package installed.${NC}  Start with:"
    echo "  export ALUMINATAI_API_KEY=alum_your_key"
    echo "  aluminatiai"
    exit 0
fi

# ── System user and directories ───────────────────────────────────────────────

step "Creating system user and directories"

if ! id -u aluminatai &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin \
            --comment "AluminatAI GPU agent" aluminatai
    ok "System user 'aluminatai' created"
else
    ok "System user 'aluminatai' already exists"
fi

install -d -m 0700 -o aluminatai -g aluminatai /var/lib/aluminatai
install -d -m 0755 -o aluminatai -g aluminatai /var/log/aluminatai
install -d -m 0750 /etc/aluminatai
ok "Directories created  (/var/lib/aluminatai, /var/log/aluminatai, /etc/aluminatai)"

# ── API key ───────────────────────────────────────────────────────────────────

step "API key"

if [[ $UNATTENDED -eq 1 ]]; then
    if [[ -z "${ALUMINATAI_API_KEY:-}" ]]; then
        die "--unattended requires ALUMINATAI_API_KEY env var to be set."
    fi
    API_KEY="$ALUMINATAI_API_KEY"
    echo "  Using API key from environment."
elif [[ -f /etc/aluminatai/agent.env ]]; then
    warn "Existing /etc/aluminatai/agent.env found — keeping it."
    warn "To replace it, run:  aluminatiai service install"
    API_KEY=""
else
    echo "  Get your key at: https://aluminatiai.com/dashboard/setup"
    echo
    while true; do
        read -rp "  Enter API Key: " API_KEY
        if [[ -n "$API_KEY" ]]; then
            break
        fi
        warn "API key cannot be empty."
    done
    if [[ ! "$API_KEY" =~ ^alum_ ]]; then
        warn "Key does not start with 'alum_' — double-check you copied it correctly."
    fi
fi

# ── Write env file ────────────────────────────────────────────────────────────

if [[ -n "$API_KEY" ]]; then
    cat > /etc/aluminatai/agent.env <<EOF
# AluminatAI Agent Configuration
# Generated by install.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Edit this file to change settings; restart the service afterwards:
#   sudo systemctl restart aluminatai-agent
#
# Full reference: https://aluminatiai.com/docs/agent#configuration

ALUMINATAI_API_KEY=$API_KEY
ALUMINATAI_API_ENDPOINT=https://aluminatiai.com/api/metrics/ingest
SAMPLE_INTERVAL=5.0
UPLOAD_INTERVAL=60
METRICS_PORT=9100
LOG_LEVEL=INFO
EOF
    chmod 600 /etc/aluminatai/agent.env
    chown root:root /etc/aluminatai/agent.env
    ok "Wrote /etc/aluminatai/agent.env (mode 600)"
fi

# ── Systemd unit ──────────────────────────────────────────────────────────────

step "Installing systemd service"

cat > /etc/systemd/system/aluminatai-agent.service <<UNIT
# AluminatAI GPU Energy Monitoring Agent
# Managed by install.sh — edit with care
[Unit]
Description=AluminatAI GPU Energy Monitoring Agent
Documentation=https://aluminatiai.com/docs/agent
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
User=aluminatai
Group=aluminatai
EnvironmentFile=/etc/aluminatai/agent.env
Environment=DATA_DIR=/var/lib/aluminatai
Environment=LOG_DIR=/var/log/aluminatai
ExecStart=$AGENT_BIN
Restart=on-failure
RestartSec=10s
TimeoutStopSec=30s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/aluminatai /var/log/aluminatai
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM
CapabilityBoundingSet=
MemoryMax=256M
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNIT

chmod 644 /etc/systemd/system/aluminatai-agent.service
ok "Wrote /etc/systemd/system/aluminatai-agent.service"

systemctl daemon-reload
systemctl enable aluminatai-agent
systemctl restart aluminatai-agent

# Give it a moment to start
sleep 3

if systemctl is-active --quiet aluminatai-agent; then
    echo
    echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}${BOLD}  AluminatAI Agent is running!${NC}"
    echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo "  Status:    sudo systemctl status aluminatai-agent"
    echo "  Logs:      sudo journalctl -u aluminatai-agent -f"
    echo "  Metrics:   curl -s localhost:9100/metrics | head -20"
    echo "  Dashboard: https://aluminatiai.com/dashboard"
    echo
else
    echo
    echo -e "${RED}Service failed to start.${NC}"
    echo "  Check logs: sudo journalctl -u aluminatai-agent -n 50"
    exit 1
fi
