#!/usr/bin/env bash
# =============================================================================
# install.sh — full from-scratch setup for guidellm-bench on Ubuntu 24.04 (noble)
#              inside the Intel XPU vLLM container (intel/vllm:0.14.1-xpu)
#
# Usage:
#   bash install.sh          # install everything
#   bash install.sh --skip-xpu-smi   # skip system package step (already done)
# =============================================================================
set -euo pipefail

SKIP_XPU_SMI=false
for arg in "$@"; do
  [[ "$arg" == "--skip-xpu-smi" ]] && SKIP_XPU_SMI=true
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── colours ──────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${G}[install]${NC} $*"; }
warn()  { echo -e "${Y}[warn]${NC}   $*"; }
die()   { echo -e "${R}[error]${NC}  $*" >&2; exit 1; }

# =============================================================================
# 1. xpu-smi (system package)
# =============================================================================
if [[ "$SKIP_XPU_SMI" == "false" ]]; then
  info "Installing xpu-smi …"

  # Sanity: we need the Intel GPU noble unified repo already configured.
  # It ships inside intel/vllm:0.14.1-xpu by default.
  if ! apt-cache show xpu-smi &>/dev/null; then
    die "Intel GPU noble repo not found. Expected it pre-configured in the container."
  fi

  # Do NOT add the jammy repo — wrong distro, causes libmetee4/5 conflict.
  # Do NOT run plain 'apt install xpu-smi': apt picks libmetee4=5.0.0 candidate
  # AND libmetee5=5.0 from kobuk PPA simultaneously — "Breaks" conflict.
  #
  # Fix: pin libmetee4 to a 4.x version at solve time, then upgrade it to 5.0.0
  # (the .so the binary actually links against) in a second pass.

  apt-get update -qq
  apt-get install -y \
    xpu-smi=1.2.42-79~24.04 \
    libmetee4=4.3.1-115~u24.04

  # Now upgrade libmetee4 to 5.0.0 — xpu-smi links against libmetee.so.5.0.0
  apt-get install -y libmetee4=5.0.0-123~u24.04

  if ! command -v xpu-smi &>/dev/null; then
    die "xpu-smi install failed (binary not on PATH)"
  fi
  info "xpu-smi $(dpkg-query -W -f='${Version}' xpu-smi) installed ✓"
else
  info "Skipping xpu-smi (--skip-xpu-smi passed)"
fi

# =============================================================================
# 2. Python — check version
# =============================================================================
PY=$(command -v python3 || true)
[[ -z "$PY" ]] && die "python3 not found"

PY_VER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Using Python $PY_VER at $PY"

# zoneinfo is stdlib from 3.9; datasets requires >= 3.8; guidellm requires >= 3.10
if python3 -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)'; then
  : # ok
else
  die "Python >= 3.10 required (found $PY_VER)"
fi

# =============================================================================
# 3. Core Python dependencies (from pyproject.toml)
# =============================================================================
info "Installing Python dependencies …"
cd "$SCRIPT_DIR"
# In the vLLM container pip is installed by debian (no RECORD file); upgrading
# it with pip itself fails. Skip the self-upgrade — the bundled pip is fine.
pip install --quiet -e ".[guidellm]"

# =============================================================================
# 4. Verify
# =============================================================================
info "Verifying imports …"
python3 - <<'EOF'
import importlib, sys

checks = [
    ("datasets",  "HuggingFace datasets"),
    ("guidellm",  "guidellm (patched fork)"),
    ("zoneinfo",  "zoneinfo"),
]
failed = []
for mod, label in checks:
    try:
        importlib.import_module(mod)
        print(f"  ✓  {label}")
    except ImportError as e:
        print(f"  ✗  {label}: {e}")
        failed.append(label)

if failed:
    sys.exit(1)
EOF

info "All checks passed ✓"
echo ""
echo "  Quick-start:"
echo "    ./bench.py --sanity       # fast smoke test (single config, 4 requests)"
echo "    ./bench.py                # full suite"
echo ""
