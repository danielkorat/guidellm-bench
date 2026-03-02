#!/usr/bin/env bash
# =============================================================================
# install.sh — full from-scratch setup for guidellm-bench
#              Runs from the HOST machine; all container steps execute via
#              'docker exec' into intel/vllm:0.14.1-xpu (container: vllm-0.14).
#
# Usage:
#   bash install.sh          # install everything (starts container if needed)
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

CONTAINER=vllm-0.14
IMAGE=intel/vllm:0.14.1-xpu
HOST_ROOT=/root/dkorat

# =============================================================================
# 0. Ensure the container is running
# =============================================================================
info "Checking container '${CONTAINER}'..."

running=$(docker inspect --format '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || echo "missing")

if [[ "$running" == "true" ]]; then
  info "Container already running ✓"
elif [[ "$running" == "false" ]]; then
  info "Starting stopped container..."
  docker start "${CONTAINER}"
else
  info "Container not found — creating from ${IMAGE}..."
  HF_TOKEN="${HF_READ_TOKEN:-}"
  docker run -t -d --shm-size 10g --net=host --ipc=host --privileged \
    -e http_proxy=http://proxy-dmz.intel.com:912 \
    -e https_proxy=http://proxy-dmz.intel.com:912 \
    -e HTTP_PROXY=http://proxy-dmz.intel.com:912 \
    -e HTTPS_PROXY=http://proxy-dmz.intel.com:912 \
    -e no_proxy=localhost,127.0.0.1,0.0.0.0 \
    -e NO_PROXY=localhost,127.0.0.1,0.0.0.0 \
    -e "HF_TOKEN=${HF_TOKEN}" \
    --name="${CONTAINER}" \
    --device /dev/dri:/dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "${HOST_ROOT}/:/root" \
    --entrypoint= \
    "${IMAGE}" /bin/bash
  info "Container '${CONTAINER}' launched ✓"
fi

# Helper: run a bash command inside the container (with oneAPI + proxy env).
cexec() {
  docker exec "${CONTAINER}" bash --login -c \
    "source /opt/intel/oneapi/setvars.sh --force && \
     export no_proxy=localhost,127.0.0.1,0.0.0.0 && \
     export NO_PROXY=localhost,127.0.0.1,0.0.0.0 && \
     $*"
}

# =============================================================================
# 1. xpu-smi (system package — inside container)
# =============================================================================
if [[ "$SKIP_XPU_SMI" == "false" ]]; then
  info "Installing xpu-smi inside container..."

  # Sanity: the Intel GPU noble unified repo must be pre-configured in the container.
  # Do NOT add the jammy repo — wrong distro, causes libmetee4/5 conflict.
  docker exec "${CONTAINER}" bash -c \
    'apt-cache show xpu-smi &>/dev/null || { echo "[error] Intel GPU noble repo not found."; exit 1; }'

  docker exec "${CONTAINER}" bash -c 'apt-get update -qq'

  # Two-step libmetee4 pinning: pin to 4.x at solve time to avoid the Breaks
  # constraint from libmetee5, then upgrade to 5.0.0 which provides the .so the
  # binary actually links against.
  docker exec "${CONTAINER}" bash -c \
    'apt-get install -y xpu-smi=1.2.42-79~24.04 libmetee4=4.3.1-115~u24.04'
  docker exec "${CONTAINER}" bash -c \
    'apt-get install -y libmetee4=5.0.0-123~u24.04'

  docker exec "${CONTAINER}" bash -c 'command -v xpu-smi' \
    || die "xpu-smi install failed (binary not on PATH)"
  info "xpu-smi installed ✓"
else
  info "Skipping xpu-smi (--skip-xpu-smi passed)"
fi

# =============================================================================
# 2. Python — check version inside container
# =============================================================================
info "Checking Python version inside container..."
PY_VER=$(docker exec "${CONTAINER}" python3 -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python ${PY_VER} ✓"

docker exec "${CONTAINER}" python3 -c \
  'import sys; exit(0 if sys.version_info >= (3,10) else 1)' \
  || die "Python >= 3.10 required inside container (found ${PY_VER})"

# =============================================================================
# 3. Python dependencies — installed inside container from the volume-mounted repo
# =============================================================================
info "Installing Python dependencies inside container..."
# The volume mount makes /root/dkorat/ → /root/ inside the container.
# In the vLLM container pip is OS-managed (PEP 668); --break-system-packages is
# required. Skipping pip self-upgrade (no RECORD file in the debian-managed pip).
docker exec "${CONTAINER}" bash -c \
  'pip install --quiet --break-system-packages -e "/root/guidellm-bench[guidellm]"'

# =============================================================================
# 4. Verify imports (container only)
# =============================================================================
info "Verifying container-side imports..."
docker exec "${CONTAINER}" python3 - <<'EOF'
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
echo "  Quick-start (from the host machine):"
echo "    ./bench.py --sanity       # fast smoke test (auto-relaunches inside container)"
echo "    ./bench.py                # full suite"
echo "    nohup ./bench.py &        # full suite in background (self-logs)"
echo ""

