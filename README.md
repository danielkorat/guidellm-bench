# guidellm-bench

Automated benchmarking for **Intel XPU vLLM** inference using [guidellm](https://github.com/vllm-project/guidellm). Runs a matrix of model × tensor-parallelism × quantization configurations and produces an interactive HTML dashboard with latency, throughput, and GPU memory metrics.

**All commands run from the host machine.** The tool automatically launches or reuses the `lsv-container` Docker container (`intel/llm-scaler-vllm:0.14.0-b8`) and dispatches every vLLM / guidellm invocation inside it via `docker exec`.

## Requirements

- Host machine running Ubuntu 24.04 (noble) with Docker CLI available
- Python ≥ 3.10 on the **host** (only for `bench.py` orchestration)
- `intel/llm-scaler-vllm:0.14.0-b8` Docker image pulled (contains vLLM with Expert Parallelism support; guidellm will be installed, xpu-smi optional)
- Intel XPU hardware with the Intel GPU noble unified apt repo pre-configured **inside** the container
- Volume mount `/root/dkorat/` → `/root/` — `bench.py` writes result files here so they are visible on both host and container

## Installation

Run the install script from the **host machine** (the container is started automatically):

```bash
git clone https://github.com/danielkorat/guidellm-bench.git /root/dkorat/guidellm-bench
cd /root/dkorat/guidellm-bench
bash install.sh
```

If `xpu-smi` is already installed inside the container, skip the system package step:

```bash
bash install.sh --skip-xpu-smi
```

The script does four things in order:

### 0. Ensure container is running

The script inspects the `lsv-container` container. If it does not exist it runs:

```bash
docker run -t -d --shm-size 32g --net=host --ipc=host --privileged \
  -e http_proxy=... -e https_proxy=... \
  -e HF_TOKEN=${HF_READ_TOKEN} \
  --name=lsv-container \
  --device /dev/dri:/dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v /root/dkorat/:/root \
  --entrypoint= intel/llm-scaler-vllm:0.14.0-b8 /bin/bash
```

### 1. `xpu-smi` (system package — installed inside container)

> **Note**: The Intel GPU noble unified repo must already be configured — it is pre-configured in `intel/llm-scaler-vllm:0.14.0-b8`.  
> **Do NOT** add the `jammy` (Ubuntu 22.04) Intel graphics repo — it causes a `libmetee` package conflict.

A two-step `libmetee4` pinning strategy is required to avoid an apt solver conflict between `libmetee4` (Intel GPU repo) and `libmetee5` (kobuk PPA) which declares `Breaks: libmetee4`:

```bash
# Step 1: pin libmetee4 to 4.x at solve time to avoid the Breaks constraint
apt-get install -y xpu-smi=1.2.42-79~24.04 libmetee4=4.3.1-115~u24.04
# Step 2: upgrade to 5.0.0 which provides libmetee.so.5.0.0 (required at runtime)
apt-get install -y libmetee4=5.0.0-123~u24.04
```

### 2. Python dependencies (installed inside container)

Defined in `pyproject.toml`:

| Package | Purpose |
|---|---|
| `datasets>=2.19.0` | Downloads AIME 2024 benchmark prompts from HuggingFace |
| `tzdata>=2024.1` | zoneinfo timezone data (Israel time for output dirs) |
| `guidellm` (patched fork) | Benchmark driver — fixes TTFT=0 bug on thinking models |

```bash
# (run from host) install inside container:
docker exec lsv-container pip install -e "/root/guidellm-bench[guidellm]"
```

The patched guidellm fork: `git+https://github.com/danielkorat/guidellm.git@fix/thinking-model-ttft`  
Fixes `delta.reasoning_content` detection in `ChatCompletionsRequestHandler`.

### 3. Verification (runs inside container)

The script imports `datasets`, `guidellm`, and `zoneinfo` inside the container to confirm everything resolved.

## Quick Start

```bash
# Sanity check (single config, 4 requests) — run from host machine
./bench.py --sanity
# Logs → sanity_results/YYYYMMDD_HHMM/bench.log
# PID  → sanity_results/YYYYMMDD_HHMM/bench.pid

# Full benchmark suite in background
nohup ./bench.py > /dev/null 2>&1 &
# Logs → results/YYYYMMDD_HHMM/bench.log  (self-logged; no redirect needed)

# Resume an interrupted run (skips configs with existing _benchmarks.json)
./bench.py --resume results/YYYYMMDD_HHMM
# Or: resume the latest run automatically
./bench.py --resume

# With Eagle3 speculative decoding (gpt-oss-120b, tp=8)
./bench.py --eagle3
```

## Default Matrix

| Parameter | Full | Sanity |
|---|---|---|
| Models | `gpt-oss-20b`, `Qwen3-30B-A3B`, `Qwen3-4B-Thinking-2507` | `Qwen3-4B-Thinking-2507` |
| TP | `[4]` | `[4]` |
| Quantization | `none`, `fp8` | `none` |
| Eager | `true` | `true` |
| Prompts | 20 (16 clean after warmup/cooldown) | 4 |
| Max model len | 16384 | 2048 |

Override any parameter via CLI: `--models`, `--tp`, `--quantization`, `--num-prompts`, `--max-model-len`, etc.

## Repository Structure

```
guidellm-bench/
├── bench.py                      # Entry point (thin — delegates to guidellm_bench/)
├── install.sh                    # Host-side install script (starts container + installs deps)
├── pyproject.toml                # Project metadata and dependencies
├── README.md
├── guidellm_bench/               # Core package
│   ├── config.py                 # Config dataclass, defaults, skip rules
│   ├── docker.py                 # Container lifecycle: ensure_container_running, docker_exec_cmd
│   ├── server.py                 # vLLM server lifecycle
│   ├── dataset.py                # AIME 2024 dataset preparation
│   ├── benchmark.py              # guidellm benchmark runner
│   └── dashboard.py              # Interactive HTML dashboard builder
└── results/             # Created at runtime
    └── YYYYMMDD_HHMM/
        ├── {cfg}_benchmarks.json
        ├── {cfg}_benchmarks.html
        ├── dashboard.html
        ├── serve_dashboard.sh
        └── logs/
```

## Outputs

Each run creates a timestamped directory (Israel time, `YYYYMMDD_HHMM`):

```bash
bash results/YYYYMMDD_HHMM/serve_dashboard.sh
```

## Dashboard Metrics

- **TTFT** (Time to First Token, ms)
- **ITL** (Inter-Token Latency, ms)
- **Throughput** (req/s and tok/s)
- **Model Weights Memory (GiB/GPU)** — parsed directly from the vLLM server log (`parse_model_mem_gib()`); `xpu-smi` is NOT used at runtime (enters uninterruptible D-state when the GPU is active)
- Per-config GPU utilisation and memory time-series charts

## Rebuild Dashboard from Partial Results

If a run is interrupted, regenerate the dashboard from completed configs:

```python
from guidellm_bench.dashboard import build_dashboard_html
from pathlib import Path
build_dashboard_html(Path("results/YYYYMMDD_HHMM"), ["cfg1_name", "cfg2_name"])
```

## Known Skip Rules

| Skipped combination | Reason |
|---|---|
| Any model, fp8 + eager=false | vLLM engine init failure |
| gpt-oss-20b + fp8 | Model has mxfp4 baked in |
| gpt-oss-20b + eager=false | XPU OOM (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) |
| Qwen3-30B + quant=none | IPEX mode-stack bug |

## Notes

- Uses `HuggingFaceH4/aime_2024` (30 AIME math problems) as realistic prompts; falls back to synthetic tokens if unavailable.
  - JSONL column is `output_tokens_count` (guidellm default) so it auto-maps to `max_tokens` in the completions request. Using `output_tokens` instead causes models to generate only 16 tokens (vLLM default when `max_tokens` is absent).
  - Cache path: `/root/dkorat/aime_2024_v2.jsonl` (host) = `/root/aime_2024_v2.jsonl` (container)
- Benchmark quality controls: `--warmup 0.1 --cooldown 0.1 --max-errors 5 --max-seconds 600`
- Thinking models use `--request-format /v1/completions` to bypass chat template (prevents TTFT=0).
- Dashboard metrics are computed from `b['requests']['successful']` per-request fields; `b['metrics']` aggregates are zero-filled in guidellm v0.6 and must not be used.
- Output directory timestamps use Israel time (`Asia/Jerusalem` via `zoneinfo`).
