# guidellm-bench — Copilot Instructions

## Purpose

Single-script benchmarking tool (`bench.py`) that runs guidellm against vLLM servers on Intel XPU hardware, across a matrix of models × tensor-parallelism × quantization × eager-mode configurations. Produces per-config JSON results, a combined interactive HTML dashboard, and GPU utilisation data.

## Repository Structure

```
/root/guidellm-bench/
├── bench.py                      # Single entry-point — server lifecycle + guidellm runner + dashboard
├── .github/copilot-instructions.md
├── guidellm_results/             # Created at runtime (full runs)
│   └── YYYYMMDD_HHMM/
│       ├── {cfg_name}_benchmarks.json
│       ├── {cfg_name}_benchmarks.html
│       ├── {cfg_name}_gpu_monitor.json
│       ├── dashboard.html
│       ├── serve_dashboard.sh
│       └── logs/
└── guidellm_sanity_results/      # Created at runtime (--sanity runs)
```

## Technology Stack

- **Language**: Python 3, single file (`bench.py`)
- **Benchmark driver**: guidellm — use the patched fork (upstream is broken for thinking models; TTFT=0):
  ```bash
  pip install git+https://github.com/danielkorat/guidellm.git@fix/thinking-model-ttft
  ```
  Fork: https://github.com/danielkorat/guidellm/tree/fix/thinking-model-ttft  
  The patch fixes `delta.reasoning`/`delta.reasoning_content` detection in `ChatCompletionsRequestHandler.add_streaming_line()`.
- **LLM server**: vLLM (Intel XPU backend, `intel/vllm:0.14.1-xpu` container)
- **Hardware**: Intel XPU — `xpu-smi` for GPU monitoring
- **Environment**: Runs **inside** the Docker container; no `docker exec` wrappers

## Critical Rules & Corrections

### 1. Quantization Flag
**RULE**: Never use `--quantization off` or `--quantization none`. Omit the flag entirely.
```python
if cfg.quant:
    parts.append(f"--quantization {cfg.quant}")
```

### 2. oneAPI Environment
**RULE**: Write a `/tmp/*.sh` script that sources `setvars.sh`, then run with `bash --login /tmp/script.sh`.
```python
_write_script("/tmp/vllm_server.sh", "source /opt/intel/oneapi/setvars.sh --force", ...)
subprocess.Popen(["bash", "--login", "/tmp/vllm_server.sh"], ...)
```

### 3. Proxy / no_proxy
**RULE**: Always export `no_proxy` and `NO_PROXY` inside the shell scripts to bypass Intel proxy for localhost connections.
```bash
export no_proxy=localhost,127.0.0.1,0.0.0.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
```

### 4. /v1/completions for Thinking Models
**RULE**: Always pass `--request-format /v1/completions` for thinking models (`gpt-oss-20b`, `Qwen3-4B-Thinking-2507`, etc.).
- Default `/v1/chat/completions` applies the chat template, injecting thinking tokens that cause mid-stream vLLM errors (`Unexpected token 200002`) → `TTFT=0ms`.
- `/v1/completions` bypasses the template entirely and produces correct TTFT/ITL.

### 5. Skip Rules (all enforced in `skip_reason()`)

| Combination | Reason |
|---|---|
| fp8 + eager=false | Engine initialization failure |
| gpt-oss-20b + fp8 | Model has mxfp4 baked in; fp8 override rejected |
| gpt-oss-20b + eager=false | `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on XPU |
| Qwen3-30B + quant=none | IPEX/XPU mode-stack bug (unquantized BF16 fails) |

### 6. Defaults: tp=4, eager=true only
**RULE**: Full runs default to `tp=[4]` and `eager=["true"]`.
- `tp=8` available via `--tp 4 8` when needed.
- `eager=false` is always skipped (OOM on gpt-oss-20b; negligible benefit elsewhere).

### 7. Benchmark Quality Controls (full runs)
```
guidellm benchmark \
  --warmup 0.1       # exclude first 10% requests (covers XPU JIT spike on req 1)
  --cooldown 0.1     # exclude last 10%
  --max-errors 5     # abort on repeated failures
  --max-seconds 600  # hard wall-clock limit per benchmark
  --num-requests 20  # → 2 warmup + 2 cooldown = 16 clean samples
```

### 8. AIME Dataset for Realistic Prompts
**RULE**: Use `HuggingFaceH4/aime_2024` dataset (30 math problems) instead of synthetic random tokens.
- Loaded once at startup via `prepare_aime_dataset()`, cached to `/tmp/aime_2024.jsonl`.
- Each JSONL row: `{"prompt": "<problem>", "output_tokens": 1024}`.
- Passed via: `--data /tmp/aime_2024.jsonl --data-column-mapper '{"text_column": "prompt", "output_tokens_count_column": "output_tokens"}' --data-samples -1 --max-requests 30`
- Falls back to synthetic tokens silently if the download fails (no internet).

### 9. GPU Monitoring via xpu-smi
**RULE**: `GpuMonitor` background thread polls every 10 seconds using:
```bash
xpu-smi dump -d -1 -m 0,1,18 -i 1 -n 1
```
- `-d -1` = all devices, `-m 0,1,18` = GPU util (%), Power (W), Memory Used (MiB)
- Readings written to `{cfg_name}_gpu_monitor.json` alongside benchmark results.
- Dashboard shows peak GPU memory bar chart + per-device time-series (with graceful fallback if xpu-smi unavailable).

### 10. Health Check
**RULE**: Use `curl -f` and check only `returncode == 0`. The `/health` endpoint returns HTTP 200 with an empty body — do NOT check stdout content.

### 11. Eagle3 120b — Opt-in Only
**RULE**: `openai/gpt-oss-120b` with Eagle3 speculative decoding is **not** in the default matrix. Append only when `--eagle3` CLI flag is passed.
```python
if not args.sanity and args.eagle3:
    configs.append(Config(model="openai/gpt-oss-120b", tp=8, quant=None, eager=True,
                          speculative_config=EAGLE3_SPECULATIVE_CONFIG))
```

### 12. Timestamped Output Directories (Israel Time)
```python
from zoneinfo import ZoneInfo
ts = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y%m%d_%H%M")
out_dir = Path(results_dir) / ts
```

### 13. Script Executability
**RULE**: `bench.py` must have `#!/usr/bin/env python3` shebang and be `chmod +x`.

## Default Configuration

### Models
```python
["openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-4B-Thinking-2507"]
```

### Full Run Defaults
| Parameter | Value |
|---|---|
| `tp` | `[4]` |
| `quant` | `["none", "fp8"]` |
| `eager` | `["true"]` |
| `input_len` | 1024 |
| `output_len` | 1024 |
| `concurrency` | 16 |
| `num_prompts` | 20 |
| `max_model_len` | 16384 |
| `timeout_startup` | 300s |

### Sanity Defaults
| Parameter | Value |
|---|---|
| `models` | `["Qwen/Qwen3-4B-Thinking-2507"]` |
| `tp` | `[4]` |
| `num_prompts` | 4 |
| `max_model_len` | 2048 |

## Common Commands

```bash
# Quick sanity check
./bench.py --sanity

# Full benchmark suite (background)
nohup ./bench.py > bench_full.log 2>&1 & echo $! > bench_full.pid

# Specific model/config
./bench.py --models openai/gpt-oss-20b --tp 4 --quantization none

# With Eagle3 (gpt-oss-120b appended)
./bench.py --eagle3

# Rebuild dashboard from completed results
python3 -c "
from bench import build_dashboard_html
from pathlib import Path
out_dir = Path('guidellm_results/YYYYMMDD_HHMM')
succeeded = ['model_tp4_quant-none_eager-true']
build_dashboard_html(out_dir, succeeded)
"

# Serve dashboard
bash guidellm_results/YYYYMMDD_HHMM/serve_dashboard.sh
```

## Known Issues

| Issue | Cause | Fix |
|---|---|---|
| TTFT = 0ms on thinking models | Chat template injects thinking tokens | Use `--request-format /v1/completions` (Rule 4) |
| OOM on gpt-oss-20b + eager=false | XPU memory exhausted by graph compilation buffers | Skip (Rule 5) |
| Qwen3-30B + fp8 mismatch | mxfp4 in model config vs fp8 override | Skip (Rule 5) |
| xpu-smi unavailable | Tool not on PATH inside container | GpuMonitor silently returns `[]`; dashboard shows fallback |

---

**Last Updated**: March 2, 2026 — initial standalone repo from vllm-bench/guidellm-bench  
**Primary Maintainer**: Daniel Korat, Intel
