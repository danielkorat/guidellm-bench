# guidellm-bench — Copilot Instructions

## Purpose

Single-script benchmarking tool (`bench.py`) that runs guidellm against vLLM servers on Intel XPU hardware, across a matrix of models × tensor-parallelism × quantization × eager-mode configurations. Produces per-config JSON results, a combined interactive HTML dashboard, and GPU utilisation data.

## Repository Structure

```
/root/dkorat/guidellm-bench/
├── bench.py                      # Entry point (thin — delegates to guidellm_bench/)
├── install.sh                    # Host-side install: starts container + installs deps inside it
├── pyproject.toml                # Python project metadata and dependencies
├── README.md
├── .github/copilot-instructions.md
├── guidellm_bench/               # Core package
│   ├── __init__.py               # Public API re-exports
│   ├── config.py                 # Config dataclass, FULL/SANITY defaults, skip_reason()
│   ├── docker.py                 # CONTAINER_NAME, docker_exec_cmd(), ensure_container_running()
│   ├── server.py                 # vLLM server lifecycle: start, health-check, stop
│   ├── monitor.py                # GpuMonitor background thread (xpu-smi)
│   ├── dataset.py                # AIME 2024 dataset download and caching
│   ├── benchmark.py              # run_guidellm() and copy_results()
│   └── dashboard.py              # build_dashboard_html() and write_serve_script()
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
- **Environment**: Runs from the **host machine**; all vLLM / guidellm / xpu-smi subprocess calls
  are dispatched into `vllm-0.14` via `docker_exec_cmd()` from `guidellm_bench/docker.py`.
  The container is started automatically by `ensure_container_running()` at the top of `main()`.
- **Volume mount**: Host `/root/dkorat/` → Container `/root/` — result files land on both

## Installation

Run `install.sh` from the **host machine** (container is created/started automatically):

```bash
bash install.sh
# or, if xpu-smi already installed inside the container:
bash install.sh --skip-xpu-smi
```

The script does **four things** in order:

### 0. Ensure container running

Inspects `vllm-0.14`. If missing, runs `docker run` with the standard args from `reference.sh`
(volume `/root/dkorat/:/root`, `--net=host`, `--privileged`, etc.).

### 1. xpu-smi (system package — inside container, Ubuntu 24.04 noble)

> **Do NOT** add `https://repositories.intel.com/graphics/ubuntu jammy client` —
> wrong distro, triggers the `libmetee4 Breaks libmetee5` conflict.

The Intel GPU noble unified repo is pre-configured in the container. The two-step
pinning trick is required because apt's solver selects `libmetee4=5.0.0-123` and
`libmetee5` (kobuk PPA) simultaneously, and `libmetee5` declares `Breaks: libmetee4`:

```bash
apt-get install -y xpu-smi=1.2.42-79~24.04 libmetee4=4.3.1-115~u24.04
apt-get install -y libmetee4=5.0.0-123~u24.04   # upgrade: binary links libmetee.so.5.0.0
```

### 2. Python dependencies

Defined in `pyproject.toml`:
- `datasets>=2.19.0` — downloads AIME 2024 prompts via HuggingFace
- `tzdata>=2024.1` — zoneinfo timezone data
- `guidellm` (optional extra `[guidellm]`) — patched fork:
  `git+https://github.com/danielkorat/guidellm.git@fix/thinking-model-ttft`

```bash
pip install -e ".[guidellm]"
```

### 3. Verification

Script imports `datasets`, `guidellm`, and `zoneinfo` to confirm everything resolved.

---

## Critical Rules & Corrections

### 0. Host-machine Operation — ALL subprocesses use docker_exec_cmd()
**RULE**: The project runs on the **host machine**. Every subprocess call that targets vLLM,
guidellm, xpu-smi, or other container-resident tools MUST be wrapped with `docker_exec_cmd()`
from `guidellm_bench/docker.py`. Never call these tools directly as bare commands.
```python
from guidellm_bench.docker import docker_exec_cmd
# CORRECT:
subprocess.Popen(docker_exec_cmd("vllm serve ..."), ...)
# WRONG:
subprocess.Popen(["bash", "--login", "/tmp/vllm_server.sh"], ...)
```
`ensure_container_running()` is called once at the start of `main()` to start/create the container.

### 1. Quantization Flag
**RULE**: Never use `--quantization off` or `--quantization none`. Omit the flag entirely.
```python
if cfg.quant:
    parts.append(f"--quantization {cfg.quant}")
```

### 2. oneAPI Environment
**RULE**: The oneAPI preamble (`source /opt/intel/oneapi/setvars.sh --force`) is baked into
`docker_exec_cmd()` automatically via `_PREAMBLE` in `docker.py`. Do NOT write `/tmp/*.sh` scripts
and run them with `bash --login`. Use `docker_exec_cmd(inner_cmd)` directly.

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
| gpt-oss-20b + tp<4 | `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on XPU — model too large for 2 GPUs |
| gpt-oss-20b + eager=false | `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on XPU |
| Qwen3-30B + quant=none | IPEX/XPU mode-stack bug (unquantized BF16 fails) |
| Qwen3-4B + quant=none | fp8 is uniformly faster (lower TTFT/ITL/lat, higher TPS — verified 20260302); skip unquantized |

### 6. Defaults: tp=[2,4], eager=true only
**RULE**: Full runs default to `tp=[2, 4]` and `eager=["true"]`.
- `tp=8` available via `--tp 4 8` when needed.
- `eager=false` is always skipped (OOM on gpt-oss-20b; negligible benefit elsewhere).
- `eager` is always `True` and is **omitted from `Config.name`** to keep experiment names short.

### 7. Benchmark Quality Controls (full runs)
```
guidellm benchmark \
  --warmup 0.1       # exclude first 10% requests (covers XPU JIT spike on req 1)
  --cooldown 0.1     # exclude last 10%
  --max-errors 5     # abort on repeated failures
  --max-seconds 900  # hard wall-clock limit per benchmark (Qwen3-30B needs ~540s for 30 reqs)
  --num-requests 20  # → 2 warmup + 2 cooldown = 16 clean samples
```

### 8. AIME Dataset for Realistic Prompts
**RULE**: Use `HuggingFaceH4/aime_2024` dataset (30 math problems) instead of synthetic random tokens.
- Loaded once at startup via `prepare_aime_dataset()`, cached to `/tmp/aime_2024_v2.jsonl`.
- Each JSONL row: `{"prompt": "<problem>", "output_tokens_count": 1024}`.
  - Column name MUST be `output_tokens_count` (guidellm default) — this maps to `max_tokens` in
    the `/v1/completions` request body. Using `output_tokens` instead results in `max_tokens` being
    absent and models generating only 16 tokens (vLLM default).
- No `--data-column-mapper` needed — `prompt` and `output_tokens_count` are auto-detected defaults.
- Passed via: `--data /tmp/aime_2024_v2.jsonl --data-samples -1 --max-requests 30`
- Falls back to synthetic tokens silently if the download fails (no internet).

### 9. GPU Monitoring via xpu-smi
**RULE**: `GpuMonitor` runs `xpu-smi` **on the host machine** directly (plain `subprocess.run`, no `docker_exec_cmd`). xpu-smi does not work inside the container.

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

### 14. Always Update Documentation
**RULE**: After ANY change that affects usage, installation, repo structure, or behaviour — update **both** `README.md` and `.github/copilot-instructions.md` in the same response.
- New file added → add to repo structure in both docs
- New CLI flag → add to `## Common Commands` / `## Quick Start`
- New install step → update `## Installation` in both docs
- Behaviour change → update relevant rules and notes

### 15. Module Structure (package layout)
**RULE**: Business logic lives in `guidellm_bench/` package. `bench.py` is a thin entry point only.

| Module | Responsibility |
|---|---|
| `config.py` | `Config` dataclass, `FULL`/`SANITY` defaults, `skip_reason()` |
| `docker.py` | `CONTAINER_NAME`, `docker_exec_cmd()`, `ensure_container_running()`, path helpers |
| `server.py` | vLLM server lifecycle: start, health-check, stop |
| `monitor.py` | `GpuMonitor` background thread (xpu-smi) |
| `dataset.py` | AIME 2024 dataset download and caching |
| `benchmark.py` | `run_guidellm()` and `copy_results()` |
| `dashboard.py` | `build_dashboard_html()` and `write_serve_script()` |

Do **not** add new logic directly to `bench.py`.

### 16. Self-Logging (bench.py writes its own log + pid)
**RULE**: After creating `out_dir`, `bench.py` tees stdout/stderr into `out_dir/bench.log` and writes its PID to `out_dir/bench.pid`. No external redirect is needed.
```bash
# Correct:
nohup ./bench.py &
# Log and pid land in guidellm_results/YYYYMMDD_HHMM/

# WRONG (old pattern):
nohup ./bench.py > bench_full.log 2>&1 & echo $! > bench_full.pid
```

## Default Configuration

### Models
```python
["openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-4B-Thinking-2507"]
```

### Full Run Defaults
| Parameter | Value |
|---|---|
| `tp` | `[2, 4]` |
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
| `timeout_startup` | 600s |

## Common Commands

```bash
# Quick sanity check
./bench.py --sanity
# Logs → ./guidellm_sanity_results/YYYYMMDD_HHMM/bench.log
# PID  → ./guidellm_sanity_results/YYYYMMDD_HHMM/bench.pid

# Full benchmark suite (background — self-logs; no redirect needed)
nohup ./bench.py &

# Resume an interrupted run (skips configs with existing _benchmarks.json)
./bench.py --resume guidellm_results/YYYYMMDD_HHMM
# Or: resume the latest run automatically
./bench.py --resume

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
| Dashboard shows 0 for all metrics | `b['metrics']` aggregates are zero-filled in guidellm v0.6 | Extract medians from `b['requests']['successful']` per-request fields |
| Models generate only 16 output tokens | `output_tokens` column not mapped to `max_tokens`; vLLM default is 16 | Use column name `output_tokens_count` (guidellm default, auto-detected) |

---

**Last Updated**: March 2, 2026 — xpu-smi runs on host directly (not via docker exec); migrated to host-machine operation with docker_exec_cmd()  
**Primary Maintainer**: Daniel Korat, Intel

---

## Lessons Learned

Mistakes that happened once and must not repeat:

| # | Mistake | Correct behaviour |
|---|---|---|
| 1 | Added `jammy` Intel graphics repo on noble (24.04) | Use only the pre-configured Intel GPU noble repo; never add jammy repos |
| 2 | `apt install xpu-smi` without pinning → `libmetee4 Breaks libmetee5` conflict | Always use the two-step pin: first `libmetee4=4.3.1-115`, then upgrade to `5.0.0-123` |
| 3 | Created/updated files without updating `README.md` | Every change must update README.md and copilot-instructions.md in the same response (Rule 14) |
| 4 | Kept all 900 lines of logic in a single `bench.py` | Business logic belongs in `guidellm_bench/` package; `bench.py` is entry-point only (Rule 15) |
| 5 | Tried to upgrade system pip inside container (`RECORD file not found`) | Skip `pip install --upgrade pip` inside the vLLM container; the bundled pip is sufficient |
| 6 | `SANITY.timeout_startup=180` too short — vLLM XPU JIT on first load takes >3 min | Use `timeout_startup=600` for both FULL and SANITY |
| 7 | Log and pid files saved in repo root via `nohup ./bench.py > bench.log` | `bench.py` self-logs into `out_dir/bench.log`; just run `nohup ./bench.py &` (Rule 16) |
| 8 | vLLM server hangs after XPU kernel registration warnings (`OperatorEntry.cpp:208 Warning: Overriding a previously registered kernel`) and never reaches `/health` | The XPU driver/runtime state is corrupted — **reboot the host**: `ssh root@10.75.137.163` then `reboot`. No amount of `pkill` or restart will fix this without a reboot. |
| 9 | Named AIME JSONL column `output_tokens` — models generated only 16 tokens | Column must be `output_tokens_count` (guidellm default); `output_tokens` is not mapped to `max_tokens` in the completions body, leaving vLLM's default of 16. Cache path is `/root/dkorat/aime_2024_v2.jsonl` (host) = `/root/aime_2024_v2.jsonl` (container). |
| 10 | Dashboard showed 0 for TTFT/ITL/req/s/tok/s | `b['metrics']` aggregates are zero-filled in guidellm v0.6; must compute medians from `b['requests']['successful'][*]` per-request fields in `_extract_sweep_points`. |
| 11 | gpt-oss-20b + tp=2 crashed with OOM (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY), guidellm reported 30 "successful" requests with empty output | Added `gpt-oss-20b + tp<4` skip rule — model requires at least 4 GPUs |
| 12 | Wrote subprocess calls that ran tools directly on the host (e.g. `pkill -f 'vllm serve'`) | All tool invocations (vLLM, guidellm, pkill) MUST use `docker_exec_cmd()` — the tools live inside the container, not on the host. **Exception: `xpu-smi` runs on the host directly** (does not work inside the container). |
| 13 | Wrapped `xpu-smi` in `docker_exec_cmd()` | `xpu-smi` must run on the host as a plain subprocess — `subprocess.run(["xpu-smi", ...])`. |
