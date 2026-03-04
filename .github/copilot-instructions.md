# guidellm-bench — Copilot Instructions

## Purpose

Benchmarking tool (`bench.py` entry point + `guidellm_bench/` package) that runs guidellm against vLLM servers on Intel XPU hardware, across a matrix of models × tensor-parallelism × quantization × eager-mode configurations. Produces per-config JSON results, a combined interactive HTML dashboard, and GPU utilisation data.

## Repository Structure

```
/root/dkorat/guidellm-bench/
├── bench.py                      # Entry point (thin — delegates to guidellm_bench/)
├── install.sh                    # Host-side install: starts container + installs deps inside it
├── pyproject.toml                # Python project metadata and dependencies
├── README.md
├── PLAN.md                       # Living plan/checklist for feature work
├── .github/copilot-instructions.md
├── guidellm_bench/               # Core package
│   ├── __init__.py               # Public API re-exports
│   ├── config.py                 # Config dataclass, FULL/SANITY defaults, skip_reason()
│   ├── docker.py                 # CONTAINER_NAME, _PREAMBLE, ensure_container_running()
│   ├── server.py                 # vLLM server lifecycle: start, health-check, stop
│   ├── dataset.py                # AIME 2024 + generic HF dataset download; long-context slicing
│   ├── benchmark.py              # run_guidellm() (+ lc_mode) and copy_results()
│   └── dashboard.py              # build_dashboard_html() and write_serve_script()
├── results/                      # Created at runtime (full runs) — gitignored
│   └── YYYYMMDD_HHMM/
│       ├── {cfg_name}_benchmarks.json
│       ├── {cfg_name}_benchmarks.html
│       ├── {cfg_name}_lc{N}k_benchmarks.json  # long-context slices (--long-contexts)
│       ├── dashboard.html
│       ├── serve_dashboard.sh
│       ├── datasets/                           # cached JSONL files for this run
│       └── logs/
└── sanity_results/               # Created at runtime (—sanity runs) — gitignored
```

## Technology Stack

- **Language**: Python 3 package (`bench.py` thin entry point + `guidellm_bench/` core package)
- **Benchmark driver**: guidellm — use the patched fork (upstream is broken for thinking models; TTFT=0):
  ```bash
  pip install git+https://github.com/danielkorat/guidellm.git@fix/thinking-model-ttft
  ```
  Fork: https://github.com/danielkorat/guidellm/tree/fix/thinking-model-ttft  
  The patch fixes `delta.reasoning`/`delta.reasoning_content` detection in `ChatCompletionsRequestHandler.add_streaming_line()`.
- **LLM server**: vLLM (Intel XPU backend, `intel/llm-scaler-vllm:0.14.0-b8` container)
- **Hardware**: Intel XPU — GPU memory is parsed from the vLLM server log (`parse_model_mem_gib()`). `xpu-smi` is installed as a system tool (see Installation) but is **never called during benchmarks** — it enters uninterruptible D-state when the GPU is active (see Rule 9).
- **Environment**: `bench.py` and `guidellm_bench/` run **inside** a Docker container. When
  invoked from the host, `bench.py` detects it is outside a container (`/.dockerenv` absent) and
  automatically re-execs itself via `docker exec` into `lsv-container` (`intel/llm-scaler-vllm:0.14.0-b8`).
  Proxy env vars from the host are forwarded via `-e` flags so HuggingFace remains reachable.

  All subprocess calls (vLLM, guidellm, pkill) are made directly inside the container — no docker exec wrapping needed.
- **Volume mount**: Host `/root/dkorat/` → Container `/root/` — result files land on both

## Installation

Run `install.sh` from the **host machine** (container is created/started automatically):

```bash
bash install.sh
# or, if xpu-smi already installed inside the container:
bash install.sh --skip-xpu-smi
```

The script does **three things** in order:

### 0. Ensure container running

Inspects `lsv-container`. If missing, runs `docker run` with proxy env vars inherited from
the host (volume `/root/dkorat/:/root`, `--net=host`, `--privileged`, `--shm-size 32g`, etc.).

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

### 2. Python dependencies (inside container)

Defined in `pyproject.toml`:
- `datasets>=2.19.0` — downloads AIME 2024 prompts via HuggingFace
- `tzdata>=2024.1` — zoneinfo timezone data
- `guidellm` (optional extra `[guidellm]`) — patched fork:
  `git+https://github.com/danielkorat/guidellm.git@fix/thinking-model-ttft`

```bash
# NOTE: --break-system-packages is required (PEP 668 / externally-managed-environment)
pip install --break-system-packages -e "/root/guidellm-bench/.[guidellm]"
```

`install.sh` handles this via `docker exec lsv-container pip install --break-system-packages ...`.
`ensure_container_running()` in `docker.py` auto-installs after creating a **new** container.
`bench.py` runs a safety-net import check at startup and installs if missing.

### 3. Verification (container only)

Script imports `datasets`, `guidellm`, and `zoneinfo` inside the container to confirm everything resolved.

---

## Critical Rules & Corrections

### 0. Execution Model — bench.py runs INSIDE the container
**RULE**: `bench.py` and `guidellm_bench/` run inside `lsv-container` (`intel/llm-scaler-vllm:0.14.0-b8`).
The re-exec guard always targets `lsv-container` and forwards host proxy env vars via `-e`:

```python
# bench.py re-exec guard (top of file, stdlib only):
if not os.path.exists("/.dockerenv"):
    _tty = ["-t"] if sys.stdout.isatty() else []
    _proxy_args = []
    for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        _val = os.environ.get(_var)
        if _val:
            _proxy_args += ["-e", f"{_var}={_val}"]
    _cmd = (["docker", "exec", "-w", "/root/guidellm-bench"]
            + _tty + _proxy_args
            + ["lsv-container", "python3", "/root/guidellm-bench/bench.py"]
            + sys.argv[1:])
    _rc = subprocess.call(_cmd)
    if _rc == 42:
        subprocess.call(["reboot"])
    sys.exit(_rc)
```

All subprocess calls (vLLM, guidellm, pkill, curl) are made **directly** — no `docker exec` wrapping.

`install.sh` installs all Python dependencies (datasets, guidellm, tzdata) **inside the container**
only. No host-side `pip install` is needed.

### 1. Quantization Flag
**RULE**: Never use `--quantization off` or `--quantization none`. Omit the flag entirely.
```python
if cfg.quant:
    parts.append(f"--quantization {cfg.quant}")
```

### 2. oneAPI Environment
**RULE**: Every subprocess that invokes vLLM or guidellm must source oneAPI first via
`bash --login -c "{_PREAMBLE} && {cmd}"` where `_PREAMBLE` is defined in `docker.py`.
Do NOT write `/tmp/*.sh` scripts and run them separately.

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
- Loaded once at startup via `prepare_aime_dataset()`, cached to `/root/aime_2024_v2.jsonl` (container path; volume-backed at host `/root/dkorat/aime_2024_v2.jsonl`).
- Each JSONL row: `{"prompt": "<problem>", "output_tokens_count": 1024}`.
  - Column name MUST be `output_tokens_count` (guidellm default) — this maps to `max_tokens` in
    the `/v1/completions` request body. Using `output_tokens` instead results in `max_tokens` being
    absent and models generating only 16 tokens (vLLM default).
- No `--data-column-mapper` needed — `prompt` and `output_tokens_count` are auto-detected defaults.
- Passed via: `--data /tmp/aime_2024_v2.jsonl --data-samples -1 --max-requests 30`
- Falls back to synthetic tokens silently if the download fails (no internet).

### 9. GPU Memory via Server Log Parsing
**RULE**: Do NOT use `xpu-smi` for GPU memory monitoring. `xpu-smi dump` enters
`D` (uninterruptible sleep) state when the GPU is in use by vLLM and **cannot be
killed even with SIGKILL**. This would accumulate zombie processes and block the host.

Instead, parse GPU memory directly from the vLLM server log:
```python
# server.py — parse_model_mem_gib(log_path: Path) -> Optional[float]
# Returns per-GPU weight memory in GiB from:
# "Model loading took X.XX GiB memory"
# Multiply by cfg.tp to get total across all devices.
```
GPU memory is passed to `build_dashboard_html()` via `model_mem_gib` dict and renders
the "Model Weights Memory (GiB)" bar chart. `monitor.py` and `GpuMonitor` have been
removed entirely.

### 10. Health Check
**RULE**: Use `curl -f` and check only `returncode == 0`. The `/health` endpoint returns HTTP 200 with an empty body — do NOT check stdout content.

### 11. Eagle3 120b — Opt-in Only
**RULE**: `openai/gpt-oss-120b` with Eagle3 speculative decoding is **not** in the default matrix. Append only when `--eagle3` CLI flag is passed.
```python
if not args.sanity and args.eagle3:
    configs.append(Config(model="openai/gpt-oss-120b", tp=8, quant=None, eager=True,
                          speculative_config=EAGLE3_SPECULATIVE_CONFIG))
```

### 12. Expert Parallelism (EP) — Opt-in Only
**RULE**: EP variants for MoE models are **not** in the default matrix. Append only when `--ep` CLI flag is passed.
```python
# Appended via --ep:
Config(model="openai/gpt-oss-20b",  tp=4, quant=None,  eager=True, expert_parallel_size=4)
Config(model="Qwen/Qwen3-30B-A3B", tp=2, quant="fp8", eager=True, expert_parallel_size=2)
Config(model="Qwen/Qwen3-30B-A3B", tp=4, quant="fp8", eager=True, expert_parallel_size=4)
```
⚠️ **Requires `intel/llm-scaler-vllm:0.14.0-b8` or later** (the single container used for all runs).
The vLLM flag emitted is: `--enable-expert-parallel` (boolean — **no** `--expert-parallel-size` parameter exists in this build).
EP is meaningful only for MoE models (gpt-oss-20b, Qwen3-30B-A3B): it distributes experts across GPUs, reducing per-expert memory and improving throughput.
**Guard**: `is_moe_model(model)` (defined in `config.py`) must return True before any EP config is added. The ep_configs loop skips non-MoE models with a printed warning.
**`Config.name` EP suffix**: `-ep` (no numeric value). `expert_parallel_size` is used as a boolean trigger only — the specific parallelism degree is implied by `tp`. Example: `Qwen_Qwen3-30B-A3B_tp4_quant-fp8-ep`.

### 13. Timestamped Output Directories (Israel Time)
```python
from zoneinfo import ZoneInfo
ts = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y%m%d_%H%M")
out_dir = Path(results_dir) / ts
```

### 14. Script Executability
**RULE**: `bench.py` must have `#!/usr/bin/env python3` shebang and be `chmod +x`.

### 15. Always Update Documentation
**RULE**: After ANY change that affects usage, installation, repo structure, or behaviour — update **both** `README.md` and `.github/copilot-instructions.md` in the same response.
- New file added → add to repo structure in both docs
- New CLI flag → add to `## Common Commands` / `## Quick Start`
- New install step → update `## Installation` in both docs
- Behaviour change → update relevant rules and notes

### 16. Module Structure (package layout)
**RULE**: Business logic lives in `guidellm_bench/` package. `bench.py` is a thin entry point only.

| Module | Responsibility |
|---|---|
| `config.py` | `Config` dataclass, `FULL`/`SANITY` defaults, `skip_reason()` |
| `docker.py` | `CONTAINER_NAME`, `_PREAMBLE`, `ensure_container_running()` (used by install.sh) |
| `server.py` | vLLM server lifecycle: start, health-check, stop |
| `dataset.py` | AIME 2024 dataset download and caching |
| `benchmark.py` | `run_guidellm()` and `copy_results()` |
| `dashboard.py` | `build_dashboard_html()` and `write_serve_script()` |

Do **not** add new logic directly to `bench.py`.

### 19. No /tmp for Logs or Temp Files
**RULE**: Never redirect logs or write temp files to `/tmp`. Always use the run's result directory
or a gitignored subdirectory of the repo.
- Bench logs → `out_dir/bench.log` (self-logged by bench.py, see Rule 20)
- guidellm scratch output → `out_dir/logs/.guidellm_out/` (benchmark.py creates per-run, auto-cleaned)
- Always use `nohup ./bench.py > /dev/null 2>&1 &` — **do NOT** redirect to `/tmp/something.log`
  and **do NOT** omit the `> /dev/null 2>&1` redirect: without it, `nohup` creates `nohup.out`
  at the repo root (Rule 25).

### 20. Result Directory Names
**RULE**: Result directories are `results/` and `sanity_results/` (no `guidellm_` prefix).
Both are gitignored.
```
results/YYYYMMDD_HHMM/        # full runs
sanity_results/YYYYMMDD_HHMM/ # --sanity runs
```
**RULE**: After creating `out_dir`, `bench.py` tees stdout/stderr into `out_dir/bench.log` and writes its PID to `out_dir/bench.pid`. No external redirect is needed, **but always add `> /dev/null 2>&1`** to prevent `nohup` from creating `nohup.out` at the repo root.
```bash
# Correct:
nohup ./bench.py > /dev/null 2>&1 &
# Log and pid land in results/YYYYMMDD_HHMM/

# WRONG (creates nohup.out at repo root):
nohup ./bench.py &

# WRONG (old pattern):
nohup ./bench.py > bench_full.log 2>&1 & echo $! > bench_full.pid
```

### 18. XPU Kernel Registration Hang — Reboot the Host
**RULE**: When the vLLM server log shows `OperatorEntry.cpp:208` kernel override warnings followed
by repeated `Still waiting... Xs elapsed` health-check messages, the XPU driver state inside
the container is corrupted and the server will **never** become healthy.

**This is now handled automatically:**
- `wait_for_server()` scans the server log at each 60s interval.
- If `OperatorEntry.cpp:208` is present AND ≥120s have elapsed with no healthy response,
  it raises `XpuKernelHangError`.
- The container-side bench.py catches this and exits with **code 42**.
- The host-side re-exec guard detects exit code 42, prints a warning, and **prompts for confirmation** before calling `reboot`.
- In non-interactive sessions (nohup), the reboot is skipped and must be done manually.

After reboot, resume manually:
```bash
./bench.py --resume   # picks up the latest run dir automatically
```

Do NOT attempt container recreation — D-state vllm processes block `docker rm -f` even with SIGKILL.

### 22. Server Reuse via server_status.json
**RULE**: `bench.py` **never stops the server at end of run**. `server_status.json` is left on
disk so the next bench invocation can skip the ~90s restart when the config matches.
`stop_server()` is only called **mid-run** when a config change requires a different server.

Before each config, `server_is_reusable(cfg, max_model_len)` in `server.py` checks:
1. `/root/guidellm-bench/server_status.json` exists and all config fields match (model, tp, quant, eager, expert_parallel_size, speculative_config, max_model_len, port).
2. The recorded PID is alive (`os.kill(pid, 0)`).
3. `/health` returns HTTP 200 (with `no_proxy` to bypass Intel proxy).

If all three pass → print "Reusing running server" and skip restart (saves ~90s per config).
On successful startup → `write_server_status(cfg, max_model_len, proc.pid, log_path)` writes the JSON.
On `stop_server()` (config change only) → `server_status.json` is deleted.

Status file path: `/root/guidellm-bench/server_status.json` (inside container; volume-backed at host `/root/dkorat/guidellm-bench/server_status.json`).

### 23. EP is MoE-Only — Enforced in Code
**RULE**: `is_moe_model(model)` (defined in `config.py`, using `_MOE_MODELS` frozenset) must return True before any `expert_parallel_size` config is built. The `ep_configs` loop in `bench.py` calls this guard and skips non-MoE models with a printed warning. Never add EP configs for dense models (e.g. Qwen3-4B-Thinking).

```python
# MoE models supporting EP:
_MOE_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b", "Qwen/Qwen3-30B-A3B"})
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
| `quant` | `["fp8"]` |
| `num_prompts` | 4 |
| `max_model_len` | 2048 |
| `timeout_startup` | 600s |

## Common Commands

```bash
# Quick sanity check
./bench.py --sanity
# Logs → ./sanity_results/YYYYMMDD_HHMM/bench.log
# PID  → ./sanity_results/YYYYMMDD_HHMM/bench.pid

# Full benchmark suite (background — self-logs; no redirect needed)
nohup ./bench.py > /dev/null 2>&1 &

# Resume an interrupted run (skips configs with existing _benchmarks.json)
./bench.py --resume results/YYYYMMDD_HHMM
# Or: resume the latest run automatically
./bench.py --resume

# Specific model/config
./bench.py --models openai/gpt-oss-20b --tp 4 --quantization none

# With Eagle3 (gpt-oss-120b appended)
./bench.py --eagle3

# With Expert Parallelism for MoE models (requires intel/llm-scaler-vllm:0.14.0-b8+)
./bench.py --ep
# Adds: gpt-oss-20b tp4+ep4, Qwen3-30B-A3B tp2+ep2, Qwen3-30B-A3B tp4+ep4

# EP comparison (non-EP + EP side-by-side in one dashboard) — mutually exclusive with --ep
./bench.py --ep-compare

# Custom HF dataset (auto-detects text column)
./bench.py --data cx-cmu/deepresearchgym-agentic-search-logs

# Long-context input-length sweep (1k/4k/8k/16k, 10 samples each, TTFT chart)
./bench.py --long-contexts

# Combined: gpt-oss-20b EP comparison with long-context slices on deepresearchgym dataset
nohup ./bench.py \
  --models openai/gpt-oss-20b \
  --tp 4 \
  --ep-compare \
  --long-contexts \
  --data cx-cmu/deepresearchgym-agentic-search-logs \
  > /dev/null 2>&1 &

# Sanity test with EP compare + long contexts (single gpt-oss-20b config)
./bench.py --sanity --models openai/gpt-oss-20b --tp 4 --ep-compare --long-contexts

# Rebuild dashboard from completed results
python3 -c "
from bench import build_dashboard_html
from pathlib import Path
out_dir = Path('results/YYYYMMDD_HHMM')
succeeded = ['model_tp4_quant-none_eager-true']
build_dashboard_html(out_dir, succeeded)
"

# Serve dashboard
bash results/YYYYMMDD_HHMM/serve_dashboard.sh
```

## Known Issues

| Issue | Cause | Fix |
|---|---|---|
| TTFT = 0ms on thinking models | Chat template injects thinking tokens | Use `--request-format /v1/completions` (Rule 4) |
| OOM on gpt-oss-20b + eager=false | XPU memory exhausted by graph compilation buffers | Skip (Rule 5) |
| Qwen3-30B + fp8 mismatch | mxfp4 in model config vs fp8 override | Skip (Rule 5) |
| Peak GPU Memory chart empty | xpu-smi enters D state when GPU in use; cannot kill | `parse_model_mem_gib()` parses server log instead; passes `model_mem_gib=` to `build_dashboard_html` |
| Dashboard shows 0 for all metrics | `b['metrics']` aggregates are zero-filled in guidellm v0.6 | Extract medians from `b['requests']['successful']` per-request fields |
| Models generate only 16 output tokens | `output_tokens` column not mapped to `max_tokens`; vLLM default is 16 | Use column name `output_tokens_count` (guidellm default, auto-detected) |
| Dashboard shows only previous run's config | `lsof` not available inside container — prior `http.server` held port 8081; new server failed to bind silently | `_serve_dashboard` now uses `fuser -k {port}/tcp` (available in container) with `lsof` as host fallback |

---

### 26. Israel Timezone — Use `_israel_now()` Not Raw `ZoneInfo`
**RULE**: Never call `datetime.now(ZoneInfo("Asia/Jerusalem"))` directly. Use `_israel_now()` in `bench.py` which wraps it in try/except and falls back to UTC+2 fixed offset if `tzdata` is not installed. This prevents silent UTC fallback that produces wrong timestamps on a fresh container.

### 27. Incomplete Run Auto-Cleanup
**RULE**: On every fresh start (not `--resume`), `_clean_incomplete_runs(results_dir)` removes subdirectories with zero `*_benchmarks.json` files. These are safe to delete (no useful data). The cleanup prints what was removed. The function is in `bench.py`.

### 28. Long-Context Benchmarks — `--long-contexts`
**RULE**: When `--long-contexts` is set:
- `max_model_len` is auto-raised to 16384 if it's lower.
- `prepare_long_context_datasets()` creates JSONL files under `out_dir/datasets/`, one per target length. Cached per run (not globally) — cache key includes source stem and length label.
- Each LC benchmark uses `lc_mode=True` in `run_guidellm()`: synchronous profile, 10 requests, no warmup/cooldown.
- Results saved as `{cfg_name}_lc{N}k_benchmarks.json` (e.g. `..._lc4k_benchmarks.json`).
- LC slices use `output_tokens_count=512` (not 1024) to keep total generation time manageable at 16k input.

### 29. Generic HF Dataset — `--data`
**RULE**: `prepare_hf_dataset(hf_name, ...)` in `dataset.py` auto-detects the text column:
1. Priority list match (prompt/text/input/content/instruction/query/question/document/context/passage/problem...).
2. Fallback: string column with longest median text length (sampled from first 50 rows).
Cached to `out_dir/datasets/{safe_name}_{split}_v1.jsonl`. Output: `{prompt, output_tokens_count}` rows.

### 30. `--ep-compare` vs `--ep`
**RULE**: `--ep-compare` is a superset of `--ep`:
- Adds EP variants (same as `--ep`).
- ALSO ensures the non-EP base configs for those same `model+tp+quant` combos are in the matrix, even if they would have been filtered by `--models`.
- **Mutually exclusive** with `--ep` — validated at startup with `sys.exit()`.
- Both `_EP_VARIANTS` list and the dedup loop live in `bench.py::main()` (not `config.py`).

**Last Updated**: March 3, 2026 — Added --ep-compare, --data, --long-contexts, _israel_now(), _clean_incomplete_runs(); updated for new features
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
| 8 | vLLM server hangs after XPU kernel registration warnings (`OperatorEntry.cpp:208 Warning: Overriding a previously registered kernel`) followed by `Still waiting... Xs elapsed` and never reaches `/health` | XPU driver state is corrupted; D-state processes block even `docker rm -f`. Exit 42 fires; host-side guard **prompts user** before rebooting (non-interactive sessions skip auto-reboot). Resume with `./bench.py --resume` after reboot. See Rule 18. |
| 9 | Named AIME JSONL column `output_tokens` — models generated only 16 tokens | Column must be `output_tokens_count` (guidellm default); `output_tokens` is not mapped to `max_tokens` in the completions body, leaving vLLM's default of 16. Cache path is `/root/aime_2024_v2.jsonl` (container) = `/root/dkorat/aime_2024_v2.jsonl` (host, volume-backed). |
| 10 | Dashboard showed 0 for TTFT/ITL/req/s/tok/s | `b['metrics']` aggregates are zero-filled in guidellm v0.6; must compute medians from `b['requests']['successful'][*]` per-request fields in `_extract_sweep_points`. |
| 11 | gpt-oss-20b + tp=2 crashed with OOM (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY), guidellm reported 30 "successful" requests with empty output | Added `gpt-oss-20b + tp<4` skip rule — model requires at least 4 GPUs |
| 12 | Installed Python deps on the host with `pip install -e .` instead of inside the container | All deps (datasets, guidellm, tzdata) must be installed **inside the container** only. `install.sh` does `docker exec lsv-container pip install -e ".[guidellm]"`. No host-side pip install. |
| 13 | Wrote subprocess calls in server.py/benchmark.py that wrapped tools with `docker_exec_cmd()` | bench.py runs inside the container; call vLLM, guidellm, pkill directly as plain subprocesses with `bash --login -c "{_PREAMBLE} && {cmd}"`. |
| 14 | Set `SANITY.quant=["none"]` — the only sanity config was immediately skipped by the Qwen3-4B+quant=none rule | SANITY must use `quant=["fp8"]` since Qwen3-4B + quant=none is always skipped. |
| 15 | `docker exec` re-exec guard omitted `-w /root/guidellm-bench` — container default cwd is `/workspace/vllm/`, so results landed there instead of the volume-mounted `/root/` | Always pass `-w /root/guidellm-bench` in the re-exec: `docker exec -w /root/guidellm-bench lsv-container python3 /root/guidellm-bench/bench.py` |
| 16 | `_serve_dashboard` used `lsof` to kill the old server, but `lsof` is not available inside the vLLM container — the kill silently no-oped and the new http.server failed to bind, leaving the old (sanity) dashboard visible | Use `fuser -k {port}/tcp 2>/dev/null || lsof -ti tcp:{port} ...` — `fuser` is available in the container |
| 17 | `pkill -f guidellm` in `stop_server()` killed bench.py itself — `/root/guidellm-bench/bench.py` contains "guidellm" in the path | Use `pkill -f 'guidellm benchmark'` to match only the guidellm CLI invocation, not bench.py's own path. |
| 18 | Used `xpu-smi dump` to monitor GPU memory during benchmarks — processes entered `D` (uninterruptible sleep) state when the GPU was in use and could not be killed even with `SIGKILL` | Never call `xpu-smi` while vLLM holds the GPU. Use `parse_model_mem_gib()` to read GPU weight memory from the vLLM server log instead. |
| 19 | Renamed gitignored dirs (e.g. `guidellm_results/` → `results/`) by updating `.gitignore` only — the old directories were no longer ignored and got accidentally committed | When renaming gitignored dirs: (1) add the new name to `.gitignore`, (2) keep the old name too (both must be ignored until old dirs are gone), (3) run `git rm -r --cached <old_dir>` BEFORE committing to untrack any previously committed files |
| 20 | Added `--expert-parallel-size N` to vLLM command — caused `unrecognized arguments` error | `intel/llm-scaler-vllm:0.14.0-b8` only supports `--enable-expert-parallel` (boolean flag). There is NO `--expert-parallel-size` parameter. `cfg.expert_parallel_size` is used as a boolean trigger only; emit `--enable-expert-parallel` with no value. |
| 21 | Health-check curl routed through Intel proxy (http_proxy forwarded via docker exec -e) → all curl calls to localhost:8000 silently failed, keeping wait_for_server stuck forever | Always set `no_proxy=localhost,127.0.0.1,0.0.0.0` and `NO_PROXY=...` in the env dict passed to any subprocess that calls curl against localhost. Already done in `wait_for_server` via `env=dict(os.environ, no_proxy="...", NO_PROXY="...")`. |
| 22 | `pip install -e ".[guidellm]"` inside container failed: "externally-managed-environment" (PEP 668) | `intel/llm-scaler-vllm:0.14.0-b8` uses a system-managed Python. Always pass `--break-system-packages` to pip inside this container. |
| 23 | `pip install -e "/root/guidellm-bench[guidellm]"` — path missing `.` → installed the package but without the `[guidellm]` extra, so guidellm itself was not installed | Correct form is `pip install --break-system-packages -e "/root/guidellm-bench/.[guidellm]"` (note the `/./`). |
| 24 | `docker run -e http_proxy="$PROXY"` where PROXY was set using `${http_proxy:-fallback}` in a non-login shell — the outer env var wasn't available to the script, resulting in an empty-proxy container | Always pass literal proxy values or verify the env var is set before using it as a docker -e argument. In `ensure_container_running()` this is handled with `os.environ.get("http_proxy", "http://proxy-dmz.intel.com:911/")`. |
| 25 | `nohup ./bench.py &` without stdout redirect creates `nohup.out` at the repo root — even though bench.py tees via `_Tee`, `nohup` creates the file before Python starts. | Always run `nohup ./bench.py > /dev/null 2>&1 &`. bench.py self-logs everything to `out_dir/bench.log`, so `/dev/null` discards nothing important. See Rule 25 / Rule 19. |
| 26 | `datetime.now(ZoneInfo("Asia/Jerusalem"))` silently falls back to UTC when `tzdata` is not installed, producing wrong directory timestamps. | Use `_israel_now()` which wraps in try/except and falls back to UTC+2 fixed offset. See Rule 26. |
| 27 | Long-context output tokens set to 1024 when input is up to 16384 — causes context overflow on large inputs (prompt + max_tokens > max_model_len). | Use `output_tokens_count=512` for LC slices (half of 1024), giving headroom. See Rule 28. |
| 28 | LC dataset cache stored globally in `/root/` — filename collision if two different source datasets happen to have the same stem. | Store LC JSONL files under `out_dir/datasets/` (per-run); include source stem in filename. |
| 29 | `Config.name` used `-ep{N}` (e.g. `-ep4`, `-ep2`) — numeric EP value in filenames conflates TP and EP degrees and violates the "EP is boolean" principle. | `ep_suffix` in `Config.name` is just `-ep` (no number). `expert_parallel_size` is a boolean trigger; the degree is implied by `tp`. |
