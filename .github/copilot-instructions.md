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
│   ├── agent/                    # deep-research agent subpackage (CONCURRENCY=1 demo)
│   │   ├── __init__.py           #   re-exports all public API
│   │   ├── constants.py          #   AGENT_MAX_MODEL_LEN, AGENT_MAX_BATCHED, CONCURRENCY, MATRIX_*, dataclasses
│   │   ├── debug.py              #   file-backed debug logger
│   │   ├── helpers.py            #   make_session, _tokenize/_detokenize, _warm_cache, _measure_ttft
│   │   ├── corpus.py             #   Corpus class, _prepare_frames_corpus, _find_arxiv_fallback
│   │   ├── matrix.py             #   measure_cell, run_ttft_matrix, _save_matrix_checkpoint
│   │   ├── scenarios.py          #   ReAct loop, run_research_session, run_agent_scenarios_frames
│   │   └── run.py                #   run_agent_bench, get_agent_server_config
│   ├── agent_bench.py            # backward-compat shim → re-exports from guidellm_bench.agent
│   └── dashboard.py              # build_dashboard_html() and write_serve_script()
├── verify_pc.py                  # Verify PC artifact: confirms LC dataset cross-run KV-cache seeding
├── results/                      # Created at runtime (full runs) — gitignored
│   └── YYYYMMDD_HHMM/
│       ├── {cfg_name}_benchmarks.json
│       ├── {cfg_name}_benchmarks.html
│       ├── {cfg_name}_lc{N}k_benchmarks.json  # long-context slices (--long-contexts)
│       ├── dashboard.html
│       ├── serve_dashboard.sh
│       ├── datasets/                           # cached JSONL files for this run
│       └── logs/
├── ablation_results/             # Created at runtime (--ablation runs) — gitignored
│   └── YYYYMMDD_HHMM/
│       ├── ablation_dashboard.html
│       ├── serve_ablation_dashboard.sh
│       ├── {cfg_name}_lc{N}k_benchmarks.json  # one per config × LC length
│       ├── datasets/                           # lc_*_v2_{N}k.jsonl — non-overlapping subsets
│       ├── bench.log
│       ├── bench.pid
│       └── logs/
├── agent_results/                # Created at runtime (--agent runs) — gitignored
├── throughput_results/           # Created at runtime (--throughput runs) — gitignored
│   └── YYYYMMDD_HHMM/
│       ├── throughput_dashboard.html
│       ├── serve_throughput_dashboard.sh
│       ├── {cfg_name}_c{c}_il{il}k_benchmarks.json  # one per server×concurrency×input_len
│       ├── datasets/                                 # throughput_*_{N}k_v1.jsonl
│       ├── bench.log
│       ├── bench.pid
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
    # Auto-start container if stopped (e.g. after host reboot — no --restart policy).
    _state = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", "lsv-container"],
        capture_output=True, text=True,
    ).stdout.strip()
    if _state != "true":
        print("[bench] lsv-container is not running — starting it...", flush=True)
        _start_rc = subprocess.call(["docker", "start", "lsv-container"])
        if _start_rc != 0:
            print("[bench] ERROR: failed to start lsv-container", flush=True)
            sys.exit(1)
        time.sleep(3)  # let container initialise
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
| `agent/` | deep-research agent subpackage (constants, debug, helpers, corpus, matrix, scenarios, run) |
| `agent_bench.py` | backward-compat shim — re-exports everything from `guidellm_bench.agent` |
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
- `wait_for_server()` scans the server log every 60s.
- Hang is declared only when **both** conditions hold after `_HANG_DETECT_AFTER_S` (300s):
  1. `OperatorEntry.cpp:208` is present in the log (driver registered XPU kernels)
  2. **No** `EngineCore` or `Worker_TP` lines exist (workers never started)
  - `OperatorEntry.cpp:208` alone is **not** a hang — it appears within the first ~5s of every healthy startup too.
- `_log_has_xpu_hang()` in `server.py` implements this two-condition check.
- The container-side bench.py catches `XpuKernelHangError` and exits with **code 42**.
- The host-side re-exec guard detects exit code 42, prints a warning, and **prompts for confirmation** before calling `reboot`.
- In non-interactive sessions (nohup), the reboot is skipped and must be done manually.

After reboot, resume manually:
```bash
./bench.py --resume   # picks up the latest run dir automatically
```

Do NOT attempt container recreation — D-state vllm processes block `docker rm -f` even with SIGKILL.

### 21b. Auto-Resume Systemd Service
**RULE**: Every new run automatically installs a systemd service so it resumes after an unexpected reboot/power-cycle without any manual action.
- `_write_resume_script(out_dir, argv)` in `bench.py` writes `{out_dir}/resume.sh` (chmod 755) and calls `_install_resume_service()`.
- `_install_resume_service(resume_script)` writes `/etc/systemd/system/guidellm-resume.service` and runs `systemctl daemon-reload && systemctl enable`.
- Service fires after `docker.service` is ready; starts the container if stopped, then `nohup ./bench.py --throughput --resume {out_dir} > /dev/null 2>&1 &`.
- `_disable_resume_service()` is called at the **end** of every run mode (`_run_throughput`, `_run_ablation`, main sweep) to prevent the service from re-firing on the next normal boot.
- Service file: `/etc/systemd/system/guidellm-resume.service`. Check with: `systemctl is-active guidellm-resume.service`.

### 21c. guidellm Watchdog Timer
**RULE**: `run_guidellm()` in `benchmark.py` installs a `threading.Timer` hard-kill at `max_seconds + 120s`. If guidellm doesn't finish before that deadline, the subprocess is SIGKILLed unconditionally. This prevents the 17h hang observed when vLLM crashes mid-run and guidellm retries forever against a dead backend, ignoring `--max-seconds`.
```python
_hard_limit = max_seconds + 120
_timer = threading.Timer(_hard_limit, lambda: proc.kill())
_timer.start()
try:
    proc.wait()
finally:
    _timer.cancel()
```

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

# Ablation study: optimal vLLM config for gpt-oss-20b on Intel XPU
# 6 configs × 4 input lengths (1k/2k/4k/8k), 5 samples each
# Ablation dims: TP=4 vs TP=8, EP, async-scheduling, prefix-caching
# Output → ./ablation_results/YYYYMMDD_HHMM/ablation_dashboard.html
./bench.py --ablation

# Ablation with custom dataset (auto-discovers from last run if omitted)
./bench.py --ablation --data cx-cmu/deepresearchgym-agentic-search-logs

# Throughput study: 2 server configs (tp8+async and tp8+async+EP)
# 4 concurrencies (c=1/16/64/128) × 4 input lengths (16k/32k/48k/96k) × output=16k
# PC disabled; exactly 1 server restart; samples = 2×c (10/32/128/256)
# Output → ./throughput_results/YYYYMMDD_HHMM/throughput_dashboard.html
./bench.py --throughput

# Throughput with custom dataset (auto-discovers arxiv-summarization if omitted)
./bench.py --throughput --data ccdv/arxiv-summarization

# Throughput run in background (self-logs; ~12 hours wall-clock)
nohup ./bench.py --throughput > /dev/null 2>&1 &

# Resume an interrupted throughput run
./bench.py --throughput --resume throughput_results/YYYYMMDD_HHMM

# Agent benchmark: TTFT matrix (24 cells) + 4 scenario simulations
# Server: gpt-oss-20b tp8+async+PC, max_model_len=200k, PC=True
# Results → ./agent_results/YYYYMMDD_HHMM/agent_bench_results.json
nohup ./bench.py --agent > /dev/null 2>&1 &

# Agent: matrix only (skip multi-turn scenario simulations)
./bench.py --agent --skip-scenarios

# Agent: scenarios only (skip the 24-cell TTFT matrix)
./bench.py --agent --skip-matrix

# Agent: use custom corpus
./bench.py --agent --data ccdv/arxiv-summarization

# Resume interrupted agent run
./bench.py --agent --resume agent_results/YYYYMMDD_HHMM

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
| Dashboard shows 0 for all metrics | `b['metrics']` aggregates zero-filled because `GenerativeMetrics.compile()` used `or 0.0` on per-request lambdas, coercing `None` to `0.0` before `from_values_function` could skip it | Root cause fixed in installed fork (`metrics.py`); workaround in `guidellm_bench/dashboard.py` reads per-request fields directly and remains as defence-in-depth |
| Models generate only 16 output tokens | `output_tokens` column not mapped to `max_tokens`; vLLM default is 16 | Use column name `output_tokens_count` (guidellm default, auto-detected) |
| Dashboard shows only previous run's config | `lsof` not available inside container — prior `http.server` held port 8081; new server failed to bind silently | `_serve_dashboard` now uses `fuser -k {port}/tcp` (available in container) with `lsof` as host fallback |

---

### 31. Ablation Mode (`--ablation`)
**RULE**: `--ablation` is a focused study for `gpt-oss-20b` Intel XPU optimization, split into two phases:

#### Phase 1 — c=1 (latency/quality, all 9 configs × 4 LC lengths)
- 5 samples per length (synchronous profile), 512 output tokens
- Input lengths: `[1024, 2048, 4096, 8192]` (`ABLATION_LC_LENGTHS` in `config.py`)
- Results saved as: `{cfg_name}_lc{N}k_benchmarks.json`
- 9 predefined configs from `get_ablation_configs()` in `config.py`:
  1. Baseline: `tp=4, quant=None, eager=True` (MXFP4 native, Intel defaults)
  2. +EP: `tp=4, expert_parallel_size=4` (`--enable-expert-parallel`)
  3. +TP8: `tp=8` (wider tensor parallelism)
  4. +TP8+EP: `tp=8, expert_parallel_size=8`
  5. +Async: `tp=4, async_scheduling=True` (`--async-scheduling`, Intel 0.14.1-xpu)
  6. +PC: `tp=4, prefix_caching=True` (KV-cache reuse, removes `--no-enable-prefix-caching`)
  7. +TP2: `tp=2, max_model_len_override=8192` (memory-constrained, LC capped at 4k)
  8. +Async+PC: `tp=4, async_scheduling=True, prefix_caching=True` (combined best)
  9. +TP8+Async+PC: `tp=8, async_scheduling=True, prefix_caching=True`
  10. +Eagle3: `tp=4, speculative_config=EAGLE3_20B_SPECULATIVE_CONFIG`

#### Phase 2 — c=16 (throughput scale, top-5 + EP + Eagle3, same 4 LC lengths)
- 20 samples per length (concurrent profile, `--profile concurrent --rate 16`), 512 output tokens
- Constants: `ABLATION_C16_CONCURRENCY=16`, `ABLATION_C16_SAMPLES=20` in `config.py`
- Results saved as: `{cfg_name}_c16_lc{N}k_benchmarks.json`
- Config selection (`_is_c16_config()` in `bench.py`):
  - ✅ All EP configs (`expert_parallel_size` set)
  - ✅ Eagle3 (`speculative_config` set)
  - ✅ Top-5 non-EP: baseline(tp4), tp8, async(tp4), asyncpc(tp4), tp8+asyncpc
  - ❌ tp2 (`max_model_len_override` set — memory constraint at c=16)
- **Ordering fix**: c16 data is loaded BEFORE `_generate_conclusions()` is called — so the EP/c16 insight cards in the Conclusions tab correctly read c16 metrics (old code had this backwards)

#### Dashboard tabs
- **Overview** (existing): 4 LC line charts for all c=1 configs
- **&#9889; Throughput (c=16)** (NEW): 4 LC line charts + 8k snapshot bar charts for c=16 configs
- **Conclusions** (merged): c=1 tuning insights + c=16 EP/throughput insights side-by-side

#### Other rules
- Results → `./ablation_results/YYYYMMDD_HHMM/` (separate from `./results/`)
- Dataset: `--data` if provided, else auto-discovered from `./results/*/datasets/*.jsonl`, else AIME 2024
- `_run_ablation()` in `bench.py` handles the loop; `build_ablation_dashboard_html()` in `dashboard.py` builds the output
- `--ablation` and `--resume` are compatible (c1 and c16 checkpoint files both checked for resume skipping)
- `--ablation` is **not** combinable with `--sanity`, `--ep`, `--ep-compare` (uses its own config matrix)

### 32. New Config Fields: `async_scheduling` and `prefix_caching`
**RULE**: `Config` dataclass has two new boolean fields:
- `async_scheduling: bool = False` → emits `--async-scheduling` in `build_vllm_cmd()`
- `prefix_caching: bool = False` → when `True`, omits `--no-enable-prefix-caching` (default is always added)
Both fields are included in `_cfg_to_status_key()` so server reuse correctly detects config changes.
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

### 34. Agent Benchmark (`--agent`)
**RULE**: `--agent` measures two things for `gpt-oss-20b tp8+async+PC`:

#### Part 1 — TTFT matrix (24 cells)
- 6 cached-context sizes × 4 new-token sizes
- `MATRIX_N_CACHED = [0, 8k, 32k, 64k, 96k, 112k]` tokens (accumulated KV cache)
- `MATRIX_N_NEW = [1k, 4k, 8k, 16k]` tokens (new tokens per iteration)
- max cell: 114,688 + 16,384 = **131,072 = AGENT_MAX_MODEL_LEN** (exact fit)
- 15 samples/cell + 3 warm-up discards; re-runs cell if CV > 0.35
- Cache warming protocol: `prefix_prompt → max_tokens=1` before each measured request
- Token-exact prompt construction: `Corpus` class calls `/tokenize` once and `/detokenize` per slice
- Matrix corpus: FRAMES Wikipedia articles (`_prepare_frames_corpus()`)

#### Part 2 — Real ReAct agent scenarios (FRAMES benchmark)
**RULE**: Scenarios are NOT emulation. They use a real ReAct loop running against the LLM:
- Dataset: `google/frames-benchmark` — 824 multi-hop research questions, each bundled with 2-15 gold Wikipedia articles (the actual search results a real agent would retrieve)
- 4 questions selected at evenly-spaced percentiles of total article content length (short → deep)
- Per scenario / per iteration:
  1. Build prompt = `AGENT_SYSTEM_PROMPT + question + conversation_so_far + "Next action:"`
  2. Warm KV cache with prefix (non-streaming, max_tokens=1)
  3. Single streaming call → TTFT measured + full generated text captured in ONE pass
  4. Parse JSON action from LLM output: `{"action":"search","query":"..."}` or `{"action":"answer","text":"..."}`
  5. If search: retrieve best-matching Wikipedia article via keyword recall over gold docs; append to conversation
  6. If answer: done. Max 20 iterations.
- `_keyword_recall(query, doc)` — fraction of query words in doc (stdlib only, no deps)
- `_find_best_doc(query, docs, used)` — returns highest-scoring unused doc
- `_parse_json_action(raw)` — extracts first `{...}` JSON object; falls back to implicit search

#### Implementation
- `guidellm_bench/agent/` — all logic (constants / debug / helpers / corpus / matrix / scenarios / run)
- `_run_agent()` in `bench.py` — server lifecycle + dispatch
- Server: `tp=8, async_scheduling=False, prefix_caching=True`
- `AGENT_MAX_MODEL_LEN = 131_072`, `AGENT_MAX_BATCHED = 8_192`, `CONCURRENCY = 1`
- `AGENT_MAX_BATCHED=8_192` keeps vLLM's warm-up dummy kernel within Intel XPU 256KB PTSS limit;
  131k-token prompts are served via chunked prefill (16 × 8k passes) — no kernel size issue
- Dataset: auto-builds FRAMES corpus via `_prepare_frames_corpus(out_dir)` (downloads once, cached per run dir)
  - **FRAMES has no article text** — only URLs (`wikipedia_link_1`…`11+`). Corpus and scenarios both
    call `_parse_frames_urls(row)` + `_fetch_wikipedia_text(url)` (Wikipedia Action API, reachable via Intel proxy).
- Results → `./agent_results/YYYYMMDD_HHMM/agent_bench_results.json`
- Supports `--resume`, `--skip-matrix`, `--skip-scenarios`

#### TTFT model (for cache-hit ratio / speedup reporting)
`cold_est = 38.3 + 62.0·N_total_k + 0.731·N_total_k²` ms  (fitted to no-PC data 1k–48k)
Speedup vs cold = `cold_est(N_new, N_cached) / measured_TTFT_with_PC`

### 33. Throughput Study (`--throughput`)
**RULE**: `--throughput` is a concurrency × input_length sweep for `gpt-oss-20b` on Intel XPU.

#### Server configurations
| Name | tp | async | EP | Concurrencies |
|---|---|---|---|---|
| Server A | 8 | ✓ | ✗ | 1, 16, 64, 128 |
| Server B | 8 | ✓ | ✓ | 16, 64, 128 |

**Exactly 1 server restart** (Server A finishes ALL concurrencies before Server B starts).

#### Constants (in `config.py`)
| Constant | Value |
|---|---|
| `THROUGHPUT_INPUT_LENGTHS` | `[16384, 32768, 49152, 98304]` (16k/32k/48k/96k) |
| `THROUGHPUT_OUTPUT_LEN` | `16384` (16k) |
| `THROUGHPUT_CONCURRENCIES` | `[1, 16, 64, 128]` |
| `THROUGHPUT_MAX_MODEL_LEN` | `131072` |
| `THROUGHPUT_MAX_NUM_BATCHED_TOKENS` | `131072` |
| `THROUGHPUT_SAMPLES` | `{1:10, 16:32, 64:128, 128:256}` (2×c rule) |
| `THROUGHPUT_MAX_SECONDS` | `10800` (3h per cell) |

#### Implementation
- `get_throughput_configs()` in `config.py` → Server A + Server B
- `prepare_throughput_dataset(source, input_len, output_len, num_samples, cache_dir)` in `dataset.py` → cyclic concatenation, safe because PC is disabled
- `_run_throughput()` in `bench.py` → outer loop = server config, inner loop = concurrency then input_len
- `build_throughput_dashboard_html(out_dir, succeeded)` in `dashboard.py` → 5 tabs: c=1/16/64/128 + Concurrency Effects
- File naming: `{cfg_name}_c{c}_il{il//1024}k_benchmarks.json`
- Results → `./throughput_results/YYYYMMDD_HHMM/`

#### Throughput dashboard tabs
- **c=1 (Latency)**: TTFT / ITL / Req/s / Tok/s vs input_len, no-EP only
- **c=16, c=64, c=128**: same 4 charts, no-EP + EP lines
- **Concurrency Effects**: TTFT and Tok/s vs concurrency [1→128], one line per (config × input_len); EP lines start at c=16

#### `max_num_batched_tokens` for large-context studies
**RULE**: Set `max_num_batched_tokens ≥ max_input_len` when running large-context (16k+) benchmarks. The default of `8192` causes vLLM to chunk a 96k-token prefill over ~12 forward passes, serialising the KV-cache write and inflating TTFT by 10-50×. `THROUGHPUT_MAX_NUM_BATCHED_TOKENS = 131072` matches the full context window.

**Last Updated**: March 8, 2026 — Upgraded `--agent` to real deep-research agent using `google/frames-benchmark`: real ReAct loop (LLM generates JSON search/answer actions; keyword retrieval over gold Wikipedia articles), FRAMES corpus for matrix, max_model_len=131_072 (was 200k), MATRIX_N_CACHED updated to [0,8k,32k,64k,96k,112k] (max cell=131072 exact fit). Old emulation (`AGENT_SCENARIOS` fixed-token loop) removed. Rule 34 fully rewritten. Previous: March 8, 2026 — Added `--agent` deep-research agent benchmark in the installed guidellm fork (Lesson 45): removed `or 0.0` from TTFT/ITL/latency lambdas in `GenerativeMetrics.compile()` so `None` propagates to `from_values_function`'s built-in None-skip. Fix applied to installed copy in container; must be committed to `fix/thinking-model-ttft` branch to survive reinstall. Previous: March 9, 2026 — Fixed empty throughput dashboard plots (Lesson 44: `chart_js_lines` missing `<script>` wrapper in `build_throughput_dashboard_html`; added dashboard HTML verification rule). Added `--no-ep` CLI flag to skip Server B (EP) in throughput runs. Previous: Added `--throughput` mode: `_run_throughput()` in `bench.py`, `build_throughput_dashboard_html()` in `dashboard.py`, `prepare_throughput_dataset()` in `dataset.py`, `get_throughput_configs()` + THROUGHPUT_* constants in `config.py`; 2 server configs (tp8+async, tp8+async+ep); exactly 1 restart; 4 concurrencies × 4 input lengths × 1 output length; 2×c samples; max_num_batched_tokens=131072; throughput_dashboard.html with per-concurrency tabs + Concurrency Effects tab; results → `./throughput_results/`; Rule 33 added; repo structure + Common Commands updated.
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
| 30 | Dashboard subtitles didn't show which container image or vLLM command was used, making results hard to reproduce. | `start_server()` writes the full `vllm serve …` command to `logs/{cfg_name}_vllm_cmd.txt`. Both `build_dashboard_html()` and `build_ablation_dashboard_html()` read that file and include **`Docker: intel/llm-scaler-vllm:0.14.0-b8`** plus the full vllm command string in every config tab subtitle. The global page subtitle always shows the docker image. |
| 31 | Intel blog (0.10.2) showed gpt-oss-20b at tp=1; we OOM at tp<4 on 0.14.0-b8. | Different vLLM version (0.14.0-b8 pre-allocates far more KV-cache memory), higher `--max-model-len=16384`, and larger `--max-num-batched-tokens=8192`. The blog likely used a much smaller context/batch. Do NOT add tp=2 to the skip-exempt list without first testing with a smaller `--max-model-len`. |
| 33 | Eagle3 ablation config crashed with `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE`: `sample_recovered_tokens_kernel` needs 292KB PTSS but XPU max is 256KB. | Eagle3 speculative decoding is unsupported on `intel/llm-scaler-vllm:0.14.0-b8` XPU hardware. Removed from `get_ablation_configs()`. Do not re-enable without a driver/firmware upgrade. |
| 32 | `--ablation` without `--data` used AIME (30 short math problems) instead of the arxiv-summarization dataset from the last run — all LC slices skipped (0 prompts ≥ min word count). | `main()` unconditionally called `prepare_aime_dataset()` before reaching the ablation branch, so `dataset_path` was never `None` and `_run_ablation`'s `_find_last_run_dataset()` discovery was dead code. Fix: skip AIME pre-load when `--ablation` is set and no `--data` given (`dataset_path = None`), letting `_run_ablation` auto-discover from `./results/*/datasets/`. |
| 34 | When `--resume`ing an ablation run on a freshly started machine, Worker_TP2 OOMed during model load of the first new config (`tp4+async-pc`) even though not all GPUs were used by the previous config (`tp2`). Process exit as seen by `pgrep` ≠ VRAM released by XPU driver. Also observed: if graceful SIGTERM shutdown times out and SIGKILL is used, xe-destroy-wq kernel workers take much longer to clear — starting a new server within 13s caused XPU kernel registration hang (OperatorEntry.cpp:208). | `stop_server()` tracks `_sigkill_used`. If SIGTERM succeeded cleanly → `time.sleep(10)`. If SIGKILL was needed → `time.sleep(60)`. This gives the XPU driver enough time to release device handles in both cases. |
| 35 | `build_ablation_dashboard_html(out_dir, succeeded)` produced N/A in the recommendation/conclusions panel when the baseline config (`tp4_quant-none`) had a server startup failure in the current run (even though its LC JSON files existed from a prior run). `lc_data` was built only from `succeeded`, so baseline was missing → `at_len(baseline, ...)` returned None everywhere. | Fixed in `build_ablation_dashboard_html`: scan `out_dir` for all existing `*_lc*_benchmarks.json` files and merge with `succeeded`. Any config whose data is on disk is included regardless of whether it's in `succeeded`. |
| 36 | Ablation PC config showed ~40% TTFT reduction at lc_2k/lc_4k/lc_8k — appeared to be a real prefix-caching benefit, but the prompts are all unique (no shared system prefix). | **Benchmark design artifact.** `prepare_long_context_datasets()` used `eligible[:N]` for every target length. Because the same long source articles qualify for ALL shorter lengths, 4-5/5 documents reappeared across consecutive LC slices (truncated from position 0). Running lc_1k→lc_2k→lc_4k→lc_8k in order with PC enabled seeds the KV cache for each subsequent run (4/5 shared docs × 50% token overlap = 40% effective TTFT reduction — exactly matching observed data). **Fix applied**: `prepare_long_context_datasets()` now maintains a `used_keys` set across target lengths processed in ascending order, ensuring each source document is assigned to at most one LC slice. Output files renamed to `*_v2.jsonl` to force cache invalidation. Verified with `verify_pc.py` (all 4 LC lengths ✓ CONFIRMED). **Rule**: any LC benchmark comparing configs with PC enabled/disabled MUST use non-overlapping document sets across target lengths. |
| 37 | Resumed an ablation run (`--resume`) after applying the `dataset.py` fix mid-run. The baseline config was already complete (from the original contaminated run) and was skipped by resume logic. Non-baseline configs ran with the new v2 non-overlapping datasets. The resulting dashboard **silently mixed** contaminated baseline data with clean non-baseline data — an invalid comparison that is harder to detect than obvious failure. | After ANY dataset-layer fix (rename, logic change, bug fix), **do NOT resume** existing runs. Always start a completely fresh ablation run so every config uses the same corrected dataset. Old result directories with mixed-generation data must not be used for cross-config comparison. |
| 38 | The old ablation throughput tab ran a flat single-benchmark (no LC lengths, bar charts only, 4 configs) and called `_generate_conclusions(lc_data, throughput_data=throughput_data)` BEFORE the `throughput_data` dict was loaded — so EP insight notes always showed "n/a". | (1) Replace flat throughput benchmark with C=16 PHASE: 8 configs × 4 LC lengths → line charts (same structure as c=1 Overview tab) + 8k snapshot bars. (2) Always load `c16_data` **before** calling `_generate_conclusions()`. Files named `{cfg_name}_c16_lc{N}k_benchmarks.json`. Constants: `ABLATION_C16_CONCURRENCY=16`, `ABLATION_C16_SAMPLES=20`. |
| 39 | After host reboot, `lsv-container` stops (no `--restart always` policy). The host-side re-exec guard called `docker exec lsv-container ...` directly — this fails silently with "container is not running", PID exits with non-zero code, stdout was `/dev/null`, and no log was ever created. | The re-exec guard now checks `docker inspect --format {{.State.Running}} lsv-container` first; if not `true`, runs `docker start lsv-container` + `time.sleep(3)` before the `docker exec`. No manual `docker start` required after a reboot. |
| 40 | `_log_has_xpu_hang` only checked for the presence of `OperatorEntry.cpp:208`, which is ALWAYS emitted by the main vLLM process within the first ~5s of ANY startup (hang or not). At `_HANG_DETECT_AFTER_S=120s`, any server still loading (gpt-oss-20b tp=8 takes ~90–150s) was falsely declared hung and killed, even though the XPU was healthy. Confirmed: OPT-125m takes ~37s from `OperatorEntry` to `Application startup complete`, gpt-oss-20b tp=8 takes significantly longer. | The real hang signature is **workers never start**: only 1 OperatorEntry occurrence, no `EngineCore` or `Worker_TP` lines. Fixed `_log_has_xpu_hang` to require BOTH: (1) OperatorEntry.cpp:208 present AND (2) no `EngineCore\|Worker_TP` lines. Raised `_HANG_DETECT_AFTER_S` from 120 → 300s as an additional safety floor. Never declare a hang based on OperatorEntry alone. |
| 41 | After reboot, `docker start lsv-container` was called while the XPU driver was still initialising. `--device /dev/dri:/dev/dri` captures only the device nodes present at container-start time. Driver initialises GPUs sequentially, so only 2 of 8 `renderD` nodes existed yet → container started with 2 GPUs → `tp=8` crashed with "device index out of range". | Added GPU count check in the re-exec guard: after ensuring the container is running, compare `ls /dev/dri/renderD* \| wc -l` on host vs inside the container. If mismatch, stop+rm the container and re-run `install.sh --skip-xpu-smi` to recreate it with the full device list. |
| 42 | guidellm hung for 17 hours on `c=64/il=16k` after vLLM crashed mid-run. guidellm retried in a loop against the dead backend, silently ignoring `--max-seconds` (which only covers time-between-requests, not total wall-clock when the server is completely unreachable). | Added `threading.Timer(_hard_limit, lambda: proc.kill())` in `run_guidellm()` with `_hard_limit = max_seconds + 120`. If the subprocess hasn't finished within that wall-clock window it is SIGKILLed unconditionally. See Rule 21c. |
| 43 | `_log_has_xpu_hang` declared a hang whenever `OperatorEntry.cpp:208` appeared in the log past 120s — but this warning is emitted by the main vLLM process within the first ~5s of **every** normal startup. gpt-oss-20b tp=8 takes 90–150s to fully load, so it was always killed as a false positive. Confirmed by running OPT-125m directly inside the container (same single OperatorEntry, then successful startup ~37s later). | Real hang = OperatorEntry present **AND** no `EngineCore`/`Worker_TP` lines (workers never registered). Raised `_HANG_DETECT_AFTER_S` 120 → 300s. See updated Rule 18 and Lesson 40. |
| 44 | `build_throughput_dashboard_html` had two bugs that both produced empty plots: (1) `chart_js_lines` appended to `conc_tab_panes` without `<script>` tags — browser treated them as text nodes. (2) `makeLineChart` function defined in a `<script>` at the **bottom** of the page, after all the tab pane `<script>` blocks that call it — browser executes inline scripts as it parses, so every call threw `makeLineChart is not defined`. | (1) Wrap chart JS in `<script>(function(){...})();</script>` inside the f-string. (2) Define `makeLineChart` in `<head>`, before any body content. **Verification rule**: `h = open(...).read(); assert h.find('function makeLineChart') < h.find('makeLineChart("thr-')` — function must appear before its first call. Also check `grep -c "makeLineChart" dashboard.html` > 0. Only open in browser after both pass. |
| 45 | `b['metrics']` aggregates (TTFT, ITL, latency, tpot) were all zero for thinking models. Root cause: `GenerativeMetrics.compile()` in guidellm's `metrics.py` used `lambda req: req.time_to_first_token_ms or 0.0` — the `or 0.0` coerces `None` to `0.0` **before** `from_values_function` sees it. `from_values_function` already skips `None` via `continue`, but never gets the chance because `None` is gone. | Fixed in the installed fork at `/usr/local/lib/python3.12/dist-packages/guidellm/benchmark/schemas/generative/metrics.py`: (1) `request_latency or 0.0` → `req.request_latency`; (2) `time_to_first_token_ms or 0.0` → `req.time_to_first_token_ms`; (3) `time_per_output_token_ms` tuple: add `if req.time_per_output_token_ms is not None else None` guard; (4) `inter_token_latency_ms` tuple: add `if req.inter_token_latency_ms is not None else None` guard. **This fix will be lost if guidellm is reinstalled** — it must be committed to the `fix/thinking-model-ttft` branch of the fork and re-installed. Applied via `/tmp/fix_metrics.py` on 2026-03-08. |
| 46 | Agent benchmark crashed with `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` when `AGENT_MAX_BATCHED=131_072` (equal to `max_model_len`). vLLM's warm-up dummy run builds a Triton kernel matching `max_num_batched_tokens`; at 131k tokens the kernel exceeds Intel XPU PTSS 256KB limit. | Fix: `AGENT_MAX_BATCHED = 8_192` (default). vLLM serves 131k-token prompts safely via chunked prefill (16 passes × 8k tokens each). Never set `max_num_batched_tokens = max_model_len` on Intel XPU when `max_model_len > ~32k`. The throughput study uses the same pattern: `THROUGHPUT_MAX_MODEL_LEN=131_072`, `THROUGHPUT_MAX_NUM_BATCHED_TOKENS=8_192`. |
| 47 | After a failed tp=8 run (wrong GPU count), the Inductor kernel cache at `/tmp/torchinductor_root/` inside the container held stale XSO binaries compiled for a different hardware state. The next run (tp=4) tried to load those cached binaries, hit `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` from `Worker_TP0` during `load_by_key_path`, and crashed — even with `--enforce-eager` (IPEX still JIT-compiles pointwise fusion kernels via Inductor on first use). | Clear the Triton/Inductor cache whenever a `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` crash occurs in a Worker process (not APIServer): `docker exec lsv-container rm -rf /tmp/torchinductor_root /tmp/triton_cache`. The stale cache is always safe to delete — vLLM recompiles on the next startup. |
| 48 | `google/frames-benchmark` does **not** contain Wikipedia article text — only URLs (`wikipedia_link_1` … `wikipedia_link_11+` and `wiki_links` columns). Code that accessed `wiki_doc`/`wiki_docs`/`documents` columns got `None` for every row → corpus and scenarios both built 0 entries → `Corpus too small: 0 tokens`. | Use `_parse_frames_urls(row)` to extract URLs and `_fetch_wikipedia_text(url)` to fetch full article text via Wikipedia Action API (`/w/api.php?action=query&prop=extracts&explaintext=1`). Wikipedia is reachable from the container via Intel proxy. Both helpers are in `guidellm_bench/agent/helpers.py`. |
