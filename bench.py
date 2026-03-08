#!/usr/bin/env python3
"""
guidellm-bench — entry point.

Usage:
    ./bench.py               # full benchmark matrix
    ./bench.py --sanity      # single config, fast smoke-test
    ./bench.py --ep          # include Expert Parallelism variants
    ./bench.py --ep-compare  # run EP-capable models WITH and WITHOUT EP for comparison
    ./bench.py --data cx-cmu/deepresearchgym-agentic-search-logs  # custom HF dataset
    ./bench.py --long-contexts  # run 1k/4k/8k/16k input-token slices + TTFT chart
    ./bench.py --models openai/gpt-oss-20b --tp 4 --quantization none

When called from the host this script auto-relaunches itself inside
lsv-container (intel/llm-scaler-vllm:0.14.0-b8) via 'docker exec'.
See guidellm_bench/ for implementation details.
"""

# ---------------------------------------------------------------------------
# Container guard — re-exec inside the container when called from the host.
# Must use only stdlib and happen before any guidellm_bench imports.
# ---------------------------------------------------------------------------
import os
import subprocess
import sys
import time

if not os.path.exists("/.dockerenv"):
    _tty = ["-t"] if sys.stdout.isatty() else []
    # Forward host proxy env vars into the container so HuggingFace / Intel
    # registries remain reachable regardless of how the container was created.
    _proxy_args: list[str] = []
    for _var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        _val = os.environ.get(_var)
        if _val:
            _proxy_args += ["-e", f"{_var}={_val}"]
    # Ensure container is running (it stops after host reboot with no --restart policy).
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
        # Give container a moment to fully initialise before exec-ing into it.
        time.sleep(3)

    # After the container is running, verify its GPU count matches the host.
    # Race condition after reboot: the XPU driver creates device nodes
    # sequentially; if docker start (or docker run) happens while the driver
    # is still initialising, the container captures only the already-created
    # nodes (e.g. 2 instead of 8).  Detect this and recreate the container.
    _host_gpus = subprocess.run(
        ["bash", "-c", "ls /dev/dri/renderD* 2>/dev/null | wc -l"],
        capture_output=True, text=True,
    ).stdout.strip()
    _ctr_gpus = subprocess.run(
        ["docker", "exec", "lsv-container", "bash", "-c",
         "ls /dev/dri/renderD* 2>/dev/null | wc -l"],
        capture_output=True, text=True,
    ).stdout.strip()
    if _host_gpus and _ctr_gpus and _host_gpus != _ctr_gpus:
        print(
            f"[bench] GPU count mismatch: host={_host_gpus}, container={_ctr_gpus}. "
            "Recreating container to capture full device list...",
            flush=True,
        )
        subprocess.call(["docker", "stop", "lsv-container"])
        subprocess.call(["docker", "rm", "lsv-container"])
        _install_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install.sh")
        _install_rc = subprocess.call(["bash", _install_sh, "--skip-xpu-smi"])
        if _install_rc != 0:
            print("[bench] ERROR: install.sh failed during container recreation", flush=True)
            sys.exit(1)
        print("[bench] Container recreated with all GPUs ✓", flush=True)

    _cmd = (
        ["docker", "exec", "-w", "/root/guidellm-bench"]
        + _tty
        + _proxy_args
        + ["lsv-container", "python3", "/root/guidellm-bench/bench.py"]
        + sys.argv[1:]
    )
    _rc = subprocess.call(_cmd)
    if _rc == 42:
        # XPU kernel hang: container state is corrupted past recovery.
        print(
            "\n[recovery] XPU kernel hang detected (exit 42).\n"
            "  A host reboot is required to clear the XPU driver state.\n"
            "  After rebooting, resume with: ./bench.py --resume",
            flush=True,
        )
        if sys.stdin.isatty():
            try:
                answer = input("  Reboot now? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer == "y":
                subprocess.call(["reboot"])
            else:
                print("  Skipping reboot. Reboot manually when ready.", flush=True)
        else:
            print("  (Non-interactive session — skipping automatic reboot. Reboot manually.)", flush=True)
    sys.exit(_rc)

# ---------------------------------------------------------------------------
# Normal imports — only reached when running inside the container.
# ---------------------------------------------------------------------------
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from guidellm_bench import (
    ABLATION_LC_LENGTHS,
    ABLATION_C16_CONCURRENCY,
    ABLATION_C16_SAMPLES,
    EAGLE3_SPECULATIVE_CONFIG,
    FULL,
    LONG_CONTEXT_LENGTHS,
    SANITY,
    THROUGHPUT_CONCURRENCIES,
    THROUGHPUT_INPUT_LENGTHS,
    THROUGHPUT_MAX_MODEL_LEN,
    THROUGHPUT_MAX_NUM_BATCHED_TOKENS,
    THROUGHPUT_MAX_SECONDS,
    THROUGHPUT_OUTPUT_LEN,
    THROUGHPUT_SAMPLES,
    Config,
    SERVER_STATUS_PATH,
    XpuKernelHangError,
    build_ablation_dashboard_html,
    build_dashboard_html,
    build_throughput_dashboard_html,
    get_ablation_configs,
    get_throughput_configs,
    is_moe_model,
    parse_model_mem_gib,
    prepare_aime_dataset,
    prepare_hf_dataset,
    prepare_long_context_datasets,
    prepare_throughput_dataset,
    run_guidellm,
    server_is_reusable,
    skip_reason,
    start_server,
    stop_server,
    wait_for_server,
    write_server_status,
)
from guidellm_bench.benchmark import copy_results
from guidellm_bench.dashboard import write_serve_script


class _Tee:
    """Duplicate writes to two file-like objects (e.g. stdout + a log file)."""

    def __init__(self, *files):
        self._files = files

    def write(self, obj: str) -> None:
        for f in self._files:
            f.write(obj)
            f.flush()

    def flush(self) -> None:
        for f in self._files:
            f.flush()

    def fileno(self) -> int:
        # Needed so subprocesses can inherit the underlying fd.
        return self._files[0].fileno()


def _israel_now() -> datetime:
    """Return current datetime in Israel time (Asia/Jerusalem, UTC+2/+3 DST).

    Falls back to a fixed UTC+2 offset if tzdata is not installed in the
    container, so ts-labelled directories are always human-readable local time.
    """
    try:
        return datetime.now(ZoneInfo("Asia/Jerusalem"))
    except Exception:
        # tzdata missing — use fixed UTC+2 as a safe fallback
        from datetime import timezone
        il_tz = timezone(timedelta(hours=2))
        print(
            "  WARNING: ZoneInfo('Asia/Jerusalem') unavailable — "
            "using UTC+2 for timestamp. Install tzdata to fix.",
            flush=True,
        )
        return datetime.now(il_tz)


def _fmt_dur(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS (or MM:SS when < 1 hour)."""
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# Module-level reference so the SIGTERM/SIGINT handler can call stop_server()
# on the currently running vLLM process even when bench.py is killed externally.
_current_server_proc = None


def _signal_handler(signum, frame):
    """Gracefully stop the vLLM server before exiting on SIGTERM/SIGINT.

    Without this, an external kill of bench.py leaves the vLLM GPU workers
    alive (or killed via SIGKILL), which causes xe-destroy-wq kernel threads
    to get stuck in D-state, requiring a host reboot.
    """
    import signal
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n  {sig_name} received — shutting down server gracefully...", flush=True)
    stop_server(_current_server_proc)
    sys.exit(1)


def _clean_incomplete_runs(results_dir: str) -> None:
    """Remove subdirs in *results_dir* that have zero *_benchmarks.json* files.

    These are leftover from crashes / early exits. Safe to delete because no
    useful data was produced. Printed to stdout so the user can see what went.
    Skip the directory that is currently being written (caller creates it after).
    """
    base = Path(results_dir)
    if not base.exists():
        return
    removed = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if list(d.glob("*_benchmarks.json")):
            continue  # has results — keep
        # No benchmark JSON → incomplete run
        import shutil as _shutil
        _shutil.rmtree(d, ignore_errors=True)
        removed.append(d.name)
    if removed:
        print(f"  Auto-cleaned {len(removed)} incomplete run dir(s) from {base}: {removed}", flush=True)


def _write_resume_script(out_dir: Path, original_argv: list) -> None:
    """Write resume.sh into *out_dir* for easy post-reboot resumption.

    The script:
    - starts lsv-container if stopped (happens after every host reboot)
    - re-executes bench.py with the original flags + --resume <out_dir>
    """
    # Strip any existing --resume / --resume <val> flags from original argv
    clean_argv: list[str] = []
    skip_next = False
    for arg in original_argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--resume":
            skip_next = True   # next token is the value
            continue
        if arg.startswith("--resume="):
            continue
        clean_argv.append(arg)

    # Relative path from repo root so the script is portable
    rel = out_dir.relative_to(Path("/root/guidellm-bench"))

    bench_args = " ".join(clean_argv + [f"--resume {rel}"])

    script = f"""#!/usr/bin/env bash
# Resume {rel} after a host reboot.
# Generated by bench.py — do not edit the --resume path.
set -e

CD_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && cd .. && pwd)"
cd "$CD_DIR"

# Auto-start container (it stops on every host reboot — no --restart policy)
STATE=$(docker inspect --format '{{{{.State.Running}}}}' lsv-container 2>/dev/null || echo 'false')
if [ "$STATE" != "true" ]; then
    echo "[resume] Starting lsv-container..."
    docker start lsv-container
    sleep 3
fi

nohup ./bench.py {bench_args} > /dev/null 2>&1 &
echo "Resumed PID: $!"
echo "Log: {rel}/bench.log"
"""

    script_path = out_dir / "resume.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    print(f"  Resume script: {rel}/resume.sh", flush=True)

    # Install + enable a systemd service so the run auto-resumes after reboot
    # without any manual intervention.  The service is disabled by bench.py
    # once the run completes (see _disable_resume_service).
    _install_resume_service(script_path)


def _install_resume_service(resume_script: Path) -> None:
    """Create and enable /etc/systemd/system/guidellm-resume.service."""
    service = f"""[Unit]
Description=Auto-resume guidellm-bench run after reboot
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
User=root
WorkingDirectory=/root/dkorat/guidellm-bench
ExecStart={resume_script}
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    svc_path = Path("/etc/systemd/system/guidellm-resume.service")
    try:
        svc_path.write_text(service)
        subprocess.run(["systemctl", "daemon-reload"], capture_output=True)
        subprocess.run(["systemctl", "enable", "guidellm-resume.service"], capture_output=True)
        print("  Auto-resume service enabled (systemd): will restart run after reboot", flush=True)
    except Exception as exc:
        print(f"  WARNING: could not install systemd service: {exc}", flush=True)


def _disable_resume_service() -> None:
    """Disable the auto-resume systemd service — called when a run completes."""
    try:
        r = subprocess.run(["systemctl", "is-enabled", "guidellm-resume.service"],
                           capture_output=True, text=True)
        if r.stdout.strip() == "enabled":
            subprocess.run(["systemctl", "disable", "guidellm-resume.service"], capture_output=True)
            print("  Auto-resume service disabled (run complete).", flush=True)
    except Exception:
        pass



def _find_last_run_dataset(base_dirs: tuple = ("./results",)) -> Optional[str]:
    """Return the path to the most recently-created non-LC JSONL dataset.

    Searches *base_dirs* in order, within each run dir's ``datasets/`` subfolder.
    Excludes ``lc_*`` slices (those are derived from the source dataset).
    """
    for base in base_dirs:
        d = Path(base)
        if not d.exists():
            continue
        for run_dir in sorted(d.iterdir(), key=lambda x: x.name, reverse=True):
            ds_dir = run_dir / "datasets"
            if not ds_dir.is_dir():
                continue
            for jsonl in sorted(ds_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
                if not jsonl.name.startswith("lc_"):
                    return str(jsonl)
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="guidellm benchmarking for vLLM model/tp/quant/eager configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sanity", action="store_true", help="Use sanity (small) defaults")
    p.add_argument("--models", nargs="+")
    p.add_argument("--tp", nargs="+", type=int)
    p.add_argument("--quantization", nargs="+", help='Use "none" to omit --quantization flag')
    p.add_argument("--enforce-eager", nargs="+", choices=["true", "false"])
    p.add_argument("--input-len", type=int)
    p.add_argument("--output-len", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--num-prompts", type=int)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--results-dir")
    p.add_argument("--timeout-startup", type=int)
    p.add_argument(
        "--eagle3", action="store_true",
        help="Append gpt-oss-120b Eagle3 speculative-decoding config (opt-in only)",
    )
    p.add_argument(
        "--ep", action="store_true",
        help="Append Expert Parallelism (EP) variants for MoE models "
             "(requires intel/llm-scaler-vllm:0.14.0-b8+). "
             "Adds gpt-oss-20b tp4+ep4 and Qwen3-30B-A3B tp2+ep2 / tp4+ep4.",
    )
    p.add_argument(
        "--ep-compare", action="store_true", dest="ep_compare",
        help="Run EP-capable models both WITH and WITHOUT Expert Parallelism so "
             "both appear in the same dashboard for direct comparison. "
             "Superset of --ep: also ensures the non-EP base configs are present. "
             "Mutually exclusive with --ep.",
    )
    p.add_argument(
        "--data", metavar="HF_DATASET",
        help="HuggingFace dataset id to use as prompt source instead of AIME 2024 "
             "(e.g. 'cx-cmu/deepresearchgym-agentic-search-logs'). "
             "The text column is auto-detected.",
    )
    p.add_argument(
        "--long-contexts", action="store_true", dest="long_contexts",
        help="Run additional mini-benchmarks at 1k/4k/8k/16k input tokens (10 samples "
             "each) and add a TTFT-vs-input-length chart to each config tab in the dashboard.",
    )
    p.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study for gpt-oss-20b: LC-only benchmarks at 1k/2k/4k/8k, "
             "5 samples each, across a predefined set of Intel XPU optimization configs "
             "(EP, tp=8, async-scheduling, prefix-caching). "
             "Results saved to ./ablation_results/. Combine with --data to use a custom dataset.",
    )
    p.add_argument(
        "--throughput", action="store_true",
        help="Run throughput study for gpt-oss-20b: 2 server configs (tp8+async and "
             "tp8+async+EP), 4 concurrencies (1/16/64/128), 4 input lengths "
             "(16k/32k/48k/96k), 16k output, PC disabled. "
             "Results → ./throughput_results/.",
    )
    p.add_argument(
        "--no-ep", action="store_true", dest="no_ep",
        help="With --throughput: skip Server B (EP config). Run only Server A "
             "(tp8+async, no expert parallelism).",
    )
    p.add_argument(
        "--resume", metavar="DIR", nargs="?", const="",
        help="Resume an interrupted run. With a DIR argument, reuses that directory. "
             "Without a DIR argument, automatically resumes the latest run in the "
             "results directory. Skips any config whose _benchmarks.json already exists.",
    )
    return p


def _ensure_guidellm_installed() -> None:
    """Safety-net: install guidellm if missing (happens on a fresh container)."""
    r = subprocess.run(["python3", "-c", "import guidellm"], capture_output=True)
    if r.returncode != 0:
        print(
            "  guidellm not found — installing now (fresh container?). "
            "This takes ~30s and will not repeat on subsequent runs.",
            flush=True,
        )
        subprocess.run(
            ["pip", "install", "--break-system-packages", "-q", "-e", ".[guidellm]"],
            check=True,
            cwd="/root/guidellm-bench",
        )
        print("  guidellm installed.", flush=True)


def main() -> None:
    import signal as _signal
    _signal.signal(_signal.SIGTERM, _signal_handler)
    _signal.signal(_signal.SIGINT, _signal_handler)

    _ensure_guidellm_installed()
    args = build_arg_parser().parse_args()
    D = SANITY if args.sanity else FULL

    def get(attr: str, key: str):
        v = getattr(args, attr.replace("-", "_"), None)
        return v if v is not None else D[key]

    models           = get("models", "models")
    tp_values        = get("tp", "tp")
    quant_list       = [None if q == "none" else q for q in get("quantization", "quant")]
    eager_list       = [e == "true" for e in get("enforce_eager", "eager")]
    input_len        = get("input_len", "input_len")
    output_len       = get("output_len", "output_len")
    concurrency      = get("concurrency", "concurrency")
    num_prompts      = get("num_prompts", "num_prompts")
    max_model_len    = get("max_model_len", "max_model_len")
    results_dir      = get("results_dir", "results_dir")
    timeout_startup  = get("timeout_startup", "timeout_startup")

    # Ablation / throughput modes use their own results directories
    if getattr(args, "ablation", False):
        results_dir = "./ablation_results"
    if getattr(args, "throughput", False):
        results_dir = "./throughput_results"

    # Auto-clean incomplete run dirs (no _benchmarks.json) from prior aborted runs.
    # Skip when resuming — the target dir may legitimately have zero files yet.
    if args.resume is None:
        _clean_incomplete_runs(results_dir)

    # Output directory: existing (--resume) or a fresh timestamped one
    if args.resume is not None:
        if args.resume == "":
            # Bare --resume: find the latest subdirectory in results_dir
            base = Path(results_dir)
            candidates = sorted(
                (d for d in base.iterdir() if d.is_dir()),
                key=lambda d: d.name,
            )
            if not candidates:
                sys.exit(f"--resume: no run directories found in {base}")
            out_dir = candidates[-1].resolve()
            print(f"Auto-resuming latest run: {out_dir}", file=sys.__stdout__)
        else:
            out_dir = Path(args.resume).resolve()
        if not out_dir.is_dir():
            sys.exit(f"--resume: directory not found: {out_dir}")
        _log_mode = "a"   # append to existing bench.log
    else:
        ts      = _israel_now().strftime("%Y%m%d_%H%M")
        out_dir = Path(results_dir) / ts
        _log_mode = "w"

    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Self-log: tee all output into out_dir/bench.log; write pid for easy kill.
    _log_file = open(out_dir / "bench.log", _log_mode, buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_file)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, _log_file)  # type: ignore[assignment]
    (out_dir / "bench.pid").write_text(str(os.getpid()))

    if args.resume is not None:
        print(f"\n{'='*60}")
        print(f"RESUMING  {out_dir}")
        print(f"{'='*60}\n")
    else:
        print(f"Results → {out_dir}\n")
        _write_resume_script(out_dir, sys.argv[1:])

    # Validate mutually exclusive EP flags
    if args.ep and args.ep_compare:
        sys.exit("--ep and --ep-compare are mutually exclusive — use --ep-compare for side-by-side EP vs no-EP")

    # Prepare prompt dataset (custom HF dataset or AIME fallback)
    if args.data:
        dataset_path = prepare_hf_dataset(
            args.data,
            output_tokens=1024,
            max_samples=1000,
            cache_dir=out_dir / "datasets",
        )
        if dataset_path is None:
            print(f"  WARNING: '{args.data}' unavailable — falling back to AIME 2024", flush=True)
            dataset_path = prepare_aime_dataset(output_tokens=1024)
    else:
        if getattr(args, "ablation", False):
            # Ablation auto-discovers the dataset from the last full run (via
            # _find_last_run_dataset inside _run_ablation).  Leave dataset_path=None
            # so that discovery runs; _run_ablation falls back to AIME if nothing found.
            dataset_path = None
        else:
            dataset_path = prepare_aime_dataset(output_tokens=1024)

    # Prepare long-context JSONL slices (if --long-contexts and we have a source dataset)
    lc_datasets: dict[int, str | None] = {}   # {token_len: path | None}
    if args.long_contexts:
        if dataset_path:
            # --long-contexts needs max_model_len >= 16384; override if needed
            if max_model_len < 16384:
                print(
                    f"  --long-contexts: max_model_len {max_model_len} < 16384 — overriding to 16384",
                    flush=True,
                )
                max_model_len = 16384
            lc_datasets = prepare_long_context_datasets(
                source_path=dataset_path,
                token_lengths=LONG_CONTEXT_LENGTHS,
                num_samples=10,
                output_tokens=512,
                cache_dir=out_dir / "datasets",
            )
        else:
            print("  WARNING: --long-contexts requires a dataset; using synthetic fallback (no JSONL available)", flush=True)

    model_mem_gib: dict[str, float] = {}
    if args.resume is not None:
        # Re-parse model memory from existing server logs
        for p in out_dir.glob("logs/*_server.log"):
            cfg_name = p.stem.replace("_server", "")
            mem = parse_model_mem_gib(p)
            if mem is not None:
                model_mem_gib[cfg_name] = mem

    # ------------------------------------------------------------------
    # Ablation mode: LC-only study; bypass the standard config matrix loop
    # ------------------------------------------------------------------
    if getattr(args, "ablation", False):
        _run_ablation(
            out_dir=out_dir,
            log_dir=log_dir,
            dataset_path=dataset_path,
            max_model_len=max_model_len,
            timeout_startup=timeout_startup,
            model_mem_gib=model_mem_gib,
            resume=args.resume,
        )
        return  # ablation handled; skip standard loop

    # ------------------------------------------------------------------
    # Throughput mode: concurrency × input-length study for gpt-oss-20b
    # ------------------------------------------------------------------
    if getattr(args, "throughput", False):
        _run_throughput(
            out_dir=out_dir,
            log_dir=log_dir,
            dataset_path=dataset_path,
            timeout_startup=timeout_startup,
            model_mem_gib=model_mem_gib,
            resume=args.resume,
            no_ep=getattr(args, "no_ep", False),
        )
        return  # throughput handled; skip standard loop

    # Build config list, applying skip rules
    configs: list[Config] = []
    for model in models:
        for tp in tp_values:
            for quant in quant_list:
                for eager in eager_list:
                    reason = skip_reason(model, quant, eager, tp)
                    if reason:
                        print(f"  SKIP  {model}  tp={tp}  quant={quant or 'none'}  eager={eager}: {reason}")
                        continue
                    configs.append(Config(model=model, tp=tp, quant=quant, eager=eager))

    # Expert Parallelism (EP) variants for MoE models — opt-in via --ep or --ep-compare
    _EP_VARIANTS = [
        # gpt-oss-20b: mxfp4 baked in (no quant), tp=4 with ep=4
        Config(model="openai/gpt-oss-20b",  tp=4, quant=None,  eager=True, expert_parallel_size=4),
        # Qwen3-30B-A3B (MoE): fp8, tp=2+ep=2 and tp=4+ep=4
        Config(model="Qwen/Qwen3-30B-A3B", tp=2, quant="fp8", eager=True, expert_parallel_size=2),
        Config(model="Qwen/Qwen3-30B-A3B", tp=4, quant="fp8", eager=True, expert_parallel_size=4),
    ]
    if args.ep or args.ep_compare:
        flag = "--ep-compare" if args.ep_compare else "--ep"
        for c in _EP_VARIANTS:
            if models and c.model not in models:
                continue
            if not is_moe_model(c.model):
                print(f"  SKIP  {c.model}  ep: EP requires MoE architecture")
                continue
            configs.append(c)
            print(f"  +    {c.model}  tp={c.tp}  quant={c.quant or 'none'}  ep  (EP via {flag})")

    # --ep-compare: also ensure the non-EP counterparts of EP variants are present in the matrix
    if args.ep_compare:
        existing_names = {c.name for c in configs}
        for c in _EP_VARIANTS:
            if models and c.model not in models:
                continue
            if not is_moe_model(c.model):
                continue
            base = Config(model=c.model, tp=c.tp, quant=c.quant, eager=c.eager)
            if base.name not in existing_names:
                reason = skip_reason(base.model, base.quant, base.eager, base.tp)
                if reason:
                    print(f"  SKIP (ep-compare base)  {base.name}: {reason}")
                else:
                    configs.append(base)
                    existing_names.add(base.name)
                    print(f"  +    {base.model}  tp={base.tp}  quant={base.quant or 'none'}  (non-EP base via --ep-compare)")

    # gpt-oss-120b Eagle3 is opt-in (tp=8, no quant — mxfp4 baked in)
    if not args.sanity and args.eagle3:
        configs.append(Config(
            model="openai/gpt-oss-120b",
            tp=8,
            quant=None,
            eager=True,
            speculative_config=EAGLE3_SPECULATIVE_CONFIG,
        ))
        print("  +    openai/gpt-oss-120b  tp=8  eagle3  (appended via --eagle3)")

    print(f"\n{len(configs)} configurations to benchmark\n")

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    # Tracks the Popen of the currently-running vLLM server so we can skip
    # tearing it down and restarting when consecutive configs are identical.
    current_server_proc = None

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(configs)}]  {cfg.name}")
        print(f"{'='*60}")

        # Checkpointing: skip configs that already have results
        checkpoint = out_dir / f"{cfg.name}_benchmarks.json"
        if args.resume is not None and checkpoint.exists():
            print(f"  ✓  already done (checkpoint found) — skipping")
            succeeded.append(cfg.name)
            continue

        t0 = time.time()

        _server_log = log_dir / f"{cfg.name}_server.log"
        if server_is_reusable(cfg, max_model_len):
            # Server already running with this exact config and passing /health.
            # Reuse it — avoid the ~90s restart cost.
            print("  Reusing running server (config matches, server is healthy)", flush=True)
            proc = None
        else:
            # Stop whatever is currently running (previous config or stray).
            stop_server(current_server_proc)
            current_server_proc = None
            global _current_server_proc
            _current_server_proc = None
            proc = start_server(cfg, max_model_len, _server_log)
            _current_server_proc = proc
            try:
                ready = wait_for_server(timeout_startup, log_path=_server_log, proc=proc)
            except XpuKernelHangError as _hang:
                print(f"  FATAL: {_hang}", flush=True)
                print(
                    "  Exiting with code 42 — host will recreate the container and resume.",
                    flush=True,
                )
                stop_server(proc)
                sys.exit(42)
            if not ready:
                stop_server(proc)
                current_server_proc = None
                _current_server_proc = None
                failed.append((cfg.name, "server startup failed"))
                continue

            # Persist running server config so next iteration can skip restart.
            write_server_status(cfg, max_model_len, proc.pid, _server_log)
            current_server_proc = proc
            _current_server_proc = proc

        data = run_guidellm(
            cfg, input_len, output_len, concurrency, num_prompts,
            log_dir / f"{cfg.name}_bench.log",
            sweep=not args.sanity,
            dataset_path=dataset_path,
        )
        # Server intentionally kept alive — reused by next config if possible.
        # Final teardown happens after all configs or before a config with a
        # different server requirement (handled above at the top of the loop).

        if data is None:
            failed.append((cfg.name, "benchmark failed or produced no output"))
            continue

        saved = copy_results(cfg.name, out_dir, log_dir / ".guidellm_out")
        for fname in saved:
            dest = out_dir / fname
            if dest.suffix == ".html":
                write_serve_script(dest)

        # Parse model weight memory from vLLM server log (reliable; no xpu-smi needed)
        mem_gib = parse_model_mem_gib(log_dir / f"{cfg.name}_server.log")
        if mem_gib is not None:
            model_mem_gib[cfg.name] = mem_gib
            print(f"  Model weights: {mem_gib:.2f} GiB/GPU × {cfg.tp} GPUs = {mem_gib * cfg.tp:.2f} GiB total")

        # Long-context slices (--long-contexts): run 1k/4k/8k/16k input-token benchmarks
        if args.long_contexts and lc_datasets:
            print(f"  Running long-context slices for {cfg.name}...", flush=True)
            for token_len, lc_path in sorted(lc_datasets.items()):
                if lc_path is None:
                    continue
                lc_label = f"{token_len // 1024}k" if token_len >= 1024 else str(token_len)
                lc_name = f"{cfg.name}_lc{lc_label}"
                lc_checkpoint = out_dir / f"{lc_name}_benchmarks.json"
                if args.resume is not None and lc_checkpoint.exists():
                    print(f"    ✓  lc{lc_label} already done — skipping", flush=True)
                    continue
                print(f"    → lc{lc_label} ({token_len} input tokens, 10 samples)", flush=True)
                lc_data = run_guidellm(
                    cfg, input_len, output_len, concurrency, num_prompts,
                    log_dir / f"{lc_name}_bench.log",
                    sweep=False,
                    dataset_path=lc_path,
                    lc_mode=True,
                )
                if lc_data is not None:
                    lc_saved = copy_results(lc_name, out_dir, log_dir / ".guidellm_out")
                    print(f"    ✓  lc{lc_label} → {', '.join(lc_saved)}", flush=True)
                else:
                    print(f"    ✗  lc{lc_label} failed", flush=True)

        elapsed = time.time() - t0
        print(f"\n  ✓  {cfg.name}  ({elapsed:.0f}s)  →  {', '.join(saved)}", flush=True)
        succeeded.append(cfg.name)

    # Server is intentionally left running so the next bench invocation can
    # reuse it without a ~90s restart (server_status.json remains valid).
    # Only stop if a run failed entirely and no server was started.
    if current_server_proc is None and not server_is_reusable.__module__:
        pass  # nothing to do
    else:
        srv_status = SERVER_STATUS_PATH.exists()
        if srv_status:
            print(f"\n  Server kept alive — server_status.json preserved for reuse.", flush=True)
        # Do NOT call stop_server here; let the next run decide whether to reuse.

    # Summary
    print(f"\n{'='*60}")
    print(f"Finished:  {len(succeeded)} succeeded  /  {len(failed)} failed  /  {len(configs)} total")
    if failed:
        print("\nFailed configurations:")
        for name, reason in failed:
            print(f"  ✗  {name}:  {reason}")
    print(f"\nResults: {out_dir}")

    if succeeded:
        build_dashboard_html(out_dir, succeeded, model_mem_gib=model_mem_gib)
        _serve_dashboard(out_dir)
    _disable_resume_service()


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def _run_ablation(
    out_dir: Path,
    log_dir: Path,
    dataset_path: Optional[str],
    max_model_len: int,
    timeout_startup: int,
    model_mem_gib: dict,
    resume,
) -> None:
    """LC-only ablation study for gpt-oss-20b on Intel XPU.

    Runs a predefined matrix of vLLM configuration variants at 4 input-token
    lengths (1k/2k/4k/8k), 5 samples each.  No main concurrency sweep is run.
    Builds ablation_dashboard.html with lineplots + auto-generated conclusions.
    """
    print(f"\n{'='*60}")
    print("ABLATION MODE: gpt-oss-20b Intel XPU optimization study")
    print(f"{'='*60}\n")

    # Dataset: caller already resolved --data / AIME.  If still None (no internet
    # and no --data), try to auto-discover a JSONL from a prior full run.
    if dataset_path is None:
        discovered = _find_last_run_dataset()
        if discovered:
            print(f"  Ablation: auto-discovered dataset from last run: {discovered}", flush=True)
            dataset_path = discovered
        else:
            dataset_path = prepare_aime_dataset(output_tokens=1024)

    # max_model_len must be able to handle 8k-token inputs + 512 output tokens
    if max_model_len < 8192:
        print(f"  Ablation: max_model_len {max_model_len} → overriding to 16384", flush=True)
        max_model_len = 16384

    # Prepare LC JSONL slices: 1k / 2k / 4k / 8k, 5 samples, 512 output tokens
    lc_datasets: dict[int, Optional[str]] = {}
    if dataset_path:
        lc_datasets = prepare_long_context_datasets(
            source_path=dataset_path,
            token_lengths=ABLATION_LC_LENGTHS,
            num_samples=5,
            output_tokens=512,
            cache_dir=out_dir / "datasets",
        )

    # Build ablation config matrix.
    # Ablation configs are manually curated — skip_reason() is NOT applied.
    # tp=2 is deliberately included (with a reduced max_model_len_override=8192)
    # to test whether the blog's tp=1 result is reproducible at tp=2 on 0.14.0-b8.
    # Eagle3 is included with a speculative decoding config.
    ablation_cfgs: list[Config] = list(get_ablation_configs())

    n_runs = len(ablation_cfgs) * len(lc_datasets)
    print(
        f"{len(ablation_cfgs)} configs × {len(lc_datasets)} LC lengths = {n_runs} benchmark runs\n",
        flush=True,
    )

    global _current_server_proc
    current_server_proc = None
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    run_start = time.time()

    for i, cfg in enumerate(ablation_cfgs, 1):
        # ── Overall progress banner (shown before every config except the first) ──
        if i > 1:
            _e = time.time() - run_start
            _avg = _e / (i - 1)
            _rem = _avg * (len(ablation_cfgs) - (i - 1))
            _eta = _israel_now() + timedelta(seconds=_rem)
            _pct = (i - 1) / len(ablation_cfgs) * 100
            print(
                f"\n  ── {i-1}/{len(ablation_cfgs)} done ({_pct:.0f}%)  "
                f"|  {_fmt_dur(_e)} elapsed  "
                f"|  ~{_fmt_dur(_rem)} remaining  "
                f"|  ETA {_eta.strftime('%H:%M')} Israel ──",
                flush=True,
            )
        print(f"\n{'='*60}")
        print(f"[{i}/{len(ablation_cfgs)}]  {cfg.name}")
        print(f"{'='*60}")

        # Per-config effective max_model_len:
        # tp=2 declares max_model_len_override=8192 to fit within 2-GPU memory budget.
        # All other configs use the global ablation default (16384).
        effective_max_model_len = cfg.max_model_len_override or max_model_len
        if cfg.max_model_len_override:
            print(f"  Using max_model_len={effective_max_model_len} (per-config override for tp={cfg.tp})", flush=True)

        _server_log = log_dir / f"{cfg.name}_server.log"
        if server_is_reusable(cfg, effective_max_model_len):
            print("  Reusing running server (config matches, server is healthy)", flush=True)
        else:
            stop_server(current_server_proc)
            current_server_proc = None
            _current_server_proc = None
            proc = start_server(cfg, effective_max_model_len, _server_log)
            _current_server_proc = proc
            try:
                ready = wait_for_server(timeout_startup, log_path=_server_log, proc=proc)
            except XpuKernelHangError as _hang:
                print(f"  FATAL: {_hang}", flush=True)
                stop_server(proc)
                sys.exit(42)
            if not ready:
                stop_server(proc)
                current_server_proc = None
                _current_server_proc = None
                failed.append((cfg.name, "server startup failed"))
                continue
            write_server_status(cfg, effective_max_model_len, proc.pid, _server_log)
            current_server_proc = proc
            _current_server_proc = proc

        # Parse GPU weight memory for the dashboard
        mem_gib = parse_model_mem_gib(_server_log)
        if mem_gib is not None:
            model_mem_gib[cfg.name] = mem_gib
            print(
                f"  Model weights: {mem_gib:.2f} GiB/GPU × {cfg.tp} GPUs"
                f" = {mem_gib * cfg.tp:.2f} GiB total",
            )

        # LC-only: no main sweep benchmark — just the input-length slices
        any_lc_success = False
        for token_len, lc_path in sorted(lc_datasets.items()):
            if lc_path is None:
                continue
            lc_label = f"{token_len // 1024}k" if token_len >= 1024 else str(token_len)
            # Skip LC lengths that cannot fit within this config's context window.
            # Guard: input_tokens + output_tokens must not exceed max_model_len.
            if token_len + 512 > effective_max_model_len:
                print(f"    SKIP lc{lc_label}: {token_len} + 512 output > max_model_len={effective_max_model_len}", flush=True)
                continue
            lc_name = f"{cfg.name}_lc{lc_label}"
            lc_checkpoint = out_dir / f"{lc_name}_benchmarks.json"
            if resume is not None and lc_checkpoint.exists():
                print(f"    ✓  lc{lc_label} already done — skipping", flush=True)
                any_lc_success = True
                continue
            print(f"    → lc{lc_label} ({token_len} tokens, 5 samples)", flush=True)
            lc_data = run_guidellm(
                cfg,
                token_len,   # input tokens (overrides global input_len for LC runs)
                512,         # output tokens
                1,           # concurrency=1 for serial profile
                5,           # num_prompts
                log_dir / f"{lc_name}_bench.log",
                sweep=False,
                dataset_path=lc_path,
                lc_mode=True,
            )
            if lc_data is not None:
                lc_saved = copy_results(lc_name, out_dir, log_dir / ".guidellm_out")
                print(f"    ✓  lc{lc_label} → {', '.join(lc_saved)}", flush=True)
                any_lc_success = True
            else:
                print(f"    ✗  lc{lc_label} failed", flush=True)

        if any_lc_success:
            succeeded.append(cfg.name)
        else:
            failed.append((cfg.name, "all LC slices failed"))
        # ── Post-config progress line ────────────────────────────────────────────
        _done = i
        _te = time.time() - run_start
        _rem_cfgs = len(ablation_cfgs) - _done
        _pct2 = _done / len(ablation_cfgs) * 100
        _sym = "✓" if any_lc_success else "✗"
        if _rem_cfgs > 0:
            _avg2 = _te / _done
            _rem2 = _avg2 * _rem_cfgs
            _eta2 = _israel_now() + timedelta(seconds=_rem2)
            print(
                f"\n  {_sym} [{_done}/{len(ablation_cfgs)}] {cfg.name}  "
                f"|  {_pct2:.0f}% done  |  {_fmt_dur(_te)} elapsed  "
                f"|  ~{_fmt_dur(_rem2)} remaining  |  ETA {_eta2.strftime('%H:%M')} Israel",
                flush=True,
            )
        else:
            print(
                f"\n  {_sym} [{_done}/{len(ablation_cfgs)}] {cfg.name}  "
                f"|  100% done  |  {_fmt_dur(_te)} total",
                flush=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # C=16 PHASE: top-5 + EP + Eagle3 configs at concurrency=16
    # ─────────────────────────────────────────────────────────────────────────
    # Same 4 input lengths as the c=1 study (1k/2k/4k/8k), ABLATION_C16_SAMPLES
    # samples each.  Splits the ablation into two use cases:
    #   c=1  → latency-minimization (serial): which config is fastest per request?
    #   c=16 → throughput-maximization:       which config handles concurrent load?
    #
    # Config selection rationale:
    #   EP hurts c=1 latency (all-to-all not amortized at batch_size≈1).
    #   At c=16, many tokens are in-flight and all-to-all cost amortizes → EP may win.
    #   Top-5 non-EP configs by expected high-concurrency performance:
    #     1. baseline (tp4)       — reference
    #     2. tp8                  — extra TP bandwidth
    #     3. async (tp4)          — CPU pipelining scales with in-flight depth
    #     4. asyncpc (tp4)        — PC+async: combined optimisation
    #     5. tp8+asyncpc          — highest ceiling (8 GPU + all opts)
    #   All EP configs included (EP shines at high concurrency).
    #   Eagle3 included (speculative accept-rate improves with deeper batches).
    #   Excluded: tp2 (max_model_len_override=8192 — memory constraint at c=16).
    def _is_c16_config(c: "Config") -> bool:
        if c.max_model_len_override is not None:
            return False   # tp2: skip (restricted context budget)
        if c.expert_parallel_size:
            return True    # all EP configs — EP is a throughput optimisation
        if c.speculative_config is not None:
            return True    # Eagle3 — accept-rate benefits grow with batch depth
        # Top-5 non-EP, non-speculative configs
        if c.tp == 4 and not c.async_scheduling and not c.prefix_caching:
            return True    # baseline
        if c.tp == 8 and not c.async_scheduling and not c.prefix_caching:
            return True    # tp8
        if c.tp == 4 and c.async_scheduling and not c.prefix_caching:
            return True    # async
        if c.tp == 4 and c.async_scheduling and c.prefix_caching:
            return True    # asyncpc
        if c.tp == 8 and c.async_scheduling and c.prefix_caching:
            return True    # tp8+asyncpc
        return False

    c16_cfgs = [c for c in ablation_cfgs if _is_c16_config(c)]
    c16_succeeded: list[str] = []
    c16_failed: list[tuple[str, str]] = []

    if c16_cfgs and lc_datasets:
        print(f"\n{'='*60}")
        print(f"C=16 PHASE: {len(c16_cfgs)} configs × {len(lc_datasets)} lengths @ concurrency={ABLATION_C16_CONCURRENCY}")
        print(f"  (top-5 + EP + Eagle3; shows which config wins at throughput scale)")
        print(f"{'='*60}")

        # Prepare c16 LC datasets: 20 samples per length (more than c=1's 5,
        # needed to sustain concurrency=16 across a full benchmark run).
        # Same cache_dir as c=1 but num_samples=20 triggers a cache upgrade.
        c16_lc_datasets: dict[int, Optional[str]] = {}
        if dataset_path:
            c16_lc_datasets = prepare_long_context_datasets(
                source_path=dataset_path,
                token_lengths=ABLATION_LC_LENGTHS,
                num_samples=ABLATION_C16_SAMPLES,
                output_tokens=512,
                cache_dir=out_dir / "datasets",
            )
        else:
            c16_lc_datasets = lc_datasets  # fall back if no dataset (may have < 20 samples)

        for cfg in c16_cfgs:
            effective_max_model_len = cfg.max_model_len_override or max_model_len
            _server_log = log_dir / f"{cfg.name}_server.log"

            # Check --resume: skip config if ALL its c16 LC slices already done
            c16_any_done = any(
                (out_dir / f"{cfg.name}_c16_lc{l // 1024}k_benchmarks.json").exists()
                for l in sorted(c16_lc_datasets)
                if c16_lc_datasets.get(l) and l + 512 <= effective_max_model_len
            )
            if resume is not None and c16_any_done:
                # Re-check if truly ALL are done (not just some)
                all_c16_done = all(
                    (out_dir / f"{cfg.name}_c16_lc{l // 1024}k_benchmarks.json").exists()
                    for l in sorted(c16_lc_datasets)
                    if c16_lc_datasets.get(l) and l + 512 <= effective_max_model_len
                )
                if all_c16_done:
                    print(f"  ✓  {cfg.name} c16 all LC slices already done — skipping", flush=True)
                    c16_succeeded.append(cfg.name)
                    continue

            print(f"\n  → {cfg.name} c16 (c={ABLATION_C16_CONCURRENCY}, {ABLATION_C16_SAMPLES} samples/length)", flush=True)
            if server_is_reusable(cfg, effective_max_model_len):
                print("    Reusing running server", flush=True)
            else:
                stop_server(current_server_proc)
                current_server_proc = None
                _current_server_proc = None
                proc = start_server(cfg, effective_max_model_len, _server_log)
                _current_server_proc = proc
                try:
                    ready = wait_for_server(timeout_startup, log_path=_server_log, proc=proc)
                except XpuKernelHangError as _hang:
                    print(f"  FATAL: {_hang}", flush=True)
                    stop_server(proc)
                    sys.exit(42)
                if not ready:
                    stop_server(proc)
                    current_server_proc = None
                    _current_server_proc = None
                    c16_failed.append((cfg.name, "server startup failed"))
                    continue
                write_server_status(cfg, effective_max_model_len, proc.pid, _server_log)
                current_server_proc = proc
                _current_server_proc = proc

            c16_any_success = False
            for token_len, lc_path in sorted(c16_lc_datasets.items()):
                if lc_path is None:
                    continue
                if token_len + 512 > effective_max_model_len:
                    print(f"    SKIP c16_lc{token_len // 1024}k: {token_len}+512 > max_model_len={effective_max_model_len}", flush=True)
                    continue
                lc_label = f"{token_len // 1024}k" if token_len >= 1024 else str(token_len)
                c16_name = f"{cfg.name}_c16_lc{lc_label}"
                c16_checkpoint = out_dir / f"{c16_name}_benchmarks.json"
                if resume is not None and c16_checkpoint.exists():
                    print(f"    ✓  c16_lc{lc_label} already done — skipping", flush=True)
                    c16_any_success = True
                    continue
                print(f"    → c16_lc{lc_label} ({token_len} tok, {ABLATION_C16_SAMPLES} samples, c={ABLATION_C16_CONCURRENCY})", flush=True)
                c16_result = run_guidellm(
                    cfg,
                    token_len,
                    512,
                    ABLATION_C16_CONCURRENCY,
                    ABLATION_C16_SAMPLES,
                    log_dir / f"{c16_name}_bench.log",
                    sweep=False,        # concurrent profile: --profile concurrent --rate N
                    dataset_path=lc_path,
                    lc_mode=False,      # NOT lc_mode: we want concurrent profile, not synchronous
                )
                if c16_result is not None:
                    c16_saved = copy_results(c16_name, out_dir, log_dir / ".guidellm_out")
                    print(f"    ✓  c16_lc{lc_label} → {', '.join(c16_saved)}", flush=True)
                    c16_any_success = True
                else:
                    print(f"    ✗  c16_lc{lc_label} failed", flush=True)

            if c16_any_success:
                c16_succeeded.append(cfg.name)
            else:
                c16_failed.append((cfg.name, "all c16 LC slices failed"))

        if c16_failed:
            print("\nFailed c16 configs:")
            for name, reason_str in c16_failed:
                print(f"  ✗  {name}: {reason_str}")

    # Ablation Summary
    print(f"\n{'='*60}")
    print(
        f"Ablation finished: {len(succeeded)} succeeded / {len(failed)} failed"
        f" / {len(ablation_cfgs)} total",
    )
    if failed:
        print("\nFailed configurations:")
        for name, reason_str in failed:
            print(f"  ✗  {name}: {reason_str}")
    print(f"\nAblation results: {out_dir}")

    if succeeded:
        abl_html = build_ablation_dashboard_html(
            out_dir, succeeded, model_mem_gib=model_mem_gib,
            c16_succeeded=c16_succeeded,
        )
        if abl_html:
            _serve_html(abl_html)
    _disable_resume_service()


def _run_throughput(
    out_dir: Path,
    log_dir: Path,
    dataset_path: Optional[str],
    timeout_startup: int,
    model_mem_gib: dict,
    resume,
    no_ep: bool = False,
) -> None:
    """Throughput study: concurrency × input_length sweep for gpt-oss-20b on Intel XPU.

    2 server configurations:
      Server A – tp=8, async-scheduling, no EP: concurrencies [1, 16, 64, 128]
      Server B – tp=8, async-scheduling, EP:    concurrencies [   16, 64, 128]

    Exactly 1 server restart between Server A and Server B.
    Input lengths: 16k / 32k / 48k / 96k (tokens).  Output: 16k.  PC disabled.
    Samples per concurrency: {1:10, 16:32, 64:128, 128:256} (2×c rule).
    """
    print(f"\n{'='*60}")
    print("THROUGHPUT MODE: gpt-oss-20b concurrency × input-length sweep")
    print(f"  tp=8 + async | 4 concurrencies | 4 input lengths | output=16k | PC disabled")
    print(f"{'='*60}\n")

    # ── Dataset resolution ───────────────────────────────────────────────────
    if dataset_path is None:
        # Try auto-discovering from the last full run
        discovered = _find_last_run_dataset()
        if discovered:
            print(f"  Throughput: auto-discovered dataset from last run: {discovered}", flush=True)
            dataset_path = discovered
        else:
            # Fall back to arxiv-summarization (long documents → realistic prefill)
            print("  No dataset supplied — downloading ccdv/arxiv-summarization …", flush=True)
            arxiv_path = prepare_hf_dataset(
                "ccdv/arxiv-summarization",
                output_tokens=THROUGHPUT_OUTPUT_LEN,
                max_samples=1000,
                cache_dir=out_dir / "datasets",
            )
            dataset_path = arxiv_path or prepare_aime_dataset(output_tokens=THROUGHPUT_OUTPUT_LEN)

    # ── Pre-build throughput JSONL slices (one per input_len, 256 rows max) ──
    max_samples_needed = max(THROUGHPUT_SAMPLES.values())   # 256 for c=128
    tp_datasets: dict[int, Optional[str]] = {}
    for il in THROUGHPUT_INPUT_LENGTHS:
        if dataset_path:
            try:
                tp_datasets[il] = prepare_throughput_dataset(
                    source_path=dataset_path,
                    input_len=il,
                    output_len=THROUGHPUT_OUTPUT_LEN,
                    num_samples=max_samples_needed,
                    cache_dir=out_dir / "datasets",
                )
            except Exception as _ds_exc:
                print(f"  WARNING: dataset prep failed for il={il//1024}k: {_ds_exc} — cell will be skipped", flush=True)
                tp_datasets[il] = None
        else:
            tp_datasets[il] = None

    # ── Config matrix ────────────────────────────────────────────────────────
    thr_cfgs = get_throughput_configs()    # [Server A, Server B]
    if no_ep:
        thr_cfgs = [c for c in thr_cfgs if not c.expert_parallel_size]
        print("  --no-ep: skipping Server B (EP config)", flush=True)
    concurrencies_for_cfg: dict[str, list[int]] = {
        thr_cfgs[0].name: list(THROUGHPUT_CONCURRENCIES),            # all
        thr_cfgs[1].name: [c for c in THROUGHPUT_CONCURRENCIES if c > 1],  # EP: c>1 only
    }
    total_cells = sum(
        len(cs) * len(THROUGHPUT_INPUT_LENGTHS)
        for cs in concurrencies_for_cfg.values()
    )
    print(
        f"{len(thr_cfgs)} server configs × {len(THROUGHPUT_CONCURRENCIES)} concurrencies "
        f"× {len(THROUGHPUT_INPUT_LENGTHS)} input lengths = {total_cells} cells total\n",
        flush=True,
    )

    global _current_server_proc
    current_server_proc = None
    succeeded: list[tuple[str, int, int]] = []   # (cfg_name, concurrency, input_len)
    failed:    list[tuple[str, str]]       = []
    run_start = time.time()

    # Outer loop = server config (ensures exactly 1 restart between A and B)
    for srv_idx, cfg in enumerate(thr_cfgs):
        concs = concurrencies_for_cfg.get(cfg.name, list(THROUGHPUT_CONCURRENCIES))
        print(f"\n{'='*60}")
        print(f"[Server {srv_idx + 1}/{len(thr_cfgs)}]  {cfg.name}")
        print(f"  Concurrencies: {concs}")
        print(f"{'='*60}")

        _server_log = log_dir / f"{cfg.name}_server.log"
        if server_is_reusable(cfg, THROUGHPUT_MAX_MODEL_LEN):
            print("  Reusing running server (config matches, server is healthy)", flush=True)
        else:
            stop_server(current_server_proc)
            current_server_proc = None
            _current_server_proc = None
            proc = start_server(
                cfg, THROUGHPUT_MAX_MODEL_LEN, _server_log,
                max_num_batched_tokens=THROUGHPUT_MAX_NUM_BATCHED_TOKENS,
            )
            _current_server_proc = proc
            try:
                ready = wait_for_server(timeout_startup, log_path=_server_log, proc=proc)
            except XpuKernelHangError as _hang:
                print(f"  FATAL: {_hang}", flush=True)
                stop_server(proc)
                sys.exit(42)
            if not ready:
                stop_server(proc)
                current_server_proc = None
                _current_server_proc = None
                failed.append((cfg.name, "server startup failed"))
                continue
            write_server_status(cfg, THROUGHPUT_MAX_MODEL_LEN, proc.pid, _server_log)
            current_server_proc = proc
            _current_server_proc = proc

        mem_gib = parse_model_mem_gib(_server_log)
        if mem_gib is not None:
            model_mem_gib[cfg.name] = mem_gib
            print(
                f"  Model weights: {mem_gib:.2f} GiB/GPU × {cfg.tp} GPUs"
                f" = {mem_gib * cfg.tp:.2f} GiB total",
            )

        # Inner loops: concurrency then input_length (server stays up throughout)
        server_ok = True   # flipped to False if server dies and cannot be restarted
        for c in sorted(concs):
            if not server_ok:
                break
            n_samples   = THROUGHPUT_SAMPLES[c]
            is_serial   = (c == 1)   # c=1 → synchronous / lc_mode; c>1 → concurrent
            for il in THROUGHPUT_INPUT_LENGTHS:
                if not server_ok:
                    break
                il_label  = f"{il // 1024}k"
                cell_name = f"{cfg.name}_c{c}_il{il_label}"
                checkpoint = out_dir / f"{cell_name}_benchmarks.json"

                if resume is not None and checkpoint.exists():
                    print(f"  ✓  {cell_name} already done — skipping", flush=True)
                    succeeded.append((cfg.name, c, il))
                    continue

                _e = time.time() - run_start
                _done_cells = len(succeeded) + len(failed)
                _avg = _e / _done_cells if _done_cells else 0
                _rem = _avg * (total_cells - _done_cells) if _avg else 0
                _eta = _israel_now() + timedelta(seconds=_rem) if _rem else None
                _eta_str = f" | ETA {_eta.strftime('%H:%M')} Israel" if _eta else ""
                print(
                    f"\n  → c={c}  il={il_label}  output={THROUGHPUT_OUTPUT_LEN // 1024}k"
                    f"  ({n_samples} samples)"
                    f"  [{_done_cells}/{total_cells} done{_eta_str}]",
                    flush=True,
                )

                result = None
                try:
                    result = run_guidellm(
                        cfg,
                        il,                          # input tokens
                        THROUGHPUT_OUTPUT_LEN,       # output tokens
                        c,                           # concurrency / rate
                        n_samples,                   # num_prompts
                        log_dir / f"{cell_name}_bench.log",
                        sweep=False,
                        dataset_path=tp_datasets.get(il),
                        data_samples=n_samples,
                        lc_mode=is_serial,
                        max_seconds=THROUGHPUT_MAX_SECONDS,
                        num_requests_override=n_samples,
                    )
                except Exception as _cell_exc:
                    print(f"  ✗  {cell_name} raised unexpected error: {_cell_exc}", flush=True)

                if result is not None:
                    saved = copy_results(cell_name, out_dir, log_dir / ".guidellm_out")
                    print(f"  ✓  {cell_name} → {', '.join(saved)}", flush=True)
                    succeeded.append((cfg.name, c, il))
                else:
                    print(f"  ✗  {cell_name} failed — checking server health …", flush=True)
                    failed.append((cell_name, "guidellm run failed"))

                    # Check whether the vLLM server process is still alive.
                    # If it died (OOM, XPU driver fault, etc.) we must restart it
                    # before attempting the next cell, otherwise the next call will
                    # hang until THROUGHPUT_MAX_SECONDS expires.
                    server_died = (
                        current_server_proc is not None
                        and current_server_proc.poll() is not None
                    )
                    if server_died:
                        print(
                            f"  Server process exited (code {current_server_proc.returncode})"
                            " — attempting restart …",
                            flush=True,
                        )
                        stop_server(current_server_proc)
                        current_server_proc = None
                        _current_server_proc = None
                        proc_new = start_server(
                            cfg, THROUGHPUT_MAX_MODEL_LEN, _server_log,
                            max_num_batched_tokens=THROUGHPUT_MAX_NUM_BATCHED_TOKENS,
                        )
                        _current_server_proc = proc_new
                        try:
                            ready_new = wait_for_server(
                                timeout_startup, log_path=_server_log, proc=proc_new
                            )
                        except XpuKernelHangError as _hang:
                            print(f"  FATAL: {_hang}", flush=True)
                            stop_server(proc_new)
                            sys.exit(42)
                        if ready_new:
                            write_server_status(
                                cfg, THROUGHPUT_MAX_MODEL_LEN, proc_new.pid, _server_log
                            )
                            current_server_proc = proc_new
                            _current_server_proc = proc_new
                            print("  Server restarted successfully — continuing.", flush=True)
                        else:
                            stop_server(proc_new)
                            current_server_proc = None
                            _current_server_proc = None
                            print(
                                "  Server restart failed — skipping remaining cells "
                                f"for {cfg.name}.",
                                flush=True,
                            )
                            server_ok = False  # break out of both inner loops

                # Rebuild dashboard incrementally after every cell (success or fail)
                cfg_names_done = sorted({s[0] for s in succeeded})
                if cfg_names_done:
                    try:
                        build_throughput_dashboard_html(out_dir, cfg_names_done)
                    except Exception as _dash_exc:
                        print(f"  (dashboard rebuild failed: {_dash_exc})", flush=True)

    # ── Final summary ────────────────────────────────────────────────────────
    total_elapsed = time.time() - run_start
    print(f"\n{'='*60}")
    print(
        f"Throughput study finished: {len(succeeded)}/{total_cells} cells done"
        f" in {_fmt_dur(total_elapsed)}",
    )
    if failed:
        print("\nFailed cells:")
        for name, reason_str in failed:
            print(f"  ✗  {name}: {reason_str}")
    print(f"\nResults: {out_dir}", flush=True)

    cfg_names_done = sorted({s[0] for s in succeeded})
    if cfg_names_done:
        thr_html = build_throughput_dashboard_html(out_dir, cfg_names_done)
        if thr_html:
            _serve_html(thr_html)
    _disable_resume_service()


def _serve_html(html_path: Path) -> None:
    """Kill any prior server on _DASHBOARD_PORT, serve *html_path*, print URL."""
    if not html_path.exists():
        return
    subprocess.run(
        ["bash", "-c",
         f"fuser -k {_DASHBOARD_PORT}/tcp 2>/dev/null || "
         f"lsof -ti tcp:{_DASHBOARD_PORT} 2>/dev/null | xargs -r kill -9"],
        capture_output=True,
    )
    time.sleep(0.5)
    subprocess.Popen(
        ["python3", "-m", "http.server", str(_DASHBOARD_PORT),
         "--directory", str(html_path.parent.resolve())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.5)
    url = f"http://localhost:{_DASHBOARD_PORT}/{html_path.name}"
    print(f"\n  Ablation dashboard served at: {url}\n", flush=True)


_DASHBOARD_PORT = 8081


def _serve_dashboard(out_dir: Path) -> None:
    """Kill any prior server on the dashboard port, start a new one, print URL."""
    dashboard = out_dir / "dashboard.html"
    if not dashboard.exists():
        return

    # Kill any process already holding the port.
    # Use fuser (available inside the vLLM container); fall back to lsof on the host.
    subprocess.run(
        ["bash", "-c",
         f"fuser -k {_DASHBOARD_PORT}/tcp 2>/dev/null || "
         f"lsof -ti tcp:{_DASHBOARD_PORT} 2>/dev/null | xargs -r kill -9"],
        capture_output=True,
    )
    time.sleep(0.5)

    # Start http.server in background; --directory makes it cwd-independent.
    subprocess.Popen(
        [
            "python3", "-m", "http.server", str(_DASHBOARD_PORT),
            "--directory", str(out_dir.resolve()),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach so it survives bench.py exiting
    )
    time.sleep(0.5)
    url = f"http://localhost:{_DASHBOARD_PORT}/dashboard.html"
    print(f"\n  Dashboard served at: {url}\n", flush=True)


if __name__ == "__main__":
    main()

