#!/usr/bin/env python3
"""
guidellm-bench — entry point.

Usage:
    ./bench.py               # full benchmark matrix
    ./bench.py --sanity      # single config, fast smoke-test
    ./bench.py --ep          # include Expert Parallelism variants
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
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from guidellm_bench import (
    EAGLE3_SPECULATIVE_CONFIG,
    FULL,
    SANITY,
    Config,
    GpuMonitor,
    SERVER_STATUS_PATH,
    XpuKernelHangError,
    build_dashboard_html,
    is_moe_model,
    parse_model_mem_gib,
    prepare_aime_dataset,
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
        "--resume", metavar="DIR", nargs="?", const="",
        help="Resume an interrupted run. With a DIR argument, reuses that directory. "
             "Without a DIR argument, automatically resumes the latest run in the "
             "results directory. Skips any config whose _benchmarks.json already exists.",
    )
    return p


def main() -> None:
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
        ts      = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y%m%d_%H%M")
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

    # Prepare AIME dataset once (cached; None → synthetic fallback)
    aime_path = prepare_aime_dataset(output_tokens=1024)

    # When resuming, seed gpu_data from any existing monitor files
    gpu_data: dict[str, list] = {}
    model_mem_gib: dict[str, float] = {}
    if args.resume is not None:
        for p in out_dir.glob("*_gpu_monitor.json"):
            cfg_name = p.stem.replace("_gpu_monitor", "")
            with open(p) as f:
                gpu_data[cfg_name] = json.load(f)
        # Re-parse model memory from existing server logs
        for p in out_dir.glob("logs/*_server.log"):
            cfg_name = p.stem.replace("_server", "")
            mem = parse_model_mem_gib(p)
            if mem is not None:
                model_mem_gib[cfg_name] = mem

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

    # Expert Parallelism (EP) variants for MoE models — opt-in via --ep
    if args.ep:
        ep_configs = [
            # gpt-oss-20b: mxfp4 baked in (no quant), tp=4 with ep=4
            Config(model="openai/gpt-oss-20b",    tp=4, quant=None,  eager=True, expert_parallel_size=4),
            # Qwen3-30B-A3B (MoE): fp8, tp=2+ep=2 and tp=4+ep=4
            Config(model="Qwen/Qwen3-30B-A3B",    tp=2, quant="fp8", eager=True, expert_parallel_size=2),
            Config(model="Qwen/Qwen3-30B-A3B",    tp=4, quant="fp8", eager=True, expert_parallel_size=4),
        ]
        for c in ep_configs:
            if models and c.model not in models:
                continue
            if not is_moe_model(c.model):
                print(f"  SKIP  {c.model}  ep={c.expert_parallel_size}: EP requires MoE architecture (not applicable to this model)")
                continue
            configs.append(c)
            print(f"  +    {c.model}  tp={c.tp}  quant={c.quant or 'none'}  ep={c.expert_parallel_size}  (EP via --ep)")

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

            proc = start_server(cfg, max_model_len, _server_log)
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
                failed.append((cfg.name, "server startup failed"))
                continue

            # Persist running server config so next iteration can skip restart.
            write_server_status(cfg, max_model_len, proc.pid, _server_log)
            current_server_proc = proc

        monitor = GpuMonitor(interval=10)
        monitor.start()

        data = run_guidellm(
            cfg, input_len, output_len, concurrency, num_prompts,
            log_dir / f"{cfg.name}_bench.log",
            sweep=not args.sanity,
            dataset_path=aime_path,
        )
        # Server intentionally kept alive — reused by next config if possible.
        # Final teardown happens after all configs or before a config with a
        # different server requirement (handled above at the top of the loop).

        gpu_readings = monitor.stop()
        gpu_data[cfg.name] = gpu_readings
        with open(out_dir / f"{cfg.name}_gpu_monitor.json", "w") as f:
            json.dump(gpu_readings, f)

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

        elapsed = time.time() - t0
        print(f"\n  ✓  {cfg.name}  ({elapsed:.0f}s)  →  {', '.join(saved)}", flush=True)
        succeeded.append(cfg.name)

    # Final cleanup — shut down the server that handled the last config.
    stop_server(current_server_proc)

    # Summary
    print(f"\n{'='*60}")
    print(f"Finished:  {len(succeeded)} succeeded  /  {len(failed)} failed  /  {len(configs)} total")
    if failed:
        print("\nFailed configurations:")
        for name, reason in failed:
            print(f"  ✗  {name}:  {reason}")
    print(f"\nResults: {out_dir}")

    if succeeded:
        build_dashboard_html(out_dir, succeeded, gpu_data=gpu_data, model_mem_gib=model_mem_gib)
        _serve_dashboard(out_dir)


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

