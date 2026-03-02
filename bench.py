#!/usr/bin/env python3
"""
guidellm-bench — entry point.

Usage:
    ./bench.py               # full benchmark matrix
    ./bench.py --sanity      # single config, fast smoke-test
    ./bench.py --models openai/gpt-oss-20b --tp 4 --quantization none

See guidellm_bench/ for implementation details.
"""

import argparse
import json
import os
import sys
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
    build_dashboard_html,
    ensure_container_running,
    prepare_aime_dataset,
    run_guidellm,
    skip_reason,
    start_server,
    stop_server,
    wait_for_server,
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
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    D = SANITY if args.sanity else FULL

    # Ensure the Docker container is running before any other operation.
    ensure_container_running()

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

    # Timestamped output directory (Israel time)
    ts      = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y%m%d_%H%M")
    out_dir = Path(results_dir) / ts
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Self-log: tee all output into out_dir/bench.log; write pid for easy kill.
    _log_file = open(out_dir / "bench.log", "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_file)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, _log_file)  # type: ignore[assignment]
    (out_dir / "bench.pid").write_text(str(os.getpid()))

    print(f"Results → {out_dir}\n")

    # Prepare AIME dataset once (cached; None → synthetic fallback)
    aime_path = prepare_aime_dataset(output_tokens=1024)

    gpu_data: dict[str, list] = {}

    # Build config list, applying skip rules
    configs: list[Config] = []
    for model in models:
        for tp in tp_values:
            for quant in quant_list:
                for eager in eager_list:
                    reason = skip_reason(model, quant, eager)
                    if reason:
                        print(f"  SKIP  {model}  tp={tp}  quant={quant or 'none'}  eager={eager}: {reason}")
                        continue
                    configs.append(Config(model=model, tp=tp, quant=quant, eager=eager))

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

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(configs)}]  {cfg.name}")
        print(f"{'='*60}")
        t0 = time.time()

        stop_server()  # clean up any lingering processes

        monitor = GpuMonitor(interval=10)
        monitor.start()

        proc = start_server(cfg, max_model_len, log_dir / f"{cfg.name}_server.log")
        if not wait_for_server(timeout_startup):
            stop_server(proc)
            monitor.stop()
            failed.append((cfg.name, "server startup failed"))
            continue

        data = run_guidellm(
            cfg, input_len, output_len, concurrency, num_prompts,
            log_dir / f"{cfg.name}_bench.log",
            sweep=not args.sanity,
            dataset_path=aime_path,
        )
        stop_server(proc)

        gpu_readings = monitor.stop()
        gpu_data[cfg.name] = gpu_readings
        with open(out_dir / f"{cfg.name}_gpu_monitor.json", "w") as f:
            json.dump(gpu_readings, f)

        if data is None:
            failed.append((cfg.name, "benchmark failed or produced no output"))
            continue

        saved = copy_results(cfg.name, out_dir)
        for fname in saved:
            dest = out_dir / fname
            if dest.suffix == ".html":
                write_serve_script(dest)

        elapsed = time.time() - t0
        print(f"\n  ✓  {cfg.name}  ({elapsed:.0f}s)  →  {', '.join(saved)}", flush=True)
        succeeded.append(cfg.name)

    # Summary
    print(f"\n{'='*60}")
    print(f"Finished:  {len(succeeded)} succeeded  /  {len(failed)} failed  /  {len(configs)} total")
    if failed:
        print("\nFailed configurations:")
        for name, reason in failed:
            print(f"  ✗  {name}:  {reason}")
    print(f"\nResults: {out_dir}")

    if succeeded:
        build_dashboard_html(out_dir, succeeded, gpu_data=gpu_data)


if __name__ == "__main__":
    main()

