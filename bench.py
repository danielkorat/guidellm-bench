#!/usr/bin/env python3
"""
guidellm-based benchmarking over the same model/tp/quant/eager matrix
defined in run_experiments.py, but driven by guidellm instead of 'vllm bench serve'.

Requirements:
    pip install git+https://github.com/danielkorat/guidellm.git@fix/thinking-model-ttft

Usage:
    ./bench.py               # full benchmark suite (sweep profile for all configs)
    ./bench.py --sanity      # single config, fast (concurrent profile)
    ./bench.py --models openai/gpt-oss-20b --tp 4 --quantization none
"""

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Default parameter sets
# ---------------------------------------------------------------------------

# Eagle3 speculative decoding config for gpt-oss-120b (always appended to full runs).
# Draft model runs at draft_tensor_parallel_size=1 regardless of base TP (Eagle3 constraint).
EAGLE3_SPECULATIVE_CONFIG = (
    '{"model": "nvidia/gpt-oss-120b-Eagle3", "num_speculative_tokens": 5,'
    ' "method": "eagle3", "draft_tensor_parallel_size": 1}'
)

FULL = dict(
    models=["openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-4B-Thinking-2507"],
    tp=[4],          # tp=8 can be added via --tp 4 8 if needed
    quant=["none", "fp8"],
    eager=["true"],   # eager=false skipped (OOM on 20b, negligible diff on Qwen)
    input_len=1024,
    output_len=1024,
    concurrency=16,
    num_prompts=20,  # 20 requests: first 2 (10%) warm-up, last 2 (10%) cool-down excluded
    max_model_len=16384,
    results_dir="./guidellm_results",
    timeout_startup=300,
)

SANITY = dict(
    models=["Qwen/Qwen3-4B-Thinking-2507"],
    tp=[4],
    quant=["none"],
    eager=["true"],
    input_len=64,
    output_len=64,
    concurrency=4,
    num_prompts=4,
    max_model_len=2048,
    results_dir="./guidellm_sanity_results",
    timeout_startup=180,
)

PORT = 8000

# ---------------------------------------------------------------------------
# Skip rules (mirrors run_experiments.py)
# ---------------------------------------------------------------------------

def skip_reason(model: str, quant: Optional[str], eager: bool) -> Optional[str]:
    if quant == "fp8" and not eager:
        return "fp8 + eager=false (known engine failure)"
    if "gpt-oss-20b" in model and quant == "fp8":
        return "gpt-oss-20b + fp8 (mxfp4 config mismatch)"
    if "gpt-oss-20b" in model and not eager:
        return "gpt-oss-20b + eager=false (OOM: UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)"
    if "Qwen3-30B" in model and quant is None:
        return "Qwen3-30B + no quant (IPEX mode-stack bug)"
    return None

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: str
    tp: int
    quant: Optional[str]
    eager: bool
    speculative_config: Optional[str] = None  # JSON string for --speculative_config

    @property
    def name(self) -> str:
        m = self.model.replace("/", "_")
        q = self.quant or "none"
        e = "true" if self.eager else "false"
        suffix = "-eagle3" if self.speculative_config else ""
        return f"{m}_tp{self.tp}_quant-{q}_eager-{e}{suffix}"

# ---------------------------------------------------------------------------
# Shell script helpers
# ---------------------------------------------------------------------------

def _write_script(path: str, *lines: str):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("source /opt/intel/oneapi/setvars.sh --force\n")
        f.write("export no_proxy=localhost,127.0.0.1,0.0.0.0\n")
        f.write("export NO_PROXY=localhost,127.0.0.1,0.0.0.0\n")
        for line in lines:
            f.write(line + "\n")
    os.chmod(path, 0o755)

def _run_tee(cmd: list[str], log_path: Path) -> subprocess.Popen:
    """Start a subprocess, tee-ing stdout+stderr to file and terminal."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    def _tee():
        with open(log_path, "w") as f:
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
    threading.Thread(target=_tee, daemon=True).start()
    return proc


def _write_serve_script(html_path: Path):
    """Write a serve_<name>.sh next to html_path that starts an HTTP server
    on a free port and opens the file in the host browser via $BROWSER."""
    script = html_path.parent / f"serve_{html_path.stem}.sh"
    script.write_text(
        '#!/bin/bash\n'
        '# Auto-generated – serves the HTML report and opens it in your browser.\n'
        'cd "$(dirname "$0")"\n'
        'PORT=$(python3 -c "import socket; s=socket.socket(); s.bind((\'\',0));'
        ' print(s.getsockname()[1]); s.close()")\n'
        f'HTML="{html_path.name}"\n'
        'echo "Serving http://localhost:$PORT/$HTML  (Ctrl-C to stop)"\n'
        'python3 -m http.server "$PORT" &\n'
        'SERVER_PID=$!\n'
        'sleep 1\n'
        '"$BROWSER" "http://localhost:$PORT/$HTML" 2>/dev/null || '
        'xdg-open "http://localhost:$PORT/$HTML" 2>/dev/null || '
        'open "http://localhost:$PORT/$HTML" 2>/dev/null || '
        'echo "Open http://localhost:$PORT/$HTML in your browser"\n'
        'wait "$SERVER_PID"\n'
    )
    script.chmod(0o755)
    print(f"  Serve script → {script}", flush=True)

# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def build_vllm_cmd(cfg: Config, max_model_len: int) -> str:
    parts = [
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve {cfg.model}",
        "--dtype=bfloat16",
        f"--port {PORT}",
        "--block-size 64",
        "--gpu-memory-util 0.9",
        "--no-enable-prefix-caching",
        "--trust-remote-code",
        "--disable-sliding-window",
        "--disable-log-requests",
        "--max-num-batched-tokens=8192",
        f"--max-model-len {max_model_len}",
        f"-tp={cfg.tp}",
    ]
    if cfg.eager:
        parts.append("--enforce-eager")
    if cfg.quant:
        parts.append(f"--quantization {cfg.quant}")
    if cfg.speculative_config:
        parts.append(f"--speculative_config '{cfg.speculative_config}'")
    return " ".join(parts)


def start_server(cfg: Config, max_model_len: int, log_path: Path) -> subprocess.Popen:
    _write_script("/tmp/vllm_server.sh", build_vllm_cmd(cfg, max_model_len))
    return _run_tee(["bash", "--login", "/tmp/vllm_server.sh"], log_path)


def wait_for_server(timeout: int) -> bool:
    print(f"  Waiting for server (timeout={timeout}s)...", flush=True)
    time.sleep(10)
    r = subprocess.run(["bash", "-c", "pgrep -f 'vllm serve' | head -1"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        print("  ERROR: vLLM process not running", flush=True)
        return False
    for elapsed in range(10, timeout, 5):
        try:
            r = subprocess.run(
                ["bash", "-c", f"curl -f -s http://localhost:{PORT}/health"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"  Server ready ({elapsed}s)", flush=True)
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
        if elapsed % 60 == 0:
            print(f"  Still waiting... {elapsed}s elapsed", flush=True)
    print(f"  ERROR: server did not become ready within {timeout}s", flush=True)
    return False


def stop_server(proc: Optional[subprocess.Popen] = None):
    if proc:
        try:
            proc.kill(); proc.wait(timeout=10)
        except Exception:
            pass
    for pat in ("'vllm serve'", "'vllm bench'", "guidellm"):
        subprocess.run(["bash", "-c", f"pkill -f {pat}"], capture_output=True)
    time.sleep(5)

# ---------------------------------------------------------------------------
# GPU monitoring (xpu-smi)
# ---------------------------------------------------------------------------

def _try_float(s: str) -> Optional[float]:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


class GpuMonitor:
    """Background xpu-smi dump monitor. Polls all devices every `interval` seconds.

    Readings are dicts: {t (elapsed s), device (str), util (%), power_w, mem_mib}
    Falls back silently if xpu-smi is unavailable.
    """
    def __init__(self, interval: int = 10):
        self._interval = interval
        self._readings: list[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self):
        self._running = True
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._interval + 5)
        return list(self._readings)

    def _run(self):
        while self._running:
            self._poll()
            for _ in range(self._interval * 10):
                if not self._running:
                    return
                time.sleep(0.1)

    def _poll(self):
        try:
            r = subprocess.run(
                ["bash", "-c", "xpu-smi dump -d -1 -m 0,1,18 -i 1 -n 1 2>/dev/null"],
                capture_output=True, text=True, timeout=8,
            )
            elapsed = round(time.time() - self._t0, 1)
            for line in r.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("Timestamp"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                self._readings.append({
                    "t":       elapsed,
                    "device":  parts[1].strip(),
                    "util":    _try_float(parts[2]),
                    "power_w": _try_float(parts[3]),
                    "mem_mib": _try_float(parts[4]) if len(parts) > 4 else None,
                })
        except Exception:
            pass  # xpu-smi unavailable; monitor silently skipped


# ---------------------------------------------------------------------------
# AIME 2024 dataset preparation
# ---------------------------------------------------------------------------

def prepare_aime_dataset(output_tokens: int = 1024) -> Optional[str]:
    """Download HuggingFaceH4/aime_2024 (30 AIME problems) to a temp JSONL.

    Each row: {"prompt": "<problem text>", "output_tokens": output_tokens}
    Returns the file path, or None on failure (caller falls back to synthetic data).
    """
    out_path = Path("/tmp/aime_2024.jsonl")
    if out_path.exists():
        print(f"  AIME dataset: using cached {out_path}", flush=True)
        return str(out_path)
    try:
        from datasets import load_dataset  # type: ignore
        print("  Downloading HuggingFaceH4/aime_2024 (30 problems)...", flush=True)
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        with open(out_path, "w") as f:
            for row in ds:
                json.dump({"prompt": row["problem"], "output_tokens": output_tokens}, f)
                f.write("\n")
        print(f"  AIME dataset ready: {len(ds)} problems → {out_path}", flush=True)
        return str(out_path)
    except Exception as e:
        print(f"  WARNING: AIME download failed ({e}) — falling back to synthetic data", flush=True)
        return None


# ---------------------------------------------------------------------------
# guidellm benchmark
# ---------------------------------------------------------------------------

def run_guidellm(
    cfg: Config,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    log_path: Path,
    sweep: bool = True,
    dataset_path: Optional[str] = None,
) -> Optional[dict]:
    """Run a guidellm benchmark and save all outputs (json, html) to tmp.

    sweep=True  → synchronous profile with warmup/cooldown (full runs).
    sweep=False → concurrent profile, single rate (sanity).
    dataset_path → use AIME JSONL instead of synthetic data.
    """
    out_tmp = Path("/tmp/guidellm_out")
    shutil.rmtree(out_tmp, ignore_errors=True)
    out_tmp.mkdir()

    # ---- data source ----
    if dataset_path:
        # Use real AIME 2024 problems; all 30 samples, output capped per row.
        data_args = [
            f"--data {dataset_path}",
            '--data-column-mapper \'{"text_column": "prompt", "output_tokens_count_column": "output_tokens"}\'',
            "--data-samples -1",
        ]
        effective_requests = 30  # all AIME problems
    else:
        data_args = [f"--data 'prompt_tokens={input_len},output_tokens={output_len}'"]
        effective_requests = num_prompts

    # ---- profile / limits ----
    if sweep:
        profile_args = [
            "--profile synchronous",
            f"--max-requests {effective_requests}",
            "--warmup 0.1",     # exclude first ~10 % of requests (XPU JIT spike)
            "--cooldown 0.1",   # exclude last  ~10 % of requests (tail effects)
            "--max-errors 5",   # abort early on repeated failures
            "--max-seconds 600",  # safety wall-clock timeout per benchmark
        ]
    else:
        # Sanity / single-rate run (fast, no warmup needed)
        profile_args = ["--profile concurrent", f"--rate {concurrency}",
                        f"--max-requests {num_prompts}"]

    cmd = " ".join([
        "guidellm benchmark run",
        f"--target http://0.0.0.0:{PORT}",
        f"--model {cfg.model}",
        *data_args,
        "--request-format /v1/completions",   # bypass chat template (critical for thinking models)
        *profile_args,
        f"--output-dir {out_tmp}",
        "--outputs json",
        "--outputs html",
        "--disable-console-interactive",
    ])
    _write_script("/tmp/guidellm_bench.sh", cmd)

    proc = subprocess.Popen(
        ["bash", "--login", "/tmp/guidellm_bench.sh"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    with open(log_path, "w") as f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
    proc.wait()

    if proc.returncode != 0:
        print(f"  guidellm exited with code {proc.returncode}", flush=True)
        return None

    result_file = out_tmp / "benchmarks.json"
    if not result_file.exists():
        print("  guidellm result file not found", flush=True)
        return None

    with open(result_file) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Combined HTML dashboard
# ---------------------------------------------------------------------------

def _scalar(v) -> Optional[float]:
    """Extract a single float from a metric that may be a number or a stats dict."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for key in ("median", "p50", "mean", "value"):
            if key in v and isinstance(v[key], (int, float)):
                return float(v[key])
    return None


def _extract_sweep_points(data: dict) -> list[dict]:
    """Return per-benchmark metric rows from a guidellm JSON report.

    guidellm JSON schema (v0.6):
      data['benchmarks'][i]['config']['strategy']['streams']  -> concurrency
      data['benchmarks'][i]['metrics']['<metric>']['successful']['median']

    Each row: {concurrency, ttft_ms, itl_ms, latency_s, throughput_rps, throughput_tps}
    """
    benchmarks = data.get("benchmarks", []) if isinstance(data, dict) else data
    rows = []
    for b in benchmarks:
        # ---- concurrency ----
        strategy = b.get("config", {}).get("strategy", {})
        conc = (strategy.get("streams")
                or strategy.get("max_concurrency")
                or strategy.get("worker_count")
                or len(rows) + 1)

        def med(metric_key: str) -> Optional[float]:
            """metrics[metric_key]['successful']['median']"""
            v = b.get("metrics", {}).get(metric_key, {})
            if isinstance(v, dict):
                s = v.get("successful")
                if isinstance(s, dict):
                    return _scalar(s.get("median"))
                return _scalar(s)
            return _scalar(v)

        rows.append({
            "concurrency":    float(conc),
            "ttft_ms":        med("time_to_first_token_ms"),
            "itl_ms":         med("inter_token_latency_ms"),
            "latency_s":      med("request_latency"),
            "throughput_rps": med("requests_per_second"),
            "throughput_tps": med("output_tokens_per_second"),
        })
    return rows


def build_dashboard_html(
    out_dir: Path,
    succeeded: list[str],
    gpu_data: Optional[dict] = None,      # {cfg.name: [{t, device, util, mem_mib, ...}]}
) -> Optional[Path]:
    """Combine all per-config guidellm outputs into a single dashboard.html.

    Overview tab  – latency/throughput bar charts + GPU memory bar + sweep line charts.
    Per-config tab – per-run stats table, 4 perf charts + 2 GPU monitoring charts.
    """
    COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
              "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"]

    records = []
    for name in succeeded:
        json_path = out_dir / f"{name}_benchmarks.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        records.append({
            "name":   name,
            "points": _extract_sweep_points(data),
        })

    if not records:
        print("  No benchmark results found for dashboard", flush=True)
        return None

    # ------------------------------------------------------------------
    # Aggregate data for charts
    # ------------------------------------------------------------------
    bar_labels, bar_ttft, bar_itl, bar_tput_rps, bar_tput_tps = [], [], [], [], []
    bar_gpu_mem: list[Optional[float]] = []   # total peak GPU memory across all devices (MiB)
    line_ttft_ds, line_tput_ds = [], []

    for i, rec in enumerate(records):
        color  = COLORS[i % len(COLORS)]
        pts    = sorted(rec["points"], key=lambda p: p["concurrency"] or 0)
        label  = rec["name"].replace("openai_", "").replace("Qwen_", "")[:60]
        bar_labels.append(label)

        peak = pts[-1] if pts else {}
        bar_ttft.append(peak.get("ttft_ms"))
        bar_itl.append(peak.get("itl_ms"))
        bar_tput_rps.append(peak.get("throughput_rps"))
        bar_tput_tps.append(peak.get("throughput_tps"))

        # GPU peak memory: sum of peak mem_mib per device across all readings for this config
        gpu_readings = (gpu_data or {}).get(rec["name"], [])
        if gpu_readings:
            by_dev: dict[str, float] = {}
            for r in gpu_readings:
                if r.get("mem_mib") is not None:
                    dev = r["device"]
                    by_dev[dev] = max(by_dev.get(dev, 0.0), r["mem_mib"])
            bar_gpu_mem.append(round(sum(by_dev.values()), 1) if by_dev else None)
        else:
            bar_gpu_mem.append(None)

        xy_ttft  = [{"x": p["concurrency"], "y": p["ttft_ms"]}        for p in pts if p.get("ttft_ms")        is not None]
        xy_tput  = [{"x": p["concurrency"], "y": p["throughput_rps"]}  for p in pts if p.get("throughput_rps") is not None]
        if xy_ttft:
            line_ttft_ds.append({"label": label, "data": xy_ttft, "borderColor": color,
                                  "backgroundColor": color + "33", "tension": 0.3, "spanGaps": True})
        if xy_tput:
            line_tput_ds.append({"label": label, "data": xy_tput,  "borderColor": color,
                                  "backgroundColor": color + "33", "tension": 0.3, "spanGaps": True})

    # ------------------------------------------------------------------
    # Tab navigation + content
    # ------------------------------------------------------------------
    tab_nav_html = '<li class="nav-item"><a class="nav-link active" data-bs-toggle="tab" href="#tab-overview">&#128202; Overview</a></li>'
    overview_content = '''<div class="tab-pane fade show active" id="tab-overview">
  <div class="row g-4 mt-2">
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">TTFT (ms) &mdash; median</div><div class="card-body"><canvas id="c-bar-ttft"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">ITL (ms) &mdash; median</div><div class="card-body"><canvas id="c-bar-itl"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">Throughput req/s</div><div class="card-body"><canvas id="c-bar-rps"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">Output tok/s</div><div class="card-body"><canvas id="c-bar-tps"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">Peak GPU Memory Used (MiB, all devices summed)</div><div class="card-body"><canvas id="c-bar-mem"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">TTFT (ms) vs Concurrency &mdash; sweep</div><div class="card-body"><canvas id="c-line-ttft" style="max-height:260px"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">Throughput (req/s) vs Concurrency &mdash; sweep</div><div class="card-body"><canvas id="c-line-rps" style="max-height:260px"></canvas></div></div></div>
  </div>
</div>'''

    config_tabs_nav = ""
    config_tabs_content = ""
    for idx, rec in enumerate(records):
        tab_id  = "tab-" + rec["name"].replace("/", "-").replace("_", "-")
        short   = rec["name"].replace("openai_", "").replace("Qwen_", "")[:55]
        color   = COLORS[idx % len(COLORS)]
        pts     = sorted(rec["points"], key=lambda p: p["concurrency"] or 0)
        concs   = [p["concurrency"] for p in pts]
        cid     = lambda s: f"cfg-{idx}-{s}"   # unique canvas id per config

        # Build a summary stats table from the last (peak) row
        peak   = pts[-1] if pts else {}
        stats  = [
            ("TTFT (ms)",       peak.get("ttft_ms")),
            ("ITL (ms)",        peak.get("itl_ms")),
            ("Latency (s)",     peak.get("latency_s")),
            ("Req/s",           peak.get("throughput_rps")),
            ("Output tok/s",    peak.get("throughput_tps")),
            ("Concurrency",     peak.get("concurrency")),
        ]
        stat_rows = "".join(
            f'<tr><th class="text-end pe-3">{lbl}</th>'
            f'<td><strong>{f"{v:.3g}" if v is not None else "—"}</strong></td></tr>'
            for lbl, v in stats
        )

        # Per-config inline charts JS (concurrency on x-axis)
        def series_js(metric: str) -> str:
            vals = [p.get(metric) for p in pts]
            return json.dumps(vals)

        config_tabs_nav     += f'<li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#{tab_id}">{short}</a></li>\n'
        # GPU readings for this config (grouped by device).
        gpu_readings = (gpu_data or {}).get(rec["name"], [])
        gpu_dev_mem: dict[str, list] = {}   # device -> [{x: t, y: mem_mib}]
        gpu_dev_util: dict[str, list] = {}  # device -> [{x: t, y: util}]
        for r in gpu_readings:
            dev = r["device"]
            if r.get("mem_mib") is not None:
                gpu_dev_mem.setdefault(dev, []).append({"x": r["t"], "y": r["mem_mib"]})
            if r.get("util") is not None:
                gpu_dev_util.setdefault(dev, []).append({"x": r["t"], "y": r["util"]})
        GPU_COLORS = ["#e15759","#4e79a7","#f28e2b","#76b7b2","#59a14f",
                      "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"]
        mem_datasets = json.dumps([
            {"label": f"GPU {dev}", "data": pts_list,
             "borderColor": GPU_COLORS[j % len(GPU_COLORS)],
             "backgroundColor": GPU_COLORS[j % len(GPU_COLORS)] + "33",
             "tension": 0.3, "pointRadius": 3, "fill": False}
            for j, (dev, pts_list) in enumerate(sorted(gpu_dev_mem.items()))
        ])
        util_datasets = json.dumps([
            {"label": f"GPU {dev}", "data": pts_list,
             "borderColor": GPU_COLORS[j % len(GPU_COLORS)],
             "backgroundColor": GPU_COLORS[j % len(GPU_COLORS)] + "33",
             "tension": 0.3, "pointRadius": 3, "fill": False}
            for j, (dev, pts_list) in enumerate(sorted(gpu_dev_util.items()))
        ])
        has_gpu = bool(gpu_dev_mem)

        config_tabs_content += f'''<div class="tab-pane fade" id="{tab_id}">
  <h6 class="mt-3 mb-2 fw-bold">{rec["name"]}</h6>
  <div class="row g-3">
    <div class="col-md-4">
      <table class="table table-sm table-bordered table-hover">
        <thead><tr><th>Metric</th><th>Median</th></tr></thead>
        <tbody>{stat_rows}</tbody>
      </table>
    </div>
    <div class="col-md-8">
      <div class="row g-2">
        <div class="col-6"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">TTFT (ms) vs Concurrency</div><div class="card-body p-2"><canvas id="{cid("ttft")}"></canvas></div></div></div>
        <div class="col-6"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">ITL (ms) vs Concurrency</div><div class="card-body p-2"><canvas id="{cid("itl")}"></canvas></div></div></div>
        <div class="col-6"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">Req/s vs Concurrency</div><div class="card-body p-2"><canvas id="{cid("rps")}"></canvas></div></div></div>
        <div class="col-6"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">Output tok/s vs Concurrency</div><div class="card-body p-2"><canvas id="{cid("tps")}"></canvas></div></div></div>
        <div class="col-6"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">GPU Memory Used (MiB) over Time</div><div class="card-body p-2">{'<canvas id="' + cid('gmem') + '"></canvas>' if has_gpu else '<p class="text-muted small m-1">No xpu-smi data</p>'}</div></div></div>
        <div class="col-6"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">GPU Utilization (%) over Time</div><div class="card-body p-2">{'<canvas id="' + cid('gutil') + '"></canvas>' if has_gpu else '<p class="text-muted small m-1">No xpu-smi data</p>'}</div></div></div>
      </div>
    </div>
  </div>
</div>
<script>
(function(){{
  const C = {json.dumps(concs)};
  const col = "{color}";
  function cfgLine(id, vals, yLabel) {{
    const el = document.getElementById(id);
    if (!el) return;
    new Chart(el, {{
      type: 'line',
      data: {{ labels: C, datasets: [{{
        label: yLabel, data: vals, borderColor: col,
        backgroundColor: col + '33', pointRadius: 4, tension: 0.3, spanGaps: true
      }}]}},
      options: {{
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Concurrency' }} }},
          y: {{ beginAtZero: false, title: {{ display: true, text: yLabel }} }}
        }}
      }}
    }});
  }}
  cfgLine("{cid("ttft")}", {series_js("ttft_ms")},        "TTFT (ms)");
  cfgLine("{cid("itl")}",  {series_js("itl_ms")},         "ITL (ms)");
  cfgLine("{cid("rps")}",  {series_js("throughput_rps")}, "Req/s");
  cfgLine("{cid("tps")}",  {series_js("throughput_tps")}, "Output tok/s");
  (function() {{
    function gpuLine(id, datasets, yLabel) {{
      const el = document.getElementById(id);
      if (!el || !datasets.length) return;
      new Chart(el, {{
        type: 'line', data: {{ datasets }},
        options: {{
          parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
          scales: {{
            x: {{ type: 'linear', title: {{ display: true, text: 'Elapsed (s)' }} }},
            y: {{ beginAtZero: false, title: {{ display: true, text: yLabel }} }}
          }},
          plugins: {{ legend: {{ display: true, position: 'top' }} }}
        }}
      }});
    }}
    gpuLine("{cid("gmem")}",  {mem_datasets},  "Memory (MiB)");
    gpuLine("{cid("gutil")}", {util_datasets}, "Util (%)");
  }})();
}})();
</script>\n'''

    title = f"guidellm Benchmark Dashboard \u2014 {out_dir.name}"
    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    body {{ background:#f8f9fa; }}
    .nav-tabs .nav-link {{ font-size:.82rem; padding:.35rem .7rem; }}
    .card-header {{ background:#e9ecef; font-size:.88rem; }}
  </style>
</head>
<body>
<div class="container-fluid py-3">
  <h4 class="mb-3 text-center fw-bold">{title}</h4>
  <ul class="nav nav-tabs flex-wrap mb-0" id="mainTabs">
    {tab_nav_html}
    {config_tabs_nav}
  </ul>
  <div class="tab-content border border-top-0 rounded-bottom bg-white p-3">
    {overview_content}
    {config_tabs_content}
  </div>
</div>
<script>
const LABELS = {json.dumps(bar_labels)};
const PAL    = {json.dumps(COLORS)};

function bar(id, data, unit) {{
  const el = document.getElementById(id);
  if (!el) return;
  new Chart(el, {{
    type: 'bar',
    data: {{ labels: LABELS,
             datasets: [{{ label: unit, data,
               backgroundColor: PAL.slice(0, data.length).map(c => c + 'cc'),
               borderColor:     PAL.slice(0, data.length), borderWidth: 1 }}] }},
    options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
  }});
}}

function line(id, datasets) {{
  const el = document.getElementById(id);
  if (!el || !datasets.length) return;
  new Chart(el, {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Concurrency' }} }},
        y: {{ beginAtZero: false }}
      }}
    }}
  }});
}}

bar('c-bar-ttft', {json.dumps(bar_ttft)},     'ms');
bar('c-bar-itl',  {json.dumps(bar_itl)},      'ms');
bar('c-bar-rps',  {json.dumps(bar_tput_rps)}, 'req/s');
bar('c-bar-tps',  {json.dumps(bar_tput_tps)}, 'tok/s');
bar('c-bar-mem',  {json.dumps(bar_gpu_mem)},  'MiB');
line('c-line-ttft', {json.dumps(line_ttft_ds)});
line('c-line-rps',  {json.dumps(line_tput_ds)});
</script>
</body>
</html>'''

    out_path = out_dir / "dashboard.html"
    out_path.write_text(page)
    print(f"\n  Dashboard → {out_path}", flush=True)
    _write_serve_script(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="guidellm benchmarking for vLLM model/tp/quant/eager configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sanity", action="store_true", help="Use sanity (small) defaults")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--tp", nargs="+", type=int)
    parser.add_argument("--quantization", nargs="+", help='Use "none" to omit --quantization flag')
    parser.add_argument("--enforce-eager", nargs="+", choices=["true", "false"])
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--results-dir")
    parser.add_argument("--timeout-startup", type=int)
    parser.add_argument("--eagle3", action="store_true",
                        help="Append gpt-oss-120b Eagle3 speculative-decoding config (disabled by default)")
    args = parser.parse_args()

    D = SANITY if args.sanity else FULL

    def get(attr: str, key: str):
        v = getattr(args, attr.replace("-", "_"), None)
        return v if v is not None else D[key]

    models       = get("models", "models")
    tp_values    = get("tp", "tp")
    quant_list   = [None if q == "none" else q for q in get("quantization", "quant")]
    eager_list   = [e == "true" for e in get("enforce_eager", "eager")]
    input_len    = get("input_len", "input_len")
    output_len   = get("output_len", "output_len")
    concurrency  = get("concurrency", "concurrency")
    num_prompts  = get("num_prompts", "num_prompts")
    max_model_len = get("max_model_len", "max_model_len")
    results_dir  = get("results_dir", "results_dir")
    timeout_startup = get("timeout_startup", "timeout_startup")

    # Timestamped output directory (Israel time)
    ts = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y%m%d_%H%M")
    out_dir = Path(results_dir) / ts
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    print(f"Results → {out_dir}\n")

    # Prepare AIME dataset once (cached; None if download fails → synthetic fallback)
    aime_path = prepare_aime_dataset(output_tokens=1024)

    # GPU monitoring data collected per config
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

    # Append gpt-oss-120b Eagle3 speculative-decoding config only when --eagle3 is passed.
    # Fixed config: tp=8, eager=true, no --quantization (mxfp4 baked into model).
    if not args.sanity and args.eagle3:
        configs.append(Config(
            model="openai/gpt-oss-120b",
            tp=8,
            quant=None,
            eager=True,
            speculative_config=EAGLE3_SPECULATIVE_CONFIG,
        ))
        print(f"  +    openai/gpt-oss-120b  tp=8  eagle3  (appended via --eagle3)")

    print(f"\n{len(configs)} configurations to benchmark\n")

    succeeded, failed = [], []

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(configs)}]  {cfg.name}")
        print(f"{'='*60}")
        t0 = time.time()

        stop_server()   # clean up any lingering processes

        monitor = GpuMonitor(interval=10)
        monitor.start()

        proc = start_server(cfg, max_model_len, log_dir / f"{cfg.name}_server.log")
        if not wait_for_server(timeout_startup):
            stop_server(proc)
            monitor.stop()
            failed.append((cfg.name, "server startup failed"))
            continue

        bench_log = log_dir / f"{cfg.name}_bench.log"
        is_sweep = not args.sanity
        data = run_guidellm(cfg, input_len, output_len, concurrency, num_prompts,
                            bench_log, sweep=is_sweep, dataset_path=aime_path)
        stop_server(proc)

        # Collect GPU readings and persist to disk
        gpu_readings = monitor.stop()
        gpu_data[cfg.name] = gpu_readings
        gpu_file = out_dir / f"{cfg.name}_gpu_monitor.json"
        with open(gpu_file, "w") as f:
            json.dump(gpu_readings, f)

        if data is None:
            failed.append((cfg.name, "benchmark failed or produced no output"))
            continue

        # Copy all guidellm output files (json, html, ...) to the results directory.
        out_tmp = Path("/tmp/guidellm_out")
        saved_files = []
        for src in sorted(out_tmp.iterdir()):
            dest = out_dir / f"{cfg.name}_{src.name}"
            shutil.copy(src, dest)
            saved_files.append(dest.name)
            if dest.suffix == ".html":
                _write_serve_script(dest)

        elapsed = time.time() - t0
        print(f"\n  ✓  {cfg.name}  ({elapsed:.0f}s)  →  {', '.join(saved_files)}", flush=True)
        succeeded.append(cfg.name)

    # Summary
    print(f"\n{'='*60}")
    print(f"Finished:  {len(succeeded)} succeeded  /  {len(failed)} failed  /  {len(configs)} total")
    if failed:
        print("\nFailed configurations:")
        for name, reason in failed:
            print(f"  ✗  {name}:  {reason}")
    print(f"\nResults: {out_dir}")

    # Build combined dashboard from all completed runs
    if succeeded:
        build_dashboard_html(out_dir, succeeded, gpu_data=gpu_data)


if __name__ == "__main__":
    main()
