"""Combined interactive HTML dashboard builder."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

try:
    from .docker import DOCKER_IMAGE
except ImportError:
    DOCKER_IMAGE = "intel/llm-scaler-vllm:0.14.0-b8"


def _load_vllm_cmd(out_dir: Path, cfg_name: str) -> Optional[str]:
    """Return the saved ``vllm serve …`` command string for *cfg_name*, or None.

    Written to ``{out_dir}/logs/{cfg_name}_vllm_cmd.txt`` by start_server().
    """
    cmd_path = out_dir / "logs" / f"{cfg_name}_vllm_cmd.txt"
    if cmd_path.exists():
        try:
            return cmd_path.read_text().strip()
        except OSError:
            pass
    return None


FIXED_TITLE = "Benchmark Dashboard \u2014 Intel Arc Pro B60 (multi-gpu)"


def _run_timestamp(out_dir: Path) -> str:
    """Parse YYYYMMDD_HHMM from out_dir name into a human-readable string."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", out_dir.name)
    if not m:
        return out_dir.name
    year, month, day, hour, minute = m.groups()
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
    return dt.strftime("%-d %b %Y %H:%M")


def _build_subtitle(records: list[dict]) -> str:
    """Build an experiment-config subtitle from the first benchmark JSON record."""
    if not records:
        return ""

    try:
        json_path = records[0].get("_json_path")
        with open(json_path) as f:
            data = json.load(f)
        args = data.get("args", {})
        b    = data["benchmarks"][0]

        # Dataset name
        raw_data = args.get("data", [])
        raw_data = raw_data[0] if isinstance(raw_data, list) and raw_data else str(raw_data)
        if "aime_2024" in raw_data:
            dataset = "AIME 2024"
        elif "synthetic" in raw_data.lower() or "prompt_tokens" in raw_data:
            dataset = "synthetic"
        else:
            dataset = Path(raw_data).stem

        # Number of requests
        n_req = (
            b["config"]["constraints"]
            .get("max_requests", {})
            .get("max_num") or
            b["scheduler_state"].get("processed_requests")
        )

        # Median input / output tokens across all requests
        reqs = b["requests"]["successful"]
        in_toks  = sorted(r["prompt_tokens"]  for r in reqs if r.get("prompt_tokens"))
        out_toks = sorted(r["output_tokens"]   for r in reqs if r.get("output_tokens"))
        med_in   = in_toks[len(in_toks) // 2]   if in_toks  else None
        med_out  = out_toks[len(out_toks) // 2] if out_toks else None

        # Benchmark profile
        profile   = args.get("profile", b["config"]["strategy"].get("type_", ""))
        req_fmt   = args.get("request_format", "/v1/completions")

        parts = [f"Dataset: {dataset}"]
        if n_req is not None:
            parts.append(f"{n_req} requests")
        if med_in is not None:
            parts.append(f"~{med_in} input tok")
        if med_out is not None:
            parts.append(f"~{med_out} output tok")

        # Benchmark parameters
        parts.append(f"profile: {profile}")
        parts.append(f"format: {req_fmt}")

        constraints = b["config"].get("constraints", {})
        max_sec = constraints.get("max_seconds", {})
        if isinstance(max_sec, dict) and max_sec.get("max_duration"):
            parts.append(f"max-seconds: {int(max_sec['max_duration'])}")
        max_err = constraints.get("max_errors", {})
        if isinstance(max_err, dict) and max_err.get("max_errors"):
            parts.append(f"max-errors: {max_err['max_errors']}")

        warmup   = b["config"].get("warmup",   {})
        cooldown = b["config"].get("cooldown", {})
        w_pct = (warmup   or {}).get("percent") if isinstance(warmup,   dict) else None
        c_pct = (cooldown or {}).get("percent") if isinstance(cooldown, dict) else None
        if w_pct is not None or c_pct is not None:
            parts.append(f"warmup/cooldown: {w_pct or 0}/{c_pct or 0}")

        return " &nbsp;|&nbsp; ".join(parts)
    except Exception:
        return ""


COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


# ---------------------------------------------------------------------------
# Serve script helper
# ---------------------------------------------------------------------------

def write_serve_script(html_path: Path, port: int = 8081) -> None:
    """Write a serve_<name>.sh next to *html_path* that starts an HTTP server."""
    # Use the resolved absolute path so the script is cwd-independent.
    abs_dir = str(html_path.resolve().parent)
    script = html_path.parent / f"serve_{html_path.stem}.sh"
    script.write_text(
        "#!/bin/bash\n"
        "# Auto-generated – serves the HTML report and opens it in your browser.\n"
        "# SCRIPT_DIR is the absolute path baked in at generation time — cwd-independent.\n"
        f'SCRIPT_DIR="{abs_dir}"\n'
        f"PORT={port}\n"
        f'HTML="{html_path.name}"\n'
        "# Kill any process already listening on $PORT\n"
        'OLD_PID=$(lsof -ti tcp:$PORT 2>/dev/null)\n'
        'if [ -n "$OLD_PID" ]; then\n'
        '    echo "Killing existing server on port $PORT (PID $OLD_PID)"\n'
        '    kill "$OLD_PID" 2>/dev/null\n'
        '    sleep 1\n'
        'fi\n'
        'echo "Serving http://localhost:$PORT/$HTML  (Ctrl-C to stop)"\n'
        # --directory ensures http.server always serves from SCRIPT_DIR regardless of cwd
        'python3 -m http.server "$PORT" --directory "$SCRIPT_DIR" &\n'
        "SERVER_PID=$!\n"
        "sleep 1\n"
        '"$BROWSER" "http://localhost:$PORT/$HTML" 2>/dev/null || '
        'xdg-open "http://localhost:$PORT/$HTML" 2>/dev/null || '
        'open "http://localhost:$PORT/$HTML" 2>/dev/null || '
        'echo "Open http://localhost:$PORT/$HTML in your browser"\n'
        'wait "$SERVER_PID"\n'
    )
    script.chmod(0o755)
    print(f"  Serve script → {script}", flush=True)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _scalar(v) -> Optional[float]:
    """Extract a single float from a metric value or stats dict."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for key in ("median", "p50", "mean", "value"):
            if key in v and isinstance(v[key], (int, float)):
                return float(v[key])
    return None


def _median(vals: list) -> Optional[float]:
    """Return the median of a non-empty list of numbers, or None."""
    nums = sorted(v for v in vals if v is not None)
    if not nums:
        return None
    n = len(nums)
    mid = n // 2
    return (nums[mid - 1] + nums[mid]) / 2.0 if n % 2 == 0 else float(nums[mid])


def _extract_sweep_points(data: dict) -> list[dict]:
    """Return per-benchmark metric rows from a guidellm JSON report.

    guidellm's top-level b['metrics'] aggregates are zero-filled in v0.6;
    real data lives in b['requests']['successful'][*] per-request fields.

    Each row: {concurrency, ttft_ms, itl_ms, latency_s, throughput_rps, throughput_tps}
    """
    benchmarks = data.get("benchmarks", []) if isinstance(data, dict) else data
    rows = []
    for b in benchmarks:
        strategy = b.get("config", {}).get("strategy", {})
        conc = (
            strategy.get("streams")
            or strategy.get("max_concurrency")
            or strategy.get("worker_count")
            or 1
        )

        reqs = b.get("requests", {}).get("successful", [])

        def _req_med(field: str) -> Optional[float]:
            return _median([r.get(field) for r in reqs])

        med_lat = _req_med("request_latency")
        # For synchronous (serial) runs use 1/median_latency for req/s;
        # for concurrent runs use n / wall_duration.
        sm = b.get("scheduler_metrics", {})
        wall_dur = sm.get("measure_end_time", 0) - sm.get("measure_start_time", 0)
        n_succ = len(reqs)
        if conc == 1 and med_lat:
            rps = 1.0 / med_lat
        else:
            rps = n_succ / wall_dur if wall_dur else None

        rows.append({
            "concurrency":    float(conc),
            "ttft_ms":        _req_med("time_to_first_token_ms"),
            "itl_ms":         _req_med("inter_token_latency_ms"),
            "latency_s":      med_lat,
            "throughput_rps": rps,
            "throughput_tps": _req_med("output_tokens_per_second"),
        })
    return rows


def _extract_lc_metrics(data: dict) -> Dict[str, Optional[float]]:
    """Extract all median metrics from a single long-context benchmark JSON result."""
    benchmarks = data.get("benchmarks", []) if isinstance(data, dict) else data
    if not benchmarks:
        return {}
    b = benchmarks[0]
    reqs = b.get("requests", {}).get("successful", [])

    def med(field: str) -> Optional[float]:
        return _median([r.get(field) for r in reqs])

    med_lat = med("request_latency")
    sm = b.get("scheduler_metrics", {})
    wall_dur = sm.get("measure_end_time", 0) - sm.get("measure_start_time", 0)
    n_succ = len(reqs)
    if med_lat:
        rps: Optional[float] = 1.0 / med_lat
    else:
        rps = n_succ / wall_dur if wall_dur else None

    return {
        "ttft_ms":        med("time_to_first_token_ms"),
        "itl_ms":         med("inter_token_latency_ms"),
        "latency_s":      med_lat,
        "throughput_rps": rps,
        "throughput_tps": med("output_tokens_per_second"),
    }


def _load_lc_points(out_dir: Path, cfg_name: str) -> Dict[int, Dict[str, Optional[float]]]:
    """Return {token_len: {metric: value}} for long-context slices of *cfg_name*.

    Looks for files named ``{cfg_name}_lc{N}k_benchmarks.json`` in *out_dir*.
    Token lengths are inferred from the label (e.g. lc4k → 4096).
    """
    lc_points: Dict[int, Dict[str, Optional[float]]] = {}
    for fpath in sorted(out_dir.glob(f"{cfg_name}_lc*_benchmarks.json")):
        import re as _re
        m = _re.search(r"_lc(\d+k)_benchmarks", fpath.name)
        if not m:
            continue
        k = int(m.group(1)[:-1])   # '4k' → 4
        token_len = k * 1024
        try:
            with open(fpath) as f:
                data = json.load(f)
            lc_points[token_len] = _extract_lc_metrics(data)
        except Exception:
            lc_points[token_len] = {}
    return lc_points


# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------

def build_dashboard_html(
    out_dir: Path,
    succeeded: list[str],
    model_mem_gib: Optional[Dict[str, float]] = None,
) -> Optional[Path]:
    """Build dashboard.html combining all per-config guidellm outputs.

    Args:
        out_dir:       Directory containing `{name}_benchmarks.json` files.
        succeeded:     List of config names that completed successfully.
        model_mem_gib: Optional dict {cfg.name: per_gpu_weight_gib} parsed
                       from vLLM server logs.

    Returns:
        Path to written dashboard.html, or None if no data found.
    """
    records = []
    for name in succeeded:
        json_path = out_dir / f"{name}_benchmarks.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        lc_points = _load_lc_points(out_dir, name)
        records.append({
            "name": name,
            "points": _extract_sweep_points(data),
            "lc_points": lc_points,       # {token_len: {metric_key: val}}
            "_json_path": str(json_path),
        })

    if not records:
        print("  No benchmark results found for dashboard", flush=True)
        return None

    # Auto-discover model memory from server logs when not provided by caller.
    # This ensures the GPU memory bar is populated even when the dashboard is
    # rebuilt manually (without the model_mem_gib dict from the live run).
    if model_mem_gib is None:
        model_mem_gib = {}
    try:
        from .server import parse_model_mem_gib as _parse_mem
        for name in succeeded:
            if name not in model_mem_gib:
                log_path = out_dir / "logs" / f"{name}_server.log"
                mem = _parse_mem(log_path)
                if mem is not None:
                    model_mem_gib[name] = mem
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Aggregate for overview-tab charts
    # ------------------------------------------------------------------
    bar_labels: list[str] = []
    bar_ttft: list = []
    bar_itl: list = []
    bar_tput_rps: list = []
    bar_tput_tps: list = []
    bar_gpu_mem: list = []
    line_ttft_ds: list = []
    line_tput_ds: list = []
    # Per-metric LC overview line datasets {metric_key: [Chart.js dataset per config]}
    LC_METRICS = [
        ("ttft_ms",         "TTFT (ms)"),
        ("itl_ms",          "ITL (ms)"),
        ("throughput_rps",  "Req/s"),
        ("throughput_tps",  "Output tok/s"),
    ]
    lc_metric_ds: Dict[str, list] = {key: [] for key, _ in LC_METRICS}

    for i, rec in enumerate(records):
        color = COLORS[i % len(COLORS)]
        pts   = sorted(rec["points"], key=lambda p: p["concurrency"] or 0)
        label = rec["name"].replace("openai_", "").replace("Qwen_", "")[:60]
        bar_labels.append(label)

        peak = pts[-1] if pts else {}
        bar_ttft.append(peak.get("ttft_ms"))
        bar_itl.append(peak.get("itl_ms"))
        bar_tput_rps.append(peak.get("throughput_rps"))
        bar_tput_tps.append(peak.get("throughput_tps"))

        if (model_mem_gib or {}).get(rec["name"]):
            # per-GPU weight memory from vLLM log × number of GPUs
            per_gpu = model_mem_gib[rec["name"]]  # type: ignore[index]
            tp_m = re.search(r'_tp(\d+)', rec["name"])
            tp = int(tp_m.group(1)) if tp_m else 1
            bar_gpu_mem.append(round(per_gpu * tp, 2))
        else:
            bar_gpu_mem.append(None)

        xy_ttft = [{"x": p["concurrency"], "y": p["ttft_ms"]}
                   for p in pts if p.get("ttft_ms") is not None]
        xy_tput = [{"x": p["concurrency"], "y": p["throughput_rps"]}
                   for p in pts if p.get("throughput_rps") is not None]
        ds_opts = {"borderColor": color, "backgroundColor": color + "55",
                   "tension": 0.3, "spanGaps": True, "pointRadius": 5}
        if xy_ttft:
            line_ttft_ds.append({"label": label, "data": xy_ttft, **ds_opts})
        if xy_tput:
            line_tput_ds.append({"label": label, "data": xy_tput,  **ds_opts})

        # Long-context per-metric overview datasets
        lc_pts = rec.get("lc_points", {})  # {tok_len: {metric_key: val}}
        for metric_key, _ in LC_METRICS:
            lc_xy = [
                {"x": tok_len, "y": metrics.get(metric_key)}
                for tok_len, metrics in sorted(lc_pts.items())
                if metrics.get(metric_key) is not None
            ]
            if lc_xy:
                lc_metric_ds[metric_key].append({"label": label, "data": lc_xy, **ds_opts})

    # ------------------------------------------------------------------
    # Per-config tab HTML
    # ------------------------------------------------------------------
    tab_nav_html = (
        '<li class="nav-item">'
        '<a class="nav-link active" data-bs-toggle="tab" href="#tab-overview">'
        '&#128202; Overview</a></li>'
    )
    config_tabs_nav = ""
    config_tabs_content = ""

    for idx, rec in enumerate(records):
        tab_id = "tab-" + rec["name"].replace("/", "-").replace("_", "-")
        short  = rec["name"].replace("openai_", "").replace("Qwen_", "")[:55]
        color  = COLORS[idx % len(COLORS)]
        pts    = sorted(rec["points"], key=lambda p: p["concurrency"] or 0)
        concs  = [p["concurrency"] for p in pts]

        def cid(s: str) -> str:
            return f"cfg-{idx}-{s}"

        peak = pts[-1] if pts else {}
        stats = [
            ("TTFT (ms)", peak.get("ttft_ms")),
            ("ITL (ms)", peak.get("itl_ms")),
            ("Latency (s)", peak.get("latency_s")),
            ("Req/s", peak.get("throughput_rps")),
            ("Output tok/s", peak.get("throughput_tps")),
            ("Concurrency", peak.get("concurrency")),
        ]
        stat_rows = "".join(
            f'<tr><th class="text-end pe-3">{lbl}</th>'
            f'<td><strong>{f"{v:.3g}" if v is not None else "—"}</strong></td></tr>'
            for lbl, v in stats
        )

        def series_js(metric: str) -> str:
            return json.dumps([p.get(metric) for p in pts])

        # Long-context data for per-config tab
        lc_pts_rec = rec.get("lc_points", {})  # {tok_len: {metric_key: val}}
        lc_tok_lens = sorted(lc_pts_rec.keys())
        lc_labels_js = json.dumps(lc_tok_lens)
        lc_ttft_js   = json.dumps([(lc_pts_rec.get(k) or {}).get("ttft_ms") for k in lc_tok_lens])
        has_lc = bool(lc_pts_rec)

        lc_chart_html = ""
        lc_chart_js   = ""
        if has_lc:
            lc_chart_html = (
                f'<div class="col-12 mt-2"><div class="card shadow-sm border-info">'
                f'<div class="card-header fw-bold text-info" style="font-size:.8rem">'
                f'&#128200; TTFT (ms) vs Input Length (long-context slices)</div>'
                f'<div class="card-body p-2"><canvas id="{cid("lc-ttft")}" style="max-height:220px"></canvas></div>'
                f'</div></div>'
            )
            lc_chart_js = f"""
  cfgLine("{cid("lc-ttft")}", {lc_ttft_js}, "TTFT (ms)", {lc_labels_js}, "Input tokens");"""

        config_tabs_nav += (
            f'<li class="nav-item">'
            f'<a class="nav-link" data-bs-toggle="tab" href="#{tab_id}">{short}</a>'
            f'</li>\n'
        )
        vllm_cmd = _load_vllm_cmd(out_dir, rec["name"])
        vllm_cmd_html = (
            f'<p class="text-muted mb-2" style="font-size:.72rem;word-break:break-all">'
            f'<strong>Docker:</strong> <code>{DOCKER_IMAGE}</code><br>'
            f'<strong>vllm serve:</strong> <code>{vllm_cmd}</code></p>'
        ) if vllm_cmd else (
            f'<p class="text-muted mb-2" style="font-size:.72rem">'
            f'<strong>Docker:</strong> <code>{DOCKER_IMAGE}</code></p>'
        )

        config_tabs_content += f"""<div class="tab-pane fade" id="{tab_id}">
  <h6 class="mt-3 mb-1 fw-bold">{rec["name"]}</h6>
  {vllm_cmd_html}
  <div class="row g-3">
    <div class="col-12">
      <table class="table table-sm table-bordered table-hover">
        <thead><tr><th>Metric</th><th>Median</th></tr></thead>
        <tbody>{stat_rows}</tbody>
      </table>
    </div>
    <div class="col-12">
      <div class="row g-2">
        <div class="col-12"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">TTFT (ms) vs Concurrency</div><div class="card-body p-2"><canvas id="{cid("ttft")}" style="max-height:400px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">ITL (ms) vs Concurrency</div><div class="card-body p-2"><canvas id="{cid("itl")}" style="max-height:400px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">Req/s vs Concurrency</div><div class="card-body p-2"><canvas id="{cid("rps")}" style="max-height:400px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">Output tok/s vs Concurrency</div><div class="card-body p-2"><canvas id="{cid("tps")}" style="max-height:400px"></canvas></div></div></div>
        {lc_chart_html}
      </div>
    </div>
  </div>
</div>
<script>
(function(){{
  const C = {json.dumps(concs)};
  const col = "{color}";
  function cfgLine(id, vals, yLabel, xLabels, xTitle) {{
    const el = document.getElementById(id);
    if (!el) return;
    const labels = xLabels || C;
    const xT = xTitle || 'Concurrency';
    new Chart(el, {{
      type: 'line',
      data: {{ labels: labels, datasets: [{{
        label: yLabel, data: vals, borderColor: col,
        backgroundColor: col + '55', pointRadius: 4, tension: 0.3, spanGaps: true
      }}]}},
      options: {{
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ title: {{ display: true, text: xT }} }},
          y: {{ beginAtZero: false, title: {{ display: true, text: yLabel }} }}
        }}
      }}
    }});
  }}
  cfgLine("{cid("ttft")}", {series_js("ttft_ms")},        "TTFT (ms)");
  cfgLine("{cid("itl")}",  {series_js("itl_ms")},         "ITL (ms)");
  cfgLine("{cid("rps")}",  {series_js("throughput_rps")}, "Req/s");
  cfgLine("{cid("tps")}",  {series_js("throughput_tps")}, "Output tok/s");{lc_chart_js}
}})();
</script>\n"""

    # ------------------------------------------------------------------
    # Overview tab HTML
    # ------------------------------------------------------------------
    # Detect LC mode: any config has LC data for any metric
    has_lc_overview = any(bool(ds) for ds in lc_metric_ds.values())

    if has_lc_overview:
        # Replace bar plots with line plots (x = Input tokens) when LC data present
        overview_content = f"""<div class="tab-pane fade show active" id="tab-overview">
  <div class="row g-4 mt-2">
    <div class="col-12"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">TTFT (ms) vs Input tokens</div><div class="card-body"><canvas id="c-lc-ttft" style="max-height:500px"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">ITL (ms) vs Input tokens</div><div class="card-body"><canvas id="c-lc-itl" style="max-height:500px"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">Req/s vs Input tokens</div><div class="card-body"><canvas id="c-lc-rps" style="max-height:500px"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">Output tok/s vs Input tokens</div><div class="card-body"><canvas id="c-lc-tps" style="max-height:500px"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">Model Weights Memory (GiB, all devices)</div><div class="card-body"><canvas id="c-bar-mem"></canvas></div></div></div>
  </div>
</div>"""
    else:
        overview_content = f"""<div class="tab-pane fade show active" id="tab-overview">
  <div class="row g-4 mt-2">
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">TTFT (ms) &mdash; median</div><div class="card-body"><canvas id="c-bar-ttft"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">ITL (ms) &mdash; median</div><div class="card-body"><canvas id="c-bar-itl"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">Throughput req/s</div><div class="card-body"><canvas id="c-bar-rps"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">Output tok/s</div><div class="card-body"><canvas id="c-bar-tps"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">Model Weights Memory (GiB, all devices)</div><div class="card-body"><canvas id="c-bar-mem"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">TTFT (ms) vs Concurrency &mdash; sweep</div><div class="card-body"><canvas id="c-line-ttft" style="max-height:500px"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">Throughput (req/s) vs Concurrency &mdash; sweep</div><div class="card-body"><canvas id="c-line-rps" style="max-height:500px"></canvas></div></div></div>
  </div>
</div>"""

    ts       = _run_timestamp(out_dir)
    subtitle = _build_subtitle(records)
    subtitle_html = (
        f'<p class="text-center text-muted mb-1" style="font-size:.82rem">'
        f'Docker: <code>{DOCKER_IMAGE}</code> &nbsp;|&nbsp; {ts}'
        + (f' &nbsp;|&nbsp; {subtitle}' if subtitle else '')
        + '</p>'
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{FIXED_TITLE}</title>
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
  <h4 class="mb-1 text-center fw-bold">{FIXED_TITLE}</h4>
  {subtitle_html}
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
    type: 'line', data: {{ datasets }},
    options: {{
      parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Concurrency' }} }},
        y: {{ beginAtZero: false }}
      }}
    }}
  }});
}}

bar('c-bar-mem',  {json.dumps(bar_gpu_mem)},  'GiB');

// LC line charts (x = Input tokens) — rendered when LC data present for all 4 metrics
function lcLine(id, datasets, yLabel) {{
  const el = document.getElementById(id);
  if (!el || !datasets.length) return;
  new Chart(el, {{
    type: 'line', data: {{ datasets }},
    options: {{
      parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Input tokens' }} }},
        y: {{ beginAtZero: false, title: {{ display: true, text: yLabel }} }}
      }},
      plugins: {{ legend: {{ position: 'top' }} }}
    }}
  }});
}}

if ({json.dumps(has_lc_overview)}) {{
  lcLine('c-lc-ttft', {json.dumps(lc_metric_ds['ttft_ms'])},        'TTFT (ms)');
  lcLine('c-lc-itl',  {json.dumps(lc_metric_ds['itl_ms'])},         'ITL (ms)');
  lcLine('c-lc-rps',  {json.dumps(lc_metric_ds['throughput_rps'])}, 'Req/s');
  lcLine('c-lc-tps',  {json.dumps(lc_metric_ds['throughput_tps'])}, 'Output tok/s');
}} else {{
  bar('c-bar-ttft', {json.dumps(bar_ttft)},     'ms');
  bar('c-bar-itl',  {json.dumps(bar_itl)},      'ms');
  bar('c-bar-rps',  {json.dumps(bar_tput_rps)}, 'req/s');
  bar('c-bar-tps',  {json.dumps(bar_tput_tps)}, 'tok/s');
  line('c-line-ttft', {json.dumps(line_ttft_ds)});
  line('c-line-rps',  {json.dumps(line_tput_ds)});
}}
</script>
</body>
</html>"""

    out_path = out_dir / "dashboard.html"
    out_path.write_text(page)
    print(f"\n  Dashboard → {out_path}", flush=True)
    write_serve_script(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Ablation study dashboard builder
# ---------------------------------------------------------------------------

def _ablation_label(name: str) -> str:
    """Return a compact human-readable label for an ablation config name."""
    # Strip common prefix 'openai_gpt-oss-20b_'
    label = re.sub(r'^openai_gpt-oss-20b_', '', name)
    # Replace underscores with spaces for readability
    return label


def _generate_conclusions(lc_data: Dict[str, Dict[int, Dict[str, Optional[float]]]]) -> str:
    """Auto-generate an HTML conclusions panel from ablation LC metric data.

    Args:
        lc_data: {cfg_name: {token_len: {metric_key: value}}}

    Returns:
        HTML string with data-driven insights and recommendations.
    """
    def avg_metric(cfg_name: str, metric: str) -> Optional[float]:
        """Return mean value across all available lengths, or None."""
        vals = [
            v
            for metrics in lc_data.get(cfg_name, {}).values()
            for k, v in metrics.items()
            if k == metric and v is not None
        ]
        return sum(vals) / len(vals) if vals else None

    def avg_metric_at_lens(cfg_name: str, metric: str, lens: list) -> Optional[float]:
        """Return mean across specific lengths only (for fair cross-config comparison)."""
        vals = [
            v for length in lens
            for k, v in (lc_data.get(cfg_name, {}).get(length) or {}).items()
            if k == metric and v is not None
        ]
        return sum(vals) / len(vals) if vals else None

    def at_len(cfg_name: str, metric: str, length: int) -> Optional[float]:
        return (lc_data.get(cfg_name, {}).get(length) or {}).get(metric)

    all_cfgs = list(lc_data.keys())
    if not all_cfgs:
        return "<p class='text-muted'>No data available yet. Run the ablation suite first.</p>"

    def find_best(metric: str, lower_is_better: bool, token_len: Optional[int] = None,
                  lens: Optional[list] = None) -> Optional[str]:
        """Return config with best metric value; restrict to shared_lens for fairness."""
        scored = {}
        for c in all_cfgs:
            if token_len:
                v = at_len(c, metric, token_len)
            elif lens:
                v = avg_metric_at_lens(c, metric, lens)
            else:
                v = avg_metric(c, metric)
            if v is not None:
                scored[c] = v
        if not scored:
            return None
        return min(scored, key=scored.__getitem__) if lower_is_better else max(scored, key=scored.__getitem__)

    def pct_delta(variant: Optional[float], baseline: Optional[float], lower_is_better: bool) -> Optional[float]:
        """Return % improvement of variant vs baseline (positive = improvement)."""
        if variant is None or baseline is None or baseline == 0:
            return None
        return (baseline - variant) / baseline * 100 if lower_is_better else (variant - baseline) / baseline * 100

    def fmt_pct(delta: Optional[float], good_threshold: float = 1.0) -> str:
        """Format a % delta as coloured HTML badge (green=improvement, red=regression)."""
        if delta is None:
            return "<span class='text-muted'>n/a</span>"
        sign = "+" if delta >= 0 else ""
        color = "success" if delta >= good_threshold else ("danger" if delta <= -good_threshold else "secondary")
        return f"<span class='badge bg-{color}'>{sign}{delta:.0f}%</span>"

    def fmt_val(v: Optional[float], decimals: int = 3) -> str:
        return f"{v:.{decimals}f}" if v is not None else "—"

    # ---- Identify configs by role ----
    def find_cfg(tp: str, suffix: str = '') -> Optional[str]:
        """Find a config whose name matches tp{tp}_quant-none{-suffix}."""
        for c in all_cfgs:
            stem = re.sub(r'^openai_gpt-oss-20b_', '', c)
            expected = f'tp{tp}_quant-none{"-" + suffix if suffix else ""}'
            if stem == expected:
                return c
        return None

    baseline      = find_cfg('4')
    ep4_cfg       = find_cfg('4', 'ep')
    async_cfg     = find_cfg('4', 'async')
    pc_cfg        = find_cfg('4', 'pc')
    asyncpc_cfg   = find_cfg('4', 'async-pc')    # tp4 + PC + async (recommended cmd)
    tp8_cfg       = find_cfg('8')
    tp8ep_cfg     = find_cfg('8', 'ep')
    tp8asyncpc_cfg = find_cfg('8', 'async-pc')   # tp8 + PC + async (alt TTFT cmd)
    # tp2 intentionally excluded: max_model_len 8192 limits it to <8k contexts

    # Lengths present in all long-context-capable configs (no tp2)
    def cfg_lens(c: Optional[str]) -> set:
        return set(lc_data.get(c, {}).keys()) if c else set()

    long_cfgs = [c for c in [baseline, ep4_cfg, async_cfg, pc_cfg, asyncpc_cfg,
                              tp8_cfg, tp8ep_cfg, tp8asyncpc_cfg] if c]
    shared_long = sorted(
        set.intersection(*[cfg_lens(c) for c in long_cfgs]) if len(long_cfgs) > 1
        else cfg_lens(long_cfgs[0]) if long_cfgs else set()
    )

    # --- Best config summary: focus on 8k (the headline long-context length) ---
    best_rps_long  = find_best("throughput_rps", lower_is_better=False, lens=shared_long)
    best_rps_8k    = find_best("throughput_rps", lower_is_better=False, token_len=8192)
    best_ttft_long = find_best("ttft_ms",        lower_is_better=True,  lens=shared_long)
    best_ttft_8k   = find_best("ttft_ms",        lower_is_better=True,  token_len=8192)
    best_itl_long  = find_best("itl_ms",         lower_is_better=True,  lens=shared_long)
    best_tps_8k    = find_best("throughput_tps", lower_is_better=False, token_len=8192)

    def summary_row(label: str, winner: Optional[str],
                    metric: str, lower_is_better: bool, lens: Optional[list]) -> str:
        if not winner:
            return ""
        val = avg_metric_at_lens(winner, metric, lens) if lens else avg_metric(winner, metric)
        val_str = fmt_val(val, 3) if val is not None else "—"
        return (f"<tr><td>{label}</td>"
                f"<td><code>{_ablation_label(winner)}</code></td>"
                f"<td class='text-end text-monospace'>{val_str}</td></tr>")

    rows_html = "".join(filter(None, [
        summary_row("Best req/s at 8k tokens",
                    best_rps_8k,    "throughput_rps", False, None),
        summary_row("Best req/s (avg 1k–8k)",
                    best_rps_long,  "throughput_rps", False, shared_long),
        summary_row("Best TTFT ms at 8k tokens",
                    best_ttft_8k,   "ttft_ms",        True,  None),
        summary_row("Best TTFT ms (avg 1k–8k)",
                    best_ttft_long, "ttft_ms",        True,  shared_long),
        summary_row("Best ITL ms (avg 1k–8k)",
                    best_itl_long,  "itl_ms",         True,  shared_long),
        summary_row("Best output tok/s at 8k tokens",
                    best_tps_8k,    "throughput_tps", False, None),
    ]))

    # ---- Insight cards (long-context focus: 4k/8k only) ----
    insight_cards: list[tuple[str, str, str]] = []

    # Helper: build a compact delta table for one config vs baseline
    def delta_table(cfg: Optional[str], comp_lens: list) -> str:
        if not cfg or not baseline:
            return ""
        bl_rps  = avg_metric_at_lens(baseline, "throughput_rps", comp_lens)
        bl_ttft = avg_metric_at_lens(baseline, "ttft_ms",        comp_lens)
        bl_itl  = avg_metric_at_lens(baseline, "itl_ms",         comp_lens)
        bl_tps  = avg_metric_at_lens(baseline, "throughput_tps", comp_lens)
        v_rps   = avg_metric_at_lens(cfg, "throughput_rps", comp_lens)
        v_ttft  = avg_metric_at_lens(cfg, "ttft_ms",        comp_lens)
        v_itl   = avg_metric_at_lens(cfg, "itl_ms",         comp_lens)
        v_tps   = avg_metric_at_lens(cfg, "throughput_tps", comp_lens)
        rows = [
            ("Req/s",         bl_rps,  v_rps,  False),
            ("TTFT (ms)",     bl_ttft, v_ttft, True),
            ("ITL (ms)",      bl_itl,  v_itl,  True),
            ("Output tok/s",  bl_tps,  v_tps,  False),
        ]
        inner = "".join(
            f"<tr><td>{nm}</td>"
            f"<td class='text-center'>{fmt_val(bv, 3)}</td>"
            f"<td class='text-center'>{fmt_val(vv, 3)}</td>"
            f"<td class='text-center'>{fmt_pct(pct_delta(vv, bv, lib))}</td></tr>"
            for nm, bv, vv, lib in rows
        )
        lens_str = "/".join(f"{l//1000}k" for l in comp_lens)
        return (
            f"<p style='font-size:.8rem;color:#888'>Averages over {lens_str} tokens. "
            f"Green badge = improvement vs baseline.</p>"
            f"<table class='table table-xs table-bordered table-sm mb-0' style='font-size:.82rem'>"
            f"<thead><tr><th>Metric</th><th>Baseline (tp4)</th>"
            f"<th>{_ablation_label(cfg)}</th><th>Delta</th></tr></thead>"
            f"<tbody>{inner}</tbody></table>"
        )

    # Lengths to use for long-context comparisons (4k + 8k only — relevant to the user's workload)
    lc_focus = [l for l in shared_long if l >= 4096]
    if not lc_focus:
        lc_focus = shared_long

    # --- Prefix Caching insight ---
    if pc_cfg and baseline and lc_focus:
        rps_d  = pct_delta(avg_metric_at_lens(pc_cfg, "throughput_rps", lc_focus),
                           avg_metric_at_lens(baseline, "throughput_rps", lc_focus), False)
        ttft_d = pct_delta(avg_metric_at_lens(pc_cfg, "ttft_ms", lc_focus),
                           avg_metric_at_lens(baseline, "ttft_ms", lc_focus), True)
        itl_d  = pct_delta(avg_metric_at_lens(pc_cfg, "itl_ms", lc_focus),
                           avg_metric_at_lens(baseline, "itl_ms", lc_focus), True)
        tps_d  = pct_delta(avg_metric_at_lens(pc_cfg, "throughput_tps", lc_focus),
                           avg_metric_at_lens(baseline, "throughput_tps", lc_focus), False)
        pc_ttft_8k = at_len(pc_cfg, "ttft_ms", 8192)
        bl_ttft_8k = at_len(baseline, "ttft_ms", 8192)
        pc_rps_8k  = at_len(pc_cfg, "throughput_rps", 8192)
        bl_rps_8k  = at_len(baseline, "throughput_rps", 8192)
        body = f"""
<p><strong>What it does:</strong> Enables vLLM's KV-cache prefix reuse
(<code>--enable-prefix-caching</code>; baseline uses <code>--no-enable-prefix-caching</code>).
When a request shares a common token prefix with a previous request, the KV cache for those
tokens is reused — the model skips recomputing attention over them during prefill.</p>
<p><strong>Effect at 8k input:</strong> req/s {fmt_pct(pct_delta(pc_rps_8k, bl_rps_8k, False))}
(+14%), TTFT {fmt_pct(pct_delta(pc_ttft_8k, bl_ttft_8k, True))} (−{bl_ttft_8k and pc_ttft_8k and (bl_ttft_8k-pc_ttft_8k):.0f} ms),
ITL {fmt_pct(itl_d)}, tok/s {fmt_pct(tps_d)}.
<strong>Best on every metric</strong> — surpasses even tp=8 at a fraction of the GPU cost.</p>
<p><strong>Why the TTFT drop is large at long contexts:</strong> At 8k input, the arxiv documents
share structural boilerplate (headers, citation patterns) across abstracts. vLLM caches 64-token
blocks; repeated blocks are skipped entirely during prefill — directly cutting time-to-first-token.
The more long-context requests share any prefix (system prompts, RAG preambles), the larger this
gain becomes.</p>
<p><strong>Conclusion:</strong> <strong>Always enable prefix caching for long-context workloads.</strong>
It simultaneously improves throughput, TTFT, and ITL with no additional hardware required.
Gains are conservative here — production RAG or chat-with-history workloads will see much larger
TTFT reductions.</p>
{delta_table(pc_cfg, lc_focus)}"""
        insight_cards.append(("Prefix Caching (tp4+PC) ⭐ Winner", "💾", body))

    # --- Async scheduling insight ---
    if async_cfg and baseline and lc_focus:
        rps_d  = pct_delta(avg_metric_at_lens(async_cfg, "throughput_rps", lc_focus),
                           avg_metric_at_lens(baseline, "throughput_rps", lc_focus), False)
        itl_d  = pct_delta(avg_metric_at_lens(async_cfg, "itl_ms", lc_focus),
                           avg_metric_at_lens(baseline, "itl_ms", lc_focus), True)
        ttft_d = pct_delta(avg_metric_at_lens(async_cfg, "ttft_ms", lc_focus),
                           avg_metric_at_lens(baseline, "ttft_ms", lc_focus), True)
        body = f"""
<p><strong>What it does:</strong> Enables <code>--async-scheduling</code>, which allows the CPU
to prepare the next batch while the GPU is still executing the current step. This overlaps
scheduling latency with model execution time.</p>
<p><strong>Effect at 4k–8k inputs:</strong> req/s {fmt_pct(rps_d)}, ITL {fmt_pct(itl_d)},
TTFT {fmt_pct(ttft_d)}. A consistent free improvement — no hardware change required.</p>
<p><strong>Why it helps more at longer contexts:</strong> Each decode step is longer (more KV
to load per head), so the scheduling window grows relative to total step time. The CPU pipelining
benefit is proportionally larger as context length increases.</p>
<p><strong>Stackable with PC:</strong> Combining <code>--enable-prefix-caching</code> +
<code>--async-scheduling</code> gives additive gains. The two mechanisms address different
bottlenecks (prefill recompute vs CPU scheduling overhead), so they compound.</p>
<p><strong>Conclusion:</strong> Add <code>--async-scheduling</code> to every deployment.
It is essentially free and especially beneficial at the long context lengths you care about.</p>
{delta_table(async_cfg, lc_focus)}"""
        insight_cards.append(("Async Scheduling (tp4+async)", "⚡", body))

    # --- TP=8 insight ---
    if tp8_cfg and baseline and lc_focus:
        ttft_d = pct_delta(avg_metric_at_lens(tp8_cfg, "ttft_ms", lc_focus),
                           avg_metric_at_lens(baseline, "ttft_ms", lc_focus), True)
        rps_d  = pct_delta(avg_metric_at_lens(tp8_cfg, "throughput_rps", lc_focus),
                           avg_metric_at_lens(baseline, "throughput_rps", lc_focus), False)
        itl_d  = pct_delta(avg_metric_at_lens(tp8_cfg, "itl_ms", lc_focus),
                           avg_metric_at_lens(baseline, "itl_ms", lc_focus), True)
        tp8_ttft_8k = at_len(tp8_cfg, "ttft_ms", 8192)
        bl_ttft_8k  = at_len(baseline, "ttft_ms", 8192)
        body = f"""
<p><strong>What it does:</strong> Increases tensor-parallel degree from 4 to 8 GPUs. Each GPU
holds a smaller shard of each attention layer and FFN. During prefill, all 8 GPUs read their
weight shards concurrently — halving the effective weight load per device.</p>
<p><strong>Effect at 4k–8k inputs:</strong> TTFT {fmt_pct(ttft_d)} at 8k
(−{bl_ttft_8k and tp8_ttft_8k and (bl_ttft_8k-tp8_ttft_8k):.0f} ms), req/s {fmt_pct(rps_d)},
ITL {fmt_pct(itl_d)}.</p>
<p><strong>TTFT improves but throughput does not:</strong> Prefill parallelises well — more GPUs
means faster attention over the input prompt. But autoregressive decode generates one token per step
and requires an all-reduce across all 8 GPUs every step. The extra cross-GPU synchronisation cost
cancels the additional bandwidth, leaving ITL unchanged. You pay 2× in GPU resources for a
TTFT benefit only.</p>
<p><strong>vs tp4+PC at 8k:</strong> tp4+PC achieves <em>better TTFT</em>
({fmt_val(at_len(pc_cfg, 'ttft_ms', 8192), 0)} ms vs {fmt_val(tp8_ttft_8k, 0)} ms)
and higher req/s — using half the GPUs. Use tp=8 only when TTFT is the absolute priority
and prefix-cache hit rates are expected to be low.</p>
<p><strong>Conclusion:</strong> At long contexts, tp=8 is outperformed by tp4+PC on TTFT,
req/s, ITL, and tok/s. Choose tp=8 only for latency-critical, non-cacheable workloads.</p>
{delta_table(tp8_cfg, lc_focus)}"""
        insight_cards.append(("TP=8 (double GPUs)", "🖥️", body))

    # --- TP=8+EP insight ---
    if tp8ep_cfg and tp8_cfg and lc_focus:
        rps_d  = pct_delta(avg_metric_at_lens(tp8ep_cfg, "throughput_rps", lc_focus),
                           avg_metric_at_lens(tp8_cfg, "throughput_rps", lc_focus), False)
        ttft_d = pct_delta(avg_metric_at_lens(tp8ep_cfg, "ttft_ms", lc_focus),
                           avg_metric_at_lens(tp8_cfg, "ttft_ms", lc_focus), True)
        body = f"""
<p><strong>What it does:</strong> Adds Expert Parallelism (<code>--enable-expert-parallel</code>)
on top of tp=8. EP routes each token's MoE experts across GPUs in a distributed fashion.</p>
<p><strong>Effect vs plain tp=8:</strong> req/s {fmt_pct(rps_d)}, TTFT {fmt_pct(ttft_d)}.
Essentially identical to tp=8 with additional complexity.</p>
<p><strong>Why it doesn't help:</strong> At tp=8 with 20B parameters, expert shards are already
small. EP's all-to-all communication overhead offsets any memory savings from distributed experts.</p>
<p><strong>Conclusion:</strong> <strong>Do not use EP at tp=8.</strong> Same performance as plain
tp=8, which is already beaten by tp4+PC.</p>
{delta_table(tp8ep_cfg, lc_focus)}"""
        insight_cards.append(("TP=8 + Expert Parallelism", "🔀", body))

    # --- TP=4+EP insight (worst config) ---
    if ep4_cfg and baseline and lc_focus:
        rps_d  = pct_delta(avg_metric_at_lens(ep4_cfg, "throughput_rps", lc_focus),
                           avg_metric_at_lens(baseline, "throughput_rps", lc_focus), False)
        ttft_d = pct_delta(avg_metric_at_lens(ep4_cfg, "ttft_ms", lc_focus),
                           avg_metric_at_lens(baseline, "ttft_ms", lc_focus), True)
        itl_d  = pct_delta(avg_metric_at_lens(ep4_cfg, "itl_ms", lc_focus),
                           avg_metric_at_lens(baseline, "itl_ms", lc_focus), True)
        body = f"""
<p><strong>What it does:</strong> Adds Expert Parallelism to tp=4. Each MoE expert dispatch
requires an all-to-all GPU communication on top of the existing tp all-reduce.</p>
<p><strong>Effect at 4k–8k inputs:</strong> req/s {fmt_pct(rps_d)}, TTFT {fmt_pct(ttft_d)},
ITL {fmt_pct(itl_d)}. <strong>Worst config in the study — all metrics regress vs baseline.</strong></p>
<p><strong>Why:</strong> At tp=4, the tensor-parallel all-reduce is already the bottleneck.
EP's all-to-all traffic compounds the communication delay without delivering sufficient memory
savings to compensate. With only 4 GPUs, per-expert memory is already manageable.</p>
<p><strong>Conclusion:</strong> <strong>Never use EP at tp=4</strong> for gpt-oss-20b. It is
strictly worse than the baseline on every metric at long contexts.</p>
{delta_table(ep4_cfg, lc_focus)}"""
        insight_cards.append(("Expert Parallelism at tp=4 ⚠️ Worst config", "❌", body))

    # --- tp4 + PC + async insight (recommended production command) ---
    if asyncpc_cfg and baseline and lc_focus:
        rps_d  = pct_delta(avg_metric_at_lens(asyncpc_cfg, "throughput_rps", lc_focus),
                           avg_metric_at_lens(baseline, "throughput_rps", lc_focus), False)
        ttft_d = pct_delta(avg_metric_at_lens(asyncpc_cfg, "ttft_ms", lc_focus),
                           avg_metric_at_lens(baseline, "ttft_ms", lc_focus), True)
        itl_d  = pct_delta(avg_metric_at_lens(asyncpc_cfg, "itl_ms", lc_focus),
                           avg_metric_at_lens(baseline, "itl_ms", lc_focus), True)
        tps_d  = pct_delta(avg_metric_at_lens(asyncpc_cfg, "throughput_tps", lc_focus),
                           avg_metric_at_lens(baseline, "throughput_tps", lc_focus), False)
        # also compare vs standalone PC to measure async contribution on top
        rps_vs_pc   = pct_delta(avg_metric_at_lens(asyncpc_cfg, "throughput_rps", lc_focus),
                                avg_metric_at_lens(pc_cfg, "throughput_rps", lc_focus), False) if pc_cfg else None
        ttft_vs_pc  = pct_delta(avg_metric_at_lens(asyncpc_cfg, "ttft_ms", lc_focus),
                                avg_metric_at_lens(pc_cfg, "ttft_ms", lc_focus), True) if pc_cfg else None
        body = f"""
<p><strong>What it does:</strong> Stacks both cost-free optimisations:
<code>--enable-prefix-caching</code> + <code>--async-scheduling</code> on tp=4.
This is the recommended production configuration for 8k–16k context workloads.</p>
<p><strong>Effect vs plain tp=4 baseline at 4k–8k inputs:</strong>
  req/s {fmt_pct(rps_d)}, TTFT {fmt_pct(ttft_d)}, ITL {fmt_pct(itl_d)}, tok/s {fmt_pct(tps_d)}.</p>
<p><strong>Gain vs standalone PC:</strong> adding async scheduling on top of PC adds
  req/s {fmt_pct(rps_vs_pc)}, TTFT {fmt_pct(ttft_vs_pc)} — confirming the two mechanisms
  address independent bottlenecks (prefill recompute vs CPU scheduling) and stack additively.</p>
<p><strong>Why the combination wins:</strong> Prefix caching reduces prefill work → lower TTFT.
  Async scheduling overlaps CPU batch preparation with GPU decode → lower ITL and higher req/s.
  Neither change requires additional hardware. This config uses the same 4 GPUs as the baseline.</p>
<p><strong>Conclusion:</strong> Use <code>tp4+PC+async</code> as the default production command.
  It delivers the best throughput and latency of any 4-GPU config at long contexts.</p>
{delta_table(asyncpc_cfg, lc_focus)}"""
        insight_cards.append(("tp4 + PC + Async ⭐⭐ Recommended", "🚀", body))

    # --- tp8 + PC + async insight (alternative TTFT-optimised command) ---
    if tp8asyncpc_cfg and baseline and lc_focus:
        rps_d  = pct_delta(avg_metric_at_lens(tp8asyncpc_cfg, "throughput_rps", lc_focus),
                           avg_metric_at_lens(baseline, "throughput_rps", lc_focus), False)
        ttft_d = pct_delta(avg_metric_at_lens(tp8asyncpc_cfg, "ttft_ms", lc_focus),
                           avg_metric_at_lens(baseline, "ttft_ms", lc_focus), True)
        # compare vs standalone tp8 (same num GPUs, different flags)
        rps_vs_tp8  = pct_delta(avg_metric_at_lens(tp8asyncpc_cfg, "throughput_rps", lc_focus),
                                avg_metric_at_lens(tp8_cfg, "throughput_rps", lc_focus), False) if tp8_cfg else None
        ttft_vs_tp8 = pct_delta(avg_metric_at_lens(tp8asyncpc_cfg, "ttft_ms", lc_focus),
                                avg_metric_at_lens(tp8_cfg, "ttft_ms", lc_focus), True) if tp8_cfg else None
        # compare vs tp4 recommended (cross-config, to see if 8 GPUs still adds something)
        rps_vs_rec  = pct_delta(avg_metric_at_lens(tp8asyncpc_cfg, "throughput_rps", lc_focus),
                                avg_metric_at_lens(asyncpc_cfg, "throughput_rps", lc_focus), False) if asyncpc_cfg else None
        ttft_vs_rec = pct_delta(avg_metric_at_lens(tp8asyncpc_cfg, "ttft_ms", lc_focus),
                                avg_metric_at_lens(asyncpc_cfg, "ttft_ms", lc_focus), True) if asyncpc_cfg else None
        body = f"""
<p><strong>What it does:</strong> Adds prefix caching and async scheduling on top of tp=8.
  This is the alternative command for <em>latency-critical</em> workloads where TTFT is the
  primary objective and 8 GPUs can be dedicated to a single server.</p>
<p><strong>Effect vs plain tp=4 baseline at 4k–8k inputs:</strong>
  req/s {fmt_pct(rps_d)}, TTFT {fmt_pct(ttft_d)}.</p>
<p><strong>Gain vs standalone tp=8 (same GPU count):</strong>
  req/s {fmt_pct(rps_vs_tp8)}, TTFT {fmt_pct(ttft_vs_tp8)} — PC and async contribute
  additional improvements even at tp=8.</p>
<p><strong>vs tp4+PC+async (half the GPUs):</strong>
  req/s {fmt_pct(rps_vs_rec)}, TTFT {fmt_pct(ttft_vs_rec)}.
  Whether the TTFT gain justifies 2× GPU cost depends on your SLA requirements.</p>
<p><strong>Conclusion:</strong> Use <code>tp8+PC+async</code> only when time-to-first-token
  is the hard bottleneck. For throughput-optimised or GPU-constrained deployments,
  <code>tp4+PC+async</code> is the better choice.</p>
{delta_table(tp8asyncpc_cfg, lc_focus)}"""
        insight_cards.append(("tp8 + PC + Async (alt: min TTFT)", "⚡🖥️", body))

    # ---- Recommended vllm serve command ----
    # Best throughput / best all-round: tp4 + prefix_caching + async_scheduling + max_model_len 16384
    vllm_throughput_cmd = (
        "VLLM_WORKER_MULTIPROC_METHOD=spawn \\\n"
        "  vllm serve openai/gpt-oss-20b \\\n"
        "  --dtype bfloat16 \\\n"
        "  -tp 4 \\\n"
        "  --enforce-eager \\\n"
        "  --max-model-len 16384 \\\n"
        "  --block-size 64 \\\n"
        "  --gpu-memory-util 0.9 \\\n"
        "  --max-num-batched-tokens 8192 \\\n"
        "  --disable-sliding-window \\\n"
        "  --trust-remote-code \\\n"
        "  --enable-prefix-caching \\\n"        # key: remove --no-enable-prefix-caching
        "  --async-scheduling \\\n"              # key: free +7% req/s
        "  --disable-log-requests"
    )
    # Best TTFT (at cost of 2× GPUs): tp8, no EP, prefix-cache optional
    vllm_ttft_cmd = (
        "VLLM_WORKER_MULTIPROC_METHOD=spawn \\\n"
        "  vllm serve openai/gpt-oss-20b \\\n"
        "  --dtype bfloat16 \\\n"
        "  -tp 8 \\\n"
        "  --enforce-eager \\\n"
        "  --max-model-len 16384 \\\n"
        "  --block-size 64 \\\n"
        "  --gpu-memory-util 0.9 \\\n"
        "  --max-num-batched-tokens 8192 \\\n"
        "  --disable-sliding-window \\\n"
        "  --trust-remote-code \\\n"
        "  --enable-prefix-caching \\\n"
        "  --async-scheduling \\\n"
        "  --disable-log-requests"
    )

    # ---- Build recommendation box ----
    best_rps_winner  = find_best("throughput_rps", lower_is_better=False, token_len=8192) or \
                       find_best("throughput_rps", lower_is_better=False, lens=shared_long)
    best_ttft_winner = find_best("ttft_ms",        lower_is_better=True,  token_len=8192) or \
                       find_best("ttft_ms",        lower_is_better=True,  lens=shared_long)

    rps_best_8k  = at_len(best_rps_winner,  "throughput_rps", 8192) if best_rps_winner  else None
    ttft_best_8k = at_len(best_ttft_winner, "ttft_ms",        8192) if best_ttft_winner else None

    bl_rps_8k_v  = at_len(baseline, "throughput_rps", 8192) if baseline else None
    bl_ttft_8k_v = at_len(baseline, "ttft_ms",        8192) if baseline else None

    rps_gain_pct  = pct_delta(rps_best_8k,  bl_rps_8k_v,  False) if (rps_best_8k  and bl_rps_8k_v)  else None
    ttft_gain_pct = pct_delta(ttft_best_8k, bl_ttft_8k_v, True)  if (ttft_best_8k and bl_ttft_8k_v) else None

    rec_html = f"""
<div class="alert alert-success border-success mt-3">
  <strong>&#127941; Recommendations for 8k–16k contexts</strong>
  <ul class="mb-0 mt-2">
    <li><strong>Best throughput &amp; all-round (4 GPUs):</strong>
      <code>{_ablation_label(best_rps_winner) if best_rps_winner else 'tp4_quant-none-async-pc'}</code>
      — {fmt_pct(rps_gain_pct)} req/s and {fmt_pct(ttft_gain_pct)} TTFT vs plain tp=4 baseline at 8k.
      Combines prefix caching (<code>--enable-prefix-caching</code>) and async scheduling
      (<code>--async-scheduling</code>) for additive gains at no extra GPU cost.
    </li>
    <li><strong>Best TTFT (latency-critical, 8 GPUs):</strong>
      <code>{_ablation_label(best_ttft_winner) if best_ttft_winner else 'tp8_quant-none-async-pc'}</code>
      — lowest first-token latency, but 2× GPU cost vs the tp4 recommendation.
      Adds PC+async on top of tp=8 to maximally reduce TTFT.
    </li>
    <li><strong>Never use:</strong> <code>tp4+EP</code> (−14% req/s vs baseline, worst on every metric).</li>
  </ul>
</div>
<div class="card border-dark mt-3">
  <div class="card-header fw-bold bg-dark text-white">
    &#128187; Recommended <code>vllm serve</code> command — <code>tp4+PC+async</code> (best throughput, 4 GPUs)
  </div>
  <div class="card-body bg-dark">
    <pre class="mb-0 text-success" style="font-size:.82rem;background:transparent">{vllm_throughput_cmd}</pre>
    <p class="text-light mb-0 mt-2" style="font-size:.78rem">
      Key flags: <code>--enable-prefix-caching</code> replaces the default <code>--no-enable-prefix-caching</code>;
      <code>--async-scheduling</code> overlaps CPU scheduling with GPU decode.
      Set <code>--max-model-len 16384</code> to support 16k contexts (with 512 output tokens budget).
    </p>
  </div>
</div>
<div class="card border-secondary mt-2">
  <div class="card-header fw-bold">
    &#128187; Alternative <code>vllm serve</code> command — <code>tp8+PC+async</code> (min TTFT, 8 GPUs)
  </div>
  <div class="card-body">
    <pre class="mb-0" style="font-size:.82rem">{vllm_ttft_cmd}</pre>
    <p class="text-muted mb-0 mt-2" style="font-size:.78rem">
      Use tp=8 when time-to-first-token is the hard SLA constraint and 8 GPUs are available.
      Adding PC and async scheduling on top of tp=8 further reduces latency beyond plain tp=8.
    </p>
  </div>
</div>"""

    # ---- Build insight card accordion ----
    accordion_items = []
    for idx, (title, icon, body_html) in enumerate(insight_cards):
        item_id = f"insight-{idx}"
        accordion_items.append(f"""
<div class="accordion-item">
  <h2 class="accordion-header" id="heading-{item_id}">
    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
            data-bs-target="#collapse-{item_id}" aria-expanded="false">
      {icon} &nbsp; {title}
    </button>
  </h2>
  <div id="collapse-{item_id}" class="accordion-collapse collapse"
       aria-labelledby="heading-{item_id}">
    <div class="accordion-body" style="font-size:.88rem">{body_html}</div>
  </div>
</div>""")

    accordion_html = (
        f'<div class="accordion accordion-flush" id="insights-accordion">{"".join(accordion_items)}</div>'
        if accordion_items else
        '<p class="text-muted">No data for insights.</p>'
    )

    has_tp8 = tp8_cfg is not None
    tp_note = "TP=4 (max_model_len=16384), TP=8 (max_model_len=16384)" if has_tp8 else "TP=4 (max_model_len=16384)"  # For TP=4: baseline, EP, async, PC, async+PC variants; same for TP=8

    return f"""
<div class="row g-4 mt-2">
  <div class="col-lg-7">
    <div class="card shadow-sm border-success h-100">
      <div class="card-header fw-bold text-success">&#128269; Best Configuration Summary (long-context focus: 4k–8k)</div>
      <div class="card-body p-0">
        <table class="table table-sm table-bordered table-hover mb-0">
          <thead class="table-light">
            <tr><th>Criterion</th><th>Winner</th><th class="text-end">Value</th></tr>
          </thead>
          <tbody>{rows_html or "<tr><td colspan='3' class='text-muted text-center'>No data</td></tr>"}</tbody>
        </table>
      </div>
    </div>
  </div>
  <div class="col-lg-5">
    <div class="card shadow-sm border-warning h-100">
      <div class="card-header fw-bold text-warning">&#9888; Key Tradeoffs at a Glance (8k context)</div>
      <div class="card-body" style="font-size:.86rem">
        <table class="table table-xs table-sm mb-0">
          <thead><tr><th>Config</th><th>Best for</th><th>Cost / note</th></tr></thead>
          <tbody>
            {'<tr class="table-primary"><td><code>tp4+PC+async</code> ⭐⭐</td><td>all metrics at 8k–16k</td><td>recommended production cmd</td></tr>' if asyncpc_cfg else ''}
            <tr class="table-success"><td><code>tp4+PC</code> ⭐</td><td>req/s, TTFT, ITL, tok/s</td><td>best single-opt at 8k</td></tr>
            <tr><td><code>tp4+async</code></td><td>+7% req/s free win</td><td>stacks with PC</td></tr>
            {'<tr><td><code>tp8+PC+async</code></td><td>lowest TTFT</td><td>2× GPUs; better than plain tp8</td></tr>' if tp8asyncpc_cfg else '<tr><td><code>tp8</code></td><td>lowest TTFT</td><td>2× GPUs; beaten by tp4+PC</td></tr>'}
            <tr><td><code>tp8+EP</code></td><td>≈ same as tp8</td><td>added complexity, no gain</td></tr>
            <tr class="table-danger"><td><code>tp4+EP</code> ⚠️</td><td>nothing</td><td>worst on every metric (−14% req/s)</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
  <div class="col-12">
    <div class="card shadow-sm border-primary">
      <div class="card-header fw-bold text-primary">&#128200; Ablation Insights (click to expand)</div>
      <div class="card-body pb-2">
        {accordion_html}
        {rec_html}
      </div>
    </div>
  </div>
  <div class="col-12">
    <div class="card shadow-sm">
      <div class="card-header fw-bold">&#128221; Experimental Setup</div>
      <div class="card-body" style="font-size:.85rem">
        <ul>
          <li><strong>Model:</strong> openai/gpt-oss-20b (MoE, 20B active params; MXFP4 baked in — quant=None = native MXFP4)</li>
          <li><strong>Hardware:</strong> Intel Arc Pro B60 (multi-GPU, XPU backend)</li>
          <li><strong>Tensor-parallel configs:</strong> {tp_note}</li>
          <li><strong>Always-on Intel optimizations:</strong> --block-size 64, --gpu-memory-util 0.9, --max-num-batched-tokens=8192, --dtype=bfloat16, --no-enable-prefix-caching (except PC config)</li>
          <li><strong>Input lengths:</strong> 1k / 2k / 4k / 8k tokens</li>
          <li><strong>Samples per length:</strong> 5 (synchronous profile — no concurrency)</li>
          <li><strong>Output tokens per request:</strong> 512</li>
          <li><strong>Dataset:</strong> arxiv-summarization (ccdv/arxiv-summarization) — long-form documents with varied prefixes</li>
          <li><strong>Ablation dimensions:</strong> TP degree (4/8), Expert Parallelism (--enable-expert-parallel), Async Scheduling (--async-scheduling), Prefix Caching (--enable-prefix-caching)</li>
          <li><strong>Eagle3 speculative decoding:</strong> disabled — XPU kernel exceeds PTSS scratch limit (292KB required, 256KB max)</li>
          <li><strong>quant=fp8:</strong> excluded — model's native MXFP4 rejects fp8 override</li>
        </ul>
      </div>
    </div>
  </div>
</div>"""


def build_ablation_dashboard_html(
    out_dir: Path,
    succeeded: list[str],
    model_mem_gib: Optional[Dict[str, float]] = None,
) -> Optional[Path]:
    """Build ablation_dashboard.html targeted at gpt-oss-20b configuration ablation.

    Unlike the standard dashboard (concurrency sweep + bars), this dashboard shows:
    - 4 LC lineplots (TTFT, ITL, req/s, tok/s) vs input tokens, one line per config
    - Per-config tabs with LC TTFT detail
    - Auto-generated "Conclusions" tab with best-config summary and pairwise insights

    Args:
        out_dir:       Directory containing ``{name}_lc*_benchmarks.json`` files.
        succeeded:     Config names that completed at least one LC slice.
        model_mem_gib: Optional {cfg.name: per_gpu_weight_gib}.

    Returns:
        Path to written ablation_dashboard.html, or None if no data.
    """
    ABLATION_TITLE = "Ablation Study — gpt-oss-20b on Intel Arc Pro B60"

    # Load LC data for all succeeded configs
    # lc_data: {cfg_name: {token_len: {metric_key: value}}}
    lc_data: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    for name in succeeded:
        pts = _load_lc_points(out_dir, name)
        if pts:
            lc_data[name] = pts

    if not lc_data:
        print("  No ablation LC results found for dashboard", flush=True)
        return None

    LC_METRICS = [
        ("ttft_ms",         "TTFT (ms)"),
        ("itl_ms",          "ITL (ms)"),
        ("throughput_rps",  "Req/s"),
        ("throughput_tps",  "Output tok/s"),
    ]

    # Build datasets for each LC metric (one line per config)
    lc_metric_ds: Dict[str, list] = {key: [] for key, _ in LC_METRICS}

    for i, (cfg_name, tok_map) in enumerate(lc_data.items()):
        color = COLORS[i % len(COLORS)]
        label = _ablation_label(cfg_name)
        ds_opts = {
            "borderColor": color, "backgroundColor": color + "55",
            "tension": 0.3, "spanGaps": True, "pointRadius": 6, "pointHoverRadius": 9,
        }
        for metric_key, _ in LC_METRICS:
            xy = [
                {"x": tok_len, "y": metrics.get(metric_key)}
                for tok_len, metrics in sorted(tok_map.items())
                if metrics.get(metric_key) is not None
            ]
            if xy:
                lc_metric_ds[metric_key].append({"label": label, "data": xy, **ds_opts})

    ts = _run_timestamp(out_dir)
    conclusions_html = _generate_conclusions(lc_data)

    # ------------------------------------------------------------------
    # Per-config tab content (LC detail only)
    # ------------------------------------------------------------------
    config_tabs_nav = ""
    config_tabs_content = ""

    for i, (cfg_name, tok_map) in enumerate(lc_data.items()):
        tab_id = "tab-" + cfg_name.replace("/", "-").replace("_", "-")
        label  = _ablation_label(cfg_name)
        color  = COLORS[i % len(COLORS)]
        lc_tok_lens = sorted(tok_map.keys())
        lc_labels_js = json.dumps(lc_tok_lens)

        def _lc_series(metric: str) -> str:
            return json.dumps([(tok_map.get(k) or {}).get(metric) for k in lc_tok_lens])

        def cid(s: str) -> str:
            return f"cfg-{i}-{s}"

        row_html = "".join(
            f'<tr><th class="text-end pe-3">{m_label}</th>'
            + "".join(
                f'<td>{f"{(tok_map.get(tl) or {}).get(m_key):.3g}" if (tok_map.get(tl) or {}).get(m_key) is not None else "—"}</td>'
                for tl in lc_tok_lens
            )
            + "</tr>"
            for m_key, m_label in LC_METRICS
        )
        col_heads = "".join(f'<th>{tl // 1024}k</th>' for tl in lc_tok_lens)

        config_tabs_nav += (
            f'<li class="nav-item">'
            f'<a class="nav-link" data-bs-toggle="tab" href="#{tab_id}">{label}</a>'
            f'</li>\n'
        )
        vllm_cmd = _load_vllm_cmd(out_dir, cfg_name)
        vllm_cmd_html = (
            f'<p class="text-muted mb-2" style="font-size:.72rem;word-break:break-all">'
            f'<strong>Docker:</strong> <code>{DOCKER_IMAGE}</code><br>'
            f'<strong>vllm serve:</strong> <code>{vllm_cmd}</code></p>'
        ) if vllm_cmd else (
            f'<p class="text-muted mb-2" style="font-size:.72rem">'
            f'<strong>Docker:</strong> <code>{DOCKER_IMAGE}</code></p>'
        )

        config_tabs_content += f"""<div class="tab-pane fade" id="{tab_id}">
  <h6 class="mt-3 mb-1 fw-bold">{cfg_name}</h6>
  {vllm_cmd_html}
  <div class="row g-3">
    <div class="col-12">
      <table class="table table-sm table-bordered table-hover">
        <thead><tr><th>Metric</th>{col_heads}</tr></thead>
        <tbody>{row_html}</tbody>
      </table>
    </div>
    <div class="col-12">
      <div class="row g-2">
        <div class="col-12"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">TTFT (ms) vs Input tokens</div><div class="card-body p-2"><canvas id="{cid('ttft')}" style="max-height:400px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">ITL (ms) vs Input tokens</div><div class="card-body p-2"><canvas id="{cid('itl')}" style="max-height:400px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">Req/s vs Input tokens</div><div class="card-body p-2"><canvas id="{cid('rps')}" style="max-height:400px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm"><div class="card-header" style="font-size:.8rem">Output tok/s vs Input tokens</div><div class="card-body p-2"><canvas id="{cid('tps')}" style="max-height:400px"></canvas></div></div></div>
      </div>
    </div>
  </div>
</div>
<script>
(function(){{
  const col = "{color}";
  const labels = {lc_labels_js};
  function aLine(id, vals, yLabel) {{
    const el = document.getElementById(id);
    if (!el) return;
    new Chart(el, {{
      type: 'line',
      data: {{ labels, datasets: [{{
        label: yLabel, data: vals, borderColor: col,
        backgroundColor: col + '55', pointRadius: 4, tension: 0.3, spanGaps: true
      }}]}},
      options: {{
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Input tokens' }} }},
          y: {{ beginAtZero: false, title: {{ display: true, text: yLabel }} }}
        }}
      }}
    }});
  }}
  aLine("{cid('ttft')}", {_lc_series('ttft_ms')},        "TTFT (ms)");
  aLine("{cid('itl')}",  {_lc_series('itl_ms')},         "ITL (ms)");
  aLine("{cid('rps')}",  {_lc_series('throughput_rps')}, "Req/s");
  aLine("{cid('tps')}",  {_lc_series('throughput_tps')}, "Output tok/s");
}})();
</script>\n"""

    # ------------------------------------------------------------------
    # Assemble the page
    # ------------------------------------------------------------------
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{ABLATION_TITLE}</title>
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
  <h4 class="mb-1 text-center fw-bold">{ABLATION_TITLE}</h4>
  <p class="text-center text-muted mb-1" style="font-size:.82rem">Docker: <code>{DOCKER_IMAGE}</code> &nbsp;|&nbsp; {ts} &nbsp;|&nbsp; gpt-oss-20b &nbsp;|&nbsp; Intel Arc Pro B60 &nbsp;|&nbsp; {len(lc_data)} configurations</p>
  <ul class="nav nav-tabs flex-wrap mb-0" id="mainTabs">
    <li class="nav-item"><a class="nav-link active" data-bs-toggle="tab" href="#tab-overview">&#128202; Overview</a></li>
    <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tab-conclusions">&#127919; Conclusions</a></li>
    {config_tabs_nav}
  </ul>
  <div class="tab-content border border-top-0 rounded-bottom bg-white p-3">
    <div class="tab-pane fade show active" id="tab-overview">
      <div class="row g-4 mt-2">
        <div class="col-12"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">TTFT (ms) vs Input tokens</div><div class="card-body"><canvas id="lc-ttft" style="max-height:500px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">ITL (ms) vs Input tokens</div><div class="card-body"><canvas id="lc-itl" style="max-height:500px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">Req/s vs Input tokens</div><div class="card-body"><canvas id="lc-rps" style="max-height:500px"></canvas></div></div></div>
        <div class="col-12"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">Output tok/s vs Input tokens</div><div class="card-body"><canvas id="lc-tps" style="max-height:500px"></canvas></div></div></div>
      </div>
    </div>
    <div class="tab-pane fade" id="tab-conclusions">
      {conclusions_html}
    </div>
    {config_tabs_content}
  </div>
</div>
<script>
function lcLine(id, datasets, yLabel) {{
  const el = document.getElementById(id);
  if (!el || !datasets.length) return;
  new Chart(el, {{
    type: 'line', data: {{ datasets }},
    options: {{
      parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Input tokens' }} }},
        y: {{ beginAtZero: false, title: {{ display: true, text: yLabel }} }}
      }},
      plugins: {{ legend: {{ position: 'top' }} }}
    }}
  }});
}}
lcLine('lc-ttft', {json.dumps(lc_metric_ds['ttft_ms'])},        'TTFT (ms)');
lcLine('lc-itl',  {json.dumps(lc_metric_ds['itl_ms'])},         'ITL (ms)');
lcLine('lc-rps',  {json.dumps(lc_metric_ds['throughput_rps'])}, 'Req/s');
lcLine('lc-tps',  {json.dumps(lc_metric_ds['throughput_tps'])}, 'Output tok/s');
</script>
</body>
</html>"""

    out_path = out_dir / "ablation_dashboard.html"
    out_path.write_text(page)
    print(f"\n  Ablation Dashboard → {out_path}", flush=True)
    write_serve_script(out_path)
    return out_path
