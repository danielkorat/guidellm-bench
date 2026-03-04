"""Combined interactive HTML dashboard builder."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict


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
        ds_opts = {"borderColor": color, "backgroundColor": color + "33",
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
        config_tabs_content += f"""<div class="tab-pane fade" id="{tab_id}">
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
        backgroundColor: col + '33', pointRadius: 4, tension: 0.3, spanGaps: true
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
    <div class="col-md-6"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">TTFT (ms) vs Input tokens</div><div class="card-body"><canvas id="c-lc-ttft" style="max-height:260px"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">ITL (ms) vs Input tokens</div><div class="card-body"><canvas id="c-lc-itl" style="max-height:260px"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">Req/s vs Input tokens</div><div class="card-body"><canvas id="c-lc-rps" style="max-height:260px"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm border-info"><div class="card-header fw-bold text-info">Output tok/s vs Input tokens</div><div class="card-body"><canvas id="c-lc-tps" style="max-height:260px"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">Model Weights Memory (GiB, all devices)</div><div class="card-body"><canvas id="c-bar-mem"></canvas></div></div></div>
  </div>
</div>"""
    else:
        overview_content = f"""<div class="tab-pane fade show active" id="tab-overview">
  <div class="row g-4 mt-2">
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">TTFT (ms) &mdash; median</div><div class="card-body"><canvas id="c-bar-ttft"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">ITL (ms) &mdash; median</div><div class="card-body"><canvas id="c-bar-itl"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">Throughput req/s</div><div class="card-body"><canvas id="c-bar-rps"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">Output tok/s</div><div class="card-body"><canvas id="c-bar-tps"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">Model Weights Memory (GiB, all devices)</div><div class="card-body"><canvas id="c-bar-mem"></canvas></div></div></div>
    <div class="col-md-6"><div class="card shadow-sm"><div class="card-header fw-bold">TTFT (ms) vs Concurrency &mdash; sweep</div><div class="card-body"><canvas id="c-line-ttft" style="max-height:260px"></canvas></div></div></div>
    <div class="col-12"><div class="card shadow-sm"><div class="card-header fw-bold">Throughput (req/s) vs Concurrency &mdash; sweep</div><div class="card-body"><canvas id="c-line-rps" style="max-height:260px"></canvas></div></div></div>
  </div>
</div>"""

    ts       = _run_timestamp(out_dir)
    subtitle = _build_subtitle(records)
    subtitle_html = (
        f'<p class="text-center text-muted mb-1" style="font-size:.82rem">'
        f'{ts}'
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
