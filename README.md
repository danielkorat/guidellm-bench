# guidellm-bench

Automated benchmarking for **Intel XPU vLLM** inference using [guidellm](https://github.com/vllm-project/guidellm). Runs a matrix of model × tensor-parallelism × quantization configurations and produces an interactive HTML dashboard with latency, throughput, and GPU memory metrics.

## Requirements

- Running inside `intel/vllm:0.14.1-xpu` Docker container
- Patched guidellm fork (fixes TTFT=0 for thinking models):
  ```bash
  pip install git+https://github.com/danielkorat/guidellm.git@fix/thinking-model-ttft
  ```

## Quick Start

```bash
# Sanity check (single config, small dataset)
./bench.py --sanity

# Full benchmark suite
./bench.py

# Full suite in background
nohup ./bench.py > bench_full.log 2>&1 & echo $! > bench_full.pid
tail -f bench_full.log

# With Eagle3 speculative decoding (gpt-oss-120b, tp=8)
./bench.py --eagle3
```

## Default Matrix

| Parameter | Full | Sanity |
|---|---|---|
| Models | `gpt-oss-20b`, `Qwen3-30B-A3B`, `Qwen3-4B-Thinking-2507` | `Qwen3-4B-Thinking-2507` |
| TP | `[4]` | `[4]` |
| Quantization | `none`, `fp8` | `none` |
| Eager | `true` | `true` |
| Prompts | 20 (16 clean after warmup/cooldown) | 4 |
| Max model len | 16384 | 2048 |

Override any parameter via CLI: `--models`, `--tp`, `--quantization`, `--num-prompts`, `--max-model-len`, etc.

## Outputs

Each run creates a timestamped directory (Israel time, `YYYYMMDD_HHMM`):

```
guidellm_results/20260302_1022/
├── {cfg}_benchmarks.json        # raw guidellm metrics
├── {cfg}_benchmarks.html        # per-config guidellm report
├── {cfg}_gpu_monitor.json       # xpu-smi GPU util/power/memory time-series
├── dashboard.html               # combined interactive dashboard
├── serve_dashboard.sh           # one-click local HTTP server
└── logs/
```

Serve the dashboard:
```bash
bash guidellm_results/YYYYMMDD_HHMM/serve_dashboard.sh
```

## Dashboard Metrics

- **TTFT** (Time to First Token, ms)
- **ITL** (Inter-Token Latency, ms)
- **Throughput** (req/s and tok/s)
- **Peak GPU Memory Used** (MiB, all devices)
- Per-config GPU utilisation and memory time-series charts

## Rebuild Dashboard from Partial Results

If a run is interrupted, regenerate the dashboard from completed configs:

```python
from bench import build_dashboard_html
from pathlib import Path
build_dashboard_html(Path("guidellm_results/YYYYMMDD_HHMM"), ["cfg1_name", "cfg2_name"])
```

## Known Skip Rules

| Skipped combination | Reason |
|---|---|
| Any model, fp8 + eager=false | vLLM engine init failure |
| gpt-oss-20b + fp8 | Model has mxfp4 baked in |
| gpt-oss-20b + eager=false | XPU OOM (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) |
| Qwen3-30B + quant=none | IPEX mode-stack bug |

## Notes

- Uses `HuggingFaceH4/aime_2024` (30 AIME math problems) as realistic prompts; falls back to synthetic tokens if the dataset is unavailable.
- Benchmark quality controls: `--warmup 0.1 --cooldown 0.1 --max-errors 5 --max-seconds 600`
- Thinking models use `--request-format /v1/completions` to bypass chat template (prevents TTFT=0).
