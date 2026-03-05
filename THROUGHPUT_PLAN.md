# Throughput Study — Design Plan (`--throughput` flag)

## Goal
Characterise gpt-oss-20b throughput and latency across a matrix of
**concurrency levels × input lengths** using two vLLM server configurations.
Surfaces trade-offs between no-EP and EP at each scalability point.

---

## Summary Numbers

| Dimension       | Values                             |
|---|---|
| Model           | openai/gpt-oss-20b                 |
| Input lengths   | 16k, 32k, 48k, 96k (tokens)       |
| Output length   | 16k tokens (fixed)                 |
| Concurrencies   | 1, 16, 64, 128                     |
| Server configs  | 2 (no-EP and EP — see below)       |
| Total cells     | 7 concurrency×config combos × 4 input_lens = 28 cells |
| Server restarts | **1 only** (Server A → Server B)   |

---

## Server Configs

### Server A — `tp8_async` (no EP)
```
vllm serve openai/gpt-oss-20b
  --dtype=bfloat16
  -tp=8
  --enforce-eager
  --async-scheduling
  --no-enable-prefix-caching        ← always disabled
  --max-model-len 131072            ← model's max_position_embeddings
  --max-num-batched-tokens 131072   ← raised from 8192 to handle 96k prefill
  --block-size 64
  --gpu-memory-util 0.9
  --disable-sliding-window
```
Runs concurrencies: **c=1, c=16, c=64, c=128**

### Server B — `tp8_async_ep` (with EP)
Same flags plus `--enable-expert-parallel`.
Runs concurrencies: **c=16, c=64, c=128** (EP skipped for c=1 per user requirement)

---

## Samples per Concurrency

Rule: `num_samples = 2 × c` to ensure the server sustains full queue depth
throughout the measurement window.

| Concurrency | num_samples | Clean window (80%) |
|---|---|---|
| c=1         | 10          | 10 (sequential)    |
| c=16        | 32          | ~26                |
| c=64        | 128         | ~102               |
| c=128       | 256         | ~205               |

---

## Execution Order (guarantees exactly 1 server restart)

```
[Start Server A — tp8_async]
  c=1:   il=16k(10req), il=32k(10req), il=48k(10req), il=96k(10req)
  c=16:  il=16k(32req), il=32k(32req), il=48k(32req), il=96k(32req)
  c=64:  il=16k(128req), il=32k(128req), il=48k(128req), il=96k(128req)
  c=128: il=16k(256req), il=32k(256req), il=48k(256req), il=96k(256req)
[Stop Server A]
[Start Server B — tp8_async_ep]
  c=16:  il=16k(32req), …
  c=64:  il=16k(128req), …
  c=128: il=16k(256req), …
[Keep Server B running for reuse]
```

---

## guidellm Profile per Concurrency

- **c=1**: `--profile synchronous --max-requests 10` (serial; lc_mode=True)
- **c>1**: `--profile concurrent --rate {c} --max-requests {num_samples}` (concurrent)
- All cells: `--max-seconds 10800` (3h ceiling — 96k×c=1 can take ~100 min)

---

## Dataset Design

Source: `ccdv/arxiv-summarization` (~1000 papers, median ~6,620 tokens each).

Prefix caching is **disabled** on both servers; cross-sample document reuse
does NOT cause KV-cache seeding artifacts (unlike Lesson 36 which involved
a server with PC enabled). Therefore strict disjointness across input
lengths is NOT required — only within a single run's N samples.

### `prepare_throughput_dataset(source_path, input_len, output_len, num_samples, cache_dir)`

New function in `dataset.py`:
1. Filter papers with ≥50% of target token length.
2. Compute `papers_per_sample = ceil(input_len / mean_paper_tokens)`.
3. Build each sample by concatenating `papers_per_sample` papers, cycling
   through the source list when exhausted (wrapping is fine — PC is off).
4. Truncate to exactly `input_len` tokens.
5. Set `output_tokens_count = output_len` in each JSONL row.
6. Write 256 samples (max needed for c=128); cache as `throughput_{stem}_{il}k_v1.jsonl`.

For smaller concurrencies (c=1:10, c=16:32, c=64:128), `run_guidellm` passes
`--data-samples {num_samples}` so only the first N rows are used — one file
serves all concurrency levels at each input length.

### max_model_len
131072 (model's `max_position_embeddings`). Limits input to 96k + 16k output = 112k,
well within 131k.

---

## File Naming Convention

```
{cfg_name}_c{concurrency}_il{input_len//1024}k_benchmarks.json
```
Example: `openai_gpt-oss-20b_tp8_quant-none-async_c64_il32k_benchmarks.json`

Results directory: `throughput_results/YYYYMMDD_HHMM/`

---

## Dashboard Tabs

`build_throughput_dashboard_html()` in `dashboard.py`:

| Tab | Content |
|---|---|
| **c=1 (Latency)** | 4 metric charts (TTFT, ITL, Req/s, Tok/s) vs input_len. Only no-EP config. |
| **c=16** | Same 4 charts. Both no-EP and EP lines. |
| **c=64** | Same 4 charts. Both no-EP and EP lines. |
| **c=128** | Same 4 charts. Both no-EP and EP lines. |
| **Concurrency Effects** | TTFT and tok/s vs concurrency [1,16,64,128], one line per (config × input_len). EP lines start at c=16, no-EP at c=1. |

---

## Code Changes

| File | Change |
|---|---|
| `guidellm_bench/config.py` | `THROUGHPUT_*` constants + `get_throughput_configs()` |
| `guidellm_bench/server.py` | `max_num_batched_tokens: int = 8192` param in `build_vllm_cmd` + `start_server` |
| `guidellm_bench/benchmark.py` | `data_samples: int = -1`, `max_seconds: int = 900`, `num_requests_override: Optional[int] = None` params in `run_guidellm` |
| `guidellm_bench/dataset.py` | New `prepare_throughput_dataset()` function |
| `guidellm_bench/__init__.py` | Export new symbols |
| `guidellm_bench/dashboard.py` | `_load_throughput_points()` + `build_throughput_dashboard_html()` |
| `bench.py` | `--throughput` CLI arg + `_run_throughput()` + `throughput_results/` dir |
| `.github/copilot-instructions.md` | New rule for throughput mode |

---

## ETA Estimate

Per-cell rough estimates (gpt-oss-20b tp=8, BF16/MXFP4):

> **Actuals from first cell (c=1, il=16k)**: generation ~32.7 tok/s → ~8.4 min/req
> (TTFT for 16k input: ~18s at 930 tok/s prefill throughput — negligible vs generation time)

| Cell | Time estimate |
|---|---|
| c=1, il=16k, 10 req | ~87 min (8.4 min/req gen + ~3 min TTFT overhead) |
| c=1, il=32k, 10 req | ~91 min (8.4 min gen + ~35s TTFT each) |
| c=1, il=48k, 10 req | ~96 min (8.4 min gen + ~53s TTFT each) |
| c=1, il=96k, 10 req | ~102 min (8.4 min gen + ~106s TTFT each) |
| c=16, any il, 32 req | ~12 min (GPU better utilised at concurrency) |
| c=64, any il, 128 req | ~30 min |
| c=128, any il, 256 req | ~60 min |

**Server A total (all 4×c, 4×il)**:
- c=1: ~(87+91+96+102) = 376 min ≈ 6.3 hours
- c=16: ~4×12 = 48 min
- c=64: ~4×30 = 120 min
- c=128: ~4×60 = 240 min
- **Server A subtotal: ~13.7 hours**

**Server B total (c=16,64,128 only — no c=1)**:
- ~(48+120+240) = 408 min ≈ **~6.8 hours**

**2 server restarts**: ~4 min total

**Total ETA: ~20 hours**  ±4 hours depending on XPU throughput scaling at higher concurrency and KV-cache pressure at 96k context.

---

## Constants Summary

```python
THROUGHPUT_INPUT_LENGTHS         = [16384, 32768, 49152, 98304]   # 16k/32k/48k/96k
THROUGHPUT_OUTPUT_LEN            = 16384                           # 16k output
THROUGHPUT_CONCURRENCIES         = [1, 16, 64, 128]
THROUGHPUT_MAX_MODEL_LEN         = 131072                          # max_position_embeddings
THROUGHPUT_MAX_NUM_BATCHED_TOKENS = 131072                         # raised from 8192
THROUGHPUT_SAMPLES               = {1: 10, 16: 32, 64: 128, 128: 256}
THROUGHPUT_MAX_SECONDS           = 10800                           # 3h ceiling per cell
```
