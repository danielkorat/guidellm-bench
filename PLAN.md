# guidellm-bench Feature Plan

Status legend: ⬜ not started · 🔄 in progress · ✅ done · ❌ blocked

---

## 1. Fix Israel Timestamps ✅
**Problem**: `ZoneInfo("Asia/Jerusalem")` silently falls back to UTC if `tzdata` is not installed in the container.  
**Fix**: Wrap in try/except; fall back to `UTC+2` fixed offset as last resort. Log a warning if the system tzdata lookup fails.  
**Files**: `bench.py` (`main()`).

---

## 2. Remove Incomplete Run Directories ✅
**Problem**: Failed/aborted runs leave empty dirs cluttering `results/` and `sanity_results/`.  
**Fix**: On startup (before creating a new `out_dir`), scan the base results dir and delete any subdirectory that contains **zero `*_benchmarks.json` files**. Print what was removed. Also add `--no-clean` flag to suppress.  
**Files**: `bench.py` (`main()`).

---

## 3. `--ep-compare` Flag ✅
**What it does**: Runs EP-capable models (gpt-oss-20b, Qwen3-30B-A3B) **both with and without EP** in the same run so results appear side-by-side in the dashboard.  
**Behaviour**:
- Acts like `--ep` but also ensures the non-EP base configs for those same `model+tp+quant` combos are in the matrix.
- Non-EP base configs that the standard matrix would skip due to `--models` filtering are force-added.
- Mutually exclusive with `--ep` to avoid confusion.  
**Files**: `bench.py`, `config.py` (add `EP_COMPARE_PAIRS` constant).

---

## 4. Auto-detect HF Dataset Column ✅
**Problem**: Different HF datasets use different column names (`text`, `prompt`, `input`, `question`, `content`, `instruction`, …).  
**Fix**: In `dataset.py`, add a generic `prepare_hf_dataset(hf_name, ...)` function that:
1. Downloads the HF dataset,
2. Inspects string columns and picks the best text column by priority list then by median length heuristic,
3. Converts to `{prompt, output_tokens_count}` JSONL.  
**CLI**: `--data <hf_dataset_id>` flag in bench.py (replaces hardcoded AIME path when provided).  
**Files**: `dataset.py`, `bench.py`.

---

## 5. `--long-contexts` Flag ✅
**What it does**: For every config in the run, runs **4 additional mini-benchmarks** with truncated input lengths: 1k, 4k, 8k, 16k tokens (10 samples each, synchronous profile).  
**Implementation**:
- Token estimation: simple whitespace-split approximation (words × 1.3).
- `prepare_long_context_datasets(texts, token_lengths, num_samples, output_tokens)` — produces one JSONL per length, cached under `out_dir/datasets/lc_{N}k.jsonl`.
- `run_guidellm_lc(cfg, lc_dataset_path, target_token_len, out_dir)` — mirrors `run_guidellm` for a single length slice. Stores result as `{cfg_name}_lc{N}k_benchmarks.json`.
- `--max-model-len` must be ≥ 16384 (enforced/overridden automatically when `--long-contexts` is set).  
**Files**: `dataset.py`, `benchmark.py`, `bench.py`.

---

## 6. Long-Contexts Dashboard Plot ✅
**What it shows**: Per-config tab gets a new **"TTFT vs Input Length"** line chart, plotting median TTFT (ms) at 1k/4k/8k/16k input tokens.  
**Overview tab**: Adds a "TTFT vs Input Length — all configs" multi-line chart so configs can be compared at a glance.  
**Data key**: `{cfg_name}_lc{N}k_benchmarks.json` loaded separately from normal benchmark files.  
**Files**: `dashboard.py`.

---

## 7. Sanity Test ⬜
```bash
./bench.py --sanity --models openai/gpt-oss-20b --tp 4 --long-contexts --ep-compare
```
Validates:
- A single gpt-oss-20b tp4 (non-EP) config runs correctly.
- An EP variant runs.
- Long-context slices (1k/4k/8k/16k) run.
- `dashboard.html` contains EP comparison bars and TTFT-vs-input-length chart.

---

## 8. Full Benchmark — gpt-oss with EP Compare + Long Contexts ⬜
```bash
nohup ./bench.py \
  --models openai/gpt-oss-20b \
  --tp 4 \
  --ep-compare \
  --long-contexts \
  --data cx-cmu/deepresearchgym-agentic-search-logs \
  > /dev/null 2>&1 &
```
Runs all gpt-oss-20b configs (non-EP + EP) with long-context slices using the deepresearchgym dataset.

---

## Implementation Order
1. ✅ Fix Israel timestamps (`bench.py` — `_israel_now()` with `ZoneInfo` + UTC+2 fallback)
2. ✅ Remove incomplete dirs (startup auto-clean via `_clean_incomplete_runs()`)
3. ✅ Auto-detect HF dataset column (`dataset.py` — `_detect_text_column()`)
4. ✅ Generic HF dataset loader (`dataset.py` — `prepare_hf_dataset()`)
5. ✅ `--long-contexts` dataset slicing (`dataset.py` — `prepare_long_context_datasets()`)
6. ✅ `--long-contexts` benchmark runner (`benchmark.py` — `lc_mode` param in `run_guidellm()`)
7. ✅ `--ep-compare` flag (`bench.py` — ensures both EP + non-EP configs in matrix)
8. ✅ Long-contexts dashboard plot (`dashboard.py` — TTFT-vs-input-length in per-config + overview)
9. ⬜ Sanity test + dashboard validation
10. ⬜ Full gpt-oss run

---

*Last updated: 2026-03-03*
