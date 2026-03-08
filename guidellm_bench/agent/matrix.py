"""TTFT matrix measurement for the agent benchmark.

Measures all (N_cached × N_new) cells defined in constants.MATRIX_N_CACHED /
MATRIX_N_NEW.  Each cell fires CONCURRENCY=1 requests serially so there is no
interference between measurements.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import requests

from .constants import (
    MATRIX_N_CACHED, MATRIX_N_NEW,
    N_WARMUPS, N_SAMPLES, CV_RERUN_THRESHOLD,
    OUTPUT_TOKENS_DEFAULT, INTER_REQUEST_SLEEP_S,
    CONCURRENCY, CellResult,
)
from .debug import _DBG, _DBG_INFO, _DBG_WARN
from .helpers import make_session, _warm_cache, _verify_token_count, _measure_ttft
from .corpus import Corpus

assert CONCURRENCY == 1, "matrix measurement assumes concurrency=1"

# ---------------------------------------------------------------------------
# Cold-TTFT estimate (fitted to no-PC ablation data)
# ---------------------------------------------------------------------------


def _cold_ttft_estimate(n_new: int, n_cached: int) -> float:
    """Rough cold-prefill TTFT at N_total = n_new + n_cached tokens.

    From fitted model (no-PC, measured on gpt-oss-20b+tp4):
      base=38.3 + alpha=62.0*N_k + beta=0.731*N_k^2
    """
    n_k = (n_new + n_cached) / 1000.0
    return 38.3 + 62.0 * n_k + 0.731 * n_k ** 2


# ---------------------------------------------------------------------------
# Single cell measurement
# ---------------------------------------------------------------------------


def measure_cell(
    session: requests.Session,
    cached_prompt: str,
    new_prompt: str,
    n_cached_target: int,
    n_new_target: int,
    n_warmups: int = N_WARMUPS,
    n_samples: int = N_SAMPLES,
    max_output_tokens: int = OUTPUT_TOKENS_DEFAULT,
    verbose: bool = True,
) -> CellResult:
    """Measure one (N_cached, N_new) cell of the TTFT matrix.

    Protocol (concurrency=1, all requests serial):
      1. Warm cache with cached_prompt (max_tokens=1).
      2. N_warmups streaming requests — discard TTFTs.
      3. Between each warm-up: re-warm cache (eviction guard).
      4. N_samples streaming requests — record TTFT.
      5. If CV > CV_RERUN_THRESHOLD, run one extra pass and merge.
    """
    full_prompt = cached_prompt + new_prompt
    cold_est = _cold_ttft_estimate(n_new_target, n_cached_target)
    c_label = f"N_cached={n_cached_target//1024}k N_new={n_new_target//1024}k"
    prefix = f"  Cell {c_label}"

    _DBG_INFO(
        f"[measure_cell] START {c_label}  cold_est={cold_est:.0f}ms  "
        f"max_output={max_output_tokens}"
    )

    if verbose:
        print(f"{prefix}: verifying token counts...", flush=True)

    actual_total  = _verify_token_count(session, full_prompt)
    actual_cached = _verify_token_count(session, cached_prompt) if n_cached_target > 0 else 0
    actual_new    = actual_total - actual_cached

    _DBG(
        f"measure_cell {c_label}: actual — "
        f"cached={actual_cached}, new={actual_new}, total={actual_total}"
    )
    if verbose:
        print(
            f"{prefix}: actual tokens: cached={actual_cached}, "
            f"new={actual_new}, total={actual_total}",
            flush=True,
        )

    def _one_measurement() -> tuple[float, float, float]:
        """Returns (ttft_ms, total_ms, warm_cache_ms). Concurrency=1."""
        wc_ms = 0.0
        if n_cached_target > 0:
            wc_ms = _warm_cache(session, cached_prompt, output_tokens=1)
            time.sleep(0.3)
        ttft, _, total_ms = _measure_ttft(
            session, full_prompt, max_tokens=max_output_tokens
        )
        time.sleep(INTER_REQUEST_SLEEP_S)
        return ttft, total_ms, wc_ms

    # Warm-up passes (discard TTFT but record for debug)
    warmup_ttft: list[float] = []
    for i in range(n_warmups):
        if verbose:
            print(f"{prefix}: warm-up {i+1}/{n_warmups}...", flush=True)
        ttft_wu, _, _ = _one_measurement()
        warmup_ttft.append(ttft_wu)
        _DBG(f"measure_cell {c_label}: warmup {i+1}/{n_warmups} TTFT={ttft_wu:.0f}ms")

    # Measured passes
    values:       list[float] = []
    total_ms_list: list[float] = []
    wc_ms_list:   list[float] = []
    for i in range(n_samples):
        ttft, total_ms, wc_ms = _one_measurement()
        values.append(ttft)
        total_ms_list.append(total_ms)
        wc_ms_list.append(wc_ms)
        _DBG(
            f"measure_cell {c_label}: sample {i+1}/{n_samples} "
            f"TTFT={ttft:.0f}ms  total={total_ms:.0f}ms  warm_cache={wc_ms:.0f}ms"
        )
        if verbose:
            print(f"{prefix}: sample {i+1:2d}/{n_samples} → {ttft:.0f} ms", flush=True)

    # Reliability check
    median_v = statistics.median(values)
    stddev_v = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = stddev_v / median_v if median_v > 0 else 0.0
    if cv > CV_RERUN_THRESHOLD:
        _DBG_WARN(
            f"measure_cell {c_label}: CV={cv:.2f} > {CV_RERUN_THRESHOLD} — "
            f"running extra pass (median={median_v:.0f}ms)"
        )
        print(f"{prefix}: ⚠ CV={cv:.2f} > {CV_RERUN_THRESHOLD} — running extra pass", flush=True)
        for j in range(n_samples):
            ttft, total_ms, wc_ms = _one_measurement()
            values.append(ttft)
            total_ms_list.append(total_ms)
            wc_ms_list.append(wc_ms)
            _DBG(
                f"measure_cell {c_label}: rerun {j+1}/{n_samples} "
                f"TTFT={ttft:.0f}ms  total={total_ms:.0f}ms"
            )
        median_v = statistics.median(values)
        stddev_v = statistics.stdev(values) if len(values) > 1 else 0.0
        cv = stddev_v / median_v if median_v > 0 else 0.0

    sorted_v = sorted(values)
    n = len(sorted_v)
    cache_hit_threshold = cold_est * 0.5
    hit_ratio = sum(1 for v in values if v < cache_hit_threshold) / len(values)

    result = CellResult(
        n_cached=n_cached_target,
        n_new=n_new_target,
        actual_cached=actual_cached,
        actual_new=actual_new,
        n_samples=len(values),
        ttft_ms_values=values,
        ttft_median=round(statistics.median(values), 1),
        ttft_p25=round(sorted_v[int(0.25 * n)], 1),
        ttft_p75=round(sorted_v[int(0.75 * n)], 1),
        ttft_p95=round(sorted_v[int(0.95 * n)], 1),
        ttft_min=round(min(values), 1),
        ttft_max=round(max(values), 1),
        ttft_cv=round(cv, 3),
        cache_hit_ratio=round(hit_ratio, 2),
        cold_ttft_estimate=round(cold_est, 1),
        warmup_ttft_ms_values=warmup_ttft,
        total_request_ms_values=total_ms_list,
        warm_cache_ms_values=wc_ms_list,
    )
    _DBG_INFO(
        f"[measure_cell] DONE {c_label}: "
        f"median={result.ttft_median:.0f}ms  p95={result.ttft_p95:.0f}ms  "
        f"CV={cv:.2f}  cache_hit={hit_ratio:.0%}  n_samples={len(values)}"
    )
    print(
        f"{prefix}: ✓ median={result.ttft_median:.0f}ms  "
        f"p95={result.ttft_p95:.0f}ms  CV={cv:.2f}  "
        f"cache_hit={hit_ratio:.0%}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Full matrix sweep
# ---------------------------------------------------------------------------


def run_ttft_matrix(
    corpus: Corpus,
    out_dir: Path,
    n_warmups: int = N_WARMUPS,
    n_samples: int = N_SAMPLES,
    max_output_tokens: int = OUTPUT_TOKENS_DEFAULT,
    resume: bool = False,
) -> list[CellResult]:
    """Measure all (N_cached × N_new) cells.  Returns list of CellResult.

    All cells are measured serially (CONCURRENCY=1).
    Checkpoints after each cell to agent_matrix.json.
    """
    session = make_session()
    results: list[CellResult] = []
    cache_file = out_dir / "agent_matrix.json"
    total_cells = len(MATRIX_N_CACHED) * len(MATRIX_N_NEW)

    _DBG_INFO(
        f"[run_ttft_matrix] START  cells={total_cells}  "
        f"n_warmups={n_warmups}  n_samples={n_samples}  "
        f"max_output_tokens={max_output_tokens}  resume={resume}  "
        f"concurrency={CONCURRENCY}"
    )

    done_cells: set[tuple[int, int]] = set()
    if resume and cache_file.exists():
        existing = json.loads(cache_file.read_text())
        for cr in existing.get("matrix", []):
            results.append(CellResult(**{k: v for k, v in cr.items()}))
            done_cells.add((cr["n_cached"], cr["n_new"]))
        _DBG_INFO(f"[run_ttft_matrix] Resumed {len(done_cells)} already-completed cells.")
        print(f"  Resumed {len(done_cells)} already-completed cells.", flush=True)

    cell_idx = 0
    t_matrix_start = time.perf_counter()

    for n_cached in MATRIX_N_CACHED:
        cached_prompt = corpus.slice_text(0, n_cached) if n_cached > 0 else ""

        for n_new in MATRIX_N_NEW:
            cell_idx += 1
            c_label = f"N_cached={n_cached//1024}k N_new={n_new//1024}k"

            if (n_cached, n_new) in done_cells:
                print(
                    f"  [{cell_idx}/{total_cells}] Skipping {c_label} (already done)",
                    flush=True,
                )
                continue

            new_prompt = corpus.slice_text(n_cached, n_cached + n_new)
            t_cell_start = time.perf_counter()
            _DBG_INFO(f"[run_ttft_matrix] [{cell_idx}/{total_cells}] START {c_label}")
            print(
                f"\n  [{cell_idx}/{total_cells}] "
                f"N_cached={n_cached//1024}k  N_new={n_new//1024}k",
                flush=True,
            )
            cr = measure_cell(
                session, cached_prompt, new_prompt,
                n_cached, n_new, n_warmups, n_samples, max_output_tokens,
            )
            results.append(cr)
            cell_elapsed = (time.perf_counter() - t_cell_start) / 60.0
            _DBG_INFO(
                f"[run_ttft_matrix] [{cell_idx}/{total_cells}] DONE {c_label}  "
                f"cell_wall={cell_elapsed:.1f}min  "
                f"matrix_wall={(time.perf_counter()-t_matrix_start)/60:.1f}min"
            )
            _save_matrix_checkpoint(results, cache_file)

    total_elapsed = (time.perf_counter() - t_matrix_start) / 60.0
    _DBG_INFO(
        f"[run_ttft_matrix] ALL DONE  "
        f"cells_measured={len(results)}  total_wall={total_elapsed:.1f}min"
    )
    return results


def _save_matrix_checkpoint(results: list[CellResult], path: Path) -> None:
    data = {"matrix": [asdict(r) for r in results]}
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def print_ttft_table(matrix: list[CellResult]) -> None:
    """Print the (N_cached × N_new) TTFT median table to stdout."""
    lookup: dict[tuple[int, int], CellResult] = {
        (c.n_cached, c.n_new): c for c in matrix
    }
    col_w = 16
    header_cells = [f"  {n//1024}k new toks" for n in MATRIX_N_NEW]
    print()
    print("=" * 80)
    print("TTFT MATRIX — gpt-oss-20b tp8+PC (measured, concurrency=1)")
    print("  Columns = new tokens per iteration | Rows = accumulated KV cache context")
    print(f"  Values = median TTFT  [p25–p75 range]  (n={N_SAMPLES} samples/cell)")
    print("=" * 80)
    print(
        f"  {'Cached context':>20} |"
        + "|".join(f"{h:>{col_w}}" for h in header_cells)
        + " |"
    )
    print("  " + "-" * 22 + ("+" + "-" * col_w) * len(MATRIX_N_NEW))

    for n_cached in MATRIX_N_CACHED:
        row_label = f"{n_cached//1024}k accumulated" if n_cached > 0 else "0k (cold)"
        row = f"  {row_label:>20} |"
        for n_new in MATRIX_N_NEW:
            cr = lookup.get((n_cached, n_new))
            if cr is None:
                row += f"{'  --':>{col_w}} |"
                continue
            med = cr.ttft_median
            cv  = cr.ttft_cv
            flag = "⚠" if cv > 0.25 else " "
            cell = f"{med/1000:.2f}s {flag}" if med >= 1000 else f"{med:.0f}ms {flag}"
            row += f"{cell:>{col_w}} |"
        print(row)

    print()
    print("  ⚠ = CV > 0.25 (high variance; interpret with caution)")
    print()
