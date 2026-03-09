"""Top-level entry point for the agent benchmark.

run_agent_bench()      — full benchmark: TTFT matrix + real ReAct scenarios
get_agent_server_config() — vLLM Config for the agent server (tp8, PC, 131k ctx)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..config import Config
from .constants import (
    AGENT_MODEL, AGENT_TP, AGENT_MAX_MODEL_LEN,
    MATRIX_N_CACHED, MATRIX_N_NEW, N_SAMPLES, N_WARMUPS, N_AGENT_SCENARIOS,
    AgentBenchResult, CellResult, ScenarioResult,
)
from .debug import _setup_debug_log, _DBG_INFO, _DBG_WARN
from .corpus import Corpus, _prepare_frames_corpus, _find_arxiv_fallback
from .helpers import make_session
from .matrix import run_ttft_matrix, print_ttft_table
from .scenarios import (
    _load_frames_questions,
    run_agent_scenarios_frames,
    print_scenario_summary,
)

__all__ = ["run_agent_bench", "get_agent_server_config"]


def get_agent_server_config(tp: Optional[int] = None) -> Config:
    """Return the vLLM Config for the agent benchmark server.

    Defaults:  model=gpt-oss-20b  tp=8  eager=True  prefix_caching=True
               max_model_len=131_072  max_num_batched_tokens=8_192 (AGENT_MAX_BATCHED)

    Context window: 131k tokens — works because AGENT_MAX_BATCHED=8_192 keeps the
    vLLM warm-up dummy run below the Intel XPU Triton PTSS 256KB limit.  Actual long
    prompts are served via chunked prefill (16 passes × 8k batched tokens).

    NOTE: async_scheduling omitted — at CONCURRENCY=1 it adds no throughput benefit
    and can trigger XPU Triton compilation failures at large context sizes.
    If ZE_RESULT_ERROR_MODULE_BUILD_FAILURE is seen on startup, clear the stale
    Inductor cache: docker exec lsv-container rm -rf /tmp/torchinductor_root

    Args:
        tp: Tensor-parallelism override.  Default = AGENT_TP (8).
            Use tp=4 when fewer than 8 GPUs are available.
    """
    return Config(
        model=AGENT_MODEL,
        tp=tp if tp is not None else AGENT_TP,
        quant=None,
        eager=True,
        async_scheduling=False,
        prefix_caching=True,
    )


def run_agent_bench(
    out_dir: Path,
    dataset_path: Optional[Path] = None,
    n_samples: int = N_SAMPLES,
    n_warmups: int = N_WARMUPS,
    skip_scenarios: bool = False,
    skip_matrix: bool = False,
    resume: bool = False,
    tp: Optional[int] = None,
) -> AgentBenchResult:
    """Run the full agent benchmark and save results to *out_dir*.

    **Part 1 — TTFT matrix (concurrency=1)**
    Measures all (N_cached × N_new) cells defined in constants.MATRIX_N_CACHED /
    MATRIX_N_NEW.  Each cell fires N_WARMUPS discarded warm-up requests then
    N_SAMPLES measured streaming requests, serial (one in-flight at a time).
    Corpus: FRAMES Wikipedia articles (or *dataset_path* override).
    Checkpoint: agent_matrix.json — supports ``--resume``.

    **Part 2 — Real ReAct agent scenarios (concurrency=1)**
    Runs a genuine ReAct loop on FRAMES multi-hop research questions.  The LLM
    generates JSON search/answer actions; we retrieve the best matching Wikipedia
    article from the gold set and accumulate context across turns.  TTFT is
    measured at each LLM call.

    Args:
        out_dir:        Directory for all output files.
        dataset_path:   Override corpus JSONL for matrix (default: FRAMES wiki corpus).
        n_samples:      Measured samples per matrix cell.
        n_warmups:      Discarded warm-up requests per cell.
        skip_scenarios: Skip Part 2 (matrix only).
        skip_matrix:    Skip Part 1 (scenarios only).
        resume:         Skip already-completed matrix cells (reads agent_matrix.json).
    """
    from datetime import datetime

    try:
        from zoneinfo import ZoneInfo
        ts = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        from datetime import timezone, timedelta
        ts = datetime.now(timezone(timedelta(hours=2))).strftime("%Y-%m-%d %H:%M")

    _tp = tp if tp is not None else AGENT_TP

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "datasets").mkdir(exist_ok=True)

    _setup_debug_log(out_dir / "agent_debug.log")
    _DBG_INFO(
        f"[run_agent_bench] START  ts={ts}  model={AGENT_MODEL}  "
        f"tp={_tp}  max_model_len={AGENT_MAX_MODEL_LEN}  PC=True  "
        f"n_samples={n_samples}  n_warmups={n_warmups}  "
        f"skip_matrix={skip_matrix}  skip_scenarios={skip_scenarios}  "
        f"resume={resume}  out_dir={out_dir}"
    )

    print(f"\n[agent_bench] Starting — {ts}", flush=True)
    print(
        f"  Model: {AGENT_MODEL}  tp={_tp}  "
        f"max_model_len={AGENT_MAX_MODEL_LEN:,}  PC=True  concurrency=1",
        flush=True,
    )
    print(
        f"  Matrix: {len(MATRIX_N_CACHED)} × {len(MATRIX_N_NEW)} cells "
        f"× {n_samples} samples  "
        f"(N_cached up to {max(MATRIX_N_CACHED)//1024}k, "
        f"N_new up to {max(MATRIX_N_NEW)//1024}k)",
        flush=True,
    )
    print(f"  Debug log: {out_dir / 'agent_debug.log'}", flush=True)

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------
    corpus: Optional[Corpus] = None
    if not skip_matrix:
        _DBG_INFO("[run_agent_bench] Phase: corpus preparation")
        if dataset_path is None:
            dataset_path = _prepare_frames_corpus(out_dir)
        if dataset_path is None:
            dataset_path = _find_arxiv_fallback()
        _DBG_INFO(f"[run_agent_bench] Corpus path: {dataset_path}")
        print(f"  Corpus: {dataset_path}", flush=True)

        session_corpus = make_session()
        print("\n[agent_bench] Building corpus (tokenising once, ~30s)...", flush=True)
        corpus = Corpus(dataset_path, session_corpus)
        _DBG_INFO(f"[run_agent_bench] Corpus built: {corpus.total_tokens} tokens available")

    result = AgentBenchResult(
        run_timestamp=ts,
        model=AGENT_MODEL,
        tp=_tp,
        prefix_caching=True,
    )

    # ------------------------------------------------------------------
    # Part 1: TTFT matrix
    # ------------------------------------------------------------------
    if not skip_matrix and corpus is not None:
        _DBG_INFO("[run_agent_bench] Phase 1/2: TTFT matrix")
        print("\n[agent_bench] Part 1/2: TTFT matrix measurement", flush=True)
        matrix = run_ttft_matrix(
            corpus, out_dir, n_warmups, n_samples, resume=resume
        )
        result.matrix = matrix
        print_ttft_table(matrix)
        _DBG_INFO(f"[run_agent_bench] Phase 1 done: {len(matrix)} cells measured")

    # ------------------------------------------------------------------
    # Part 2: real ReAct scenarios
    # ------------------------------------------------------------------
    if not skip_scenarios:
        _DBG_INFO("[run_agent_bench] Phase 2/2: real ReAct scenarios (FRAMES)")
        print(
            "\n[agent_bench] Part 2/2: Real deep-research agent scenarios (FRAMES)",
            flush=True,
        )
        frames_questions = _load_frames_questions(N_AGENT_SCENARIOS)
        if frames_questions:
            _DBG_INFO(f"[run_agent_bench] Loaded {len(frames_questions)} FRAMES questions")
            scenarios = run_agent_scenarios_frames(frames_questions, out_dir)
        else:
            _DBG_WARN("[run_agent_bench] FRAMES unavailable — skipping scenarios")
            print("  WARNING: FRAMES unavailable — skipping scenarios.", flush=True)
            scenarios = []
        result.scenarios = scenarios
        if scenarios:
            print_scenario_summary(scenarios)
            _DBG_INFO(f"[run_agent_bench] Phase 2 done: {len(scenarios)} scenarios")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    json_path = out_dir / "agent_bench_results.json"
    json_path.write_text(json.dumps(_result_to_dict(result), indent=2))
    print(f"\n[agent_bench] Results saved → {json_path}", flush=True)
    return result


def _result_to_dict(r: AgentBenchResult) -> dict:
    return {
        "run_timestamp":  r.run_timestamp,
        "model":          r.model,
        "tp":             r.tp,
        "prefix_caching": r.prefix_caching,
        "matrix":         [asdict(c) for c in r.matrix],
        "scenarios":      [asdict(s) for s in r.scenarios],
    }
