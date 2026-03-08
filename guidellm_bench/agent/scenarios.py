"""Real ReAct agent scenarios using the FRAMES benchmark.

Each scenario is a genuine multi-turn research loop:
  - LLM generates JSON search/answer actions (concurrency=1, serial).
  - Search: retrieve best-matching Wikipedia article from the gold set.
  - Answer: record final result and stop.
  - TTFT measured at each LLM call with KV-cache warming.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Optional

import requests

from .constants import (
    AGENT_DATASET, N_AGENT_SCENARIOS, AGENT_SYSTEM_PROMPT,
    OUTPUT_TOKENS_SCENARIO, INTER_REQUEST_SLEEP_S,
    AGENT_MAX_MODEL_LEN, CONCURRENCY, ScenarioResult,
)
from .debug import _DBG, _DBG_INFO, _DBG_WARN
from .helpers import make_session, _warm_cache, _measure_ttft
from .matrix import _cold_ttft_estimate

assert CONCURRENCY == 1, "scenarios assume concurrency=1"

__all__ = [
    "_load_frames_questions",
    "run_research_session",
    "run_agent_scenarios_frames",
    "print_scenario_summary",
]


# ---------------------------------------------------------------------------
# FRAMES dataset loading
# ---------------------------------------------------------------------------


def _load_frames_questions(n_select: int = N_AGENT_SCENARIOS) -> list[dict]:
    """Download FRAMES benchmark and select *n_select* diverse questions.

    Each returned dict has:
      question (str), wiki_docs (list[str]), n_docs (int),
      total_chars (int), answer (str), reasoning_types (str).

    Selection strategy: sort all valid FRAMES questions by total wiki content
    length and pick at evenly-spaced percentiles so we cover a range from
    short (2-3 docs) to long (10-15 docs) sessions.
    """
    try:
        from datasets import load_dataset  # type: ignore

        print(f"  Loading {AGENT_DATASET} ...", flush=True)
        ds = load_dataset(AGENT_DATASET, split="test")
    except Exception as exc:
        print(f"  WARNING: FRAMES load failed ({exc}) — scenarios skipped.", flush=True)
        return []

    questions: list[dict] = []
    for row in ds:
        prompt = (
            row.get("Prompt")
            or row.get("prompt")
            or row.get("question")
            or row.get("task")
            or ""
        )
        raw_wiki = (
            row.get("wiki_doc")
            or row.get("wiki_docs")
            or row.get("documents")
            or ""
        )
        if isinstance(raw_wiki, str):
            parts = [p.strip() for p in raw_wiki.split("\n\n\n") if p.strip()]
            wiki_docs = parts if parts else ([raw_wiki] if raw_wiki.strip() else [])
        elif isinstance(raw_wiki, list):
            wiki_docs = [str(d).strip() for d in raw_wiki if d]
        else:
            wiki_docs = []

        if not prompt or not wiki_docs:
            continue

        total_chars = sum(len(d) for d in wiki_docs)
        questions.append(
            {
                "question":       prompt,
                "wiki_docs":      wiki_docs,
                "n_docs":         len(wiki_docs),
                "total_chars":    total_chars,
                "answer":         str(row.get("Answer") or row.get("answer") or ""),
                "reasoning_types": str(row.get("reasoning_types") or ""),
            }
        )

    if not questions:
        print("  WARNING: No valid FRAMES questions found.", flush=True)
        return []

    questions.sort(key=lambda q: q["total_chars"])
    n = len(questions)
    percentiles = [i / (n_select + 1) for i in range(1, n_select + 1)]
    selected = [questions[int(p * n)] for p in percentiles]

    print(f"  FRAMES: {n} valid questions → selected {n_select}:", flush=True)
    for i, q in enumerate(selected):
        print(
            f"    [{i+1}] {q['n_docs']} docs, {q['total_chars']//1000}k chars | "
            f"{q['question'][:80]}...",
            flush=True,
        )
    return selected


# ---------------------------------------------------------------------------
# ReAct helpers
# ---------------------------------------------------------------------------


def _parse_json_action(raw: str) -> dict:
    """Extract the first valid JSON object from *raw* LLM output.

    Falls back to {"action": "search", "query": raw[:100]} on parse failure
    so the agent loop always progresses.
    """
    raw = raw.strip()
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    import json as _json
                    return _json.loads(raw[start : i + 1])
                except Exception:
                    start = -1
    return {"action": "search", "query": raw[:120].replace("\n", " ")}


def _keyword_recall(query: str, doc: str) -> float:
    """Fraction of query words present in *doc* (simple recall score, no deps)."""
    q_words = set(query.lower().split())
    if not q_words:
        return 0.0
    d_words = set(doc.lower().split())
    return len(q_words & d_words) / len(q_words)


def _find_best_doc(
    query: str,
    docs: list[str],
    used: set[int],
) -> tuple[int, str]:
    """Return (index, text) of the highest-scoring *unused* document for *query*.

    Scores using keyword recall over the first 2000 chars (title + lede).
    Falls back to first unused doc if all scores are zero.
    """
    best_i, best_score = -1, -1.0
    for i, doc in enumerate(docs):
        if i in used:
            continue
        score = _keyword_recall(query, doc[:2_000])
        if score > best_score:
            best_score, best_i = score, i
    if best_i == -1:
        for i in range(len(docs)):
            if i not in used:
                return i, docs[i]
        return 0, docs[0] if docs else ""
    return best_i, docs[best_i]


# ---------------------------------------------------------------------------
# Real ReAct research loop (concurrency=1)
# ---------------------------------------------------------------------------


def run_research_session(
    session: requests.Session,
    scenario_name: str,
    question: str,
    wiki_docs: list[str],
    max_iterations: int = 20,
) -> ScenarioResult:
    """Run a real ReAct research loop on *question* using *wiki_docs* as the search index.

    All LLM calls are serial (CONCURRENCY=1).  Each iteration:
      1. Build prompt = system_prompt + question + conversation so far + "Next action:"
      2. Warm KV cache with the stable prefix (conversation up to "Next action:").
      3. Single streaming call — capture TTFT + full generated text in one pass.
      4. Parse JSON action: search → retrieve doc; answer → stop.
      5. Track per-iteration metrics.
    """
    sr = ScenarioResult(
        name=scenario_name,
        description=f"FRAMES: {question[:100]}",
        n_calls=len(wiki_docs),
    )

    base_prefix = (
        f"{AGENT_SYSTEM_PROMPT}\n\n"
        f"---\nResearch Question: {question}\n\n---\n"
    )
    conversation_body = ""
    used_docs: set[int] = set()
    ttfts: list[float] = []
    prev_prompt_tokens = 0

    for iteration in range(max_iterations):
        prefix_for_cache = base_prefix + conversation_body
        full_prompt = prefix_for_cache + "Next action: "

        _DBG(
            f"run_research_session [{scenario_name}] iter={iteration}  "
            f"prefix_chars={len(prefix_for_cache):,}  prev_tokens={prev_prompt_tokens}"
        )

        wc_ms = _warm_cache(session, prefix_for_cache, output_tokens=1)
        time.sleep(0.3)

        t0_req = time.perf_counter()
        ttft_ms, actual_ptok, response_text, _total_ms_inner = _measure_ttft(
            session, full_prompt, max_tokens=OUTPUT_TOKENS_SCENARIO, return_text=True
        )
        total_req_ms = (time.perf_counter() - t0_req) * 1000.0
        ttfts.append(ttft_ms)
        time.sleep(INTER_REQUEST_SLEEP_S)

        _DBG(
            f"run_research_session [{scenario_name}] iter={iteration}  "
            f"TTFT={ttft_ms:.0f}ms  total_req={total_req_ms:.0f}ms  "
            f"warm_cache={wc_ms:.0f}ms  actual_ptok={actual_ptok}"
        )
        _DBG(
            f"run_research_session [{scenario_name}] iter={iteration}  "
            f"raw_output={response_text!r}"
        )

        n_new_tokens = max(0, actual_ptok - prev_prompt_tokens)
        cold_est = _cold_ttft_estimate(n_new_tokens, prev_prompt_tokens)

        action = _parse_json_action(response_text)
        action_type = action.get("action", "search")
        search_query = action.get("query", "") if action_type == "search" else ""

        iter_info = {
            "iteration":        iteration,
            "n_cached_tokens":  prev_prompt_tokens,
            "n_new_tokens":     n_new_tokens,
            "n_total_tokens":   actual_ptok,
            "n_output_tokens":  OUTPUT_TOKENS_SCENARIO,
            "ttft_ms":          round(ttft_ms, 1),
            "total_request_ms": round(total_req_ms, 1),
            "warm_cache_ms":    round(wc_ms, 1),
            "cold_ttft_est_ms": round(cold_est, 1),
            "speedup_vs_cold":  round(cold_est / max(ttft_ms, 1.0), 2),
            "action":           action_type,
            "search_query":     search_query,
            "raw_response_text": response_text,
        }
        sr.iters.append(iter_info)
        print(
            f"    iter {iteration:2d}: cached={prev_prompt_tokens//1024:4d}k "
            f"new={n_new_tokens//1024:3d}k  ttft={ttft_ms:.0f}ms  "
            f"cold_est={cold_est:.0f}ms  speedup={cold_est/max(ttft_ms,1):.1f}×  "
            f"action={action_type}"
            + (f"  q=[{search_query[:50]}]" if search_query else ""),
            flush=True,
        )

        prev_prompt_tokens = actual_ptok

        if action_type == "answer":
            print(
                f"    → Agent answered after {iteration+1} iteration(s).", flush=True
            )
            _DBG_INFO(
                f"[run_research_session] ANSWERED scenario={scenario_name!r} "
                f"after {iteration+1} iters  final_tokens={actual_ptok}  "
                f"TTFT_last={ttft_ms:.0f}ms"
            )
            break

        doc_idx, doc_text = _find_best_doc(
            search_query or question, wiki_docs, used_docs
        )
        used_docs.add(doc_idx)
        doc_snip = doc_text[:3_000]
        conversation_body += (
            f"[Search: {search_query[:120]}]\n"
            f"[Result ({doc_idx+1}/{len(wiki_docs)})]: {doc_snip}\n\n"
        )

        # Safety: stop if approaching the context window
        if actual_ptok + OUTPUT_TOKENS_SCENARIO + 500 > AGENT_MAX_MODEL_LEN - 4_096:
            _DBG_WARN(
                f"run_research_session [{scenario_name}] iter={iteration}: "
                f"context ({actual_ptok:,} tok) nearing window limit — stopping"
            )
            print(
                f"    Iteration {iteration}: context ({actual_ptok:,} tok) "
                f"nearing window limit — stopping.",
                flush=True,
            )
            break

        if len(used_docs) >= len(wiki_docs):
            conversation_body += (
                "[All available documents retrieved. Please provide your final answer.]\n\n"
            )

    if ttfts:
        sr.ttft_median_first = round(ttfts[0], 1)
        sr.ttft_median_last  = round(ttfts[-1], 1)
        sr.ttft_median_all   = round(statistics.median(ttfts), 1)
        sr.total_context_k   = round(prev_prompt_tokens / 1024, 1)

    _DBG_INFO(
        f"[run_research_session] DONE scenario={scenario_name!r}  "
        f"n_iters={len(sr.iters)}  "
        f"ttft_first={sr.ttft_median_first}ms  ttft_last={sr.ttft_median_last}ms  "
        f"total_context={sr.total_context_k}k"
    )
    return sr


def run_agent_scenarios_frames(
    questions: list[dict],
    out_dir: Optional[Path] = None,
) -> list[ScenarioResult]:
    """Run the real ReAct agent on each FRAMES question, return ScenarioResults.

    All calls are serial (CONCURRENCY=1).
    """
    session = make_session()
    results: list[ScenarioResult] = []

    for i, q in enumerate(questions):
        name = f"frames_{i+1}"
        print(
            f"\n  === Scenario '{name}' ({q['n_docs']} docs, "
            f"{q['total_chars']//1000}k chars) ===",
            flush=True,
        )
        print(f"  Q: {q['question'][:120]}...", flush=True)
        sr = run_research_session(
            session,
            scenario_name=name,
            question=q["question"],
            wiki_docs=q["wiki_docs"],
        )
        results.append(sr)

    return results


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def print_scenario_summary(scenarios: list[ScenarioResult]) -> None:
    print()
    print("=" * 80)
    print("AGENT SCENARIO SUMMARY")
    print("=" * 80)
    print(
        f"  {'Scenario':<18} {'Calls':>6} {'Total ctx':>12} "
        f"{'TTFT iter0':>12} {'TTFT last':>12} {'Median all':>12}"
    )
    print("  " + "-" * 80)
    for sr in scenarios:
        print(
            f"  {sr.name:<18} {sr.n_calls:>6} {sr.total_context_k:>10.0f}k "
            f"{sr.ttft_median_first:>11.0f}ms "
            f"{sr.ttft_median_last:>11.0f}ms "
            f"{sr.ttft_median_all:>11.0f}ms"
        )
    print()
    for sr in scenarios:
        if not sr.iters:
            continue
        print(f"  {sr.name} — per-iteration TTFT:")
        print(
            f"    {'iter':>4}  {'cached_k':>8}  {'new_k':>6}  "
            f"{'ttft_ms':>8}  {'cold_est':>9}  {'speedup':>7}  action"
        )
        for it in sr.iters:
            action_lbl = it.get("action", "")[:6]
            q_snip = (
                f"  [{it.get('search_query','')[:40]}]"
                if action_lbl == "search"
                else ""
            )
            print(
                f"    {it['iteration']:4d}  {it['n_cached_tokens']//1024:8d}k  "
                f"{it['n_new_tokens']//1024:6d}k  "
                f"{it['ttft_ms']:8.0f}  "
                f"{it['cold_ttft_est_ms']:9.0f}  "
                f"{it['speedup_vs_cold']:7.1f}×  {action_lbl}{q_snip}"
            )
        print()
