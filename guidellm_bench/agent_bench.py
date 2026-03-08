"""Deep-research agent benchmark.

Measures TTFT vs (N_new, N_cached) matrix and simulates realistic multi-turn
agent sessions, using vLLM's prefix caching to model accumulated context growth.

Key design:
  - Uses /v1/completions (not chat) to avoid chat-template token injection.
  - Token-exact prompt construction via vLLM /tokenize + /detokenize endpoints.
  - Cache warming: send prefix with max_tokens=1 before each measured request.
  - Streaming TTFT: time.perf_counter() from request send to first text token.
  - N_WARMUPS=3 requests discarded before each cell; N_SAMPLES=15.
  - Reliability guard: re-runs cell if CV (stddev/median) > 0.35.

Runs inside lsv-container (intel/llm-scaler-vllm:0.14.0-b8).
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

from .config import Config, PORT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_MODEL          = "openai/gpt-oss-20b"
AGENT_TP             = 8
AGENT_MAX_MODEL_LEN  = 131_072          # model's physical context window
AGENT_MAX_BATCHED    = 131_072          # single-pass prefill — must match max_model_len
N_WARMUPS            = 3               # warm-up requests discarded before each cell
N_SAMPLES            = 15              # measured samples per cell
CV_RERUN_THRESHOLD   = 0.35            # re-run cell if stddev/median > this
OUTPUT_TOKENS_DEFAULT  = 256           # generation length during matrix cell measurements
OUTPUT_TOKENS_SCENARIO = 200           # max tokens per agent turn (JSON action is short)
INTER_REQUEST_SLEEP_S  = 0.5           # avoid thermal-throttle artefacts

# TTFT matrix axes (in tokens).
# max N_cached + max N_new = 114_688 + 16_384 = 131_072 = AGENT_MAX_MODEL_LEN (exact fit)
MATRIX_N_CACHED = [0, 8_192, 32_768, 65_536, 98_304, 114_688]  # 0/8k/32k/64k/96k/112k
MATRIX_N_NEW    = [1_024, 4_096, 8_192, 16_384]                 # 1k/4k/8k/16k

# Real deep-research agent scenarios use FRAMES (multi-hop questions + Wikipedia articles)
AGENT_DATASET = "google/frames-benchmark"  # 824 questions, each with 2-15 gold Wikipedia docs
N_AGENT_SCENARIOS = 4   # how many FRAMES questions to run as agent scenarios

# System prompt for the ReAct agent loop (instructs JSON-only output)
AGENT_SYSTEM_PROMPT = (
    "You are a deep research assistant. Answer complex multi-hop research questions "
    "by searching for information iteratively.\n"
    "For each response, output EXACTLY ONE JSON object (no surrounding text):\n"
    '  To retrieve a document: {"action": "search", "query": "your specific search query"}\n'
    '  To give your final answer: {"action": "answer", "text": "your comprehensive answer"}\n'
    "Issue multiple searches before answering. Be specific in your search queries."
)

_BASE_URL = f"http://localhost:{PORT}"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    n_cached:      int
    n_new:         int
    actual_cached: int           # verified from server
    actual_new:    int           # verified from server
    n_samples:     int
    ttft_ms_values: list[float]  = field(default_factory=list)
    ttft_median:   float         = 0.0
    ttft_p25:      float         = 0.0
    ttft_p75:      float         = 0.0
    ttft_p95:      float         = 0.0
    ttft_min:      float         = 0.0
    ttft_max:      float         = 0.0
    ttft_cv:       float         = 0.0   # stddev / median
    cache_hit_ratio: float       = 0.0   # fraction of samples with TTFT < cold baseline
    cold_ttft_estimate: float    = 0.0   # expected TTFT without PC


@dataclass
class ScenarioResult:
    name:        str
    description: str
    n_calls:     int
    iters: list[dict]   = field(default_factory=list)   # per-iteration metrics
    # Summary
    ttft_median_first:  float = 0.0   # TTFT at iteration 0 (cold-ish)
    ttft_median_last:   float = 0.0   # TTFT at final iteration
    ttft_median_all:    float = 0.0
    total_context_k:    float = 0.0   # accumulated context at final iteration


@dataclass
class AgentBenchResult:
    run_timestamp:   str
    model:           str
    tp:              int
    prefix_caching:  bool
    matrix:          list[CellResult]  = field(default_factory=list)
    scenarios:       list[ScenarioResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level vLLM API helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False   # bypass http_proxy for localhost
    return s


def _tokenize(session: requests.Session, text: str) -> list[int]:
    """Return the token ID list for *text* using vLLM /tokenize."""
    resp = session.post(
        f"{_BASE_URL}/tokenize",
        json={"model": AGENT_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    # vLLM returns {"tokens": [...]} (list of ints)
    return payload["tokens"]


def _detokenize(session: requests.Session, tokens: list[int]) -> str:
    """Return the decoded text for a list of token IDs via vLLM /detokenize."""
    resp = session.post(
        f"{_BASE_URL}/detokenize",
        json={"model": AGENT_MODEL, "tokens": tokens},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["prompt"]


def _verify_token_count(session: requests.Session, text: str) -> int:
    """Return actual token count for *text* (non-streaming completions, max_tokens=1)."""
    resp = session.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": AGENT_MODEL,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["usage"]["prompt_tokens"]


def _warm_cache(session: requests.Session, prompt: str, output_tokens: int = 1) -> None:
    """Send a short non-streaming request to prime the KV cache for *prompt*."""
    session.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": AGENT_MODEL,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=600,
    )


def _measure_ttft(
    session: requests.Session,
    prompt: str,
    max_tokens: int = OUTPUT_TOKENS_DEFAULT,
    return_text: bool = False,
) -> tuple:
    """Measure TTFT (ms) for a streaming /v1/completions request.

    Returns (ttft_ms, prompt_tokens_actual) when return_text=False.
    Returns (ttft_ms, prompt_tokens_actual, full_text) when return_text=True.
    Captures TTFT, prompt token count, and optionally the full generated text
    — all in a single streaming pass.
    """
    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    prompt_tokens = 0
    full_text = ""

    with session.post(
        f"{_BASE_URL}/v1/completions",
        json={
            "model": AGENT_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line: str = raw_line if isinstance(raw_line, str) else raw_line.decode()
            if not line.startswith("data:"):
                continue
            payload_str = line[5:].strip()
            if payload_str == "[DONE]":
                break
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            # Capture TTFT from the very first chunk that has non-empty text
            if ttft_ms is None and choices and choices[0].get("text"):
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            # Accumulate full generated text (for action parsing in agent loop)
            if return_text and choices:
                full_text += choices[0].get("text", "")
            # Capture prompt_tokens from usage chunk
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

    if ttft_ms is None:
        ttft_ms = (time.perf_counter() - t0) * 1000.0   # fallback: full generation time
    if return_text:
        return ttft_ms, prompt_tokens, full_text
    return ttft_ms, prompt_tokens


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------

class Corpus:
    """Lazy-loaded token array from arxiv-summarization, sliceable to exact N tokens."""

    def __init__(self, dataset_path: Path, session: requests.Session, max_chars: int = 900_000):
        self._session = session
        raw_texts = []
        # Concatenate articles until we have enough chars for the largest cell
        for line in dataset_path.read_text().splitlines():
            row = json.loads(line)
            text = row.get("prompt") or row.get("text") or ""
            if text:
                raw_texts.append(text.strip())
            if sum(len(t) for t in raw_texts) >= max_chars:
                break

        self._full_text = "\n\n---\n\n".join(raw_texts)
        print(f"  Corpus: {len(raw_texts)} docs, {len(self._full_text):,} chars", flush=True)
        print("  Tokenising corpus (one-time, ~30s)...", flush=True)
        self._tokens: list[int] = _tokenize(session, self._full_text[:max_chars])
        print(f"  Corpus tokenised: {len(self._tokens):,} tokens", flush=True)
        if len(self._tokens) < max(MATRIX_N_CACHED) + max(MATRIX_N_NEW):
            needed = max(MATRIX_N_CACHED) + max(MATRIX_N_NEW)
            raise RuntimeError(
                f"Corpus too small: {len(self._tokens):,} tokens, need {needed:,}. "
                "Expand dataset_path or increase max_chars."
            )

    def slice_text(self, start_token: int, end_token: int) -> str:
        """Return the decoded text for tokens[start_token:end_token]."""
        sub = self._tokens[start_token:end_token]
        return _detokenize(self._session, sub)

    def n_tokens(self) -> int:
        return len(self._tokens)


# ---------------------------------------------------------------------------
# TTFT matrix measurement
# ---------------------------------------------------------------------------

def _cold_ttft_estimate(n_new: int, n_cached: int) -> float:
    """Rough cold-prefill TTFT at N_total = n_new + n_cached tokens.

    From fitted model (no-PC): base=38.3 + alpha=62.0*N_k + beta=0.731*N_k^2
    """
    n_k = (n_new + n_cached) / 1000.0
    return 38.3 + 62.0 * n_k + 0.731 * n_k ** 2


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

    Protocol:
      1. Warm cache with cached_prompt (max_tokens=1).
      2. N_warmups streaming requests with full prompt — discard results.
      3. Between each warm-up: re-warm cache (it may have been evicted by generation).
      4. N_samples streaming requests — record TTFT.
      5. If CV > threshold, run one extra pass and merge.
    """
    full_prompt = cached_prompt + new_prompt
    cold_est = _cold_ttft_estimate(n_new_target, n_cached_target)

    if verbose:
        prefix = f"  Cell N_cached={n_cached_target//1024}k N_new={n_new_target//1024}k"
        print(f"{prefix}: warming cache...", flush=True)

    # Verify actual token counts on first non-streaming call
    actual_total = _verify_token_count(session, full_prompt)
    actual_cached = _verify_token_count(session, cached_prompt) if n_cached_target > 0 else 0
    actual_new = actual_total - actual_cached

    if verbose:
        print(f"{prefix}: actual tokens: cached={actual_cached}, new={actual_new}, total={actual_total}", flush=True)

    def _one_measurement() -> float:
        # Re-warm the prefix cache before each measurement
        if n_cached_target > 0:
            _warm_cache(session, cached_prompt, output_tokens=1)
            time.sleep(0.3)
        ttft, _ = _measure_ttft(session, full_prompt, max_tokens=max_output_tokens)
        time.sleep(INTER_REQUEST_SLEEP_S)
        return ttft

    # Warm-up passes (discard)
    for i in range(n_warmups):
        if verbose:
            print(f"{prefix}: warm-up {i+1}/{n_warmups}...", flush=True)
        _one_measurement()

    # Measured passes
    values: list[float] = []
    for i in range(n_samples):
        ttft = _one_measurement()
        values.append(ttft)
        if verbose:
            print(f"{prefix}: sample {i+1:2d}/{n_samples} → {ttft:.0f} ms", flush=True)

    # Reliability check: re-run if CV is too high
    median_v = statistics.median(values)
    stddev_v = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = stddev_v / median_v if median_v > 0 else 0.0
    if cv > CV_RERUN_THRESHOLD:
        print(f"{prefix}: ⚠ CV={cv:.2f} > {CV_RERUN_THRESHOLD} — running extra pass", flush=True)
        extra = [_one_measurement() for _ in range(n_samples)]
        values.extend(extra)
        median_v = statistics.median(values)
        stddev_v = statistics.stdev(values) if len(values) > 1 else 0.0
        cv = stddev_v / median_v if median_v > 0 else 0.0

    sorted_v = sorted(values)
    n = len(sorted_v)

    # Cache hit ratio: fraction of samples below 50% of cold estimate
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
    )
    print(
        f"{prefix}: ✓ median={result.ttft_median:.0f}ms  "
        f"p95={result.ttft_p95:.0f}ms  CV={cv:.2f}  "
        f"cache_hit={hit_ratio:.0%}",
        flush=True,
    )
    return result


def run_ttft_matrix(
    corpus: Corpus,
    out_dir: Path,
    n_warmups: int = N_WARMUPS,
    n_samples: int = N_SAMPLES,
    max_output_tokens: int = OUTPUT_TOKENS_DEFAULT,
    resume: bool = False,
) -> list[CellResult]:
    """Measure all (N_cached × N_new) cells. Returns list of CellResult."""
    session = _make_session()
    results: list[CellResult] = []
    cache_file = out_dir / "agent_matrix.json"

    # Load previously completed cells when resuming
    done_cells: set[tuple[int, int]] = set()
    if resume and cache_file.exists():
        existing = json.loads(cache_file.read_text())
        for cr in existing.get("matrix", []):
            results.append(CellResult(**{k: v for k, v in cr.items()}))
            done_cells.add((cr["n_cached"], cr["n_new"]))
        print(f"  Resumed {len(done_cells)} already-completed cells.", flush=True)

    total_cells = len(MATRIX_N_CACHED) * len(MATRIX_N_NEW)
    cell_idx = 0

    for n_cached in MATRIX_N_CACHED:
        # Build cached prefix once per N_cached (shared across N_new columns)
        cached_prompt = corpus.slice_text(0, n_cached) if n_cached > 0 else ""

        for n_new in MATRIX_N_NEW:
            cell_idx += 1
            if (n_cached, n_new) in done_cells:
                print(f"  [{cell_idx}/{total_cells}] Skipping N_cached={n_cached//1024}k N_new={n_new//1024}k (already done)", flush=True)
                continue

            # New tokens start after the cached block — ensures independence
            new_prompt = corpus.slice_text(n_cached, n_cached + n_new)

            print(f"\n  [{cell_idx}/{total_cells}] N_cached={n_cached//1024}k  N_new={n_new//1024}k", flush=True)
            cr = measure_cell(
                session, cached_prompt, new_prompt,
                n_cached, n_new, n_warmups, n_samples, max_output_tokens,
            )
            results.append(cr)
            # Save checkpoint after each cell
            _save_matrix_checkpoint(results, cache_file)

    return results


def _save_matrix_checkpoint(results: list[CellResult], path: Path) -> None:
    data = {"matrix": [asdict(r) for r in results]}
    path.write_text(json.dumps(data, indent=2))



# ---------------------------------------------------------------------------
# Report: print TTFT table + save JSON
# ---------------------------------------------------------------------------

def print_ttft_table(matrix: list[CellResult]) -> None:
    """Print the (N_cached × N_new) TTFT median table to stdout."""
    # Index by (n_cached, n_new)
    lookup: dict[tuple[int,int], CellResult] = {(c.n_cached, c.n_new): c for c in matrix}

    col_w = 16
    header_cells = [f"  {n//1024}k new toks" for n in MATRIX_N_NEW]
    print()
    print("=" * 80)
    print("TTFT MATRIX — gpt-oss-20b tp8+async+PC (measured)")
    print(f"  Columns = new tokens per iteration | Rows = accumulated KV cache context")
    print(f"  Values = median TTFT  [p25–p75 range]  (n={N_SAMPLES} samples/cell)")
    print("=" * 80)
    print(f"  {'Cached context':>20} |" + "|".join(f"{h:>{col_w}}" for h in header_cells) + " |")
    print("  " + "-" * 22 + ("+" + "-"*col_w) * len(MATRIX_N_NEW))

    for n_cached in MATRIX_N_CACHED:
        row_label = f"{n_cached//1024}k accumulated" if n_cached > 0 else "0k (cold)"
        row = f"  {row_label:>20} |"
        for n_new in MATRIX_N_NEW:
            cr = lookup.get((n_cached, n_new))
            if cr is None:
                row += f"{'  --':>{col_w}} |"
                continue
            med = cr.ttft_median
            p25 = cr.ttft_p25
            p75 = cr.ttft_p75
            cv  = cr.ttft_cv
            flag = "⚠" if cv > 0.25 else " "
            if med < 1000:
                cell = f"{med:.0f}ms {flag}"
            else:
                cell = f"{med/1000:.2f}s {flag}"
            row += f"{cell:>{col_w}} |"
        print(row)

    print()
    print("  ⚠ = CV > 0.25 (high variance; interpret with caution)")
    print()


def print_scenario_summary(scenarios: list[ScenarioResult]) -> None:
    print()
    print("=" * 80)
    print("AGENT SCENARIO SUMMARY")
    print("=" * 80)
    print(f"  {'Scenario':<18} {'Calls':>6} {'Total ctx':>12} {'TTFT iter0':>12} {'TTFT last':>12} {'Median all':>12}")
    print("  " + "-"*80)
    for sr in scenarios:
        print(
            f"  {sr.name:<18} {sr.n_calls:>6} {sr.total_context_k:>10.0f}k "
            f"{sr.ttft_median_first:>11.0f}ms "
            f"{sr.ttft_median_last:>11.0f}ms "
            f"{sr.ttft_median_all:>11.0f}ms"
        )
    print()
    # Per-scenario iteration detail
    for sr in scenarios:
        if not sr.iters:
            continue
        print(f"  {sr.name} — per-iteration TTFT:")
        print(f"    {'iter':>4}  {'cached_k':>8}  {'new_k':>6}  {'ttft_ms':>8}  {'cold_est':>9}  {'speedup':>7}  action")
        for it in sr.iters:
            action_lbl = it.get("action", "")[:6]
            q_snip = f"  [{it.get('search_query','')[:40]}]" if action_lbl == "search" else ""
            print(
                f"    {it['iteration']:4d}  {it['n_cached_tokens']//1024:8d}k  "
                f"{it['n_new_tokens']//1024:6d}k  "
                f"{it['ttft_ms']:8.0f}  "
                f"{it['cold_ttft_est_ms']:9.0f}  "
                f"{it['speedup_vs_cold']:7.1f}\u00d7  {action_lbl}{q_snip}"
            )
        print()


# ---------------------------------------------------------------------------
# FRAMES dataset helpers \u2014 real deep-research agent queries + Wikipedia articles
# ---------------------------------------------------------------------------

def _load_frames_questions(n_select: int = N_AGENT_SCENARIOS) -> list[dict]:
    """Download FRAMES benchmark and select *n_select* diverse questions.

    Each returned dict has:
      question (str), wiki_docs (list[str]), n_docs (int),
      total_chars (int), answer (str), reasoning_types (str).

    Selection strategy: sort all valid FRAMES questions by total wiki content
    length and pick at evenly-spaced percentiles so we get a range from
    short (2-3 docs) to long (10-15 docs) research sessions.
    """
    try:
        from datasets import load_dataset  # type: ignore
        print(f"  Loading {AGENT_DATASET} ...", flush=True)
        ds = load_dataset(AGENT_DATASET, split="test")
    except Exception as exc:
        print(f"  WARNING: FRAMES load failed ({exc}) \u2014 scenarios skipped.", flush=True)
        return []

    questions: list[dict] = []
    for row in ds:
        prompt = (row.get("Prompt") or row.get("prompt")
                  or row.get("question") or row.get("task") or "")
        # Wiki documents may be a single concatenated string or a list
        raw_wiki = (row.get("wiki_doc") or row.get("wiki_docs")
                    or row.get("documents") or "")
        if isinstance(raw_wiki, str):
            # Articles in FRAMES are usually separated by triple newlines
            parts = [p.strip() for p in raw_wiki.split("\n\n\n") if p.strip()]
            wiki_docs = parts if parts else ([raw_wiki] if raw_wiki.strip() else [])
        elif isinstance(raw_wiki, list):
            wiki_docs = [str(d).strip() for d in raw_wiki if d]
        else:
            wiki_docs = []

        if not prompt or not wiki_docs:
            continue

        total_chars = sum(len(d) for d in wiki_docs)
        questions.append({
            "question":       prompt,
            "wiki_docs":      wiki_docs,
            "n_docs":         len(wiki_docs),
            "total_chars":    total_chars,
            "answer":         str(row.get("Answer") or row.get("answer") or ""),
            "reasoning_types": str(row.get("reasoning_types") or ""),
        })

    if not questions:
        print("  WARNING: No valid FRAMES questions found.", flush=True)
        return []

    questions.sort(key=lambda q: q["total_chars"])
    n = len(questions)
    # Evenly-spaced percentiles — gives a representative difficulty spread
    percentiles = [i / (n_select + 1) for i in range(1, n_select + 1)]
    selected = [questions[int(p * n)] for p in percentiles]

    print(f"  FRAMES: {n} valid questions \u2192 selected {n_select}:", flush=True)
    for i, q in enumerate(selected):
        print(
            f"    [{i+1}] {q['n_docs']} docs, {q['total_chars']//1000}k chars | "
            f"{q['question'][:80]}...",
            flush=True,
        )
    return selected


def _prepare_frames_corpus(out_dir: Path) -> Optional[Path]:
    """Write a corpus JSONL from all FRAMES wiki_doc texts for matrix measurements.

    Returns path to the JSONL, or None on failure (caller falls back to arxiv).
    """
    corpus_path = out_dir / "datasets" / "frames_corpus_v1.jsonl"
    if corpus_path.exists():
        print(f"  FRAMES corpus: using cached {corpus_path}", flush=True)
        return corpus_path
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(AGENT_DATASET, split="test")
        n_docs = 0
        with open(corpus_path, "w") as f:
            for row in ds:
                raw_wiki = (row.get("wiki_doc") or row.get("wiki_docs")
                            or row.get("documents") or "")
                if isinstance(raw_wiki, str) and raw_wiki.strip():
                    json.dump({"prompt": raw_wiki.strip()}, f)
                    f.write("\n")
                    n_docs += 1
                elif isinstance(raw_wiki, list):
                    for d in raw_wiki:
                        if d and str(d).strip():
                            json.dump({"prompt": str(d).strip()}, f)
                            f.write("\n")
                            n_docs += 1
        print(f"  FRAMES corpus: {n_docs} documents \u2192 {corpus_path}", flush=True)
        return corpus_path
    except Exception as exc:
        print(f"  WARNING: FRAMES corpus build failed ({exc})", flush=True)
        return None


# ---------------------------------------------------------------------------
# Real ReAct agent loop
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
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    # Could not parse valid JSON \u2014 treat as implicit search
    return {"action": "search", "query": raw[:120].replace("\n", " ")}


def _keyword_recall(query: str, doc: str) -> float:
    """Fraction of query words present in *doc* (simple recall score, no deps)."""
    q_words = set(query.lower().split())
    if not q_words:
        return 0.0
    d_words = set(doc.lower().split())
    return len(q_words & d_words) / len(q_words)


def _find_best_doc(query: str, docs: list[str], used: set[int]) -> tuple[int, str]:
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
        # All docs used \u2014 return first unused (shouldn\u2019t happen in normal flow)
        for i in range(len(docs)):
            if i not in used:
                return i, docs[i]
        return 0, docs[0] if docs else ""
    return best_i, docs[best_i]


def run_research_session(
    session: requests.Session,
    scenario_name: str,
    question: str,
    wiki_docs: list[str],
    max_iterations: int = 20,
) -> ScenarioResult:
    """Run a real ReAct research loop on *question* using *wiki_docs* as the search index.

    Each iteration:
      1. Build prompt = system_prompt + question + conversation so far + "Next action:"
      2. Warm KV cache with the prefix (conversation up to "Next action:").
      3. Streaming LLM call \u2014 capture TTFT + full generated text in one pass.
      4. Parse JSON action from text:
           search  \u2192 retrieve best matching Wikipedia doc, append to conversation.
           answer  \u2192 stop.
      5. Track per-iteration metrics.
    """
    sr = ScenarioResult(
        name=scenario_name,
        description=f"FRAMES: {question[:100]}",
        n_calls=len(wiki_docs),  # upper-bound on expected iterations
    )

    # Stable prefix: never changes across iterations
    base_prefix = (
        f"{AGENT_SYSTEM_PROMPT}\n\n"
        f"---\nResearch Question: {question}\n\n---\n"
    )
    # conversation_body accumulates [Search / Result] turns
    conversation_body = ""
    used_docs: set[int] = set()
    ttfts: list[float] = []
    prev_prompt_tokens = 0   # tokens before this iteration\u2019s new content

    for iteration in range(max_iterations):
        # Prompt = base + turns so far + cue
        prefix_for_cache = base_prefix + conversation_body
        full_prompt = prefix_for_cache + "Next action: "

        # Prime KV cache with stable prefix before the measured call
        _warm_cache(session, prefix_for_cache, output_tokens=1)
        time.sleep(0.3)

        # Single streaming call: TTFT + full text
        ttft_ms, actual_ptok, response_text = _measure_ttft(
            session, full_prompt, max_tokens=OUTPUT_TOKENS_SCENARIO, return_text=True
        )
        ttfts.append(ttft_ms)
        time.sleep(INTER_REQUEST_SLEEP_S)

        n_new_tokens = max(0, actual_ptok - prev_prompt_tokens)
        cold_est = _cold_ttft_estimate(n_new_tokens, prev_prompt_tokens)

        action = _parse_json_action(response_text)
        action_type = action.get("action", "search")
        search_query = action.get("query", "") if action_type == "search" else ""

        iter_info = {
            "iteration":         iteration,
            "n_cached_tokens":   prev_prompt_tokens,
            "n_new_tokens":      n_new_tokens,
            "n_total_tokens":    actual_ptok,
            "n_output_tokens":   OUTPUT_TOKENS_SCENARIO,
            "ttft_ms":           round(ttft_ms, 1),
            "cold_ttft_est_ms":  round(cold_est, 1),
            "speedup_vs_cold":   round(cold_est / max(ttft_ms, 1.0), 2),
            "action":            action_type,
            "search_query":      search_query,
        }
        sr.iters.append(iter_info)
        print(
            f"    iter {iteration:2d}: cached={prev_prompt_tokens//1024:4d}k "
            f"new={n_new_tokens//1024:3d}k  ttft={ttft_ms:.0f}ms  "
            f"cold_est={cold_est:.0f}ms  speedup={cold_est/max(ttft_ms,1):.1f}\u00d7  "
            f"action={action_type}"
            + (f"  q=[{search_query[:50]}]" if search_query else ""),
            flush=True,
        )

        prev_prompt_tokens = actual_ptok

        if action_type == "answer":
            print(f"    \u2192 Agent answered after {iteration+1} iteration(s).", flush=True)
            break

        # Retrieve best matching doc and append to conversation
        doc_idx, doc_text = _find_best_doc(search_query or question, wiki_docs, used_docs)
        used_docs.add(doc_idx)
        # Truncate individual doc to ~3 000 chars so context grows gradually
        doc_snip = doc_text[:3_000]
        conversation_body += (
            f"[Search: {search_query[:120]}]\n"
            f"[Result ({doc_idx+1}/{len(wiki_docs)})]: {doc_snip}\n\n"
        )

        # Safety: stop if we\u2019re approaching the context window
        if actual_ptok + OUTPUT_TOKENS_SCENARIO + 500 > AGENT_MAX_MODEL_LEN - 4_096:
            print(
                f"    Iteration {iteration}: context ({actual_ptok:,} tok) nearing "
                f"window limit \u2014 stopping.",
                flush=True,
            )
            break

        # If all docs exhausted, nudge agent to answer
        if len(used_docs) >= len(wiki_docs):
            conversation_body += (
                "[All available documents retrieved. Please provide your final answer.]\n\n"
            )

    if ttfts:
        sr.ttft_median_first = round(ttfts[0], 1)
        sr.ttft_median_last  = round(ttfts[-1], 1)
        sr.ttft_median_all   = round(statistics.median(ttfts), 1)
        sr.total_context_k   = round(prev_prompt_tokens / 1024, 1)

    return sr


def run_agent_scenarios_frames(
    questions: list[dict],
    out_dir: Optional[Path] = None,
) -> list[ScenarioResult]:
    """Run the real ReAct agent on each FRAMES question, return ScenarioResults."""
    session = _make_session()
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
# Top-level entry point
# ---------------------------------------------------------------------------

def run_agent_bench(
    out_dir: Path,
    dataset_path: Optional[Path] = None,
    n_samples: int = N_SAMPLES,
    n_warmups: int = N_WARMUPS,
    skip_scenarios: bool = False,
    skip_matrix: bool = False,
    resume: bool = False,
) -> AgentBenchResult:
    """Run the full agent benchmark and save results to *out_dir*.

    Part 1 — TTFT matrix: measures (N_cached × N_new) cells using a large
    token corpus built from FRAMES Wikipedia articles (or *dataset_path* override).

    Part 2 — Real agent scenarios: runs a genuine ReAct loop on FRAMES multi-hop
    questions. The LLM actually generates search queries; we retrieve the best
    matching Wikipedia article from the gold set and accumulate context across
    turns. TTFT is measured at each LLM call.

    Args:
        out_dir:        Directory for all output files.
        dataset_path:   Override corpus JSONL for matrix (default: FRAMES wiki corpus).
        n_samples:      Measured samples per matrix cell.
        n_warmups:      Discarded warm-up requests per cell.
        skip_scenarios: Skip real agent scenarios (matrix only).
        skip_matrix:    Skip TTFT matrix (scenarios only).
        resume:         Skip already-completed matrix cells.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        ts = datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n[agent_bench] Starting — {ts}", flush=True)
    print(f"  Model: {AGENT_MODEL}  tp={AGENT_TP}  max_model_len={AGENT_MAX_MODEL_LEN}  PC=True", flush=True)
    print(
        f"  Matrix: {len(MATRIX_N_CACHED)} × {len(MATRIX_N_NEW)} cells × {n_samples} samples  "
        f"(N_cached up to {max(MATRIX_N_CACHED)//1024}k, N_new up to {max(MATRIX_N_NEW)//1024}k)",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "datasets").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ corpus
    if not skip_matrix:
        if dataset_path is None:
            # Try FRAMES corpus first; fall back to arxiv if FRAMES unavailable
            dataset_path = _prepare_frames_corpus(out_dir)
        if dataset_path is None:
            dataset_path = _find_arxiv_fallback()
        print(f"  Corpus: {dataset_path}", flush=True)

        session_corpus = _make_session()
        print("\n[agent_bench] Building corpus (tokenising once, ~30s)...", flush=True)
        corpus = Corpus(dataset_path, session_corpus)
    else:
        corpus = None  # type: ignore

    result = AgentBenchResult(
        run_timestamp=ts,
        model=AGENT_MODEL,
        tp=AGENT_TP,
        prefix_caching=True,
    )

    # -------------------------------------------------------------- Part 1: matrix
    if not skip_matrix and corpus is not None:
        print("\n[agent_bench] Part 1/2: TTFT matrix measurement", flush=True)
        matrix = run_ttft_matrix(corpus, out_dir, n_warmups, n_samples, resume=resume)
        result.matrix = matrix
        print_ttft_table(matrix)

    # --------------------------------------------- Part 2: real ReAct scenarios
    if not skip_scenarios:
        print("\n[agent_bench] Part 2/2: Real deep-research agent scenarios (FRAMES)", flush=True)
        print(f"  Dataset: {AGENT_DATASET}", flush=True)
        frames_questions = _load_frames_questions(N_AGENT_SCENARIOS)
        if frames_questions:
            scenarios = run_agent_scenarios_frames(frames_questions, out_dir)
        else:
            print(
                "  WARNING: FRAMES unavailable — skipping scenarios.",
                flush=True,
            )
            scenarios = []
        result.scenarios = scenarios
        if scenarios:
            print_scenario_summary(scenarios)

    # ------------------------------------------------------------------- save
    json_path = out_dir / "agent_bench_results.json"
    json_path.write_text(json.dumps(_result_to_dict(result), indent=2))
    print(f"\n[agent_bench] Results saved → {json_path}", flush=True)
    return result


def _result_to_dict(r: AgentBenchResult) -> dict:
    return {
        "run_timestamp": r.run_timestamp,
        "model":         r.model,
        "tp":            r.tp,
        "prefix_caching": r.prefix_caching,
        "matrix":        [asdict(c) for c in r.matrix],
        "scenarios":     [asdict(s) for s in r.scenarios],
    }


def _find_arxiv_fallback() -> Optional[Path]:
    """Last-resort corpus: discover arxiv-summarization JSONL from previous runs."""
    import glob as _glob
    candidates = sorted(
        _glob.glob("results/*/datasets/ccdv__arxiv-summarization_train_v2.jsonl") +
        _glob.glob("ablation_results/*/datasets/ccdv__arxiv-summarization_train_v2.jsonl") +
        _glob.glob("throughput_results/*/datasets/ccdv__arxiv-summarization_train_v2.jsonl"),
        reverse=True,
    )
    if candidates:
        print(f"  Arxiv fallback corpus: {candidates[0]}", flush=True)
        return Path(candidates[0])
    print("  No local arxiv corpus found — downloading...", flush=True)
    try:
        from .dataset import prepare_hf_dataset
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="agent_bench_corpus_"))
        return prepare_hf_dataset("ccdv/arxiv-summarization", tmp_dir, n_samples=1000)
    except Exception as exc:
        print(f"  ERROR: arxiv fallback failed ({exc})", flush=True)
        return None


# ---------------------------------------------------------------------------
# Server management for agent bench
# ---------------------------------------------------------------------------

def get_agent_server_config() -> Config:
    """Return the Config for the agent benchmark server (tp8, async+PC, 131k context)."""
    return Config(
        model=AGENT_MODEL,
        tp=AGENT_TP,
        quant=None,
        eager=True,
        async_scheduling=True,
        prefix_caching=True,
    )
