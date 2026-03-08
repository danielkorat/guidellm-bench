"""Low-level vLLM API helpers: tokenize, detokenize, warm_cache, measure_ttft.

All requests are serialised (CONCURRENCY=1) — one in-flight HTTP call at a time.
The requests.Session has trust_env=False so it bypasses any http_proxy set in the
environment and hits localhost:8000 directly.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import requests

from .constants import _BASE_URL, AGENT_MODEL, OUTPUT_TOKENS_DEFAULT, CONCURRENCY
from .debug import _DBG, _DBG_WARN, _DBG_ERR

__all__ = [
    "make_session",
    "_tokenize",
    "_detokenize",
    "_verify_token_count",
    "_warm_cache",
    "_measure_ttft",
]

assert CONCURRENCY == 1, "agent benchmark must run with concurrency=1"


def make_session() -> requests.Session:
    """Return a requests.Session with proxy disabled (localhost must bypass proxy)."""
    s = requests.Session()
    s.trust_env = False  # bypass http_proxy for localhost
    return s


# Keep the historical alias used in tests and other modules
_make_session = make_session


def _tokenize(session: requests.Session, text: str) -> list[int]:
    """Return the token ID list for *text* using vLLM /tokenize.

    Concurrency: 1 (synchronous, serial).
    """
    t0 = time.perf_counter()
    _DBG(f"tokenize: request len={len(text):,} chars")
    try:
        resp = session.post(
            f"{_BASE_URL}/tokenize",
            json={"model": AGENT_MODEL, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        tokens = resp.json()["tokens"]
        _DBG(f"tokenize: {len(text):,} chars → {len(tokens):,} tokens in {(time.perf_counter()-t0)*1000:.0f}ms")
        return tokens
    except Exception as exc:
        _DBG_ERR(f"tokenize FAILED after {(time.perf_counter()-t0)*1000:.0f}ms: {exc}")
        raise


def _detokenize(session: requests.Session, tokens: list[int]) -> str:
    """Return the decoded text for a list of token IDs via vLLM /detokenize."""
    t0 = time.perf_counter()
    _DBG(f"detokenize: request {len(tokens):,} tokens")
    try:
        resp = session.post(
            f"{_BASE_URL}/detokenize",
            json={"model": AGENT_MODEL, "tokens": tokens},
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["prompt"]
        _DBG(f"detokenize: {len(tokens):,} tokens → {len(text):,} chars in {(time.perf_counter()-t0)*1000:.0f}ms")
        return text
    except Exception as exc:
        _DBG_ERR(f"detokenize FAILED ({len(tokens)} tokens) after {(time.perf_counter()-t0)*1000:.0f}ms: {exc}")
        raise


def _verify_token_count(session: requests.Session, text: str) -> int:
    """Return actual token count for *text* (non-streaming completions, max_tokens=1)."""
    t0 = time.perf_counter()
    _DBG(f"verify_token_count: prompt={len(text):,} chars")
    try:
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
        n_tok = resp.json()["usage"]["prompt_tokens"]
        _DBG(f"verify_token_count: {len(text):,} chars → {n_tok} tokens in {(time.perf_counter()-t0)*1000:.0f}ms")
        return n_tok
    except Exception as exc:
        _DBG_ERR(f"verify_token_count FAILED after {(time.perf_counter()-t0)*1000:.0f}ms: {exc}")
        raise


def _warm_cache(session: requests.Session, prompt: str, output_tokens: int = 1) -> float:
    """Send a short non-streaming request to prime the KV cache for *prompt*.

    Returns elapsed ms so callers can store it for raw-measurement logging.
    Concurrency: 1 (single request, synchronous).
    """
    t0 = time.perf_counter()
    _DBG(f"warm_cache: prompt={len(prompt):,} chars, output_tokens={output_tokens}")
    try:
        resp = session.post(
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
        elapsed = (time.perf_counter() - t0) * 1000.0
        _DBG(f"warm_cache: done in {elapsed:.0f}ms, HTTP {resp.status_code}")
        return elapsed
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        _DBG_WARN(f"warm_cache: exception after {elapsed:.0f}ms: {exc}")
        return elapsed


def _measure_ttft(
    session: requests.Session,
    prompt: str,
    max_tokens: int = OUTPUT_TOKENS_DEFAULT,
    return_text: bool = False,
) -> tuple:
    """Measure TTFT (ms) for a streaming /v1/completions request (concurrency=1).

    Returns (ttft_ms, prompt_tokens_actual, total_ms) when return_text=False.
    Returns (ttft_ms, prompt_tokens_actual, full_text, total_ms) when return_text=True.

    total_ms = wall-clock from send to last SSE byte (includes generation time).
    Uses /v1/completions (not /v1/chat/completions) to avoid chat-template token injection.
    """
    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    prompt_tokens = 0
    full_text = ""
    n_chunks = 0
    n_empty_chunks = 0

    _DBG(
        f"measure_ttft: prompt={len(prompt):,} chars, max_tokens={max_tokens}, "
        f"return_text={return_text}"
    )

    try:
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
                    n_empty_chunks += 1
                    continue
                line: str = raw_line if isinstance(raw_line, str) else raw_line.decode()
                if not line.startswith("data:"):
                    _DBG(f"measure_ttft: non-data SSE line: {line[:120]!r}")
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError as je:
                    _DBG_WARN(f"measure_ttft: JSON decode error on chunk: {payload_str[:80]!r} — {je}")
                    continue
                n_chunks += 1
                choices = chunk.get("choices", [])
                # Capture TTFT from the very first chunk that has non-empty text
                if ttft_ms is None and choices and choices[0].get("text"):
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                    _DBG(f"measure_ttft: TTFT={ttft_ms:.1f}ms at chunk #{n_chunks}")
                if return_text and choices:
                    full_text += choices[0].get("text", "")
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    _DBG(
                        f"measure_ttft: usage — prompt={prompt_tokens} "
                        f"completion={usage.get('completion_tokens', 0)}"
                    )
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        _DBG_ERR(f"measure_ttft: EXCEPTION after {total_ms:.0f}ms: {exc}")
        raise

    total_ms = (time.perf_counter() - t0) * 1000.0
    if ttft_ms is None:
        _DBG_WARN(
            f"measure_ttft: no non-empty text chunk — "
            f"falling back to total_ms={total_ms:.0f}ms as TTFT. "
            f"n_chunks={n_chunks}, prompt_tokens={prompt_tokens}"
        )
        ttft_ms = total_ms

    _DBG(
        f"measure_ttft: DONE  TTFT={ttft_ms:.1f}ms  total={total_ms:.1f}ms  "
        f"prompt_tokens={prompt_tokens}  n_chunks={n_chunks}"
    )

    if return_text:
        return ttft_ms, prompt_tokens, full_text, total_ms
    return ttft_ms, prompt_tokens, total_ms
