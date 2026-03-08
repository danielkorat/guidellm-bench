#!/usr/bin/env python3
"""Sanity test: run agent_bench against a tiny in-process dummy HTTP server.

Exercises (in this order):
  1. Corpus: tokenize + detokenize endpoints
  2. measure_cell: warm_cache (non-stream) + streaming TTFT measurement
  3. run_ttft_matrix: 1 cell (N_cached=0, N_new=128), 1 warm-up, 2 samples
  4. run_research_session: 1 question, LLM returns immediate JSON answer

Usage: python3 test_agent_sanity.py
Exit 0 if all assertions pass, non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Dummy server configuration
# ---------------------------------------------------------------------------

DUMMY_PORT   = 18001           # must not conflict with real vLLM (8000)
DUMMY_ANSWER = '{"action": "answer", "text": "The answer is 42."}'


class _DummyHandler(BaseHTTPRequestHandler):
    """Minimal vLLM-API emulation for unit testing."""

    def log_message(self, *_):
        pass  # suppress per-request log noise

    # ---- helpers -----------------------------------------------------------

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def _send_json(self, obj: dict, code: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, text_chunks: list[str], prompt_tokens: int) -> None:
        """Write SSE text chunks + usage chunk + [DONE] in one write."""
        parts: list[bytes] = []
        # text chunks — first chunk triggers TTFT in _measure_ttft
        for ch in text_chunks:
            payload = json.dumps({
                "choices": [{"text": ch, "finish_reason": None}],
                "usage": None,
            })
            parts.append(f"data: {payload}\n\n".encode())
        # usage chunk
        usage_payload = json.dumps({
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": sum(len(c) for c in text_chunks),
            },
        })
        parts.append(f"data: {usage_payload}\n\n".encode())
        parts.append(b"data: [DONE]\n\n")

        body = b"".join(parts)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- GET ---------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    # ---- POST --------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()

        if self.path == "/tokenize":
            prompt = body.get("prompt", "")
            # ~1 token per 4 chars — good enough for test purposes
            n_tokens = max(1, len(prompt) // 4)
            self._send_json({"tokens": list(range(n_tokens))})

        elif self.path == "/detokenize":
            tokens = body.get("tokens", [])
            # fake but deterministic: "w0 w1 w2 ..."
            text = " ".join(f"w{t}" for t in tokens)
            self._send_json({"prompt": text})

        elif self.path == "/v1/completions":
            prompt = body.get("prompt", "")
            pt = max(1, len(prompt) // 4)   # approximate prompt_tokens
            stream = body.get("stream", False)

            if not stream:
                # warm-cache or verify_token_count call
                self._send_json({
                    "usage": {"prompt_tokens": pt, "completion_tokens": 1},
                    "choices": [{"text": "ok", "finish_reason": "stop"}],
                })
            else:
                # _measure_ttft: return answer JSON split across 3 SSE chunks
                text = DUMMY_ANSWER
                third = max(1, len(text) // 3)
                chunks = [text[:third], text[third:2*third], text[2*third:]]
                self._send_sse(chunks, pt)

        else:
            self.send_response(404)
            self.end_headers()


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------

def _start_dummy_server() -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", DUMMY_PORT), _DummyHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    # wait until the /health endpoint responds
    import requests
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{DUMMY_PORT}/health", timeout=1)
            if r.status_code == 200:
                return srv
        except Exception:
            pass
        time.sleep(0.05)
    raise RuntimeError("Dummy server did not become healthy within 5s")


# ---------------------------------------------------------------------------
# Tiny test corpus JSONL (inline, ~600 chars, > 128*4 = 512 chars needed)
# ---------------------------------------------------------------------------

_CORPUS_TEXT = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines, "
    "as opposed to the natural intelligence displayed by animals including humans. "
    "AI research has been defined as the field of study of intelligent agents, "
    "which refers to any system that perceives its environment and takes actions "
    "that maximize its chance of achieving its goals. The term may also be applied "
    "to any machine that exhibits traits associated with a human mind such as "
    "learning and problem-solving. The field was founded on the assumption that "
    "human intelligence can be so precisely described that a machine can be made "
    "to simulate it. This raises philosophical arguments about the mind and the "
    "ethics of creating artificial beings endowed with human-like intelligence. "
    "These issues have been explored by myth, fiction and philosophy since antiquity."
)


def _write_corpus_jsonl(path: Path) -> None:
    path.write_text(
        json.dumps({"prompt": _CORPUS_TEXT}) + "\n"
        + json.dumps({"prompt": _CORPUS_TEXT * 3}) + "\n"  # pad to ensure enough tokens
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def _check(label: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        print(f"  \033[32mPASS\033[0m  {label}")
        _PASS += 1
    else:
        print(f"  \033[31mFAIL\033[0m  {label}" + (f" — {detail}" if detail else ""))
        _FAIL += 1


# ---------------------------------------------------------------------------
# Main test driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("agent_bench sanity test (dummy server)")
    print("=" * 60)

    # ---- 1. Start dummy server -------------------------------------------
    print("\n[1] Starting dummy HTTP server on port", DUMMY_PORT, "...", end=" ", flush=True)
    srv = _start_dummy_server()
    print("OK")

    # ---- 2. Patch agent_bench constants -----------------------------------
    # Import AFTER the server is up so _BASE_URL can be overridden cleanly.
    import guidellm_bench.agent_bench as ab

    _orig_base_url     = ab._BASE_URL
    _orig_n_cached     = ab.MATRIX_N_CACHED
    _orig_n_new        = ab.MATRIX_N_NEW
    _orig_n_warmups    = ab.N_WARMUPS
    _orig_n_samples    = ab.N_SAMPLES
    _orig_cv_threshold = ab.CV_RERUN_THRESHOLD
    _orig_sleep        = ab.INTER_REQUEST_SLEEP_S
    _orig_model        = ab.AGENT_MODEL

    ab._BASE_URL           = f"http://127.0.0.1:{DUMMY_PORT}"
    ab.MATRIX_N_CACHED     = [0]       # single cache-miss cell only
    ab.MATRIX_N_NEW        = [128]     # tiny prompt
    ab.N_WARMUPS           = 1
    ab.N_SAMPLES           = 2
    ab.CV_RERUN_THRESHOLD  = 99.0      # disable re-run logic
    ab.INTER_REQUEST_SLEEP_S = 0.0     # no sleep between iterations
    ab.AGENT_MODEL         = "openai/gpt-oss-20b"  # keep same (server ignores model name)

    print(
        f"[2] Patched: _BASE_URL={ab._BASE_URL!r}, "
        f"MATRIX_N_CACHED={ab.MATRIX_N_CACHED}, "
        f"MATRIX_N_NEW={ab.MATRIX_N_NEW}, "
        f"N_WARMUPS={ab.N_WARMUPS}, N_SAMPLES={ab.N_SAMPLES}"
    )

    try:
        with tempfile.TemporaryDirectory() as tmpstr:
            out_dir = Path(tmpstr)
            (out_dir / "datasets").mkdir()

            # ---- Set up debug log ----------------------------------------
            ab._setup_debug_log(out_dir / "agent_debug.log")
            print(f"[3] Debug log: {out_dir}/agent_debug.log")

            import requests as req_lib
            session = ab._make_session()

            # ---- 4. Test /tokenize -----------------------------------------
            print("\n[4] Testing tokenize endpoint ...")
            tokens = ab._tokenize(session, "Hello world " * 50)
            _check("tokenize returns list[int]", isinstance(tokens, list) and all(isinstance(t, int) for t in tokens))
            _check("tokenize non-empty", len(tokens) > 0, f"got {len(tokens)}")

            # ---- 5. Test /detokenize ----------------------------------------
            print("\n[5] Testing detokenize endpoint ...")
            text = ab._detokenize(session, tokens[:10])
            _check("detokenize returns str", isinstance(text, str))
            _check("detokenize non-empty", len(text) > 0)

            # ---- 6. Test _warm_cache ----------------------------------------
            print("\n[6] Testing _warm_cache (non-streaming POST) ...")
            wc_ms = ab._warm_cache(session, "The quick brown fox")
            _check("_warm_cache returns float", isinstance(wc_ms, float))
            _check("_warm_cache > 0ms", wc_ms > 0, f"got {wc_ms}")

            # ---- 7. Test _measure_ttft (return_text=False) ------------------
            print("\n[7] Testing _measure_ttft (return_text=False) ...")
            result = ab._measure_ttft(session, "Test prompt for TTFT", max_tokens=16)
            _check("_measure_ttft 3-tuple", isinstance(result, tuple) and len(result) == 3,
                   f"got {type(result).__name__} len={len(result) if isinstance(result, tuple) else '?'}")
            ttft_ms, prompt_tokens, total_ms = result
            _check("ttft_ms > 0", ttft_ms > 0, f"got {ttft_ms}")
            _check("prompt_tokens >= 0", prompt_tokens >= 0, f"got {prompt_tokens}")
            _check("total_ms >= ttft_ms", total_ms >= ttft_ms, f"total={total_ms:.1f} ttft={ttft_ms:.1f}")

            # ---- 8. Test _measure_ttft (return_text=True) -------------------
            print("\n[8] Testing _measure_ttft (return_text=True) ...")
            result4 = ab._measure_ttft(session, "Test prompt", max_tokens=32, return_text=True)
            _check("_measure_ttft 4-tuple", isinstance(result4, tuple) and len(result4) == 4,
                   f"got len={len(result4) if isinstance(result4, tuple) else '?'}")
            ttft_ms4, pt4, full_text4, total_ms4 = result4
            _check("full_text non-empty", len(full_text4) > 0, f"got {full_text4!r}")
            _check("answer JSON in text", "answer" in full_text4 or "42" in full_text4,
                   f"text={full_text4!r}")

            # ---- 9. Build Corpus -------------------------------------------
            print("\n[9] Building Corpus from JSONL ...")
            corpus_file = out_dir / "datasets" / "test_corpus.jsonl"
            _write_corpus_jsonl(corpus_file)
            corpus = ab.Corpus(corpus_file, session, max_chars=50_000)
            _check("corpus tokenized", corpus.n_tokens() >= 128,
                   f"only {corpus.n_tokens()} tokens (need >=128)")
            sliced = corpus.slice_text(0, 64)
            _check("slice_text returns str", isinstance(sliced, str) and len(sliced) > 0)

            # ---- 10. measure_cell ------------------------------------------
            print("\n[10] Testing measure_cell (N_cached=0, N_new=128) ...")
            cached_prompt = corpus.slice_text(0, 0)       # empty (N_cached=0)
            new_prompt    = corpus.slice_text(0, 128)     # 128-token span
            cr = ab.measure_cell(session, cached_prompt, new_prompt, 0, 128,
                                 n_warmups=1, n_samples=2)
            _check("measure_cell returns CellResult", isinstance(cr, ab.CellResult))
            _check("cr.ttft_median > 0", cr.ttft_median > 0, f"got {cr.ttft_median}")
            _check("cr.n_samples == 2", cr.n_samples == 2,
                   f"got {cr.n_samples}")
            _check("raw total_request_ms populated",
                   len(cr.total_request_ms_values) == 2,
                   f"got len={len(cr.total_request_ms_values)}")
            _check("raw warm_cache_ms populated",
                   len(cr.warm_cache_ms_values) == 2,
                   f"got len={len(cr.warm_cache_ms_values)}")
            _check("raw warmup_ttft populated",
                   len(cr.warmup_ttft_ms_values) == 1,
                   f"got len={len(cr.warmup_ttft_ms_values)}")

            # ---- 11. run_ttft_matrix ----------------------------------------
            print("\n[11] Testing run_ttft_matrix (1 cell) ...")
            results = ab.run_ttft_matrix(corpus, out_dir, n_warmups=1, n_samples=2)
            _check("run_ttft_matrix returns list", isinstance(results, list))
            _check("1 result for 1 cell", len(results) == 1,
                   f"got {len(results)}")
            cell = results[0]
            _check("matrix cell has expected n_new", cell.n_new == 128,
                   f"got n_new={cell.n_new}")
            _check("matrix cell ttft_median > 0", cell.ttft_median > 0)

            # Load checkpoint JSON
            checkpoint = out_dir / "agent_matrix.json"
            _check("checkpoint file written", checkpoint.exists())
            if checkpoint.exists():
                ckpt_data = json.loads(checkpoint.read_text())
                _check("checkpoint non-empty", len(ckpt_data) > 0,
                       f"keys={list(ckpt_data.keys())}")

            # ---- 12. run_research_session -----------------------------------
            print("\n[12] Testing run_research_session (1 question, instant answer) ...")
            wiki_docs = [
                "Artificial intelligence is intelligence demonstrated by machines.",
                "Machine learning is a subset of AI.",
            ]
            sr = ab.run_research_session(
                session,
                scenario_name="sanity_test",
                question="What is artificial intelligence?",
                wiki_docs=wiki_docs,
                max_iterations=5,
            )
            _check("run_research_session returns ScenarioResult",
                   isinstance(sr, ab.ScenarioResult))
            _check("at least 1 iteration logged", len(sr.iters) >= 1,
                   f"got {len(sr.iters)}")
            first_iter = sr.iters[0]
            _check("iter ttft_ms present", "ttft_ms" in first_iter)
            _check("iter total_request_ms present", "total_request_ms" in first_iter)
            _check("iter warm_cache_ms present", "warm_cache_ms" in first_iter)
            _check("iter raw_response_text present", "raw_response_text" in first_iter)
            final_action = sr.iters[-1].get("action", "")
            _check("session ended with answer",
                   final_action == "answer",
                   f"last_action={final_action!r}")

            # ---- 13. debug log was written ----------------------------------
            print("\n[13] Checking debug log ...")
            log_path = out_dir / "agent_debug.log"
            _check("debug log exists", log_path.exists())
            if log_path.exists():
                log_size = log_path.stat().st_size
                _check("debug log non-empty", log_size > 0, f"size={log_size}")
                # spot-check for key log entries
                log_text = log_path.read_text()
                _check("tokenize logged",    "tokenize:" in log_text)
                _check("measure_ttft logged", "measure_ttft:" in log_text)
                _check("warm_cache logged",   "warm_cache:" in log_text)

    finally:
        # Restore patched constants
        ab._BASE_URL           = _orig_base_url
        ab.MATRIX_N_CACHED     = _orig_n_cached
        ab.MATRIX_N_NEW        = _orig_n_new
        ab.N_WARMUPS           = _orig_n_warmups
        ab.N_SAMPLES           = _orig_n_samples
        ab.CV_RERUN_THRESHOLD  = _orig_cv_threshold
        ab.INTER_REQUEST_SLEEP_S = _orig_sleep
        ab.AGENT_MODEL         = _orig_model
        srv.shutdown()

    # ---- Summary -----------------------------------------------------------
    total = _PASS + _FAIL
    print("\n" + "=" * 60)
    if _FAIL == 0:
        print(f"\033[32mAll {_PASS}/{total} checks PASSED\033[0m")
    else:
        print(f"\033[31m{_FAIL}/{total} checks FAILED  ({_PASS} passed)\033[0m")
    print("=" * 60)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
