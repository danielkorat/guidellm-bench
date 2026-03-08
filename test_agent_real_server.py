#!/usr/bin/env python3
"""Sanity test: run agent_bench against a real vLLM server with facebook/opt-125m.

Starts vLLM inside lsv-container, exercises every agent_bench code path with
tiny matrix / scenario parameters, then tears down the server.

Re-execs itself inside lsv-container when run from the host.

Usage (from host or inside container):
    python3 test_agent_real_server.py
Exit 0 if all assertions pass, non-zero on any failure.
"""

from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-exec guard: run this script inside lsv-container if on the host
# ---------------------------------------------------------------------------
if not os.path.exists("/.dockerenv"):
    _proxy_args = []
    for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        _val = os.environ.get(_v)
        if _val:
            _proxy_args += ["-e", f"{_v}={_val}"]
    _tty = ["-t"] if sys.stdout.isatty() else []
    _cmd = (
        ["docker", "exec", "-w", "/root/guidellm-bench"]
        + _tty + _proxy_args
        + ["lsv-container", "python3", "/root/guidellm-bench/test_agent_real_server.py"]
    )
    sys.exit(subprocess.call(_cmd))

# ---------------------------------------------------------------------------
# From here on we are inside the container
# ---------------------------------------------------------------------------

from guidellm_bench.docker import _PREAMBLE

SANITY_PORT    = 18002
SANITY_MODEL   = "facebook/opt-125m"
MAX_MODEL_LEN  = 512    # keeps startup fast and limits request size
SERVER_TIMEOUT = 180    # seconds to wait for /health

# Tiny matrix parameters (2 cells total: cold   + warm-cached)
SANITY_N_CACHED = [0, 128]
SANITY_N_NEW    = [64]
SANITY_WARMUPS  = 1
SANITY_SAMPLES  = 2
SANITY_OUT_TOK  = 16    # very short generation to keep tests fast

# One scenario with 2 wiki docs, stop after 2 iterations regardless
SANITY_QUESTION  = "What is opt-125m?"
SANITY_WIKI_DOCS = [
    "OPT (Open Pre-trained Transformer) is a suite of decoder-only pre-trained "
    "transformers by Meta AI. The 125M parameter variant is the smallest member "
    "of the family, useful for research and experimentation.",
    "Language models are trained on large corpora of text and learn statistical "
    "associations between tokens. Small models like OPT-125M take only seconds "
    "to load and run on consumer hardware.",
]

# ── Corpus text ── needs enough chars for 128+64=192 tokens of opt-125m
_CORPUS = " ".join([
    "Artificial intelligence (AI) is intelligence demonstrated by machines. "
    "AI research has been defined as the field of study of intelligent agents, "
    "which refers to any system that perceives its environment and takes actions "
    "that maximise its chance of achieving its goals. The field was founded on "
    "the assumption that human intelligence can be so precisely described that "
    "a machine can be made to simulate it. This raises philosophical questions "
    "about the mind and the ethics of creating artificial beings endowed with "
    "human-like intelligence. These issues have been explored by myth, fiction "
    "and philosophy since antiquity. Modern AI includes machine learning, deep "
    "neural networks, natural language processing, computer vision, and robotics. "
    "Large language models such as GPT and OPT are trained on internet-scale text "
    "and can perform diverse tasks through in-context learning. OPT-125M is a small "
    "but capable baseline model released by Meta AI as open research infrastructure. "
    "Transformer architectures use self-attention mechanisms to model long-range "
    "dependencies in sequences, enabling unprecedented language understanding. "
    "Pre-training on diverse data followed by fine-tuning or RLHF allows models "
    "to align with human preferences and follow complex instructions accurately. "
    "Scaling laws suggest that performance improves predictably with compute. "
] * 4)   # repeat to ensure we exceed 200+ tokens easily


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

_PASS = _FAIL = 0

def _check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        print(f"  \033[32mPASS\033[0m  {label}")
        _PASS += 1
    else:
        print(f"  \033[31mFAIL\033[0m  {label}" + (f" — {detail}" if detail else ""))
        _FAIL += 1


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def _start_vllm(log_path: Path) -> subprocess.Popen:
    """Start vLLM with opt-125m on SANITY_PORT; return the Popen object."""
    cmd = (
        f"{_PREAMBLE} && "
        f"python3 -m vllm.entrypoints.openai.api_server "
        f"    --model {SANITY_MODEL} "
        f"    --port {SANITY_PORT} "
        f"    --tensor-parallel-size 1 "
        f"    --max-model-len {MAX_MODEL_LEN} "
        f"    --max-num-batched-tokens {MAX_MODEL_LEN} "
        f"    --enable-prefix-caching "
        f"    --disable-log-requests"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    print(f"  Starting vLLM ({SANITY_MODEL}) on port {SANITY_PORT}…", flush=True)
    proc = subprocess.Popen(
        ["bash", "--login", "-c", cmd],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=dict(os.environ,
                 no_proxy="localhost,127.0.0.1,0.0.0.0",
                 NO_PROXY="localhost,127.0.0.1,0.0.0.0"),
    )
    return proc


def _wait_healthy(timeout: int = SERVER_TIMEOUT) -> bool:
    """Poll /health until 200 or timeout."""
    import requests
    url = f"http://localhost:{SANITY_PORT}/health"
    deadline = time.time() + timeout
    last_msg_t = 0.0
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        elapsed = int(time.time() - (deadline - timeout))
        if time.time() - last_msg_t >= 15:
            print(f"  Still waiting for /health… {elapsed}s elapsed", flush=True)
            last_msg_t = time.time()
        time.sleep(2)
    return False


def _stop_vllm(proc: subprocess.Popen) -> None:
    """Gracefully stop vLLM, fall back to SIGKILL after 10s."""
    if proc.poll() is not None:
        return
    print("  Stopping vLLM server…", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    # Also kill any stray workers by port
    subprocess.call(
        ["bash", "-c",
         f"fuser -k {SANITY_PORT}/tcp 2>/dev/null; "
         f"ss -tlnp 2>/dev/null | grep :{SANITY_PORT} | "
         f"grep -oP 'pid=\\K[0-9]+' | xargs -r kill -9 2>/dev/null; "
         "true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  vLLM stopped.", flush=True)


# ---------------------------------------------------------------------------
# Main test driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print(f"agent_bench real-server sanity test  (model={SANITY_MODEL})")
    print("=" * 64)

    proc = None
    with tempfile.TemporaryDirectory() as tmpstr:
        out_dir   = Path(tmpstr)
        log_path  = out_dir / "logs" / "vllm_sanity.log"
        corpus_f  = out_dir / "datasets" / "corpus.jsonl"

        # ── Write corpus ──────────────────────────────────────────────────
        corpus_f.parent.mkdir(parents=True, exist_ok=True)
        corpus_f.write_text(json.dumps({"prompt": _CORPUS}) + "\n"
                            + json.dumps({"prompt": _CORPUS}) + "\n")

        # ── Kill any stale process on the sanity port ──────────────────────
        subprocess.call(
            ["bash", "-c",
             f"fuser -k {SANITY_PORT}/tcp 2>/dev/null; "
             f"ss -tlnp 2>/dev/null | grep :{SANITY_PORT} | "
             f"grep -oP 'pid=\\K[0-9]+' | xargs -r kill -9 2>/dev/null; "
             "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

        # ── [1] Start vLLM ────────────────────────────────────────────────
        print("\n[1] Starting vLLM server…")
        proc = _start_vllm(log_path)
        healthy = _wait_healthy()
        _check(f"vLLM healthy on :{SANITY_PORT}", healthy,
               f"check {log_path} for errors")
        if not healthy:
            print(f"\n  Last 30 lines of server log ({log_path}):")
            try:
                lines = log_path.read_text().splitlines()[-30:]
                print("\n".join(f"    {l}" for l in lines))
            except Exception:
                pass
            _stop_vllm(proc)
            _print_summary()
            return 1

        print(f"  vLLM up and healthy  (PID={proc.pid})\n", flush=True)

        # ── [2] Patch agent_bench constants ───────────────────────────────
        print("[2] Patching agent_bench constants…")
        import guidellm_bench.agent_bench as ab

        saved = {
            "_BASE_URL":            ab._BASE_URL,
            "AGENT_MODEL":          ab.AGENT_MODEL,
            "MATRIX_N_CACHED":      ab.MATRIX_N_CACHED,
            "MATRIX_N_NEW":         ab.MATRIX_N_NEW,
            "N_WARMUPS":            ab.N_WARMUPS,
            "N_SAMPLES":            ab.N_SAMPLES,
            "CV_RERUN_THRESHOLD":   ab.CV_RERUN_THRESHOLD,
            "INTER_REQUEST_SLEEP_S": ab.INTER_REQUEST_SLEEP_S,
            "OUTPUT_TOKENS_DEFAULT":  ab.OUTPUT_TOKENS_DEFAULT,
            "OUTPUT_TOKENS_SCENARIO": ab.OUTPUT_TOKENS_SCENARIO,
            "AGENT_MAX_MODEL_LEN":  ab.AGENT_MAX_MODEL_LEN,
        }

        ab._BASE_URL            = f"http://localhost:{SANITY_PORT}"
        ab.AGENT_MODEL          = SANITY_MODEL
        ab.MATRIX_N_CACHED      = SANITY_N_CACHED
        ab.MATRIX_N_NEW         = SANITY_N_NEW
        ab.N_WARMUPS            = SANITY_WARMUPS
        ab.N_SAMPLES            = SANITY_SAMPLES
        ab.CV_RERUN_THRESHOLD   = 99.0      # disable re-run
        ab.INTER_REQUEST_SLEEP_S = 0.0      # no sleep between requests
        ab.OUTPUT_TOKENS_DEFAULT  = SANITY_OUT_TOK
        ab.OUTPUT_TOKENS_SCENARIO = SANITY_OUT_TOK
        ab.AGENT_MAX_MODEL_LEN  = MAX_MODEL_LEN

        print(
            f"  _BASE_URL={ab._BASE_URL}  model={ab.AGENT_MODEL}  "
            f"N_CACHED={ab.MATRIX_N_CACHED}  N_NEW={ab.MATRIX_N_NEW}  "
            f"N_WARMUPS={ab.N_WARMUPS}  N_SAMPLES={ab.N_SAMPLES}"
        )

        try:
            # ── [3] Debug log ─────────────────────────────────────────────
            ab._setup_debug_log(out_dir / "agent_debug.log")

            session = ab._make_session()

            # ── [4] /tokenize ─────────────────────────────────────────────
            print("\n[4] Testing /tokenize…")
            tokens = ab._tokenize(session, "Hello world, this is a test sentence.")
            _check("tokenize → list[int]",
                   isinstance(tokens, list) and all(isinstance(t, int) for t in tokens))
            _check("tokenize non-empty", len(tokens) > 0, f"got {len(tokens)}")

            # ── [5] /detokenize ───────────────────────────────────────────
            print("\n[5] Testing /detokenize…")
            text = ab._detokenize(session, tokens)
            _check("detokenize → str", isinstance(text, str) and len(text) > 0)

            # ── [6] _warm_cache ───────────────────────────────────────────
            print("\n[6] Testing _warm_cache (non-streaming)…")
            wc_ms = ab._warm_cache(session, "The quick brown fox jumped over the lazy dog.")
            _check("_warm_cache → float", isinstance(wc_ms, float))
            _check("_warm_cache > 0 ms", wc_ms > 0, f"got {wc_ms}")

            # ── [7] _measure_ttft (return_text=False, 3-tuple) ────────────
            print("\n[7] Testing _measure_ttft (return_text=False)…")
            r3 = ab._measure_ttft(session, "Once upon a time", max_tokens=4)
            _check("_measure_ttft 3-tuple",
                   isinstance(r3, tuple) and len(r3) == 3,
                   f"got {type(r3).__name__}[{len(r3) if isinstance(r3,tuple) else '?'}]")
            ttft3, ptok3, total3 = r3
            _check("ttft_ms > 0", ttft3 > 0, f"got {ttft3}")
            _check("prompt_tokens > 0", ptok3 > 0, f"got {ptok3}")
            _check("total_ms >= ttft_ms", total3 >= ttft3,
                   f"total={total3:.1f} ttft={ttft3:.1f}")

            # ── [8] _measure_ttft (return_text=True, 4-tuple) ─────────────
            print("\n[8] Testing _measure_ttft (return_text=True)…")
            r4 = ab._measure_ttft(session, "The capital of France is", max_tokens=4,
                                  return_text=True)
            _check("_measure_ttft 4-tuple",
                   isinstance(r4, tuple) and len(r4) == 4,
                   f"got len={len(r4) if isinstance(r4,tuple) else '?'}")
            ttft4, ptok4, txt4, total4 = r4
            _check("full_text non-empty", len(txt4) > 0, f"got {txt4!r}")

            # ── [9] Corpus ────────────────────────────────────────────────
            print("\n[9] Building Corpus…")
            corpus = ab.Corpus(corpus_f, session, max_chars=50_000)
            min_tokens = max(ab.MATRIX_N_CACHED) + max(ab.MATRIX_N_NEW)
            _check(f"corpus has ≥ {min_tokens} tokens",
                   corpus.n_tokens() >= min_tokens,
                   f"only {corpus.n_tokens()} tokens")
            sliced = corpus.slice_text(0, 32)
            _check("slice_text(0,32) → non-empty str",
                   isinstance(sliced, str) and len(sliced) > 0)

            # ── [10] measure_cell (N_cached=0, cold cell) ─────────────────
            print("\n[10] Testing measure_cell (N_cached=0, cold)…")
            new_prompt = corpus.slice_text(0, SANITY_N_NEW[0])
            cr0 = ab.measure_cell(
                session, "", new_prompt, 0, SANITY_N_NEW[0],
                n_warmups=SANITY_WARMUPS, n_samples=SANITY_SAMPLES,
                max_output_tokens=SANITY_OUT_TOK,
            )
            _check("CellResult returned", isinstance(cr0, ab.CellResult))
            _check("ttft_median > 0", cr0.ttft_median > 0, f"got {cr0.ttft_median}")
            _check("n_samples == SANITY_SAMPLES",
                   cr0.n_samples == SANITY_SAMPLES, f"got {cr0.n_samples}")
            _check("ttft_ms_values len == n_samples",
                   len(cr0.ttft_ms_values) == SANITY_SAMPLES,
                   f"got {len(cr0.ttft_ms_values)}")
            _check("total_request_ms_values populated",
                   len(cr0.total_request_ms_values) == SANITY_SAMPLES,
                   f"got {len(cr0.total_request_ms_values)}")
            _check("warmup_ttft_ms_values populated",
                   len(cr0.warmup_ttft_ms_values) == SANITY_WARMUPS,
                   f"got {len(cr0.warmup_ttft_ms_values)}")
            _check("all ttft_ms > 0",
                   all(v > 0 for v in cr0.ttft_ms_values),
                   f"values={cr0.ttft_ms_values}")

            # ── [11] measure_cell (N_cached=128, warm cell) ───────────────
            print("\n[11] Testing measure_cell (N_cached=128, warm)…")
            cached_prompt = corpus.slice_text(0, SANITY_N_CACHED[1])
            new_prompt2   = corpus.slice_text(SANITY_N_CACHED[1],
                                              SANITY_N_CACHED[1] + SANITY_N_NEW[0])
            cr1 = ab.measure_cell(
                session, cached_prompt, new_prompt2,
                SANITY_N_CACHED[1], SANITY_N_NEW[0],
                n_warmups=SANITY_WARMUPS, n_samples=SANITY_SAMPLES,
                max_output_tokens=SANITY_OUT_TOK,
            )
            _check("warm CellResult returned", isinstance(cr1, ab.CellResult))
            _check("warm ttft_median > 0", cr1.ttft_median > 0)
            _check("warm warm_cache_ms_values populated",
                   len(cr1.warm_cache_ms_values) == SANITY_SAMPLES
                   and all(v > 0 for v in cr1.warm_cache_ms_values),
                   f"got {cr1.warm_cache_ms_values}")

            # ── [12] run_ttft_matrix ──────────────────────────────────────
            print("\n[12] Testing run_ttft_matrix (2 cells)…")
            matrix_results = ab.run_ttft_matrix(
                corpus, out_dir,
                n_warmups=SANITY_WARMUPS,
                n_samples=SANITY_SAMPLES,
                max_output_tokens=SANITY_OUT_TOK,
            )
            _check("returns list", isinstance(matrix_results, list))
            _check("2 cells returned", len(matrix_results) == 2,
                   f"got {len(matrix_results)}")
            _check("checkpoint written", (out_dir / "agent_matrix.json").exists())
            ckpt = json.loads((out_dir / "agent_matrix.json").read_text())
            _check("checkpoint has 2 entries",
                   len(ckpt.get("matrix", [])) == 2,
                   f"got {len(ckpt.get('matrix', []))}")

            # ── [13] run_research_session ─────────────────────────────────
            # opt-125m won't produce valid JSON → fallback to search each iter
            # We just verify the data structures are correct.
            print("\n[13] Testing run_research_session (≤2 iterations)…")
            sr = ab.run_research_session(
                session,
                scenario_name="sanity_real",
                question=SANITY_QUESTION,
                wiki_docs=SANITY_WIKI_DOCS,
                max_iterations=2,
            )
            _check("ScenarioResult returned", isinstance(sr, ab.ScenarioResult))
            _check("≥1 iteration logged",
                   len(sr.iters) >= 1, f"got {len(sr.iters)}")
            fi = sr.iters[0]
            for field in ("ttft_ms", "total_request_ms", "warm_cache_ms",
                          "raw_response_text", "n_total_tokens"):
                _check(f"iter[0] has '{field}'", field in fi, str(fi.keys()))
            _check("iter[0] ttft_ms > 0",
                   fi["ttft_ms"] > 0, f"got {fi['ttft_ms']}")
            _check("iter[0] total_request_ms > 0",
                   fi["total_request_ms"] > 0, f"got {fi['total_request_ms']}")
            _check("iter[0] raw_response_text non-empty",
                   len(fi.get("raw_response_text","")) > 0)

            # ── [14] Debug log ────────────────────────────────────────────
            print("\n[14] Checking debug log…")
            dlog = out_dir / "agent_debug.log"
            _check("debug log exists", dlog.exists())
            if dlog.exists():
                txt = dlog.read_text()
                _check("log non-empty", len(txt) > 0)
                for marker in ("tokenize:", "measure_ttft:", "warm_cache:",
                               "measure_cell", "run_ttft_matrix"):
                    _check(f"log contains '{marker}'", marker in txt)

        finally:
            # ── Restore constants ─────────────────────────────────────────
            for k, v in saved.items():
                setattr(ab, k, v)
            _stop_vllm(proc)

    return _print_summary()


def _print_summary() -> int:
    global _PASS, _FAIL
    total = _PASS + _FAIL
    print("\n" + "=" * 64)
    if _FAIL == 0:
        print(f"\033[32mAll {_PASS}/{total} checks PASSED\033[0m")
    else:
        print(f"\033[31m{_FAIL}/{total} checks FAILED  ({_PASS} passed)\033[0m")
    print("=" * 64)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
