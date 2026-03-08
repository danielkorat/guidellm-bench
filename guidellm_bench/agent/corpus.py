"""Corpus preparation for agent benchmark matrix measurements.

Provides the Corpus class (lazy-loaded, sliceable token array) and helpers
to build a document corpus from the FRAMES dataset or arxiv as fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests

from .constants import AGENT_DATASET, MATRIX_N_CACHED, MATRIX_N_NEW
from .debug import _DBG_INFO, _DBG_WARN, _DBG_ERR
from .helpers import _tokenize, _detokenize, _fetch_wikipedia_text, _parse_frames_urls

__all__ = ["Corpus", "_prepare_frames_corpus", "_find_arxiv_fallback"]


class Corpus:
    """Lazy-loaded token array from a JSONL corpus, sliceable to exact N tokens.

    Usage::

        corpus = Corpus(dataset_path, session)
        text   = corpus.slice_text(0, 32_768)   # first 32k tokens as decoded text
    """

    def __init__(
        self,
        dataset_path: Path,
        session: requests.Session,
        max_chars: int = 900_000,
    ) -> None:
        self._session = session
        raw_texts: list[str] = []
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

        needed = max(MATRIX_N_CACHED) + max(MATRIX_N_NEW)
        if len(self._tokens) < needed:
            raise RuntimeError(
                f"Corpus too small: {len(self._tokens):,} tokens, need {needed:,}. "
                "Expand dataset_path or increase max_chars."
            )
        _DBG_INFO(f"[Corpus] ready: {len(self._tokens):,} tokens (need {needed:,})")

    def slice_text(self, start_token: int, end_token: int) -> str:
        """Return the decoded text for tokens[start_token:end_token]."""
        sub = self._tokens[start_token:end_token]
        return _detokenize(self._session, sub)

    def n_tokens(self) -> int:
        return len(self._tokens)

    @property
    def total_tokens(self) -> int:
        """Total token count — alias for n_tokens()."""
        return len(self._tokens)


def _prepare_frames_corpus(out_dir: Path) -> Optional[Path]:
    """Write a corpus JSONL from all FRAMES wiki_doc texts for matrix measurements.

    Returns path to the JSONL, or None on failure (caller falls back to arxiv).
    """
    corpus_path = out_dir / "datasets" / "frames_corpus_v1.jsonl"
    if corpus_path.exists() and corpus_path.stat().st_size > 100:
        print(f"  FRAMES corpus: using cached {corpus_path}", flush=True)
        return corpus_path
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(AGENT_DATASET, split="test")

        # FRAMES only provides Wikipedia URLs, not article text.
        # Collect unique URLs from all rows, then fetch their content.
        seen_urls: set[str] = set()
        url_list: list[str] = []
        for row in ds:
            for url in _parse_frames_urls(row):
                if url not in seen_urls:
                    seen_urls.add(url)
                    url_list.append(url)

        print(
            f"  FRAMES corpus: fetching Wikipedia text for "
            f"{min(len(url_list), 60)} / {len(url_list)} unique URLs ...",
            flush=True,
        )

        n_docs = 0
        with open(corpus_path, "w") as f:
            for url in url_list[:60]:  # 60 articles is plenty for 131k tokens
                text = _fetch_wikipedia_text(url)
                if text:
                    import json as _json
                    _json.dump({"prompt": text.strip()}, f)
                    f.write("\n")
                    n_docs += 1

        print(f"  FRAMES corpus: {n_docs} Wikipedia articles → {corpus_path}", flush=True)
        if n_docs == 0:
            corpus_path.unlink(missing_ok=True)
            return None
        return corpus_path
    except Exception as exc:
        _DBG_WARN(f"FRAMES corpus build failed ({exc})")
        print(f"  WARNING: FRAMES corpus build failed ({exc})", flush=True)
        if corpus_path.exists() and corpus_path.stat().st_size == 0:
            corpus_path.unlink(missing_ok=True)
        return None


def _find_arxiv_fallback() -> Optional[Path]:
    """Last-resort corpus: discover arxiv-summarization JSONL from previous runs."""
    import glob as _glob

    candidates = sorted(
        _glob.glob("results/*/datasets/ccdv__arxiv-summarization_train_v2.jsonl")
        + _glob.glob("ablation_results/*/datasets/ccdv__arxiv-summarization_train_v2.jsonl")
        + _glob.glob("throughput_results/*/datasets/ccdv__arxiv-summarization_train_v2.jsonl"),
        reverse=True,
    )
    if candidates:
        print(f"  Arxiv fallback corpus: {candidates[0]}", flush=True)
        return Path(candidates[0])
    print("  No local arxiv corpus found — downloading...", flush=True)
    try:
        from ..dataset import prepare_hf_dataset
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp(prefix="agent_bench_corpus_"))
        return prepare_hf_dataset("ccdv/arxiv-summarization", tmp_dir, n_samples=1000)
    except Exception as exc:
        _DBG_ERR(f"arxiv fallback failed ({exc})")
        print(f"  ERROR: arxiv fallback failed ({exc})", flush=True)
        return None
