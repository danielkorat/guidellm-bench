"""Dataset preparation for realistic benchmark prompts.

Supports:
  - AIME 2024 (default, 30 math problems)
  - Any HuggingFace dataset via ``prepare_hf_dataset(hf_name, ...)``
  - Long-context slices via ``prepare_long_context_datasets(...)``
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# AIME 2024 (original default)
# ---------------------------------------------------------------------------

# v2: column renamed output_tokens → output_tokens_count so guidellm auto-detects
# and maps it to max_tokens in the completions request body.
# Path is container-native; /root/ is volume-backed (host: /root/dkorat/).
_AIME_CACHE_PATH = Path("/root/aime_2024_v2.jsonl")


def prepare_aime_dataset(output_tokens: int = 1024) -> Optional[str]:
    """Download HuggingFaceH4/aime_2024 (30 AIME math problems) to a JSONL file.

    Each row: {"prompt": "<problem text>", "output_tokens_count": output_tokens}
    'output_tokens_count' is the guidellm default for output length and maps to
    max_tokens in the /v1/completions request body.

    Returns the file path on success, or None on failure (caller falls back to
    synthetic data).
    """
    if _AIME_CACHE_PATH.exists():
        print(f"  AIME dataset: using cached {_AIME_CACHE_PATH}", flush=True)
        return str(_AIME_CACHE_PATH)

    try:
        from datasets import load_dataset  # type: ignore
        print("  Downloading HuggingFaceH4/aime_2024 (30 problems)...", flush=True)
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        with open(_AIME_CACHE_PATH, "w") as f:
            for row in ds:
                json.dump({"prompt": row["problem"], "output_tokens_count": output_tokens}, f)
                f.write("\n")
        print(f"  AIME dataset ready: {len(ds)} problems → {_AIME_CACHE_PATH}", flush=True)
        return str(_AIME_CACHE_PATH)
    except Exception as e:
        print(
            f"  WARNING: AIME download failed ({e}) — falling back to synthetic data",
            flush=True,
        )
        return None


# ---------------------------------------------------------------------------
# Generic HF dataset loader (any dataset, auto-detects text column)
# ---------------------------------------------------------------------------

# Ordered priority list of column names likely to contain the prompt text.
_TEXT_COLUMN_CANDIDATES = [
    "prompt", "text", "input", "content", "instruction",
    "query", "question", "document", "context", "passage",
    "problem", "task", "message", "conversation",
]


def _detect_text_column(dataset) -> Optional[str]:
    """Return the best text column from a HuggingFace dataset row sample.

    Strategy:
      1. Match against priority name list (first wins).
      2. Fall back to the string column with the longest median text length.
    """
    first_row = dataset[0] if len(dataset) > 0 else {}

    # Priority name match (exact, case-insensitive)
    col_lower = {k.lower(): k for k in first_row.keys()}
    for candidate in _TEXT_COLUMN_CANDIDATES:
        if candidate in col_lower:
            chosen = col_lower[candidate]
            print(f"  Auto-detected text column: '{chosen}' (priority match)", flush=True)
            return chosen

    # Heuristic: find string columns, sample up to 50 rows, pick longest median
    string_cols = [k for k, v in first_row.items() if isinstance(v, str)]
    if not string_cols:
        print("  WARNING: No string columns found in dataset", flush=True)
        return None

    sample_size = min(50, len(dataset))
    sample = dataset.select(range(sample_size))
    best_col, best_med = None, 0.0
    for col in string_cols:
        texts = [row[col] or "" for row in sample]
        lengths = sorted(len(t) for t in texts)
        med = lengths[len(lengths) // 2] if lengths else 0.0
        if med > best_med:
            best_med, best_col = med, col

    print(
        f"  Auto-detected text column: '{best_col}' "
        f"(longest median text: {best_med:.0f} chars)",
        flush=True,
    )
    return best_col


def prepare_hf_dataset(
    hf_name: str,
    split: str = "train",
    text_column: Optional[str] = None,
    output_tokens: int = 1024,
    max_samples: int = 100,
    cache_dir: Optional[Path] = None,
) -> Optional[str]:
    """Download any HuggingFace dataset and convert to guidellm JSONL.

    Args:
        hf_name:      HuggingFace dataset id (e.g. 'cx-cmu/deepresearchgym-agentic-search-logs').
        split:        Dataset split to use (default: 'train').
        text_column:  Column name for the prompt text. Auto-detected if None.
        output_tokens: Value written to 'output_tokens_count' in each JSONL row.
        max_samples:  Maximum number of samples to include.
        cache_dir:    Directory to cache the JSONL file. Defaults to /root/.

    Returns:
        Path to the JSONL file, or None on failure.
    """
    if cache_dir is None:
        cache_dir = Path("/root")
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_name = hf_name.replace("/", "__")
    cache_path = cache_dir / f"{safe_name}_{split}_v1.jsonl"

    if cache_path.exists():
        print(f"  HF dataset: using cached {cache_path}", flush=True)
        return str(cache_path)

    try:
        from datasets import load_dataset  # type: ignore

        print(f"  Downloading HF dataset '{hf_name}' split='{split}'...", flush=True)
        ds = load_dataset(hf_name, split=split)
        n_orig = len(ds)

        col = text_column or _detect_text_column(ds)
        if col is None:
            print(f"  WARNING: Cannot detect text column in '{hf_name}'", flush=True)
            return None

        # Filter out short / empty entries (fewer than 50 chars)
        ds = ds.filter(lambda row: bool(row.get(col, "")) and len(row[col]) >= 50)
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        with open(cache_path, "w") as f:
            for row in ds:
                json.dump({"prompt": row[col], "output_tokens_count": output_tokens}, f)
                f.write("\n")

        print(
            f"  HF dataset ready: {len(ds)}/{n_orig} samples → {cache_path} "
            f"(col='{col}')",
            flush=True,
        )
        return str(cache_path)

    except Exception as e:
        print(
            f"  WARNING: HF dataset '{hf_name}' download failed ({e}) — "
            "falling back to synthetic data",
            flush=True,
        )
        return None


# ---------------------------------------------------------------------------
# Long-context dataset slicing
# ---------------------------------------------------------------------------

# Input-length targets for --long-contexts mode (in approximate tokens)
LONG_CONTEXT_LENGTHS: List[int] = [1024, 4096, 8192, 16384]

_WORDS_PER_TOKEN = 0.75  # conservative: 1 token ≈ 1.33 words → words ≈ tokens × 0.75


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return text truncated to approximately *max_tokens* tokens.

    Uses simple whitespace-split word count: words × _WORDS_PER_TOKEN ≈ tokens.
    This is language-agnostic and has no external dependencies.
    """
    max_words = int(max_tokens * _WORDS_PER_TOKEN)
    words = text.split()
    return " ".join(words[:max_words])


def prepare_long_context_datasets(
    source_path: str,
    token_lengths: Sequence[int] = LONG_CONTEXT_LENGTHS,
    num_samples: int = 10,
    output_tokens: int = 512,
    cache_dir: Optional[Path] = None,
) -> Dict[int, Optional[str]]:
    """Slice a source JSONL into multiple JSONL files, one per token-length target.

    Reads *source_path* (a JSONL with 'prompt' column), filters prompts long
    enough for each target, truncates, and writes separate JSONL files.

    Args:
        source_path:   Path to source JSONL (must have 'prompt' column).
        token_lengths: Target input lengths in tokens.
        num_samples:   Number of samples per length slice.
        output_tokens: 'output_tokens_count' value written to each row.
        cache_dir:     Directory to write lc_*.jsonl files.

    Returns:
        Dict mapping each token length to its JSONL path (or None on failure).
    """
    if cache_dir is None:
        cache_dir = Path("/root")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Read source prompts once
    try:
        source_rows = []
        with open(source_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if row.get("prompt"):
                        source_rows.append(row["prompt"])
    except Exception as e:
        print(f"  WARNING: Cannot read long-context source '{source_path}': {e}", flush=True)
        return {k: None for k in token_lengths}

    results: Dict[int, Optional[str]] = {}
    for tlen in token_lengths:
        label = f"{tlen // 1024}k" if tlen >= 1024 else str(tlen)
        # Determine the source basename for a stable cache filename
        src_stem = Path(source_path).stem
        out_path = cache_dir / f"lc_{src_stem}_{label}.jsonl"

        if out_path.exists():
            print(f"  Long-context dataset lc_{label}: using cached {out_path}", flush=True)
            results[tlen] = str(out_path)
            continue

        # Filter and truncate
        min_words = int(tlen * _WORDS_PER_TOKEN * 0.5)  # at least 50% of target length
        eligible = [t for t in source_rows if len(t.split()) >= min_words]

        if not eligible:
            print(
                f"  WARNING: No prompts long enough for lc_{label} "
                f"(need ≥{min_words} words, source has {len(source_rows)} rows) — skipping",
                flush=True,
            )
            results[tlen] = None
            continue

        selected = eligible[:num_samples]
        written = 0
        with open(out_path, "w") as f:
            for text in selected:
                truncated = _truncate_to_tokens(text, tlen)
                json.dump({"prompt": truncated, "output_tokens_count": output_tokens}, f)
                f.write("\n")
                written += 1

        print(
            f"  Long-context dataset lc_{label}: {written} samples → {out_path}",
            flush=True,
        )
        results[tlen] = str(out_path)

    return results
