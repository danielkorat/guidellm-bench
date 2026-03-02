"""AIME 2024 dataset preparation for realistic benchmark prompts."""

import json
from pathlib import Path
from typing import Optional


# v2: column renamed output_tokens → output_tokens_count so guidellm auto-detects
# and maps it to max_tokens in the completions request body.
# Written to the volume-mounted host path so it is visible inside the container
# at /root/aime_2024_v2.jsonl (volume: /root/dkorat/ → /root/).
_CACHE_PATH = Path("/root/dkorat/aime_2024_v2.jsonl")


def prepare_aime_dataset(output_tokens: int = 1024) -> Optional[str]:
    """Download HuggingFaceH4/aime_2024 (30 AIME math problems) to a temp JSONL.

    Each row: {"prompt": "<problem text>", "output_tokens_count": output_tokens}
    Column name 'output_tokens_count' is guidellm's default for output length and
    flows directly to max_tokens in the /v1/completions request body.

    Returns the file path on success, or None on failure (caller falls back to
    synthetic token data transparently).
    """
    if _CACHE_PATH.exists():
        print(f"  AIME dataset: using cached {_CACHE_PATH}", flush=True)
        return str(_CACHE_PATH)

    try:
        from datasets import load_dataset  # type: ignore
        print("  Downloading HuggingFaceH4/aime_2024 (30 problems)...", flush=True)
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        with open(_CACHE_PATH, "w") as f:
            for row in ds:
                json.dump({"prompt": row["problem"], "output_tokens_count": output_tokens}, f)
                f.write("\n")
        print(f"  AIME dataset ready: {len(ds)} problems → {_CACHE_PATH}", flush=True)
        return str(_CACHE_PATH)
    except Exception as e:
        print(
            f"  WARNING: AIME download failed ({e}) — falling back to synthetic data",
            flush=True,
        )
        return None
