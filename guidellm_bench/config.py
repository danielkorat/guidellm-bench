"""Configuration dataclasses, default parameter sets, and skip rules."""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORT = 8000

# Eagle3 speculative decoding config for gpt-oss-120b.
# Draft model always runs at draft_tensor_parallel_size=1 (Eagle3 constraint).
EAGLE3_SPECULATIVE_CONFIG = (
    '{"model": "nvidia/gpt-oss-120b-Eagle3", "num_speculative_tokens": 5,'
    ' "method": "eagle3", "draft_tensor_parallel_size": 1}'
)

# ---------------------------------------------------------------------------
# Default parameter sets
# ---------------------------------------------------------------------------

FULL = dict(
    models=["openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-4B-Thinking-2507"],
    tp=[2, 4],        # tp=8 available via --tp 4 8; tp=2 added to search space
    quant=["none", "fp8"],
    eager=["true"],   # eager=false always skipped (OOM on 20b, negligible gain elsewhere)
    input_len=1024,
    output_len=1024,
    concurrency=16,
    num_prompts=20,   # 20 requests: first 2 (10%) warm-up, last 2 (10%) cool-down excluded
    max_model_len=16384,
    results_dir="./guidellm_results",
    timeout_startup=300,
)

SANITY = dict(
    models=["Qwen/Qwen3-4B-Thinking-2507"],
    tp=[4],
    quant=["fp8"],
    eager=["true"],
    input_len=64,
    output_len=64,
    concurrency=4,
    num_prompts=4,
    max_model_len=2048,
    results_dir="./guidellm_sanity_results",
    timeout_startup=600,
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: str
    tp: int
    quant: Optional[str]
    eager: bool
    speculative_config: Optional[str] = None       # JSON string for --speculative_config
    expert_parallel_size: Optional[int] = None     # --expert-parallel-size N (MoE EP)

    @property
    def name(self) -> str:
        # eager is always True — omit from name to keep it short
        m = self.model.replace("/", "_")
        q = self.quant or "none"
        suffix = "-eagle3" if self.speculative_config else ""
        ep_suffix = f"-ep{self.expert_parallel_size}" if self.expert_parallel_size else ""
        return f"{m}_tp{self.tp}_quant-{q}{suffix}{ep_suffix}"


# ---------------------------------------------------------------------------
# Skip rules
# ---------------------------------------------------------------------------

def skip_reason(model: str, quant: Optional[str], eager: bool, tp: int = 4) -> Optional[str]:
    """Return a human-readable reason to skip this combination, or None to proceed."""
    if quant == "fp8" and not eager:
        return "fp8 + eager=false (known engine failure)"
    if "gpt-oss-20b" in model and quant == "fp8":
        return "gpt-oss-20b + fp8 (mxfp4 config mismatch)"
    if "gpt-oss-20b" in model and tp < 4:
        return "gpt-oss-20b + tp<4 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY — model too large for 2 GPUs)"
    if "gpt-oss-20b" in model and not eager:
        return "gpt-oss-20b + eager=false (OOM: UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)"
    if "Qwen3-30B" in model and quant is None:
        return "Qwen3-30B + no quant (IPEX mode-stack bug)"
    if "Qwen3-4B" in model and quant is None:
        return "Qwen3-4B + no quant (fp8 is uniformly faster: lower TTFT/ITL/lat, higher TPS — verified 20260302)"
    return None
