"""Configuration dataclasses, default parameter sets, and skip rules."""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORT = 8000

# Models that support Expert Parallelism (MoE architecture).
# Only these models should be used with --ep / expert_parallel_size.
_MOE_MODELS: frozenset[str] = frozenset({
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-30B-A3B",
})


def is_moe_model(model: str) -> bool:
    """Return True if *model* has a Mixture-of-Experts architecture (supports EP)."""
    return model in _MOE_MODELS

# Eagle3 speculative decoding config for gpt-oss-120b.
# Draft model always runs at draft_tensor_parallel_size=1 (Eagle3 constraint).
EAGLE3_SPECULATIVE_CONFIG = (
    '{"model": "nvidia/gpt-oss-120b-Eagle3", "num_speculative_tokens": 5,'
    ' "method": "eagle3", "draft_tensor_parallel_size": 1}'
)

# Eagle3 speculative decoding config for gpt-oss-20b.
# Model: RedHatAI/gpt-oss-20b-speculator.eagle3
# num_speculative_tokens=3 per model card (not 5 — smaller draft model).
EAGLE3_20B_SPECULATIVE_CONFIG = (
    '{"model": "RedHatAI/gpt-oss-20b-speculator.eagle3", "num_speculative_tokens": 3,'
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
    results_dir="./results",
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
    results_dir="./sanity_results",
    timeout_startup=600,
)

# ---------------------------------------------------------------------------
# Ablation study: optimal vLLM configuration for gpt-oss-20b on Intel XPU
# ---------------------------------------------------------------------------
#
# Research findings (Intel docs / blog, vLLM 0.14.0-b8 XPU):
#  • gpt-oss-20b has MXFP4 *baked in* → omit --quantization; quant=None IS MXFP4.
#    Passing --quantization fp8 is rejected (skip rule). mxfp4 is the only valid quant.
#  • Requires tp >= 4 (OOM with tp=2 on our hardware). tp=2 is intentionally omitted.
#  • Intel-recommended server flags (already hardcoded in build_vllm_cmd):
#      --block-size 64, --gpu-memory-util 0.9, --no-enable-prefix-caching,
#      --max-num-batched-tokens=8192
#  • Remaining ablation dimensions:
#      (a) Expert Parallelism (EP) — --enable-expert-parallel
#      (b) Tensor parallelism: tp=4 vs tp=8
#      (c) Async scheduling — --async-scheduling (Intel 0.14.1-xpu: reduces CPU overhead)
#      (d) Prefix caching — enable to test whether it helps long-context TTFT
#
# Ablation LC lengths: [1k, 2k, 4k, 8k] (shorter than full LC run, 5 samples each)
# Results dir: ablation_results/YYYYMMDD_HHMM/

ABLATION_CONFIGS: list = []  # populated lazily via get_ablation_configs() below


def get_ablation_configs() -> list:
    """Return the ablation config matrix for gpt-oss-20b on Intel XPU."""
    return [
        # 1. Baseline: tp=4, no EP, Intel defaults (MXFP4 native, eager=True)
        Config(model="openai/gpt-oss-20b", tp=4, quant=None, eager=True),

        # 2. Expert Parallelism ON (tp=4): experts distributed across 4 GPUs
        #    vLLM flag: --enable-expert-parallel (no size argument on this build)
        Config(model="openai/gpt-oss-20b", tp=4, quant=None, eager=True, expert_parallel_size=4),

        # 3. TP=8: tensor parallelism scaled to 8 GPUs (if available on the system)
        #    May fail at server startup if only 4 GPUs present — skipped gracefully
        Config(model="openai/gpt-oss-20b", tp=8, quant=None, eager=True),

        # 4. TP=8 + EP: combine both parallelism strategies on 8 GPUs
        Config(model="openai/gpt-oss-20b", tp=8, quant=None, eager=True, expert_parallel_size=8),

        # 5. Async scheduling (Intel 0.14.1-xpu): overlaps scheduling with model execution
        #    Intel docs: "may help reduce the CPU overheads, leading to better latency"
        Config(model="openai/gpt-oss-20b", tp=4, quant=None, eager=True, async_scheduling=True),

        # 6. Prefix caching enabled: test whether KV-cache reuse helps LC TTFT
        #    Default is disabled (--no-enable-prefix-caching); this removes that flag
        Config(model="openai/gpt-oss-20b", tp=4, quant=None, eager=True, prefix_caching=True),

        # 7. TP=2 with reduced max_model_len=8192: test whether the blog's tp=1 result
        #    (vllm 0.10.2-xpu) can be reproduced at tp=2 with a smaller context window.
        #    LC runs capped at 4k input (4096 + 512 output = 4608 < 8192).
        #    Uses max_model_len_override=8192 to stay within 2-GPU memory budget.
        Config(model="openai/gpt-oss-20b", tp=2, quant=None, eager=True,
               max_model_len_override=8192),

        # 7. Combined best (tp=4): prefix caching + async scheduling
        #    The recommended production command — stacks both free/cheap optimisations.
        Config(model="openai/gpt-oss-20b", tp=4, quant=None, eager=True,
               prefix_caching=True, async_scheduling=True),

        # 8. Combined tp=8: tp=8 + prefix caching + async scheduling
        #    Alternative for latency-critical deployments; 2× GPU cost.
        Config(model="openai/gpt-oss-20b", tp=8, quant=None, eager=True,
               prefix_caching=True, async_scheduling=True),

        # 9. Eagle3 speculative decoding (tp=4): DISABLED — XPU hardware limit
        #    sample_recovered_tokens_kernel requires 292KB PTSS; XPU max is 256KB.
        #    ZE_RESULT_ERROR_MODULE_BUILD_FAILURE on intel/llm-scaler-vllm:0.14.0-b8.
        #    Kept commented out for reference; do not re-enable without driver upgrade.
        # Config(model="openai/gpt-oss-20b", tp=4, quant=None, eager=True,
        #        speculative_config=EAGLE3_20B_SPECULATIVE_CONFIG),
    ]


# LC input-length sweep for the ablation study (shorter than full LC run)
# 1k / 2k / 4k / 8k — 16k is excluded to keep total run time manageable
ABLATION_LC_LENGTHS: list = [1024, 2048, 4096, 8192]


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
    async_scheduling: bool = False                  # --async-scheduling (Intel XPU optimisation)
    prefix_caching: bool = False                    # enable prefix caching (default: off via --no-enable-prefix-caching)
    max_model_len_override: Optional[int] = None   # per-config --max-model-len override (ablation only; e.g. tp=2 needs smaller value)

    @property
    def name(self) -> str:
        # eager is always True — omit from name to keep it short
        m = self.model.replace("/", "_")
        q = self.quant or "none"
        suffix = "-eagle3" if self.speculative_config else ""
        ep_suffix = "-ep" if self.expert_parallel_size else ""
        async_suffix = "-async" if self.async_scheduling else ""
        pc_suffix = "-pc" if self.prefix_caching else ""
        return f"{m}_tp{self.tp}_quant-{q}{suffix}{ep_suffix}{async_suffix}{pc_suffix}"


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
