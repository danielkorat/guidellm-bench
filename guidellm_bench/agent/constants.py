"""Agent benchmark constants, configuration, and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import PORT

# ---------------------------------------------------------------------------
# Server defaults
# ---------------------------------------------------------------------------

AGENT_MODEL         = "openai/gpt-oss-20b"
AGENT_TP            = 8                # default tensor-parallelism (override with --agent-tp 4)

# Context window
# max_model_len = 131_072 — works because max_num_batched_tokens is kept at 8_192.
# vLLM's warm-up dummy run uses max_num_batched_tokens tokens as the input size; at
# 8_192 the Triton kernel fits within Intel XPU's 256KB PTSS limit.  Actual 131k prompts
# are served via chunked prefill (16 passes of 8192 tokens each) — no kernel change.
#
# Past failures (now fixed):
#   AGENT_MAX_BATCHED = 131_072  →  warm-up uses 131k kernel → ZE_RESULT_ERROR_MODULE_BUILD_FAILURE
#   AGENT_MAX_BATCHED = 32_768   →  same crash at 32k kernel
#   AGENT_MAX_BATCHED = 8_192   →  ✓ warm-up uses 8k kernel (fits in PTSS 256KB)
AGENT_MAX_MODEL_LEN = 131_072          # full 131k context window
AGENT_MAX_BATCHED   = 8_192            # keep warm-up kernel ≤ 256KB PTSS on Intel XPU

# ---------------------------------------------------------------------------
# Measurement parameters
# ---------------------------------------------------------------------------

CONCURRENCY             = 1            # demo: always single-request serial (concurrency=1)
N_WARMUPS               = 3            # warm-up requests discarded before each cell
N_SAMPLES               = 15           # measured samples per cell
CV_RERUN_THRESHOLD      = 0.35         # re-run cell if stddev/median > this
OUTPUT_TOKENS_DEFAULT   = 256          # generation length during matrix cell measurements
OUTPUT_TOKENS_SCENARIO  = 200          # max tokens per agent turn (JSON action is short)
INTER_REQUEST_SLEEP_S   = 0.5          # avoid thermal-throttle artefacts

# ---------------------------------------------------------------------------
# TTFT matrix axes (tokens)
# max N_cached + max N_new = 114_688 + 16_384 = 131_072 = AGENT_MAX_MODEL_LEN (exact fit)
# ---------------------------------------------------------------------------

MATRIX_N_CACHED = [0, 8_192, 32_768, 65_536, 98_304, 114_688]  # 0/8k/32k/64k/96k/112k
MATRIX_N_NEW    = [1_024, 4_096, 8_192, 16_384]                 # 1k/4k/8k/16k

# ---------------------------------------------------------------------------
# FRAMES dataset
# ---------------------------------------------------------------------------

AGENT_DATASET     = "google/frames-benchmark"   # 824 multi-hop questions + Wikipedia gold docs
N_AGENT_SCENARIOS = 4

AGENT_SYSTEM_PROMPT = (
    "You are a deep research assistant. Answer complex multi-hop research questions "
    "by searching for information iteratively.\n"
    "For each response, output EXACTLY ONE JSON object (no surrounding text):\n"
    '  To retrieve a document: {"action": "search", "query": "your specific search query"}\n'
    '  To give your final answer: {"action": "answer", "text": "your comprehensive answer"}\n'
    "Issue multiple searches before answering. Be specific in your search queries."
)

# Internal — base URL for vLLM API calls
_BASE_URL = f"http://localhost:{PORT}"

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    n_cached:      int
    n_new:         int
    actual_cached: int
    actual_new:    int
    n_samples:     int
    ttft_ms_values:          list[float] = field(default_factory=list)
    warmup_ttft_ms_values:   list[float] = field(default_factory=list)
    total_request_ms_values: list[float] = field(default_factory=list)
    warm_cache_ms_values:    list[float] = field(default_factory=list)
    ttft_median:         float = 0.0
    ttft_p25:            float = 0.0
    ttft_p75:            float = 0.0
    ttft_p95:            float = 0.0
    ttft_min:            float = 0.0
    ttft_max:            float = 0.0
    ttft_cv:             float = 0.0
    cache_hit_ratio:     float = 0.0
    cold_ttft_estimate:  float = 0.0


@dataclass
class ScenarioResult:
    name:        str
    description: str
    n_calls:     int
    iters: list[dict] = field(default_factory=list)
    ttft_median_first: float = 0.0
    ttft_median_last:  float = 0.0
    ttft_median_all:   float = 0.0
    total_context_k:   float = 0.0


@dataclass
class AgentBenchResult:
    run_timestamp:   str
    model:           str
    tp:              int
    prefix_caching:  bool
    matrix:    list[CellResult]    = field(default_factory=list)
    scenarios: list[ScenarioResult] = field(default_factory=list)
