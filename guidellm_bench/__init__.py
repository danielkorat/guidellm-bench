"""guidellm-bench — Intel XPU vLLM benchmarking package."""

from .config import (
    Config, FULL, SANITY, EAGLE3_SPECULATIVE_CONFIG, EAGLE3_20B_SPECULATIVE_CONFIG,
    PORT, skip_reason, is_moe_model,
    get_ablation_configs, ABLATION_LC_LENGTHS,
    ABLATION_C16_CONCURRENCY, ABLATION_C16_SAMPLES,
    get_throughput_configs,
    THROUGHPUT_INPUT_LENGTHS, THROUGHPUT_OUTPUT_LEN, THROUGHPUT_CONCURRENCIES,
    THROUGHPUT_MAX_MODEL_LEN, THROUGHPUT_MAX_NUM_BATCHED_TOKENS,
    THROUGHPUT_SAMPLES, THROUGHPUT_MAX_SECONDS,
)
from .docker import ensure_container_running
from .server import (
    start_server, wait_for_server, stop_server, build_vllm_cmd,
    parse_model_mem_gib, XpuKernelHangError,
    write_server_status, server_is_reusable, SERVER_STATUS_PATH,
)
from .dataset import (
    prepare_aime_dataset,
    prepare_hf_dataset,
    prepare_long_context_datasets,
    prepare_throughput_dataset,
    LONG_CONTEXT_LENGTHS,
)
from .benchmark import run_guidellm
from .dashboard import build_dashboard_html, build_ablation_dashboard_html, build_throughput_dashboard_html
from .agent_bench import (
    run_agent_bench, get_agent_server_config,
    MATRIX_N_CACHED, MATRIX_N_NEW, AGENT_DATASET, N_AGENT_SCENARIOS,
    AGENT_MAX_MODEL_LEN, AGENT_MAX_BATCHED,
    AgentBenchResult, CellResult, ScenarioResult,
)

__all__ = [
    "Config", "FULL", "SANITY", "EAGLE3_SPECULATIVE_CONFIG", "EAGLE3_20B_SPECULATIVE_CONFIG",
    "PORT", "skip_reason", "is_moe_model",
    "get_ablation_configs", "ABLATION_LC_LENGTHS",
    "ABLATION_C16_CONCURRENCY", "ABLATION_C16_SAMPLES",
    "get_throughput_configs",
    "THROUGHPUT_INPUT_LENGTHS", "THROUGHPUT_OUTPUT_LEN", "THROUGHPUT_CONCURRENCIES",
    "THROUGHPUT_MAX_MODEL_LEN", "THROUGHPUT_MAX_NUM_BATCHED_TOKENS",
    "THROUGHPUT_SAMPLES", "THROUGHPUT_MAX_SECONDS",
    "ensure_container_running",
    "start_server", "wait_for_server", "stop_server", "build_vllm_cmd", "parse_model_mem_gib",
    "XpuKernelHangError", "write_server_status", "server_is_reusable", "SERVER_STATUS_PATH",
    "prepare_aime_dataset", "prepare_hf_dataset", "prepare_long_context_datasets",
    "prepare_throughput_dataset",
    "LONG_CONTEXT_LENGTHS",
    "run_guidellm",
    "build_dashboard_html", "build_ablation_dashboard_html", "build_throughput_dashboard_html",
]
