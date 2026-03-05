"""guidellm-bench — Intel XPU vLLM benchmarking package."""

from .config import (
    Config, FULL, SANITY, EAGLE3_SPECULATIVE_CONFIG, EAGLE3_20B_SPECULATIVE_CONFIG,
    PORT, skip_reason, is_moe_model,
    get_ablation_configs, ABLATION_LC_LENGTHS,
    ABLATION_C16_CONCURRENCY, ABLATION_C16_SAMPLES,
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
    LONG_CONTEXT_LENGTHS,
)
from .benchmark import run_guidellm
from .dashboard import build_dashboard_html, build_ablation_dashboard_html

__all__ = [
    "Config", "FULL", "SANITY", "EAGLE3_SPECULATIVE_CONFIG", "EAGLE3_20B_SPECULATIVE_CONFIG",
    "PORT", "skip_reason", "is_moe_model",
    "get_ablation_configs", "ABLATION_LC_LENGTHS",
    "ABLATION_C16_CONCURRENCY", "ABLATION_C16_SAMPLES",
    "ensure_container_running",
    "start_server", "wait_for_server", "stop_server", "build_vllm_cmd", "parse_model_mem_gib",
    "XpuKernelHangError", "write_server_status", "server_is_reusable", "SERVER_STATUS_PATH",
    "prepare_aime_dataset", "prepare_hf_dataset", "prepare_long_context_datasets",
    "LONG_CONTEXT_LENGTHS",
    "run_guidellm",
    "build_dashboard_html", "build_ablation_dashboard_html",
]
