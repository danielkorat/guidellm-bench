"""guidellm-bench — Intel XPU vLLM benchmarking package."""

from .config import Config, FULL, SANITY, EAGLE3_SPECULATIVE_CONFIG, PORT, skip_reason
from .docker import ensure_container_running
from .server import start_server, wait_for_server, stop_server, build_vllm_cmd, parse_model_mem_gib, XpuKernelHangError
from .monitor import GpuMonitor
from .dataset import prepare_aime_dataset
from .benchmark import run_guidellm
from .dashboard import build_dashboard_html

__all__ = [
    "Config", "FULL", "SANITY", "EAGLE3_SPECULATIVE_CONFIG", "PORT", "skip_reason",
    "ensure_container_running",
    "start_server", "wait_for_server", "stop_server", "build_vllm_cmd", "parse_model_mem_gib",
    "XpuKernelHangError",
    "GpuMonitor",
    "prepare_aime_dataset",
    "run_guidellm",
    "build_dashboard_html",
]
