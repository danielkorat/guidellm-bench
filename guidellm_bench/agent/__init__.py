"""guidellm_bench.agent — Deep-research agent benchmark.

Public API:

    run_agent_bench(out_dir, ...)     — full benchmark (TTFT matrix + ReAct scenarios)
    get_agent_server_config(tp=None)  — vLLM Config for the agent server

Constants:
    AGENT_MODEL, AGENT_TP, AGENT_MAX_MODEL_LEN, AGENT_MAX_BATCHED,
    CONCURRENCY, MATRIX_N_CACHED, MATRIX_N_NEW, AGENT_DATASET, N_AGENT_SCENARIOS

Dataclasses:
    AgentBenchResult, CellResult, ScenarioResult

Sub-modules (internal):
    constants  — constants and dataclasses
    debug      — file-backed debug logger (_DBG*, _setup_debug_log)
    helpers    — low-level vLLM API (tokenize, detokenize, warm_cache, measure_ttft)
    corpus     — Corpus class + FRAMES/arxiv corpus builders
    matrix     — TTFT matrix measurement (measure_cell, run_ttft_matrix)
    scenarios  — ReAct agent loop (run_research_session, run_agent_scenarios_frames)
    run        — top-level entry point
"""

from .constants import (
    AGENT_MODEL,
    AGENT_TP,
    AGENT_MAX_MODEL_LEN,
    AGENT_MAX_BATCHED,
    CONCURRENCY,
    MATRIX_N_CACHED,
    MATRIX_N_NEW,
    AGENT_DATASET,
    N_AGENT_SCENARIOS,
    AgentBenchResult,
    CellResult,
    ScenarioResult,
)
from .run import run_agent_bench, get_agent_server_config

__all__ = [
    # Entry points
    "run_agent_bench",
    "get_agent_server_config",
    # Constants
    "AGENT_MODEL",
    "AGENT_TP",
    "AGENT_MAX_MODEL_LEN",
    "AGENT_MAX_BATCHED",
    "CONCURRENCY",
    "MATRIX_N_CACHED",
    "MATRIX_N_NEW",
    "AGENT_DATASET",
    "N_AGENT_SCENARIOS",
    # Dataclasses
    "AgentBenchResult",
    "CellResult",
    "ScenarioResult",
]
