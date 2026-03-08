"""Backward-compatibility shim.

All public symbols have moved to guidellm_bench.agent (the agent/ subpackage).
This file exists only so that existing imports of the form
    import guidellm_bench.agent_bench as ab
continue to work without changes.

New code should import directly from guidellm_bench.agent.
"""
from guidellm_bench.agent import *                          # noqa: F401, F403
from guidellm_bench.agent import (                          # noqa: F401
    run_agent_bench, get_agent_server_config,
    AGENT_MODEL, AGENT_TP, AGENT_MAX_MODEL_LEN, AGENT_MAX_BATCHED,
    CONCURRENCY, MATRIX_N_CACHED, MATRIX_N_NEW,
    AGENT_DATASET, N_AGENT_SCENARIOS,
    AgentBenchResult, CellResult, ScenarioResult,
)
from guidellm_bench.agent.corpus import Corpus, _prepare_frames_corpus, _find_arxiv_fallback  # noqa: F401
from guidellm_bench.agent.matrix import measure_cell, run_ttft_matrix, print_ttft_table  # noqa: F401
from guidellm_bench.agent.scenarios import (                # noqa: F401
    run_research_session, run_agent_scenarios_frames, print_scenario_summary,
)
from guidellm_bench.agent.helpers import (                  # noqa: F401
    make_session as _make_session, _tokenize, _detokenize,
    _verify_token_count, _warm_cache, _measure_ttft,
)
from guidellm_bench.agent.debug import (                    # noqa: F401
    _setup_debug_log, _DBG, _DBG_INFO, _DBG_WARN, _DBG_ERR,
)
