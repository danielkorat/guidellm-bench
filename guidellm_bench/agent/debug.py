"""File-backed debug logger for agent benchmarks.

All _DBG* helpers write to a per-run log file (agent_debug.log) set up by
_setup_debug_log().  _DBG_INFO/WARN/ERR also print to stdout/stderr.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_dbg: Optional[logging.Logger] = None


def _setup_debug_log(log_path: Path) -> logging.Logger:
    """Create (or reopen) the file debug logger for this run.

    Logs DEBUG-and-above to *log_path*.  Does NOT print to stdout —
    user-facing summaries still use print().
    """
    global _dbg
    logger = logging.getLogger("agent_bench")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # prevent duplicate handlers on re-run
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d  %(levelname)-5s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(fh)
    _dbg = logger
    return logger


def _DBG(msg: str) -> None:
    """Write a DEBUG line to the run log (no-op if logger not initialised)."""
    if _dbg is not None:
        _dbg.debug(msg)


def _DBG_INFO(msg: str) -> None:
    """Write an INFO line to the run log AND stdout."""
    if _dbg is not None:
        _dbg.info(msg)
    print(msg, flush=True)


def _DBG_WARN(msg: str) -> None:
    """Write a WARNING line to the run log AND stdout."""
    if _dbg is not None:
        _dbg.warning(msg)
    print(f"WARNING: {msg}", flush=True)


def _DBG_ERR(msg: str) -> None:
    """Write an ERROR line to the run log AND stderr."""
    if _dbg is not None:
        _dbg.error(msg)
    print(f"ERROR: {msg}", flush=True, file=sys.stderr)
