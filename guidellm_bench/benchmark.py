"""guidellm benchmark runner.

Runs inside the intel/vllm:0.14.1-xpu container. All paths are
container-native; no docker exec wrapping needed.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import Config, PORT
from .docker import _PREAMBLE

# Temp directory for guidellm output files (ephemeral per benchmark run).
_OUT_TMP = Path("/tmp/guidellm_out_tmp")


def run_guidellm(
    cfg: Config,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    log_path: Path,
    sweep: bool = True,
    dataset_path: Optional[str] = None,
) -> Optional[dict]:
    """Run a guidellm benchmark and return the parsed JSON result dict.

    Args:
        cfg:          Target configuration.
        input_len:    Synthetic prompt token length (ignored when dataset_path set).
        output_len:   Synthetic output token count (ignored when dataset_path set).
        concurrency:  Number of concurrent requests (sanity mode).
        num_prompts:  Request count (sanity mode).
        log_path:     File to capture stdout/stderr.
        sweep:        True → synchronous profile with warmup/cooldown (full runs).
                      False → concurrent profile, single rate (sanity).
        dataset_path: Path to AIME JSONL file; None → synthetic data.

    Returns:
        Parsed benchmarks.json dict, or None on failure.
    """
    shutil.rmtree(_OUT_TMP, ignore_errors=True)
    _OUT_TMP.mkdir(parents=True, exist_ok=True)

    # ---- data source ------------------------------------------------
    # dataset_path is a container-native path from prepare_aime_dataset().
    if dataset_path:
        data_args = [
            f"--data {dataset_path}",
            # 'prompt' and 'output_tokens_count' are guidellm defaults — no mapper needed.
            # output_tokens_count maps to max_tokens in the completions request.
            "--data-samples -1",
        ]
        effective_requests = 30  # all AIME problems
    else:
        data_args = [f"--data 'prompt_tokens={input_len},output_tokens={output_len}'"]
        effective_requests = num_prompts

    # ---- profile / limits -------------------------------------------
    if sweep:
        profile_args = [
            "--profile synchronous",
            f"--max-requests {effective_requests}",
            "--warmup 0.1",      # exclude first ~10% of requests (XPU JIT spike)
            "--cooldown 0.1",    # exclude last  ~10% of requests (tail effects)
            "--max-errors 5",    # abort early on repeated failures
            "--max-seconds 900", # hard wall-clock limit per benchmark (Qwen3-30B needs ~18s/req × 30)
        ]
    else:
        profile_args = [
            "--profile concurrent",
            f"--rate {concurrency}",
            f"--max-requests {num_prompts}",
        ]

    cmd = " ".join([
        "guidellm benchmark run",
        f"--target http://0.0.0.0:{PORT}",
        f"--model {cfg.model}",
        *data_args,
        "--request-format /v1/completions",  # bypass chat template (critical for thinking models)
        *profile_args,
        f"--output-dir {_OUT_TMP}",
        "--outputs json",
        "--outputs html",
        "--disable-console-interactive",
    ])
    proc = subprocess.Popen(
        ["bash", "--login", "-c", f"{_PREAMBLE} && {cmd}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    with open(log_path, "w") as f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
    proc.wait()

    if proc.returncode != 0:
        print(f"  guidellm exited with code {proc.returncode}", flush=True)
        return None

    result_file = _OUT_TMP / "benchmarks.json"
    if not result_file.exists():
        print("  guidellm result file not found", flush=True)
        return None

    with open(result_file) as f:
        return json.load(f)


def copy_results(cfg_name: str, out_dir: Path) -> list[str]:
    """Copy all guidellm output files from the temp dir to *out_dir*.

    Returns list of saved filenames.
    """
    saved = []
    for src in sorted(_OUT_TMP.iterdir()):
        dest = out_dir / f"{cfg_name}_{src.name}"
        shutil.copy(src, dest)
        saved.append(dest.name)
    return saved
