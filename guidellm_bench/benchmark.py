"""guidellm benchmark runner.

Runs inside lsv-container (intel/llm-scaler-vllm:0.14.0-b8). All paths are
container-native; no docker exec wrapping needed.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import Config, PORT
from .docker import _PREAMBLE

def run_guidellm(
    cfg: Config,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    log_path: Path,
    sweep: bool = True,
    dataset_path: Optional[str] = None,
    lc_mode: bool = False,
    data_samples: int = -1,
    max_seconds: int = 900,
    num_requests_override: Optional[int] = None,
) -> Optional[dict]:
    """Run a guidellm benchmark and return the parsed JSON result dict.

    Args:
        cfg:                   Target configuration.
        input_len:             Synthetic prompt token length (ignored when dataset_path set).
        output_len:            Synthetic output token count (ignored when dataset_path set).
        concurrency:           Number of concurrent requests (sanity mode).
        num_prompts:           Request count (sanity mode).
        log_path:              File to capture stdout/stderr.
        sweep:                 True → synchronous profile with warmup/cooldown (full runs).
                               False → concurrent profile, single rate (sanity/throughput).
        dataset_path:          Path to JSONL file; None → synthetic data.
        lc_mode:               True → long-context slice (synchronous, no warmup/cooldown).
                               Overrides sweep and num_prompts for the slice.
        data_samples:          --data-samples value (-1 = all rows).  Pass the exact
                               request count to avoid reading more rows than needed
                               (e.g. 256-row file but only 32 needed at c=16).
        max_seconds:           --max-seconds wall-clock budget per run.  Default 900s;
                               raise to 10800 for long-context throughput cells.
        num_requests_override: Override the effective request count in lc_mode
                               (default 10).  Used by the throughput study for c=1
                               serial runs with a custom sample count.

    Returns:
        Parsed benchmarks.json dict, or None on failure.
    """
    # Scratch dir for guidellm output files — kept inside the run's log dir,
    # never in /tmp, so results stay with the run even on unexpected exit.
    out_tmp = log_path.parent / ".guidellm_out"
    shutil.rmtree(out_tmp, ignore_errors=True)
    out_tmp.mkdir(parents=True, exist_ok=True)

    # ---- data source ------------------------------------------------
    # dataset_path is a container-native path from prepare_aime_dataset().
    if dataset_path:
        data_args = [
            f"--data {dataset_path}",
            # 'prompt' and 'output_tokens_count' are guidellm defaults — no mapper needed.
            # output_tokens_count maps to max_tokens in the completions request.
            f"--data-samples {data_samples}",
        ]
        effective_requests = num_requests_override or (10 if lc_mode else 30)
    else:
        data_args = [f"--data 'prompt_tokens={input_len},output_tokens={output_len}'"]
        effective_requests = num_prompts

    # ---- profile / limits -------------------------------------------
    if lc_mode:
        # Long-context slice: serial, no warmup/cooldown (small sample count)
        profile_args = [
            "--profile synchronous",
            f"--max-requests {effective_requests}",
            "--max-errors 5",
            f"--max-seconds {max_seconds}",
        ]
    elif sweep:
        profile_args = [
            "--profile synchronous",
            f"--max-requests {effective_requests}",
            "--warmup 0.1",      # exclude first ~10% of requests (XPU JIT spike)
            "--cooldown 0.1",    # exclude last  ~10% of requests (tail effects)
            "--max-errors 5",    # abort early on repeated failures
            f"--max-seconds {max_seconds}", # hard wall-clock limit per benchmark
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
        f"--output-dir {out_tmp}",
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

    result_file = out_tmp / "benchmarks.json"
    if not result_file.exists():
        print("  guidellm result file not found", flush=True)
        return None

    with open(result_file) as f:
        return json.load(f)


def copy_results(cfg_name: str, out_dir: Path, out_tmp: Path) -> list[str]:
    """Copy all guidellm output files from *out_tmp* to *out_dir*.

    Returns list of saved filenames.
    """
    saved = []
    for src in sorted(out_tmp.iterdir()):
        dest = out_dir / f"{cfg_name}_{src.name}"
        shutil.copy(src, dest)
        saved.append(dest.name)
    return saved
