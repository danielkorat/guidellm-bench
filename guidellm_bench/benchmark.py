"""guidellm benchmark runner."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import Config, PORT
from .docker import HOST_ROOT, docker_exec_cmd, host_to_container

# Output written here by guidellm (container-side path); accessible on the host
# via the volume mount /root/dkorat/ → /root/.
_OUT_TMP_HOST = HOST_ROOT + "/guidellm_out_tmp"         # host path for shutil / Path ops
_OUT_TMP_CONTAINER = "/root/guidellm_out_tmp"           # container path for --output-dir arg
_OUT_TMP = _OUT_TMP_HOST  # backward-compat alias


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
    shutil.rmtree(_OUT_TMP_HOST, ignore_errors=True)
    Path(_OUT_TMP_HOST).mkdir(parents=True, exist_ok=True)

    # ---- data source ------------------------------------------------
    # dataset_path comes from prepare_aime_dataset() as a host path;
    # guidellm runs inside the container, so convert to the container-side path.
    container_dataset_path = host_to_container(dataset_path) if dataset_path else None
    if container_dataset_path:
        data_args = [
            f"--data {container_dataset_path}",
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
            "--max-seconds 600", # hard wall-clock limit per benchmark
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
        f"--output-dir {_OUT_TMP_CONTAINER}",
        "--outputs json",
        "--outputs html",
        "--disable-console-interactive",
    ])
    proc = subprocess.Popen(
        docker_exec_cmd(cmd),
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

    result_file = Path(_OUT_TMP_HOST) / "benchmarks.json"
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
    for src in sorted(Path(_OUT_TMP_HOST).iterdir()):
        dest = out_dir / f"{cfg_name}_{src.name}"
        shutil.copy(src, dest)
        saved.append(dest.name)
    return saved
