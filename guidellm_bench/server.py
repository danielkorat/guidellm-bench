"""vLLM server lifecycle: start, health-check, stop."""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .config import Config, PORT


# ---------------------------------------------------------------------------
# Shell script helpers
# ---------------------------------------------------------------------------

def _write_script(path: str, *lines: str) -> None:
    """Write a bash script that sources oneAPI env then runs *lines*."""
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("source /opt/intel/oneapi/setvars.sh --force\n")
        f.write("export no_proxy=localhost,127.0.0.1,0.0.0.0\n")
        f.write("export NO_PROXY=localhost,127.0.0.1,0.0.0.0\n")
        for line in lines:
            f.write(line + "\n")
    os.chmod(path, 0o755)


def _run_tee(cmd: list[str], log_path: Path) -> subprocess.Popen:
    """Start *cmd*, tee-ing stdout+stderr to *log_path* and the terminal."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def _tee():
        with open(log_path, "w") as f:
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()

    threading.Thread(target=_tee, daemon=True).start()
    return proc


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def build_vllm_cmd(cfg: Config, max_model_len: int) -> str:
    """Return the vllm serve shell command for *cfg*."""
    parts = [
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve {cfg.model}",
        "--dtype=bfloat16",
        f"--port {PORT}",
        "--block-size 64",
        "--gpu-memory-util 0.9",
        "--no-enable-prefix-caching",
        "--trust-remote-code",
        "--disable-sliding-window",
        "--disable-log-requests",
        "--max-num-batched-tokens=8192",
        f"--max-model-len {max_model_len}",
        f"-tp={cfg.tp}",
    ]
    if cfg.eager:
        parts.append("--enforce-eager")
    if cfg.quant:
        parts.append(f"--quantization {cfg.quant}")
    if cfg.speculative_config:
        parts.append(f"--speculative_config '{cfg.speculative_config}'")
    return " ".join(parts)


def start_server(cfg: Config, max_model_len: int, log_path: Path) -> subprocess.Popen:
    """Write a launch script and start the vLLM server process."""
    _write_script("/tmp/vllm_server.sh", build_vllm_cmd(cfg, max_model_len))
    return _run_tee(["bash", "--login", "/tmp/vllm_server.sh"], log_path)


def wait_for_server(timeout: int) -> bool:
    """Poll /health until the server responds or *timeout* seconds elapse."""
    print(f"  Waiting for server (timeout={timeout}s)...", flush=True)
    time.sleep(10)

    r = subprocess.run(
        ["bash", "-c", "pgrep -f 'vllm serve' | head -1"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        print("  ERROR: vLLM process not running", flush=True)
        return False

    for elapsed in range(10, timeout, 5):
        try:
            r = subprocess.run(
                ["bash", "-c", f"curl -f -s http://localhost:{PORT}/health"],
                capture_output=True, text=True, timeout=10,
            )
            # /health returns HTTP 200 with an empty body — check returncode only.
            if r.returncode == 0:
                print(f"  Server ready ({elapsed}s)", flush=True)
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
        if elapsed % 60 == 0:
            print(f"  Still waiting... {elapsed}s elapsed", flush=True)

    print(f"  ERROR: server did not become ready within {timeout}s", flush=True)
    return False


def stop_server(proc: Optional[subprocess.Popen] = None) -> None:
    """Kill *proc* (if given) and any lingering vllm / guidellm processes."""
    if proc:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass
    for pat in ("'vllm serve'", "'vllm bench'", "guidellm"):
        subprocess.run(["bash", "-c", f"pkill -f {pat}"], capture_output=True)
    time.sleep(5)
