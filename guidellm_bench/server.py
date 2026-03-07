"""vLLM server lifecycle: start, health-check, stop.

Runs inside lsv-container (intel/llm-scaler-vllm:0.14.0-b8). Subprocesses are called
directly (no docker exec wrapper); oneAPI is sourced via bash --login -c.
"""

import json
import os
import re as _re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .config import Config, PORT
from .docker import _PREAMBLE

# Path where the currently-running server's config + PID are persisted.
# bench.py writes this after a successful startup and reads it to decide
# whether to reuse the running server rather than restart it.
SERVER_STATUS_PATH = Path("/root/guidellm-bench/server_status.json")

# ---------------------------------------------------------------------------
# XPU kernel hang detection
# ---------------------------------------------------------------------------

# When IPEX overrides PyTorch XPU kernels AND the server never becomes healthy,
# the internal XPU driver is in a corrupted state that only a container
# recreation can fix.  We detect this by looking for BOTH conditions:
#   1. OperatorEntry.cpp:208 appears (IPEX kernel override — always present even
#      in normal starts, so alone it is NOT sufficient to declare a hang).
#   2. No EngineCore or Worker_TP lines — workers never initialised.
# Condition 2 is the true discriminator: in a real startup, all tp workers
# emit OperatorEntry warnings and EngineCore lines within ~60s of launch.
_HANG_PATTERN     = _re.compile(r"OperatorEntry\.cpp:208")
_WORKER_PATTERN   = _re.compile(r"EngineCore|Worker_TP")
_HANG_DETECT_AFTER_S = 300  # seconds before checking — gpt-oss-20b tp=8 can take ~150s


class XpuKernelHangError(RuntimeError):
    """Raised when an XPU kernel-registration hang is detected.

    This means the XPU driver inside the container is corrupted.  The only
    recovery is to remove and recreate the container.  bench.py exits with
    code 42 so the host-side re-exec guard can perform the recovery and
    resume automatically.
    """


def _log_has_xpu_hang(log_path: Optional[Path]) -> bool:
    """Return True iff the server log shows the XPU kernel-registration hang.

    A genuine hang has TWO simultaneous symptoms:
      - OperatorEntry.cpp:208 appears (IPEX kernel override — emitted by the
        main process within the first ~5 s of ANY startup, so alone it is
        not a reliable signal).
      - No EngineCore or Worker_TP lines — worker sub-processes never started.

    If workers have started (EngineCore lines present) the server is loading
    normally and we must keep waiting, regardless of how long it takes.
    """
    if log_path is None or not log_path.exists():
        return False
    try:
        text = log_path.read_text()
        if not _HANG_PATTERN.search(text):
            return False          # OperatorEntry warning not yet emitted
        if _WORKER_PATTERN.search(text):
            return False          # workers started — legitimate slow startup
        return True               # OperatorEntry present, workers never started
    except OSError:
        return False


def _log_has_startup_complete(log_path: Optional[Path]) -> bool:
    """Return True if the server log contains 'Application startup complete'.

    This confirms that uvicorn has finished initialising — the server is about
    to respond to /health.  If this message is present, the server is NOT hung
    and we must keep polling rather than raising XpuKernelHangError.
    """
    if log_path is None or not log_path.exists():
        return False
    try:
        return "Application startup complete" in log_path.read_text()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Server status persistence (server_status.json)
# ---------------------------------------------------------------------------

def _cfg_to_status_key(cfg: Config, max_model_len: int) -> dict:
    """Return the subset of fields used to decide if a server is reusable."""
    return {
        "model": cfg.model,
        "tp": cfg.tp,
        "quant": cfg.quant,
        "eager": cfg.eager,
        "expert_parallel_size": cfg.expert_parallel_size,
        "speculative_config": cfg.speculative_config,
        "async_scheduling": cfg.async_scheduling,
        "prefix_caching": cfg.prefix_caching,
        "max_model_len": max_model_len,
        "port": PORT,
    }


def write_server_status(cfg: Config, max_model_len: int, pid: int, log_path: Path) -> None:
    """Persist the running server's config + PID to SERVER_STATUS_PATH.

    Called immediately after wait_for_server() returns True so future configs
    can skip the expensive startup if the server already has what they need.
    """
    payload = _cfg_to_status_key(cfg, max_model_len)
    payload["pid"] = pid
    payload["log_path"] = str(log_path)
    payload["status"] = "ready"
    try:
        SERVER_STATUS_PATH.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        print(f"  WARNING: could not write server_status.json: {exc}", flush=True)


def server_is_reusable(cfg: Config, max_model_len: int) -> bool:
    """Return True if a vLLM server is already running with exactly this config.

    Checks (in order):
      1. SERVER_STATUS_PATH exists and all config fields match.
      2. The recorded PID is alive (os.kill sentinel).
      3. /health returns HTTP 200 (with no_proxy so Intel proxy is bypassed).
    """
    if not SERVER_STATUS_PATH.exists():
        return False
    try:
        status = json.loads(SERVER_STATUS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    desired = _cfg_to_status_key(cfg, max_model_len)
    for key, val in desired.items():
        if status.get(key) != val:
            return False

    pid = status.get("pid", 0)
    if pid:
        try:
            os.kill(pid, 0)  # raises OSError if process is dead/gone
        except OSError:
            return False

    _env = dict(os.environ, no_proxy="localhost,127.0.0.1,0.0.0.0",
                NO_PROXY="localhost,127.0.0.1,0.0.0.0")
    try:
        r = subprocess.run(
            ["bash", "-c", f"curl -f -s http://localhost:{PORT}/health"],
            capture_output=True, text=True, timeout=10, env=_env,
        )
        return r.returncode == 0
    except Exception:
        return False


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

def build_vllm_cmd(cfg: Config, max_model_len: int, max_num_batched_tokens: int = 8192) -> str:
    """Return the vllm serve shell command for *cfg*.

    Args:
        max_num_batched_tokens: Override for --max-num-batched-tokens.  Default 8192
            (Intel-recommended for normal runs).  The throughput study raises this to
            max_model_len (131072) so vLLM can process a 96k-token prefill in a single
            forward pass instead of chunking it across multiple iterations.
    """
    parts = [
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve {cfg.model}",
        "--dtype=bfloat16",
        f"--port {PORT}",
        "--block-size 64",
        "--gpu-memory-util 0.9",
        "--trust-remote-code",
        "--disable-sliding-window",
        "--disable-log-requests",
        f"--max-num-batched-tokens={max_num_batched_tokens}",
        f"--max-model-len {max_model_len}",
        f"-tp={cfg.tp}",
    ]
    # Prefix caching: disabled by default per Intel XPU recommendation;
    # enabled when cfg.prefix_caching=True (ablation variant).
    if not cfg.prefix_caching:
        parts.append("--no-enable-prefix-caching")
    # Async scheduling: optional Intel 0.14.1-xpu feature that overlaps
    # scheduling with model execution, reducing CPU overhead.
    if cfg.async_scheduling:
        parts.append("--async-scheduling")
    if cfg.eager:
        parts.append("--enforce-eager")
    if cfg.quant:
        parts.append(f"--quantization {cfg.quant}")
    if cfg.speculative_config:
        parts.append(f"--speculative_config '{cfg.speculative_config}'")
    if cfg.expert_parallel_size:
        parts.append("--enable-expert-parallel")
    return " ".join(parts)


def start_server(cfg: Config, max_model_len: int, log_path: Path,
                 max_num_batched_tokens: int = 8192) -> subprocess.Popen:
    """Start the vLLM server (runs directly inside the container).

    Also persists the full ``vllm serve …`` command to
    ``{log_path.parent}/{cfg.name}_vllm_cmd.txt`` so dashboards can display it.

    Args:
        max_num_batched_tokens: Forwarded to build_vllm_cmd.  Use the default 8192
            for ablation/full runs; pass THROUGHPUT_MAX_NUM_BATCHED_TOKENS (131072)
            for the throughput study so large prefills process in one pass.
    """
    vllm_cmd = build_vllm_cmd(cfg, max_model_len, max_num_batched_tokens)
    # Write the command for later dashboard display (docker image + full flags)
    try:
        cmd_path = log_path.parent / f"{cfg.name}_vllm_cmd.txt"
        cmd_path.write_text(vllm_cmd + "\n")
    except OSError:
        pass
    return _run_tee(
        ["bash", "--login", "-c", f"{_PREAMBLE} && {vllm_cmd}"],
        log_path,
    )


def wait_for_server(
    timeout: int,
    log_path: Optional[Path] = None,
    proc: Optional[subprocess.Popen] = None,
) -> bool:
    """Poll /health until the server responds or *timeout* seconds elapse.

    Raises XpuKernelHangError only when BOTH conditions hold:
      1. The OperatorEntry.cpp:208 pattern is in the log (XPU driver warning).
      2. The vLLM process is still alive (not crashed) but /health never responds.
    A clean server crash (proc has exited) returns False without raising.
    """
    print(f"  Waiting for server (timeout={timeout}s)...", flush=True)
    time.sleep(10)

    r = subprocess.run(
        ["pgrep", "-f", "vllm serve"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        print("  ERROR: vLLM process not running", flush=True)
        return False

    for elapsed in range(10, timeout, 5):
        try:
            # Bypass proxy for localhost (Rule 3): curl inherits http_proxy from
            # the bench process which was forwarded via docker exec -e.
            _env = dict(os.environ, no_proxy="localhost,127.0.0.1,0.0.0.0",
                        NO_PROXY="localhost,127.0.0.1,0.0.0.0")
            r = subprocess.run(
                ["bash", "-c", f"curl -f -s http://localhost:{PORT}/health"],
                capture_output=True, text=True, timeout=10, env=_env,
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
            # After _HANG_DETECT_AFTER_S with no health, check for the XPU
            # kernel-registration hang pattern in the server log.
            if elapsed >= _HANG_DETECT_AFTER_S and _log_has_xpu_hang(log_path):
                # If "Application startup complete" is already in the log,
                # the server is in the final init phase — NOT a true hang.
                # Keep polling; the /health endpoint will succeed momentarily.
                if _log_has_startup_complete(log_path):
                    print("  Server logged 'Application startup complete' — not a hang, continuing to poll...", flush=True)
                    continue
                # Only a true XPU hang if the process is still alive AND
                # not actively consuming CPU (a loading server has high CPU;
                # a hung server idles at ~0%).
                proc_alive = (proc is None) or (proc.poll() is None)
                if not proc_alive:
                    print("  Server process has exited (crash, not XPU hang) — treating as startup failure.", flush=True)
                    return False
                # Check CPU usage: if vLLM workers are actively running,
                # they're making progress — keep polling rather than rebooting.
                try:
                    cpu_out = subprocess.run(
                        ["bash", "-c", "ps -eo pid,pcpu --no-headers | awk '$2+0 > 5 {sum += $2} END {print sum+0}'"],
                        capture_output=True, text=True, timeout=5,
                    )
                    total_cpu = float(cpu_out.stdout.strip() or "0")
                    if total_cpu > 20:
                        print(f"  OperatorEntry.cpp:208 present but total CPU={total_cpu:.0f}% — server is actively loading, not hung. Continuing to poll...", flush=True)
                        continue
                except Exception:
                    pass
                raise XpuKernelHangError(
                    f"XPU kernel registration hang detected after {elapsed}s "
                    f"(OperatorEntry.cpp:208 in server log, process alive but never healthy). "
                    f"Container must be recreated."
                )

    print(f"  ERROR: server did not become ready within {timeout}s", flush=True)
    return False


_GRACEFUL_TIMEOUT_S = 45  # seconds to wait for SIGTERM before escalating to SIGKILL


def stop_server(proc: Optional[subprocess.Popen] = None) -> None:
    """Gracefully stop vLLM and any lingering benchmark processes.

    Uses SIGTERM + wait before SIGKILL so the XPU driver has time to release
    GPU device handles cleanly.  Abrupt SIGKILL leaves xe-destroy-wq kernel
    workers stuck in D-state, requiring a host reboot to recover.

    Also removes SERVER_STATUS_PATH so the next config doesn't falsely detect
    a running server.
    """
    print("  Stopping server (SIGTERM → wait → SIGKILL)...", flush=True)

    _sigkill_used = False

    # Step 1: SIGTERM the Popen handle (the bash --login wrapper)
    if proc:
        try:
            proc.terminate()  # SIGTERM
        except Exception:
            pass

    # Step 2: SIGTERM all vllm/guidellm child processes
    for pat in ("vllm serve", "vllm worker", "guidellm benchmark"):
        subprocess.run(["pkill", "-TERM", "-f", pat], capture_output=True)

    # Step 3: Wait up to _GRACEFUL_TIMEOUT_S for clean exit
    deadline = time.monotonic() + _GRACEFUL_TIMEOUT_S
    if proc:
        remaining = deadline - time.monotonic()
        try:
            proc.wait(timeout=max(remaining, 1))
        except subprocess.TimeoutExpired:
            pass

    # Poll until all target processes exit or deadline is reached
    while time.monotonic() < deadline:
        still_running = False
        for pat in ("vllm serve", "vllm worker"):
            r = subprocess.run(["pgrep", "-f", pat], capture_output=True)
            if r.returncode == 0 and r.stdout.strip():
                still_running = True
                break
        if not still_running:
            break
        time.sleep(2)
    else:
        # Deadline exceeded — escalate to SIGKILL as last resort
        print("  WARNING: graceful shutdown timed out — sending SIGKILL", flush=True)
        _sigkill_used = True
        if proc:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass
        for pat in ("vllm serve", "vllm worker", "guidellm benchmark"):
            subprocess.run(["pkill", "-KILL", "-f", pat], capture_output=True)
        time.sleep(3)

    try:
        SERVER_STATUS_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    # XPU driver releases VRAM asynchronously after process exit.  Without this
    # pause, the next vllm server attempting to load the model may OOM or hit an
    # XPU kernel registration hang (OperatorEntry.cpp:208) because xe-destroy-wq
    # workers haven't finished cleaning up GPU device handles.
    #
    # After a clean SIGTERM exit, 10s is sufficient (Lesson 34).
    # After a forced SIGKILL the driver cleanup is much slower — use 60s to be safe.
    _drain = 60 if _sigkill_used else 10
    print(f"  Waiting {_drain}s for XPU driver VRAM drain...", flush=True)
    time.sleep(_drain)
    print("  Server stopped.", flush=True)


# ---------------------------------------------------------------------------
# Server log parsing
# ---------------------------------------------------------------------------

import re as _re


def parse_model_mem_gib(log_path: Path) -> Optional[float]:
    """Parse 'Model loading took X.XX GiB memory' from vLLM server log (TP0).

    vLLM logs this line once per Worker_TP0 after weights are loaded.  The
    value is per-GPU; multiply by cfg.tp to get total across all devices.

    Returns per-GPU weight memory in GiB, or None if the line is not found.
    This is the most reliable source for GPU memory info — it doesn't require
    xpu-smi (which hangs in D state while the GPU is in use).
    """
    pat = _re.compile(r'Model loading took ([\d.]+)\s+GiB memory')
    try:
        with open(log_path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None
