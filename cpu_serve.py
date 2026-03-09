#!/usr/bin/env python3
"""cpu_serve.py — Serve meta-llama/Llama-3.1-8B-Instruct on CPU using the official vLLM CPU image.

Runs on the host; manages the container lifecycle.

Usage
-----
  # Start server (foreground, streaming logs):
  python3 cpu_serve.py

  # Custom port:
  python3 cpu_serve.py --port 8080

  # Override number of CPU threads used by vLLM:
  python3 cpu_serve.py --num-cpu-threads 64

  # Stop a running server:
  python3 cpu_serve.py --stop

  # Print the OpenAI-compatible API URL:
  python3 cpu_serve.py --url

Hardware: Intel Xeon 6730P (128 cores, ~3 TB RAM)

vLLM command used
-----------------
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --disable-log-stats \
    --tensor-parallel-size 2 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --port <port>

Env optimisations
-----------------
  - VLLM_CPU_KVCACHE_SPACE=40      : 40 GiB KV-cache DRAM allocation (tunable via --kv-cache-gib)
  - VLLM_CPU_OMP_THREADS_BIND=all  : bind OpenMP threads to all cores for max throughput
  - OMP_NUM_THREADS=<nproc>        : full logical core count

The container used is: public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:latest
It is pre-pulled on this machine.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL            = "meta-llama/Llama-3.1-8B-Instruct"

CPU_IMAGE        = "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:latest"
CONTAINER_NAME   = "vllm-cpu-llama"
DEFAULT_PORT     = 8081
DEFAULT_THREADS  = None           # None → use all logical CPUs (nproc)

# KV-cache DRAM budget in GiB (can be overridden via env VLLM_CPU_KVCACHE_SPACE)
DEFAULT_KV_CACHE_GIB = 40

# Proxy for HuggingFace model downloads
_PROXY = os.environ.get("http_proxy", "http://proxy-dmz.intel.com:911/")

# Status file (host path) — mirrors the pattern in server.py
_STATUS_FILE = Path(__file__).parent / "cpu_server_status.json"

# ---------------------------------------------------------------------------
# Re-exec guard — this script runs on the HOST only; container runs vllm serve
# directly (entrypoint = vllm serve).  No re-exec needed on the container side.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve meta-llama/Llama-3.1-8B-Instruct on CPU via vLLM CPU container.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="Port for the vLLM OpenAI-compatible API")
    p.add_argument("--num-cpu-threads", type=int, default=DEFAULT_THREADS,
                   help="Number of OMP threads vLLM may use (default: all cores)")
    p.add_argument("--kv-cache-gib", type=int, default=DEFAULT_KV_CACHE_GIB,
                   help="DRAM budget (GiB) for KV cache")
    p.add_argument("--stop", action="store_true",
                   help="Stop the running container and exit")
    p.add_argument("--url", action="store_true",
                   help="Print the API base URL and exit")
    p.add_argument("--bench", action="store_true",
                   help="Run vllm bench serve against the running server")
    p.add_argument("--concurrency", type=int, default=32,
                   help="--max-concurrency for vllm bench serve")
    p.add_argument("--num-prompts", type=int, default=200,
                   help="--num-prompts for vllm bench serve")
    p.add_argument("--input-len", type=int, default=128,
                   help="--random-input-len for vllm bench serve")
    p.add_argument("--output-len", type=int, default=128,
                   help="--random-output-len for vllm bench serve")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                   help="HuggingFace token (falls back to $HF_TOKEN)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------

def _container_running() -> bool:
    r = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def _stop_container() -> None:
    print(f"[cpu_serve] Stopping container '{CONTAINER_NAME}'...", flush=True)
    subprocess.run(["docker", "stop", "-t", "10", CONTAINER_NAME],
                   capture_output=True)
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True)
    _STATUS_FILE.unlink(missing_ok=True)
    print(f"[cpu_serve] Container stopped.", flush=True)


def _start_container(args: argparse.Namespace) -> subprocess.Popen:
    """Launch the CPU vLLM container and return the Popen handle.

    Overrides the container entrypoint to use:
      python -m vllm.entrypoints.openai.api_server
    with the exact flags requested.
    """
    # Physical cores only: HT siblings share the memory controller and add no
    # bandwidth for memory-bound inference. 128 logical / 2 HT = 64 physical.
    n_threads = min(args.num_cpu_threads or 64, 64)

    # For TP=2 the env var must be two |-separated groups, one per rank.
    # NUMA topology: socket 0 → physical cores 0-31 (NUMA 0+1)
    #                socket 1 → physical cores 32-63 (NUMA 2+3)
    # Each worker stays NUMA-local to its socket for maximum DDR5 bandwidth.
    tp = 2
    cores_per_worker = n_threads // tp   # 32 physical cores/socket
    bind_groups = "|".join(
        f"{i * cores_per_worker}-{(i + 1) * cores_per_worker - 1}"
        for i in range(tp)
    )  # e.g. "0-31|32-63"

    # vLLM CPU env vars (documented in vllm/executor/cpu_executor.py)
    env_flags = [
        # Reserve DRAM for KV cache (GiB)
        "-e", f"VLLM_CPU_KVCACHE_SPACE={args.kv_cache_gib}",
        # TP=2: bind worker-0 to socket-0 cores, worker-1 to socket-1 cores
        "-e", f"VLLM_CPU_OMP_THREADS_BIND={bind_groups}",
        # OMP thread count per worker (not total)
        "-e", f"OMP_NUM_THREADS={cores_per_worker}",
        # Proxy for HF model download
        "-e", f"http_proxy={_PROXY}",
        "-e", f"https_proxy={_PROXY.replace('911', '912')}",
        "-e", f"HTTP_PROXY={_PROXY}",
        "-e", f"HTTPS_PROXY={_PROXY.replace('911', '912')}",
        "-e", "no_proxy=localhost,127.0.0.1,0.0.0.0",
        "-e", "NO_PROXY=localhost,127.0.0.1,0.0.0.0",
    ]
    if args.hf_token:
        env_flags += ["-e", f"HF_TOKEN={args.hf_token}"]

    # HuggingFace cache shared with host
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    Path(hf_cache).mkdir(parents=True, exist_ok=True)

    # Exact vLLM command. TP=2 splits across the 2 sockets so each process
    # uses one socket's NUMA-local DDR5 bandwidth (nodes 0+1 and 2+3).
    vllm_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--dtype", "bfloat16",
        "--distributed-executor-backend", "mp",
        "--trust-remote-code",
        "--disable-log-stats",
        "--tensor-parallel-size", "2",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "4096",
        "--max-num-seqs", "256",
        "--port", str(args.port),
    ]

    # Use physical cores only (0-63): HT siblings (64-127) share the same
    # memory controller and add no bandwidth for memory-bound inference.
    # --privileged enables NUMA memory policy (numa_migrate_pages / set_mempolicy).
    docker_cmd = [
        "docker", "run", "-d",
        "--rm",                    # auto-remove when stopped
        "--name", CONTAINER_NAME,
        "--net=host",
        "--ipc=host",
        "--privileged",            # needed for NUMA memory policy (set_mempolicy)
        "--entrypoint", "",        # override vllm serve entrypoint
        # Physical cores only — no HT siblings (0-63 on a 2×32-core machine)
        "--cpuset-cpus=0-63",
        # Volume: shared HF model cache so re-runs skip the download
        "-v", f"{hf_cache}:/root/.cache/huggingface",
        # Volume: repo dir so logs can be written to results/
        "-v", f"{Path(__file__).parent}:/workspace",
    ] + env_flags + [CPU_IMAGE] + vllm_cmd

    print(f"[cpu_serve] Launching container '{CONTAINER_NAME}'...", flush=True)
    print(f"[cpu_serve] Command: {' '.join(vllm_cmd[2:])}", flush=True)
    proc = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    return proc


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------

def _health_check(port: int, timeout: int = 600) -> bool:
    """Poll /health until the server responds HTTP 200 or timeout expires."""
    url = f"http://localhost:{port}/health"
    _env = dict(os.environ,
                no_proxy="localhost,127.0.0.1,0.0.0.0",
                NO_PROXY="localhost,127.0.0.1,0.0.0.0")
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        r = subprocess.run(
            ["curl", "-f", "-s", "--max-time", "5", url],
            capture_output=True, text=True, env=_env,
        )
        if r.returncode == 0:
            print(f"\n[cpu_serve] Server is healthy after {attempt} attempts.", flush=True)
            return True
        elapsed = int(time.time() - (deadline - timeout))
        if attempt % 10 == 0:
            print(f"[cpu_serve] Still waiting... {elapsed}s elapsed", flush=True)
        else:
            print(".", end="", flush=True)
        time.sleep(6)
    return False


# ---------------------------------------------------------------------------
# Container log streaming
# ---------------------------------------------------------------------------

def _stream_container_logs(container: str) -> None:
    """Stream the container logs to stdout using `docker logs -f`."""
    subprocess.run(["docker", "logs", "-f", container])


def _run_bench(args: argparse.Namespace) -> None:
    """Run vllm bench serve inside the container against the running server."""
    bench_cmd = [
        "vllm", "bench", "serve",
        "--host", "localhost",
        "--port", str(args.port),
        "--request-rate", "inf",
        "--max-concurrency", str(args.concurrency),
        "--model", MODEL,
        "--backend", "vllm",
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--ignore-eos",
        "--num-prompts", str(args.num_prompts),
    ]
    docker_cmd = [
        "docker", "exec", "-it", CONTAINER_NAME,
        "env",
        "no_proxy=localhost,127.0.0.1,0.0.0.0",
        "NO_PROXY=localhost,127.0.0.1,0.0.0.0",
    ] + bench_cmd
    print(f"[cpu_serve] Running benchmark:", flush=True)
    print(f"  {' '.join(bench_cmd)}", flush=True)
    print("", flush=True)
    subprocess.run(docker_cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    machine_ip = "10.75.137.163"  # fixed IP for this machine (see memories/repo/machine.md)

    if args.url:
        print(f"http://{machine_ip}:{args.port}/v1", flush=True)
        return

    if args.bench:
        if not _container_running():
            print(f"[cpu_serve] ERROR: container '{CONTAINER_NAME}' is not running. Start it first.",
                  flush=True)
            sys.exit(1)
        _run_bench(args)
        return

    if args.stop:
        if _container_running():
            _stop_container()
        else:
            print(f"[cpu_serve] Container '{CONTAINER_NAME}' is not running.", flush=True)
        return

    # --- Check if already running ---
    if _container_running():
        print(f"[cpu_serve] Container '{CONTAINER_NAME}' is already running.", flush=True)
        print(f"[cpu_serve] API: http://{machine_ip}:{args.port}/v1", flush=True)
        print(f"[cpu_serve] Streaming logs (Ctrl-C to detach)...", flush=True)
        _stream_container_logs(CONTAINER_NAME)
        return

    # --- Start container ---
    _proc = _start_container(args)

    # The container ID is captured from the `docker run -d` stdout on success
    container_id = _proc.stdout.read().strip() if _proc.stdout else ""
    print(f"[cpu_serve] Container ID: {container_id[:12]}", flush=True)

    # --- Health check --- (model download may take minutes on first run)
    n_threads = args.num_cpu_threads or os.cpu_count() or 64
    print(f"[cpu_serve] Waiting for server on port {args.port} "
          f"(model download + compile may take several minutes on first run)...", flush=True)

    healthy = _health_check(args.port, timeout=900)

    if not healthy:
        print(f"\n[cpu_serve] ERROR: server did not become healthy in time.", flush=True)
        print(f"[cpu_serve] Showing last 40 lines of container logs:", flush=True)
        subprocess.run(["docker", "logs", "--tail", "40", CONTAINER_NAME])
        _stop_container()
        sys.exit(1)

    print(f"[cpu_serve] Model:   {MODEL}", flush=True)
    print(f"[cpu_serve] API URL: http://{machine_ip}:{args.port}/v1", flush=True)
    print(f"[cpu_serve] Threads: {n_threads} / {os.cpu_count()} logical CPUs", flush=True)
    print(f"[cpu_serve] KV RAM:  {args.kv_cache_gib} GiB", flush=True)
    print(f"", flush=True)
    print(f"[cpu_serve] Quick test:", flush=True)
    print(f"  curl --noproxy '*' http://{machine_ip}:{args.port}/v1/chat/completions \\", flush=True)
    print(f'    -H "Content-Type: application/json" \\', flush=True)
    print(f'    -d \'{{"model":"{MODEL}","messages":[{{"role":"user","content":"Hello"}}],"max_tokens":50}}\'', flush=True)
    print(f"", flush=True)
    print(f"[cpu_serve] Streaming logs (Ctrl-C to detach without stopping the server)...",
          flush=True)

    # Stream logs until Ctrl-C
    try:
        _stream_container_logs(CONTAINER_NAME)
    except KeyboardInterrupt:
        print(f"\n[cpu_serve] Detached from logs. Server is still running.", flush=True)
        print(f"[cpu_serve] To stop:  python3 cpu_serve.py --stop", flush=True)
        print(f"[cpu_serve] To reconnect: docker logs -f {CONTAINER_NAME}", flush=True)


if __name__ == "__main__":
    main()
