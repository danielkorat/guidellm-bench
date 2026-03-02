"""Docker container helpers — launch and exec into intel/vllm:0.14.1-xpu."""

import os
import subprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_NAME = "vllm-0.14"
DOCKER_IMAGE = "intel/vllm:0.14.1-xpu"

# Volume mount: host /root/dkorat/ → container /root/
HOST_ROOT = "/root/dkorat"
CONTAINER_ROOT = "/root"

# The bash preamble run before every command inside the container.
_PREAMBLE = (
    "source /opt/intel/oneapi/setvars.sh --force && "
    "export no_proxy=localhost,127.0.0.1,0.0.0.0 && "
    "export NO_PROXY=localhost,127.0.0.1,0.0.0.0"
)


# ---------------------------------------------------------------------------
# Path translation
# ---------------------------------------------------------------------------

def host_to_container(path: str) -> str:
    """Convert a host-side path under HOST_ROOT to its container-side equivalent."""
    if path.startswith(HOST_ROOT):
        return CONTAINER_ROOT + path[len(HOST_ROOT):]
    return path


# ---------------------------------------------------------------------------
# Command helpers
# ---------------------------------------------------------------------------

def docker_exec_cmd(inner_cmd: str) -> list[str]:
    """Return argv list: docker exec <container> bash --login -c '<preamble> && <inner_cmd>'."""
    full = f"{_PREAMBLE} && {inner_cmd}"
    return ["docker", "exec", CONTAINER_NAME, "bash", "--login", "-c", full]


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------

def ensure_container_running() -> None:
    """Start the container if it is not already running.

    Three cases:
      1. Running — nothing to do.
      2. Exists but stopped — ``docker start``.
      3. Does not exist — ``docker run`` with the standard parameters.
    """
    r = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True, text=True,
    )
    if r.returncode == 0 and r.stdout.strip() == "true":
        print(f"  Container '{CONTAINER_NAME}' is already running.", flush=True)
        return

    if r.returncode == 0:
        # Container exists but is stopped.
        print(f"  Starting stopped container '{CONTAINER_NAME}'...", flush=True)
        subprocess.run(["docker", "start", CONTAINER_NAME], check=True)
        print(f"  Container '{CONTAINER_NAME}' started.", flush=True)
        return

    # Container does not exist — create it.
    print(f"  Launching container '{CONTAINER_NAME}' from image {DOCKER_IMAGE}...", flush=True)
    hf_token = os.environ.get("HF_READ_TOKEN", "")
    hf_cache = os.path.expanduser("~/.cache/huggingface")

    subprocess.run(
        [
            "docker", "run", "-t", "-d",
            "--shm-size", "10g",
            "--net=host",
            "--ipc=host",
            "--privileged",
            "-e", "http_proxy=http://proxy-dmz.intel.com:912",
            "-e", "https_proxy=http://proxy-dmz.intel.com:912",
            "-e", "HTTP_PROXY=http://proxy-dmz.intel.com:912",
            "-e", "HTTPS_PROXY=http://proxy-dmz.intel.com:912",
            "-e", "no_proxy=localhost,127.0.0.1,0.0.0.0",
            "-e", "NO_PROXY=localhost,127.0.0.1,0.0.0.0",
            "-e", f"HF_TOKEN={hf_token}",
            "--name", CONTAINER_NAME,
            "--device", "/dev/dri:/dev/dri",
            "-v", f"{hf_cache}:/root/.cache/huggingface",
            "-v", "/dev/dri/by-path:/dev/dri/by-path",
            "-v", f"{HOST_ROOT}/:/root",
            "--entrypoint", "",
            DOCKER_IMAGE, "/bin/bash",
        ],
        check=True,
    )
    print(f"  Container '{CONTAINER_NAME}' launched.", flush=True)
