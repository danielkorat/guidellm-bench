#!/usr/bin/env python3
"""Host-side GPU monitor — runs on the HOST where xpu-smi has device access.

Usage:
    python3 host_gpu_monitor.py <output.jsonl>

Polls all XPU devices every INTERVAL seconds and appends one JSON line per
device per poll to <output.jsonl>.  Each line:
    {"ts": <unix_float>, "device": "<id>", "util": <pct>, "power_w": <W>, "mem_mib": <MiB>}

bench.py's host-side re-exec guard launches this as a background Popen and
passes the output path to the container via the GPU_MONITOR_FILE env var.
GpuMonitor (inside the container) reads the file on stop(), filtering by the
config's wall-clock start/end timestamps.

xpu-smi is wrapped in `bash timeout 5` to guarantee we never hang even if
the Intel driver is temporarily unresponsive.
"""

import json
import subprocess
import sys
import time

INTERVAL = 10  # seconds between polls


def _poll(output: str) -> None:
    """Run one xpu-smi sample and append readings to output."""
    try:
        r = subprocess.run(
            ["bash", "-c",
             "timeout 5 xpu-smi dump -d -1 -m 0,1,18 -i 1 -n 1 2>/dev/null"],
            capture_output=True, text=True, timeout=8,
        )
        ts = time.time()
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("Timestamp"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            reading: dict = {
                "ts":      ts,
                "device":  parts[1],
                "util":    None,
                "power_w": None,
                "mem_mib": None,
            }
            try:
                reading["util"]    = float(parts[2])
            except (ValueError, IndexError):
                pass
            try:
                reading["power_w"] = float(parts[3])
            except (ValueError, IndexError):
                pass
            try:
                reading["mem_mib"] = float(parts[4]) if len(parts) > 4 else None
            except (ValueError, IndexError):
                pass
            with open(output, "a") as f:
                f.write(json.dumps(reading) + "\n")
    except Exception:
        pass  # xpu-smi unavailable or timed out — silently skip


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: host_gpu_monitor.py <output.jsonl>", file=sys.stderr)
        sys.exit(1)

    output = sys.argv[1]
    # Truncate / create the file so GpuMonitor always sees a valid path.
    open(output, "w").close()

    while True:
        _poll(output)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
