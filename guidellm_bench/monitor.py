"""Background GPU monitor via xpu-smi."""

import subprocess
import threading
import time
from typing import Optional


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


class GpuMonitor:
    """Background xpu-smi dump monitor. Polls all devices every *interval* seconds.

    Each reading is a dict: {t (elapsed s), device (str), util (%), power_w, mem_mib}.
    Falls back silently if xpu-smi is not on PATH.
    """

    def __init__(self, interval: int = 10):
        self._interval = interval
        self._readings: list[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self) -> None:
        self._running = True
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        """Stop monitoring and return all collected readings."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._interval + 5)
        return list(self._readings)

    def _run(self) -> None:
        while self._running:
            self._poll()
            for _ in range(self._interval * 10):
                if not self._running:
                    return
                time.sleep(0.1)

    def _poll(self) -> None:
        try:
            r = subprocess.run(
                ["bash", "-c", "xpu-smi dump -d -1 -m 0,1,18 -i 1 -n 1 2>/dev/null"],
                capture_output=True, text=True, timeout=8,
            )
            elapsed = round(time.time() - self._t0, 1)
            for line in r.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("Timestamp"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                self._readings.append({
                    "t":       elapsed,
                    "device":  parts[1].strip(),
                    "util":    _try_float(parts[2]),
                    "power_w": _try_float(parts[3]),
                    "mem_mib": _try_float(parts[4]) if len(parts) > 4 else None,
                })
        except Exception:
            pass  # xpu-smi unavailable; silently skipped
