"""Background GPU monitor via xpu-smi.

NOTE: xpu-smi enters D (uninterruptible sleep) state when the GPU is in use by
vLLM, so it cannot reliably collect data during active benchmark runs.

GPU memory is now sourced from the vLLM server log via parse_model_mem_gib()
in server.py, which parses "Model loading took X.XX GiB memory" from TP0's log.
This class is kept as a no-op stub for backward compatibility.
"""


class GpuMonitor:
    """Best-effort GPU monitor stub.  Returns empty readings in all cases.

    xpu-smi hangs in D state when called while vLLM holds the GPU resources,
    so polling is disabled.  GPU memory data comes from parse_model_mem_gib().
    """

    def __init__(self, interval: int = 10):
        self._interval = interval

    def start(self) -> None:
        pass  # no-op

    def stop(self) -> list[dict]:
        """Return empty readings."""
        return []

