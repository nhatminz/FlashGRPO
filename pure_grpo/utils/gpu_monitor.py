from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

import torch


@dataclass
class GpuSnapshot:
    timestamp: float
    utilization_gpu: float | None = None
    memory_used_mb: float | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None
    source: str = "unavailable"


class GpuMonitor:
    def __init__(self, enabled: bool = True, min_interval_s: float = 5.0):
        self.enabled = enabled
        self.min_interval_s = float(min_interval_s)
        self._last_poll = 0.0
        self._pynvml = None
        self._nvml_handle = None
        if enabled:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._pynvml = pynvml
                index = torch.cuda.current_device() if torch.cuda.is_available() else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                self._pynvml = None
                self._nvml_handle = None

    def sample(self, force: bool = False) -> GpuSnapshot:
        now = time.time()
        if not self.enabled or (not force and now - self._last_poll < self.min_interval_s):
            return GpuSnapshot(timestamp=now)
        self._last_poll = now
        allocated = reserved = None
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
        if self._pynvml is not None and self._nvml_handle is not None:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return GpuSnapshot(
                    timestamp=now,
                    utilization_gpu=float(util.gpu),
                    memory_used_mb=mem.used / 1024**2,
                    memory_allocated_mb=allocated,
                    memory_reserved_mb=reserved,
                    source="pynvml",
                )
            except Exception:
                pass
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            first = out.strip().splitlines()[0].split(",")
            return GpuSnapshot(
                timestamp=now,
                utilization_gpu=float(first[0].strip()),
                memory_used_mb=float(first[1].strip()),
                memory_allocated_mb=allocated,
                memory_reserved_mb=reserved,
                source="nvidia-smi",
            )
        except Exception:
            return GpuSnapshot(
                timestamp=now,
                memory_allocated_mb=allocated,
                memory_reserved_mb=reserved,
                source="torch",
            )
