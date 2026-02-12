"""GPU telemetry via NVML — power, temperature, clocks, utilization, VRAM.

Works alongside PyTorch; call ``snapshot(device_index)`` for a point-in-time
reading, or use ``Monitor`` as a context manager to capture peak values
during a workload.

Requires: ``nvidia-ml-py``
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

try:
    import pynvml

    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


# ── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class GpuSnapshot:
    """Point-in-time GPU telemetry."""

    index: int
    name: str = ""
    gpu_util_pct: float = 0.0
    mem_util_pct: float = 0.0
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    mem_used_pct: float = 0.0
    power_w: float = 0.0
    power_limit_w: float = 0.0
    temperature_c: int = 0
    clock_gpu_mhz: int = 0
    clock_mem_mhz: int = 0
    clock_gpu_max_mhz: int = 0
    clock_mem_max_mhz: int = 0
    fan_speed_pct: int = 0
    pcie_gen: int = 0
    pcie_width: int = 0
    pcie_tx_kbps: int = 0
    pcie_rx_kbps: int = 0

    def __str__(self) -> str:
        lines = [
            f"GPU {self.index}: {self.name}",
            f"  Utilization   : GPU {self.gpu_util_pct:.0f}%  ·  Mem ctrl {self.mem_util_pct:.0f}%",
            f"  VRAM          : {self.mem_used_gb:.1f} / {self.mem_total_gb:.1f} GB ({self.mem_used_pct:.0f}%)",
            f"  Power         : {self.power_w:.0f} / {self.power_limit_w:.0f} W",
            f"  Temperature   : {self.temperature_c}°C",
            f"  Clocks        : GPU {self.clock_gpu_mhz} / {self.clock_gpu_max_mhz} MHz  ·  Mem {self.clock_mem_mhz} / {self.clock_mem_max_mhz} MHz",
        ]
        if self.fan_speed_pct > 0:
            lines.append(f"  Fan           : {self.fan_speed_pct}%")
        if self.pcie_gen > 0:
            lines.append(
                f"  PCIe          : Gen{self.pcie_gen} x{self.pcie_width}  "
                f"TX {self.pcie_tx_kbps / 1024:.0f} MB/s  RX {self.pcie_rx_kbps / 1024:.0f} MB/s"
            )
        return "\n".join(lines)

    def short(self) -> str:
        return (
            f"GPU={self.gpu_util_pct:.0f}%  "
            f"VRAM={self.mem_used_gb:.1f}/{self.mem_total_gb:.1f}GB  "
            f"Power={self.power_w:.0f}W  "
            f"Temp={self.temperature_c}°C  "
            f"Clk={self.clock_gpu_mhz}MHz"
        )


@dataclass
class MonitorStats:
    """Aggregated stats from a monitoring session."""

    samples: int = 0
    peak_gpu_util: float = 0
    peak_mem_util: float = 0
    peak_mem_gb: float = 0
    peak_power_w: float = 0
    peak_temp_c: int = 0
    peak_clock_gpu_mhz: int = 0
    avg_gpu_util: float = 0
    avg_power_w: float = 0
    avg_temp_c: float = 0
    last: Optional[GpuSnapshot] = None

    def summary(self) -> str:
        return (
            f"GPU util: avg {self.avg_gpu_util:.0f}% / peak {self.peak_gpu_util:.0f}%  ·  "
            f"VRAM peak: {self.peak_mem_gb:.1f}GB  ·  "
            f"Power: avg {self.avg_power_w:.0f}W / peak {self.peak_power_w:.0f}W  ·  "
            f"Temp: avg {self.avg_temp_c:.0f}°C / peak {self.peak_temp_c}°C  ·  "
            f"Clock peak: {self.peak_clock_gpu_mhz}MHz  "
            f"({self.samples} samples)"
        )


# ── NVML interface ───────────────────────────────────────────────────────────


def _safe(fn, *args, default=0):
    try:
        return fn(*args)
    except Exception:
        return default


def init() -> bool:
    """Initialise NVML.  Returns True if available."""
    if not _NVML_AVAILABLE:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def shutdown() -> None:
    if _NVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def snapshot(index: int) -> GpuSnapshot:
    """Take a single point-in-time telemetry reading for GPU *index*."""
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()

    util = _safe(pynvml.nvmlDeviceGetUtilizationRates, h)
    mem = _safe(pynvml.nvmlDeviceGetMemoryInfo, h)
    power = _safe(pynvml.nvmlDeviceGetPowerUsage, h, default=0)
    power_limit = _safe(pynvml.nvmlDeviceGetPowerManagementLimit, h, default=0)
    temp = _safe(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU, default=0)
    clock_gpu = _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_GRAPHICS, default=0)
    clock_mem = _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_MEM, default=0)
    clock_gpu_max = _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_GRAPHICS, default=0)
    clock_mem_max = _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_MEM, default=0)
    fan = _safe(pynvml.nvmlDeviceGetFanSpeed, h, default=0)
    pcie_gen = _safe(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, h, default=0)
    pcie_width = _safe(pynvml.nvmlDeviceGetCurrPcieLinkWidth, h, default=0)
    pcie_tx = _safe(pynvml.nvmlDeviceGetPcieThroughput, h, pynvml.NVML_PCIE_UTIL_TX_BYTES, default=0)
    pcie_rx = _safe(pynvml.nvmlDeviceGetPcieThroughput, h, pynvml.NVML_PCIE_UTIL_RX_BYTES, default=0)

    mem_used = mem.used / 1024**3 if mem else 0
    mem_total = mem.total / 1024**3 if mem else 0

    return GpuSnapshot(
        index=index, name=name,
        gpu_util_pct=util.gpu if util else 0,
        mem_util_pct=util.memory if util else 0,
        mem_used_gb=mem_used, mem_total_gb=mem_total,
        mem_used_pct=(mem_used / mem_total * 100) if mem_total > 0 else 0,
        power_w=power / 1000, power_limit_w=power_limit / 1000,
        temperature_c=temp,
        clock_gpu_mhz=clock_gpu, clock_mem_mhz=clock_mem,
        clock_gpu_max_mhz=clock_gpu_max, clock_mem_max_mhz=clock_mem_max,
        fan_speed_pct=fan,
        pcie_gen=pcie_gen, pcie_width=pcie_width,
        pcie_tx_kbps=pcie_tx, pcie_rx_kbps=pcie_rx,
    )


# ── Background monitor ──────────────────────────────────────────────────────


class Monitor:
    """Background thread that samples GPU telemetry at a fixed interval.

    Usage::

        with Monitor(gpu_index=0) as mon:
            # ... run workload ...
        print(mon.stats.summary())
    """

    def __init__(self, gpu_index: int = 0, interval: float = 0.1):
        self.gpu_index = gpu_index
        self.interval = interval
        self.stats = MonitorStats()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sum_gpu_util = 0.0
        self._sum_power = 0.0
        self._sum_temp = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                s = snapshot(self.gpu_index)
                self.stats.samples += 1
                self.stats.last = s
                self.stats.peak_gpu_util = max(self.stats.peak_gpu_util, s.gpu_util_pct)
                self.stats.peak_mem_util = max(self.stats.peak_mem_util, s.mem_util_pct)
                self.stats.peak_mem_gb = max(self.stats.peak_mem_gb, s.mem_used_gb)
                self.stats.peak_power_w = max(self.stats.peak_power_w, s.power_w)
                self.stats.peak_temp_c = max(self.stats.peak_temp_c, s.temperature_c)
                self.stats.peak_clock_gpu_mhz = max(self.stats.peak_clock_gpu_mhz, s.clock_gpu_mhz)
                self._sum_gpu_util += s.gpu_util_pct
                self._sum_power += s.power_w
                self._sum_temp += s.temperature_c
            except Exception:
                pass
            self._stop.wait(self.interval)
        if self.stats.samples > 0:
            self.stats.avg_gpu_util = self._sum_gpu_util / self.stats.samples
            self.stats.avg_power_w = self._sum_power / self.stats.samples
            self.stats.avg_temp_c = self._sum_temp / self.stats.samples

    def __enter__(self) -> "Monitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# ── Convenience for other modules ───────────────────────────────────────────


def run_monitored(gpu_index: int, fn, *args, nvml_ok: bool = False, **kwargs) -> tuple:
    """Run *fn* while optionally sampling GPU telemetry.

    Returns ``(fn_result, MonitorStats | None)``.
    """
    if nvml_ok:
        with Monitor(gpu_index=gpu_index, interval=0.1) as mon:
            result = fn(*args, **kwargs)
        return result, mon.stats
    return fn(*args, **kwargs), None

