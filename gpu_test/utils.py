"""Shared utilities — CUDA timing, GPU info, cleanup helpers.

All benchmark and stress modules import from here to avoid duplication.
"""

from __future__ import annotations

import gc
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator

import torch


# ── GPU information ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GpuDevice:
    """Immutable snapshot of static GPU properties."""

    index: int
    name: str
    vram_gb: float
    compute_capability: str

    @property
    def sm_tag(self) -> str:
        return f"sm_{self.compute_capability.replace('.', '')}"

    def __str__(self) -> str:
        return f"GPU {self.index}: {self.name}  ({self.vram_gb} GB, {self.sm_tag})"


def detect_gpus() -> list[GpuDevice]:
    """Return a list of all CUDA devices, or exit if none found."""
    n = torch.cuda.device_count()
    if n == 0:
        print("✗ No CUDA devices found.", file=sys.stderr)
        sys.exit(1)
    devices: list[GpuDevice] = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        devices.append(
            GpuDevice(
                index=i,
                name=p.name,
                vram_gb=round(p.total_mem / 1024**3, 1),
                compute_capability=f"{p.major}.{p.minor}",
            )
        )
    return devices


def require_gpus(minimum: int = 1) -> list[GpuDevice]:
    """Detect GPUs and exit if fewer than *minimum* are available."""
    gpus = detect_gpus()
    if len(gpus) < minimum:
        print(f"✗ Need ≥ {minimum} CUDA GPU(s), found {len(gpus)}.", file=sys.stderr)
        sys.exit(1)
    return gpus


# ── CUDA timing ─────────────────────────────────────────────────────────────


@contextmanager
def cuda_timer(device: torch.device) -> Generator[list[float], None, None]:
    """Context manager that yields a 1-element list; on exit stores elapsed seconds.

    Uses CUDA events for accurate GPU timing when *device* is CUDA,
    falls back to perf_counter for CPU.
    """
    result: list[float] = [0.0]
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        yield result
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        result[0] = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.perf_counter()
        yield result
        result[0] = time.perf_counter() - t0


# ── Memory cleanup ──────────────────────────────────────────────────────────


def cleanup(device: torch.device) -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


# ── Test result ─────────────────────────────────────────────────────────────


@dataclass
class TestResult:
    """Structured result from a single test run on a single GPU.

    Fields:
        suite:      "bench" or "stress"
        test_name:  human-readable test label
        gpu_index:  device index
        gpu_name:   device name
        status:     "ok", "oom", or "error"
        elapsed_s:  wall-clock seconds
        metrics:    test-specific key→value (e.g. tok_per_s, tflops, gb_per_s)
        config:     parameters that were used (batch, seq_len, etc.)
        telemetry:  NVML readings captured during the test
        timestamp:  ISO 8601 when the result was recorded
    """

    suite: str
    test_name: str
    gpu_index: int
    gpu_name: str
    status: str = "ok"
    elapsed_s: float = 0.0
    peak_vram_gb: float = 0.0
    # Per-test targeted metrics
    metrics: dict[str, Any] = field(default_factory=dict)
    # Config params used for this run
    config: dict[str, Any] = field(default_factory=dict)
    # NVML telemetry (filled by monitor)
    telemetry: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary_line(self) -> str:
        """One-line summary for console output."""
        m = "  ".join(f"{k}={v}" for k, v in self.metrics.items())
        c = "  ".join(f"{k}={v}" for k, v in self.config.items())
        parts = [f"GPU {self.gpu_index}: {m or self.status}"]
        if self.elapsed_s > 0:
            parts.append(f"{self.elapsed_s:.3f}s")
        if self.peak_vram_gb > 0:
            parts.append(f"peak VRAM={self.peak_vram_gb:.1f}GB")
        if c:
            parts.append(f"({c})")
        return "  ".join(parts)


# For backward compat — alias
BenchResult = TestResult


# ── Formatting helpers ──────────────────────────────────────────────────────


def banner(title: str) -> None:
    print(f"\n{'━' * 72}")
    print(f"  {title}")
    print(f"{'━' * 72}")


def section(title: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def note(msg: str) -> None:
    """Indented info line (e.g. OOM fallback messages)."""
    print(f"    ↳ {msg}")


def print_header(title: str, gpus: list[GpuDevice]) -> None:
    """Print a standardised startup header."""
    print("=" * 72)
    print(f"  {title}  ·  PyTorch {torch.__version__}")
    if torch.version.cuda:
        print(f"  CUDA {torch.version.cuda}  ·  {len(gpus)} GPU(s)")
    print("=" * 72)
