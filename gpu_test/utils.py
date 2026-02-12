"""Shared utilities — CUDA timing, GPU info, cleanup helpers.

All benchmark and stress modules import from here to avoid duplication.
"""

from __future__ import annotations

import gc
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

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


# ── Benchmark result ────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    """Structured result from a single workload run."""

    name: str
    device: str
    elapsed_s: float
    throughput: str = ""
    extra: dict = field(default_factory=dict)


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
