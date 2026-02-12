"""Multi-GPU stress testing suite — 10 tests designed to push high-VRAM GPUs.

All tests support OOM-safe execution and report NVML telemetry when available.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpu_test.monitor import (
    Monitor, init as nvml_init, run_monitored, shutdown as nvml_shutdown, snapshot,
)
from gpu_test.utils import (
    GpuDevice, cleanup, cuda_timer, detect_gpus, print_header, require_gpus, section,
)

_NVML: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gpus(minimum: int = 1) -> list[GpuDevice]:
    return require_gpus(minimum)


def _monitored(gpu_idx: int, fn):
    """Run fn() with optional NVML monitoring."""
    run_monitored(gpu_idx, fn, nvml_ok=_NVML)


# ── 1. Throughput ────────────────────────────────────────────────────────────

def test_throughput(n_iters: int = 30, batch_size: int = 32) -> None:
    """Transformer inference throughput (tok/s)."""
    section("Throughput  (transformer inference, tok/s)")
    gpus = _gpus()
    d_model, n_heads, seq_len = 2048, 16, 2048

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")

        def _run(dev=dev, i=gpu.index):
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True
            ).to(dev).eval()
            x = torch.randn(batch_size, seq_len, d_model, device=dev)
            with torch.no_grad():
                _ = layer(x)
            torch.cuda.synchronize(dev)

            with cuda_timer(dev) as elapsed:
                with torch.no_grad():
                    for _ in range(n_iters):
                        _ = layer(x)

            tps = (n_iters * batch_size * seq_len) / elapsed[0] if elapsed[0] > 0 else 0
            print(f"  GPU {i} ({gpu.name}):  {tps:,.0f} tok/s  ({elapsed[0]:.3f}s)")
            del layer, x

        _monitored(gpu.index, _run)
        cleanup(dev)


# ── 2. Memory Stress ────────────────────────────────────────────────────────

def test_memory() -> None:
    """Progressive VRAM fill (1 GB BF16 chunks)."""
    section("Memory Stress  (fill VRAM with BF16 model-weight tensors)")
    gpus = _gpus()

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        torch.cuda.reset_peak_memory_stats(dev)
        chunk = 1024 * 1024 * 1024 // 2  # 1 GB in bf16
        tensors: list[torch.Tensor] = []
        try:
            while True:
                tensors.append(torch.randn(chunk, dtype=torch.bfloat16, device=dev))
        except torch.cuda.OutOfMemoryError:
            pass
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        print(f"  GPU {gpu.index} ({gpu.name}):  {peak:.1f} / {gpu.vram_gb} GB ({peak / gpu.vram_gb * 100:.0f}%)")
        del tensors
        cleanup(dev)


# ── 3. Sustained Load ──────────────────────────────────────────────────────

def test_sustained(duration: int = 30) -> None:
    """Continuous AMP training for N seconds."""
    section(f"Sustained Load  ({duration}s continuous AMP training)")
    gpus = _gpus()
    d_model, n_heads, seq_len, batch = 2048, 16, 1024, 16

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")

        def _run(dev=dev, i=gpu.index):
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True
            ).to(dev)
            opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler(device=dev.type)
            x = torch.randn(batch, seq_len, d_model, device=dev)
            iters, errors = 0, 0
            t0 = time.monotonic()
            while time.monotonic() - t0 < duration:
                try:
                    opt.zero_grad()
                    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16):
                        out = layer(x)
                    scaler.scale(out.sum()).backward()
                    scaler.step(opt); scaler.update()
                    torch.cuda.synchronize(dev)
                    iters += 1
                except Exception as exc:
                    errors += 1
                    if errors >= 10:
                        print(f"  GPU {i}: aborting — {exc}")
                        break
            wall = time.monotonic() - t0
            print(f"  GPU {i} ({gpu.name}):  {iters} steps in {wall:.1f}s  ({iters / wall:.1f} step/s)  errors={errors}")
            del layer, opt, scaler, x

        _monitored(gpu.index, _run)
        cleanup(dev)


# ── 4. P2P Bandwidth ───────────────────────────────────────────────────────

def test_p2p(size_mb: int = 1024) -> None:
    """GPU-to-GPU peer-to-peer transfer bandwidth."""
    section(f"P2P GPU Bandwidth  ({size_mb} MB transfers)")
    gpus = _gpus(2)
    numel = size_mb * 1024 * 1024 // 4
    for src in gpus:
        for dst in gpus:
            if src.index == dst.index:
                continue
            a = torch.randn(numel, device=f"cuda:{src.index}")
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); b = a.to(f"cuda:{dst.index}"); e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e)
            bw = (size_mb / 1000) / (ms / 1000) if ms > 0 else 0
            print(f"  GPU {src.index} → GPU {dst.index}:  {bw:.2f} GB/s  ({ms:.1f}ms)")
            del a, b; torch.cuda.empty_cache()


# ── 5. Multi-GPU Scaling ───────────────────────────────────────────────────

def test_scaling(n_iters: int = 20, batch_size: int = 64) -> None:
    """Multi-GPU scaling efficiency (1..N GPUs)."""
    section("Multi-GPU Scaling Efficiency")
    gpus = _gpus()
    d_model, n_heads, seq_len = 2048, 16, 1024
    single_time: float | None = None

    for num_gpus in range(1, len(gpus) + 1):
        per_gpu = batch_size // num_gpus
        devices = [torch.device(f"cuda:{gpus[j].index}") for j in range(num_gpus)]
        layers = [nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True).to(d).eval() for d in devices]
        xs = [torch.randn(per_gpu, seq_len, d_model, device=d) for d in devices]
        with torch.no_grad():
            for l, x in zip(layers, xs): _ = l(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                for l, x in zip(layers, xs): _ = l(x)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if single_time is None:
            single_time, eff = elapsed, 100.0
        else:
            eff = (single_time / num_gpus) / elapsed * 100 if elapsed > 0 else 0

        tps = (n_iters * batch_size * seq_len) / elapsed if elapsed > 0 else 0
        print(f"  {num_gpus} GPU(s): {elapsed:.3f}s  {tps:,.0f} tok/s  efficiency={eff:.1f}%")
        del layers, xs; torch.cuda.empty_cache()


# ── 6. Compute Precision ───────────────────────────────────────────────────

def test_compute(matrix_size: int = 16384, n_iters: int = 20) -> None:
    """Matmul TFLOPS at FP32 / FP16 / BF16."""
    section(f"Compute  ({matrix_size}×{matrix_size} matmul TFLOPS)")
    gpus = _gpus()
    dtypes = [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]
    fpm = 2 * matrix_size ** 3

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        parts: list[str] = []
        for label, dt in dtypes:
            a = torch.randn(matrix_size, matrix_size, device=dev, dtype=dt)
            b = torch.randn(matrix_size, matrix_size, device=dev, dtype=dt)
            _ = torch.mm(a, b); torch.cuda.synchronize(dev)
            with cuda_timer(dev) as elapsed:
                for _ in range(n_iters): _ = torch.mm(a, b)
            tflops = (fpm * n_iters) / (elapsed[0] * 1e12) if elapsed[0] > 0 else 0
            parts.append(f"{label}={tflops:.1f}")
            del a, b
        torch.cuda.empty_cache()
        print(f"  GPU {gpu.index} ({gpu.name}):  {'  ·  '.join(parts)}  TFLOPS")


# ── 7. Memory Bandwidth ───────────────────────────────────────────────────

def test_bandwidth(size_mb: int = 2048, n_iters: int = 50) -> None:
    """Streaming memory read + write bandwidth."""
    section(f"Memory Bandwidth  ({size_mb} MB streaming r/w)")
    gpus = _gpus()
    numel = size_mb * 1024 * 1024 // 4
    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        a = torch.randn(numel, device=dev); b = torch.empty_like(a)
        b.copy_(a); torch.cuda.synchronize(dev)
        with cuda_timer(dev) as elapsed:
            for _ in range(n_iters): b.copy_(a)
        bw = (2 * size_mb * n_iters / 1000) / elapsed[0] if elapsed[0] > 0 else 0
        print(f"  GPU {gpu.index} ({gpu.name}):  {bw:.1f} GB/s")
        del a, b; torch.cuda.empty_cache()


# ── 8. KV-Cache Simulation ─────────────────────────────────────────────────

def test_kv_cache() -> None:
    """Simulate 70B-class LLM KV-cache growth until OOM."""
    section("KV-Cache Stress  (simulated LLM serving, growing context)")
    gpus = _gpus()
    n_layers, n_heads, head_dim = 40, 32, 128

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        torch.cuda.reset_peak_memory_stats(dev)
        seq_len, max_seq = 1024, 0
        kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            while True:
                for _ in range(n_layers):
                    k = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=dev)
                    v = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=dev)
                    kv.append((k, v))
                max_seq += seq_len
        except torch.cuda.OutOfMemoryError:
            pass
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        print(f"  GPU {gpu.index} ({gpu.name}):  KV cache → {max_seq:,} tokens  ({peak:.1f} / {gpu.vram_gb} GB)  ({n_layers}L × {n_heads}H × {head_dim}d)")
        del kv; cleanup(dev)


# ── 9. Gradient Accumulation Bomb ──────────────────────────────────────────

def test_gradient(n_accum: int = 16, n_iters: int = 5) -> None:
    """Forward + backward with gradient accumulation over a 6-layer transformer."""
    section(f"Gradient Bomb  ({n_accum}-step accumulation, 6-layer transformer)")
    gpus = _gpus()
    d_model, n_heads, ff_dim = 4096, 32, 16384
    seq_len, batch = 2048, 4

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        torch.cuda.reset_peak_memory_stats(dev)

        def _run(dev=dev, i=gpu.index):
            model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=ff_dim, batch_first=True),
                num_layers=6,
            ).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler(device=dev.type)
            x = torch.randn(batch, seq_len, d_model, device=dev)
            torch.cuda.synchronize(dev)

            t0 = time.perf_counter()
            for _ in range(n_iters):
                opt.zero_grad()
                for _ in range(n_accum):
                    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16):
                        out = model(x)
                    scaler.scale(out.sum() / n_accum).backward()
                scaler.step(opt); scaler.update()
                torch.cuda.synchronize(dev)
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated(dev) / 1024**3
            print(f"  GPU {i} ({gpu.name}):  {n_iters} steps  (eff. batch={batch * n_accum})  {elapsed:.1f}s  peak VRAM={peak:.1f}GB")
            del model, opt, scaler, x

        _monitored(gpu.index, _run)
        cleanup(dev)


# ── 10. Attention Stress ───────────────────────────────────────────────────

def test_attention() -> None:
    """Flash attention at extreme sequence lengths (4K → 128K), BF16."""
    section("Flash Attention Stress  (extreme sequence lengths, BF16)")
    gpus = _gpus()
    n_heads, head_dim = 32, 128
    seq_lengths = [4096, 8192, 16384, 32768, 65536, 131072]

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        results: list[str] = []
        for seq_len in seq_lengths:
            torch.cuda.reset_peak_memory_stats(dev)
            try:
                q = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=dev)
                k, v = torch.randn_like(q), torch.randn_like(q)
                with torch.no_grad():
                    _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize(dev)
                with cuda_timer(dev) as elapsed:
                    with torch.no_grad():
                        for _ in range(5):
                            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                peak = torch.cuda.max_memory_allocated(dev) / 1024**3
                results.append(f"{seq_len // 1024}K={elapsed[0] * 1000 / 5:.0f}ms/{peak:.0f}GB")
                del q, k, v; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                results.append(f"{seq_len // 1024}K=OOM")
                torch.cuda.empty_cache(); break
        print(f"  GPU {gpu.index} ({gpu.name}):  {' · '.join(results)}")


# ── Test registry ──────────────────────────────────────────────────────────

ALL_TESTS: dict[str, callable] = {
    "throughput": test_throughput,
    "memory": test_memory,
    "sustained": test_sustained,
    "p2p": test_p2p,
    "scaling": test_scaling,
    "compute": test_compute,
    "bandwidth": test_bandwidth,
    "kv_cache": test_kv_cache,
    "gradient": test_gradient,
    "attention": test_attention,
}


# ── Entry point ────────────────────────────────────────────────────────────

def run(args: argparse.Namespace | None = None) -> None:
    """Run selected stress tests."""
    global _NVML

    if args is None:
        parser = _build_parser()
        args = parser.parse_args()

    gpus = detect_gpus()
    _NVML = nvml_init()

    print_header("GPU Stress Test Suite", gpus)

    for gpu in gpus:
        print(f"\n  {gpu}")
        if _NVML:
            print(f"  {snapshot(gpu.index)}")

    tests = list(ALL_TESTS) if "all" in args.tests else args.tests

    for name in tests:
        fn = ALL_TESTS[name]
        if name == "sustained":
            fn(duration=args.duration)
        elif name in ("p2p", "scaling") and len(gpus) < 2:
            print(f"\n  ⚠ Skipping {name} (needs ≥ 2 GPUs)")
        else:
            fn()

    if _NVML:
        section("Final GPU State")
        for gpu in gpus:
            print(f"  {snapshot(gpu.index)}")
        nvml_shutdown()

    print(f"\n{'=' * 72}")
    print("  All tests complete.")
    print(f"{'=' * 72}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-GPU stress test suite — CUDA 13.0+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tests", nargs="+", choices=[*ALL_TESTS, "all"], default=["all"],
        help="Tests to run (default: all)",
    )
    parser.add_argument("--duration", type=int, default=30, help="Sustained test duration (s)")
    return parser
