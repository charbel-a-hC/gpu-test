"""Multi-GPU stress testing suite — 10 tests designed to push high-VRAM GPUs.

All tests return structured ``TestResult`` objects with per-test targeted
metrics, config parameters, and NVML telemetry.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpu_test.monitor import (
    MonitorStats, init as nvml_init, run_monitored, shutdown as nvml_shutdown, snapshot,
)
from gpu_test.utils import (
    GpuDevice, TestResult, cleanup, cuda_timer, detect_gpus, print_header, require_gpus, section,
)


_NVML: bool = False


# ── Telemetry helper ────────────────────────────────────────────────────────


def _telem(stats: MonitorStats | None) -> dict[str, Any]:
    if stats is None or stats.samples == 0:
        return {}
    return {
        "avg_gpu_util_pct": round(stats.avg_gpu_util, 1),
        "peak_gpu_util_pct": round(stats.peak_gpu_util, 1),
        "peak_mem_gb": round(stats.peak_mem_gb, 1),
        "avg_power_w": round(stats.avg_power_w, 0),
        "peak_power_w": round(stats.peak_power_w, 0),
        "avg_temp_c": round(stats.avg_temp_c, 0),
        "peak_temp_c": stats.peak_temp_c,
        "peak_clock_mhz": stats.peak_clock_gpu_mhz,
        "samples": stats.samples,
    }


def _print_telem(tl: dict) -> None:
    if tl:
        print(
            f"    ⚡ GPU={tl.get('avg_gpu_util_pct', '?')}→{tl.get('peak_gpu_util_pct', '?')}%  "
            f"Power={tl.get('avg_power_w', '?')}→{tl.get('peak_power_w', '?')}W  "
            f"Temp={tl.get('avg_temp_c', '?')}→{tl.get('peak_temp_c', '?')}°C"
        )


# ── 1. Throughput ────────────────────────────────────────────────────────────


def test_throughput(n_iters: int = 30, batch_size: int = 32) -> list[TestResult]:
    """Transformer inference throughput (tok/s)."""
    section("Throughput  (transformer inference, tok/s)")
    gpus = require_gpus()
    d_model, n_heads, seq_len = 2048, 16, 2048
    results: list[TestResult] = []

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")

        def _work(dev=dev):
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True
            ).to(dev).eval()
            x = torch.randn(batch_size, seq_len, d_model, device=dev)
            with torch.no_grad(): _ = layer(x)
            torch.cuda.synchronize(dev)
            with cuda_timer(dev) as elapsed:
                with torch.no_grad():
                    for _ in range(n_iters): _ = layer(x)
            del layer, x
            return elapsed

        elapsed, stats = run_monitored(gpu.index, _work, nvml_ok=_NVML)
        tps = (n_iters * batch_size * seq_len) / elapsed[0] if elapsed[0] > 0 else 0
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        r = TestResult(
            suite="stress", test_name="throughput", gpu_index=gpu.index, gpu_name=gpu.name,
            elapsed_s=elapsed[0], peak_vram_gb=peak,
            metrics={"tok_per_s": round(tps), "batch_size": batch_size, "seq_len": seq_len},
            config={"d_model": d_model, "n_heads": n_heads, "n_iters": n_iters},
            telemetry=_telem(stats),
        )
        results.append(r)
        print(f"  {r.summary_line()}")
        _print_telem(r.telemetry)
        cleanup(dev)

    return results


# ── 2. Memory Stress ────────────────────────────────────────────────────────


def test_memory() -> list[TestResult]:
    """Progressive VRAM fill (1 GB BF16 chunks)."""
    section("Memory Stress  (fill VRAM with BF16 model-weight tensors)")
    gpus = require_gpus()
    results: list[TestResult] = []

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        torch.cuda.reset_peak_memory_stats(dev)
        chunk = 1024 * 1024 * 1024 // 2
        tensors: list[torch.Tensor] = []
        try:
            while True:
                tensors.append(torch.randn(chunk, dtype=torch.bfloat16, device=dev))
        except torch.cuda.OutOfMemoryError:
            pass
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        pct = (peak / gpu.vram_gb * 100) if gpu.vram_gb > 0 else 0
        r = TestResult(
            suite="stress", test_name="memory", gpu_index=gpu.index, gpu_name=gpu.name,
            peak_vram_gb=peak,
            metrics={"filled_gb": round(peak, 1), "total_gb": gpu.vram_gb, "filled_pct": round(pct, 1), "chunk_gb": 1.0},
        )
        results.append(r)
        print(f"  {r.summary_line()}")
        del tensors; cleanup(dev)

    return results


# ── 3. Sustained Load ──────────────────────────────────────────────────────


def test_sustained(duration: int = 30) -> list[TestResult]:
    """Continuous AMP training for N seconds."""
    section(f"Sustained Load  ({duration}s continuous AMP training)")
    gpus = require_gpus()
    d_model, n_heads, seq_len, batch = 2048, 16, 1024, 16
    results: list[TestResult] = []

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")

        def _work(dev=dev):
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
                except Exception:
                    errors += 1
                    if errors >= 10: break
            wall = time.monotonic() - t0
            del layer, opt, scaler, x
            return {"iters": iters, "wall": wall, "errors": errors, "steps_per_s": iters / wall if wall > 0 else 0}

        data, stats = run_monitored(gpu.index, _work, nvml_ok=_NVML)
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        r = TestResult(
            suite="stress", test_name="sustained", gpu_index=gpu.index, gpu_name=gpu.name,
            elapsed_s=data["wall"], peak_vram_gb=peak,
            metrics={"steps": data["iters"], "steps_per_s": round(data["steps_per_s"], 1), "errors": data["errors"]},
            config={"duration_s": duration, "d_model": d_model, "batch": batch, "seq_len": seq_len},
            telemetry=_telem(stats),
        )
        results.append(r)
        print(f"  {r.summary_line()}")
        _print_telem(r.telemetry)
        cleanup(dev)

    return results


# ── 4. P2P Bandwidth ───────────────────────────────────────────────────────


def test_p2p(size_mb: int = 1024) -> list[TestResult]:
    """GPU-to-GPU peer-to-peer transfer bandwidth."""
    section(f"P2P GPU Bandwidth  ({size_mb} MB transfers)")
    gpus = require_gpus(2)
    numel = size_mb * 1024 * 1024 // 4
    results: list[TestResult] = []

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
            r = TestResult(
                suite="stress", test_name="p2p", gpu_index=src.index, gpu_name=src.name,
                elapsed_s=ms / 1000,
                metrics={"bandwidth_gb_s": round(bw, 2), "size_mb": size_mb, "dst_gpu": dst.index},
            )
            results.append(r)
            print(f"  GPU {src.index} → GPU {dst.index}:  {bw:.2f} GB/s  ({ms:.1f}ms)")
            del a, b; torch.cuda.empty_cache()

    return results


# ── 5. Multi-GPU Scaling ───────────────────────────────────────────────────


def test_scaling(n_iters: int = 20, batch_size: int = 64) -> list[TestResult]:
    """Multi-GPU scaling efficiency (1..N GPUs)."""
    section("Multi-GPU Scaling Efficiency")
    gpus = require_gpus()
    d_model, n_heads, seq_len = 2048, 16, 1024
    single_time: float | None = None
    results: list[TestResult] = []

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
        r = TestResult(
            suite="stress", test_name="scaling", gpu_index=num_gpus, gpu_name=f"{num_gpus}xGPU",
            elapsed_s=elapsed,
            metrics={"tok_per_s": round(tps), "efficiency_pct": round(eff, 1), "num_gpus": num_gpus},
            config={"batch_size": batch_size, "n_iters": n_iters},
        )
        results.append(r)
        print(f"  {num_gpus} GPU(s): {elapsed:.3f}s  {tps:,.0f} tok/s  efficiency={eff:.1f}%")
        del layers, xs; torch.cuda.empty_cache()

    return results


# ── 6. Compute Precision ───────────────────────────────────────────────────


def test_compute(matrix_size: int = 16384, n_iters: int = 20) -> list[TestResult]:
    """Matmul TFLOPS at FP32 / FP16 / BF16."""
    section(f"Compute  ({matrix_size}×{matrix_size} matmul TFLOPS)")
    gpus = require_gpus()
    dtypes = [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]
    fpm = 2 * matrix_size ** 3
    results: list[TestResult] = []

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
            r = TestResult(
                suite="stress", test_name=f"compute_{label.lower()}", gpu_index=gpu.index, gpu_name=gpu.name,
                elapsed_s=elapsed[0],
                metrics={"tflops": round(tflops, 2), "dtype": label},
                config={"matrix_size": matrix_size, "n_iters": n_iters},
            )
            results.append(r)
            del a, b
        torch.cuda.empty_cache()
        print(f"  GPU {gpu.index} ({gpu.name}):  {'  ·  '.join(parts)}  TFLOPS")

    return results


# ── 7. Memory Bandwidth ───────────────────────────────────────────────────


def test_bandwidth(size_mb: int = 2048, n_iters: int = 50) -> list[TestResult]:
    """Streaming memory read + write bandwidth."""
    section(f"Memory Bandwidth  ({size_mb} MB streaming r/w)")
    gpus = require_gpus()
    numel = size_mb * 1024 * 1024 // 4
    results: list[TestResult] = []

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        a = torch.randn(numel, device=dev); b = torch.empty_like(a)
        b.copy_(a); torch.cuda.synchronize(dev)
        with cuda_timer(dev) as elapsed:
            for _ in range(n_iters): b.copy_(a)
        bw = (2 * size_mb * n_iters / 1000) / elapsed[0] if elapsed[0] > 0 else 0
        r = TestResult(
            suite="stress", test_name="bandwidth", gpu_index=gpu.index, gpu_name=gpu.name,
            elapsed_s=elapsed[0],
            metrics={"bandwidth_gb_s": round(bw, 1), "size_mb": size_mb},
            config={"n_iters": n_iters},
        )
        results.append(r)
        print(f"  GPU {gpu.index} ({gpu.name}):  {bw:.1f} GB/s")
        del a, b; torch.cuda.empty_cache()

    return results


# ── 8. KV-Cache Simulation ─────────────────────────────────────────────────


def test_kv_cache() -> list[TestResult]:
    """Simulate 70B-class LLM KV-cache growth until OOM."""
    section("KV-Cache Stress  (simulated LLM serving, growing context)")
    gpus = require_gpus()
    n_layers, n_heads, head_dim = 40, 32, 128
    results: list[TestResult] = []

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
        r = TestResult(
            suite="stress", test_name="kv_cache", gpu_index=gpu.index, gpu_name=gpu.name,
            peak_vram_gb=peak,
            metrics={"max_context_tokens": max_seq, "filled_gb": round(peak, 1)},
            config={"n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim, "chunk_tokens": seq_len},
        )
        results.append(r)
        print(f"  GPU {gpu.index} ({gpu.name}):  KV → {max_seq:,} tokens  ({peak:.1f} / {gpu.vram_gb} GB)")
        del kv; cleanup(dev)

    return results


# ── 9. Gradient Accumulation Bomb ──────────────────────────────────────────


def test_gradient(n_accum: int = 16, n_iters: int = 5) -> list[TestResult]:
    """Forward + backward with gradient accumulation over a 6-layer transformer."""
    section(f"Gradient Bomb  ({n_accum}-step accumulation, 6-layer transformer)")
    gpus = require_gpus()
    d_model, n_heads, ff_dim = 4096, 32, 16384
    seq_len, batch = 2048, 4
    results: list[TestResult] = []

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        torch.cuda.reset_peak_memory_stats(dev)

        def _work(dev=dev):
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
            del model, opt, scaler, x
            return elapsed

        elapsed, stats = run_monitored(gpu.index, _work, nvml_ok=_NVML)
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        eff_batch = batch * n_accum
        r = TestResult(
            suite="stress", test_name="gradient", gpu_index=gpu.index, gpu_name=gpu.name,
            elapsed_s=elapsed, peak_vram_gb=peak,
            metrics={"optimizer_steps": n_iters, "effective_batch": eff_batch, "steps_per_s": round(n_iters / elapsed, 2) if elapsed > 0 else 0},
            config={"n_accum": n_accum, "d_model": d_model, "n_heads": n_heads, "seq_len": seq_len, "batch": batch, "num_layers": 6},
            telemetry=_telem(stats),
        )
        results.append(r)
        print(f"  {r.summary_line()}")
        _print_telem(r.telemetry)
        cleanup(dev)

    return results


# ── 10. Attention Stress ───────────────────────────────────────────────────


def test_attention() -> list[TestResult]:
    """Flash attention at extreme sequence lengths (4K → 128K), BF16."""
    section("Flash Attention Stress  (extreme sequence lengths, BF16)")
    gpus = require_gpus()
    n_heads, head_dim = 32, 128
    seq_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    results: list[TestResult] = []

    for gpu in gpus:
        dev = torch.device(f"cuda:{gpu.index}")
        line: list[str] = []
        for seq_len in seq_lengths:
            torch.cuda.reset_peak_memory_stats(dev)
            try:
                q = torch.randn(1, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=dev)
                k, v = torch.randn_like(q), torch.randn_like(q)
                with torch.no_grad(): _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize(dev)
                with cuda_timer(dev) as elapsed:
                    with torch.no_grad():
                        for _ in range(5):
                            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                peak = torch.cuda.max_memory_allocated(dev) / 1024**3
                ms_per = elapsed[0] * 1000 / 5
                r = TestResult(
                    suite="stress", test_name=f"attention_{seq_len // 1024}K", gpu_index=gpu.index, gpu_name=gpu.name,
                    elapsed_s=elapsed[0], peak_vram_gb=peak,
                    metrics={"seq_len": seq_len, "ms_per_call": round(ms_per, 1), "peak_gb": round(peak, 1)},
                    config={"n_heads": n_heads, "head_dim": head_dim},
                )
                results.append(r)
                line.append(f"{seq_len // 1024}K={ms_per:.0f}ms/{peak:.0f}GB")
                del q, k, v; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                r = TestResult(
                    suite="stress", test_name=f"attention_{seq_len // 1024}K", gpu_index=gpu.index, gpu_name=gpu.name,
                    status="oom", metrics={"seq_len": seq_len},
                )
                results.append(r)
                line.append(f"{seq_len // 1024}K=OOM")
                torch.cuda.empty_cache(); break
        print(f"  GPU {gpu.index} ({gpu.name}):  {' · '.join(line)}")

    return results


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


def run(args: argparse.Namespace | None = None) -> list[TestResult]:
    """Run selected stress tests. Returns all results."""
    global _NVML

    if args is None:
        parser = build_parser()
        args = parser.parse_args()

    gpus = detect_gpus()
    _NVML = nvml_init()

    print_header("GPU Stress Test Suite", gpus)

    for gpu in gpus:
        print(f"\n  {gpu}")
        if _NVML:
            print(f"  {snapshot(gpu.index)}")

    tests = list(ALL_TESTS) if "all" in args.tests else args.tests
    all_results: list[TestResult] = []

    for name in tests:
        fn = ALL_TESTS[name]
        if name == "sustained":
            all_results.extend(fn(duration=args.duration))
        elif name in ("p2p", "scaling") and len(gpus) < 2:
            print(f"\n  ⚠ Skipping {name} (needs ≥ 2 GPUs)")
        else:
            all_results.extend(fn())

    if _NVML:
        section("Final GPU State")
        for gpu in gpus:
            print(f"  {snapshot(gpu.index)}")
        nvml_shutdown()

    print(f"\n{'=' * 72}")
    print(f"  All tests complete — {len(all_results)} results collected.")
    print(f"{'=' * 72}")
    return all_results


def build_parser() -> argparse.ArgumentParser:
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
