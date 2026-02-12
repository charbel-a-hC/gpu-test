"""Smart GPU benchmark — modern AI workloads with OOM-safe progressive fallback.

Each workload starts with aggressive parameters and automatically reduces them
on OOM, safely finding the maximum stress point for each GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpu_test.monitor import run_monitored, init as nvml_init, shutdown as nvml_shutdown, snapshot
from gpu_test.utils import (
    BenchResult, GpuDevice, banner, cleanup, cuda_timer, detect_gpus, note, print_header,
)


# ── Workloads ────────────────────────────────────────────────────────────────


def _transformer_block(device: torch.device, n_iters: int = 10) -> BenchResult:
    """Multi-head self-attention + FFN (LLM / ViT core)."""
    configs = [
        {"d_model": 4096, "n_heads": 32, "seq_len": 8192, "batch": 16},
        {"d_model": 4096, "n_heads": 32, "seq_len": 4096, "batch": 16},
        {"d_model": 4096, "n_heads": 32, "seq_len": 4096, "batch": 8},
        {"d_model": 4096, "n_heads": 32, "seq_len": 2048, "batch": 8},
        {"d_model": 2048, "n_heads": 16, "seq_len": 2048, "batch": 8},
        {"d_model": 2048, "n_heads": 16, "seq_len": 1024, "batch": 4},
    ]
    for cfg in configs:
        cleanup(device)
        try:
            attn = nn.MultiheadAttention(cfg["d_model"], cfg["n_heads"], batch_first=True).to(device)
            ffn = nn.Sequential(
                nn.Linear(cfg["d_model"], cfg["d_model"] * 4), nn.GELU(),
                nn.Linear(cfg["d_model"] * 4, cfg["d_model"]),
            ).to(device)
            ln1 = nn.LayerNorm(cfg["d_model"]).to(device)
            ln2 = nn.LayerNorm(cfg["d_model"]).to(device)
            x = torch.randn(cfg["batch"], cfg["seq_len"], cfg["d_model"], device=device)

            with torch.no_grad():
                h = ln1(x); h, _ = attn(h, h, h); h = x + h; h = h + ffn(ln2(h))
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            with cuda_timer(device) as elapsed:
                with torch.no_grad():
                    for _ in range(n_iters):
                        h = ln1(x); h, _ = attn(h, h, h); h = x + h; h = h + ffn(ln2(h))

            tokens = cfg["batch"] * cfg["seq_len"] * n_iters
            tps = tokens / elapsed[0] if elapsed[0] > 0 else 0
            return BenchResult("Transformer Block", str(device), elapsed[0], f"{tps:,.0f} tok/s", cfg)
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at batch={cfg['batch']} seq={cfg['seq_len']} d={cfg['d_model']}, reducing…")
            cleanup(device)
    return BenchResult("Transformer Block", str(device), 0, "all configs OOM")


def _sdpa_attention(device: torch.device, n_iters: int = 10) -> BenchResult:
    """Scaled dot-product / flash attention at long context (BF16)."""
    seq_lengths = [65536, 32768, 16384, 8192, 4096]
    n_heads, head_dim, batch = 32, 128, 4

    for seq_len in seq_lengths:
        cleanup(device)
        try:
            q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
            k, v = torch.randn_like(q), torch.randn_like(q)

            with torch.no_grad():
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            with cuda_timer(device) as elapsed:
                with torch.no_grad():
                    for _ in range(n_iters):
                        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            tokens = batch * seq_len * n_iters
            tps = tokens / elapsed[0] if elapsed[0] > 0 else 0
            return BenchResult(
                "SDPA / Flash Attention (BF16)", str(device), elapsed[0], f"{tps:,.0f} tok/s",
                {"seq": seq_len, "heads": n_heads, "head_dim": head_dim, "batch": batch},
            )
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at seq_len={seq_len}, reducing…")
            cleanup(device)
    return BenchResult("SDPA / Flash Attention (BF16)", str(device), 0, "all configs OOM")


def _large_matmul(device: torch.device, n_iters: int = 10) -> BenchResult:
    """Large GEMM (BF16) — LLM forward-pass bottleneck."""
    sizes = [(32768, 32768, 16384), (16384, 16384, 16384), (16384, 16384, 8192), (8192, 8192, 8192)]
    for m, n, k in sizes:
        cleanup(device)
        try:
            a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
            _ = torch.mm(a, b)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            with cuda_timer(device) as elapsed:
                for _ in range(n_iters):
                    _ = torch.mm(a, b)
            flops = 2 * m * n * k * n_iters
            tflops = flops / (elapsed[0] * 1e12) if elapsed[0] > 0 else 0
            return BenchResult("Large GEMM (BF16)", str(device), elapsed[0], f"{tflops:.1f} TFLOPS",
                               {"shape": f"[{m}×{k}] @ [{k}×{n}]"})
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at [{m}×{k}] @ [{k}×{n}], reducing…")
            cleanup(device)
    return BenchResult("Large GEMM (BF16)", str(device), 0, "all configs OOM")


def _mixed_precision_train(device: torch.device, n_iters: int = 10) -> BenchResult:
    """Full forward + backward with AMP (mixed precision training step)."""
    configs = [
        {"d_model": 4096, "n_heads": 32, "seq_len": 4096, "batch": 16},
        {"d_model": 4096, "n_heads": 32, "seq_len": 2048, "batch": 16},
        {"d_model": 2048, "n_heads": 16, "seq_len": 2048, "batch": 16},
        {"d_model": 2048, "n_heads": 16, "seq_len": 2048, "batch": 8},
        {"d_model": 2048, "n_heads": 16, "seq_len": 1024, "batch": 8},
    ]
    for cfg in configs:
        cleanup(device)
        try:
            layer = nn.TransformerEncoderLayer(
                cfg["d_model"], cfg["n_heads"], dim_feedforward=cfg["d_model"] * 4, batch_first=True,
            ).to(device)
            opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler(device=device.type)
            x = torch.randn(cfg["batch"], cfg["seq_len"], cfg["d_model"], device=device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = layer(x)
            out.sum().backward(); opt.zero_grad()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            with cuda_timer(device) as elapsed:
                for _ in range(n_iters):
                    opt.zero_grad()
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        out = layer(x)
                    scaler.scale(out.sum()).backward()
                    scaler.step(opt); scaler.update()

            sps = n_iters / elapsed[0] if elapsed[0] > 0 else 0
            return BenchResult("Mixed-Precision Training", str(device), elapsed[0], f"{sps:.1f} steps/s", cfg)
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at batch={cfg['batch']} seq={cfg['seq_len']} d={cfg['d_model']}, reducing…")
            cleanup(device)
    return BenchResult("Mixed-Precision Training", str(device), 0, "all configs OOM")


def _conv_resnet(device: torch.device, n_iters: int = 20) -> BenchResult:
    """Heavy ResNet-style conv stack at high resolution."""
    configs = [
        {"batch": 128, "res": 512}, {"batch": 64, "res": 512}, {"batch": 32, "res": 512},
        {"batch": 64, "res": 256}, {"batch": 32, "res": 256},
    ]
    for cfg in configs:
        cleanup(device)
        try:
            model = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1000),
            ).to(device).eval()
            x = torch.randn(cfg["batch"], 3, cfg["res"], cfg["res"], device=device)
            with torch.no_grad():
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            with cuda_timer(device) as elapsed:
                with torch.no_grad():
                    for _ in range(n_iters):
                        _ = model(x)
            ips = (cfg["batch"] * n_iters) / elapsed[0] if elapsed[0] > 0 else 0
            return BenchResult("Conv Stack (ResNet)", str(device), elapsed[0], f"{ips:,.0f} img/s",
                               {"batch": cfg["batch"], "res": f"{cfg['res']}×{cfg['res']}"})
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at batch={cfg['batch']} res={cfg['res']}, reducing…")
            cleanup(device)
    return BenchResult("Conv Stack (ResNet)", str(device), 0, "all configs OOM")


def _vram_fill(device: torch.device) -> BenchResult:
    """Fill VRAM to measure usable capacity."""
    cleanup(device)
    tensors: list[torch.Tensor] = []
    try:
        while True:
            tensors.append(torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=device))
    except torch.cuda.OutOfMemoryError:
        pass
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    total_gb = torch.cuda.get_device_properties(device.index or 0).total_mem / 1024**3
    pct = (peak_gb / total_gb) * 100
    del tensors
    cleanup(device)
    return BenchResult("VRAM Fill", str(device), 0, f"{peak_gb:.1f} / {total_gb:.1f} GB ({pct:.0f}%)")


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_BENCHMARKS: list[tuple[str, callable]] = [
    ("Transformer Block (LLM-style)", _transformer_block),
    ("SDPA / Flash Attention (BF16)", _sdpa_attention),
    ("Large GEMM (BF16)", _large_matmul),
    ("Mixed-Precision Training Step", _mixed_precision_train),
    ("Conv Stack (Vision)", _conv_resnet),
    ("VRAM Capacity Fill", _vram_fill),
]


# ── Entry point ──────────────────────────────────────────────────────────────


def run() -> None:
    """Run all benchmarks on every detected GPU."""
    gpus = detect_gpus()
    nvml_ok = nvml_init()

    print_header("Smart GPU Benchmark", gpus)

    for gpu in gpus:
        print(f"\n  {gpu}")
        if nvml_ok:
            print(f"  {snapshot(gpu.index)}")

    for label, fn in ALL_BENCHMARKS:
        banner(label)
        for gpu in gpus:
            dev = torch.device(f"cuda:{gpu.index}")
            torch.cuda.reset_peak_memory_stats(dev)
            try:
                def _work(dev=dev):
                    return fn(dev)

                result = run_monitored(gpu.index, _work, nvml_ok=nvml_ok)
                mem_gb = torch.cuda.max_memory_allocated(dev) / 1024**3
                extra = "  ".join(f"{k}={v}" for k, v in result.extra.items())
                print(
                    f"  GPU {gpu.index}: {result.throughput:<28s}  "
                    f"{result.elapsed_s:.3f}s  peak VRAM={mem_gb:.1f}GB"
                    + (f"  ({extra})" if extra else "")
                )
            except Exception as exc:
                print(f"  GPU {gpu.index}: ERROR — {exc}")
            finally:
                cleanup(dev)

    if nvml_ok:
        nvml_shutdown()

    print(f"\n{'=' * 72}")
    print("  Benchmark complete.")
    print(f"{'=' * 72}")
