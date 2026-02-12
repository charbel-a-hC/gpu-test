"""Smart GPU benchmark — modern AI workloads with OOM-safe progressive fallback.

Each workload starts with aggressive parameters and automatically reduces them
on OOM, safely finding the maximum stress point for each GPU.

Returns structured ``TestResult`` objects; ``run()`` returns the full list.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpu_test.monitor import (
    MonitorStats, init as nvml_init, run_monitored, shutdown as nvml_shutdown, snapshot,
)
from gpu_test.utils import (
    GpuDevice, TestResult, banner, cleanup, cuda_timer, detect_gpus, note, print_header,
)


# ── Telemetry helper ────────────────────────────────────────────────────────


def _telem_dict(stats: MonitorStats | None) -> dict:
    """Convert MonitorStats to a flat dict for the CSV."""
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


# ── Workloads ────────────────────────────────────────────────────────────────
# Each returns (TestResult, MonitorStats | None)


def _transformer_block(
    gpu: GpuDevice, device: torch.device, nvml_ok: bool, n_iters: int = 10,
) -> TestResult:
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
            torch.cuda.synchronize(device)

            stats: MonitorStats | None = None

            def _work():
                nonlocal stats
                with cuda_timer(device) as elapsed:
                    with torch.no_grad():
                        for _ in range(n_iters):
                            h2 = ln1(x); h2, _ = attn(h2, h2, h2); h2 = x + h2; h2 = h2 + ffn(ln2(h2))
                return elapsed

            elapsed, stats = run_monitored(gpu.index, _work, nvml_ok=nvml_ok)

            tokens = cfg["batch"] * cfg["seq_len"] * n_iters
            tps = tokens / elapsed[0] if elapsed[0] > 0 else 0
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            return TestResult(
                suite="bench", test_name="Transformer Block", gpu_index=gpu.index, gpu_name=gpu.name,
                elapsed_s=elapsed[0], peak_vram_gb=peak,
                metrics={"tok_per_s": round(tps), "tokens_total": tokens, "iters": n_iters},
                config=cfg, telemetry=_telem_dict(stats),
            )
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at batch={cfg['batch']} seq={cfg['seq_len']}, reducing…")
            cleanup(device)

    return TestResult(suite="bench", test_name="Transformer Block", gpu_index=gpu.index, gpu_name=gpu.name, status="oom")


def _sdpa_attention(
    gpu: GpuDevice, device: torch.device, nvml_ok: bool, n_iters: int = 10,
) -> TestResult:
    seq_lengths = [65536, 32768, 16384, 8192, 4096]
    n_heads, head_dim, batch = 32, 128, 4

    for seq_len in seq_lengths:
        cleanup(device)
        try:
            q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
            k, v = torch.randn_like(q), torch.randn_like(q)
            with torch.no_grad():
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize(device)

            def _work():
                with cuda_timer(device) as elapsed:
                    with torch.no_grad():
                        for _ in range(n_iters):
                            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                return elapsed

            elapsed, stats = run_monitored(gpu.index, _work, nvml_ok=nvml_ok)
            tokens = batch * seq_len * n_iters
            tps = tokens / elapsed[0] if elapsed[0] > 0 else 0
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            return TestResult(
                suite="bench", test_name="SDPA / Flash Attention (BF16)",
                gpu_index=gpu.index, gpu_name=gpu.name,
                elapsed_s=elapsed[0], peak_vram_gb=peak,
                metrics={"tok_per_s": round(tps), "seq_len": seq_len},
                config={"seq_len": seq_len, "heads": n_heads, "head_dim": head_dim, "batch": batch},
                telemetry=_telem_dict(stats),
            )
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at seq_len={seq_len}, reducing…")
            cleanup(device)

    return TestResult(suite="bench", test_name="SDPA / Flash Attention (BF16)", gpu_index=gpu.index, gpu_name=gpu.name, status="oom")


def _large_matmul(
    gpu: GpuDevice, device: torch.device, nvml_ok: bool, n_iters: int = 10,
) -> TestResult:
    sizes = [(32768, 32768, 16384), (16384, 16384, 16384), (16384, 16384, 8192), (8192, 8192, 8192)]
    for m, n, k in sizes:
        cleanup(device)
        try:
            a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
            _ = torch.mm(a, b); torch.cuda.synchronize(device)

            def _work():
                with cuda_timer(device) as elapsed:
                    for _ in range(n_iters):
                        _ = torch.mm(a, b)
                return elapsed

            elapsed, stats = run_monitored(gpu.index, _work, nvml_ok=nvml_ok)
            flops = 2 * m * n * k * n_iters
            tflops = flops / (elapsed[0] * 1e12) if elapsed[0] > 0 else 0
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            return TestResult(
                suite="bench", test_name="Large GEMM (BF16)",
                gpu_index=gpu.index, gpu_name=gpu.name,
                elapsed_s=elapsed[0], peak_vram_gb=peak,
                metrics={"tflops": round(tflops, 2), "flops_total": flops},
                config={"M": m, "N": n, "K": k},
                telemetry=_telem_dict(stats),
            )
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at [{m}×{k}]@[{k}×{n}], reducing…")
            cleanup(device)

    return TestResult(suite="bench", test_name="Large GEMM (BF16)", gpu_index=gpu.index, gpu_name=gpu.name, status="oom")


def _mixed_precision_train(
    gpu: GpuDevice, device: torch.device, nvml_ok: bool, n_iters: int = 10,
) -> TestResult:
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
            out.sum().backward(); opt.zero_grad(); torch.cuda.synchronize(device)

            def _work():
                with cuda_timer(device) as elapsed:
                    for _ in range(n_iters):
                        opt.zero_grad()
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                            out2 = layer(x)
                        scaler.scale(out2.sum()).backward()
                        scaler.step(opt); scaler.update()
                return elapsed

            elapsed, stats = run_monitored(gpu.index, _work, nvml_ok=nvml_ok)
            sps = n_iters / elapsed[0] if elapsed[0] > 0 else 0
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            return TestResult(
                suite="bench", test_name="Mixed-Precision Training",
                gpu_index=gpu.index, gpu_name=gpu.name,
                elapsed_s=elapsed[0], peak_vram_gb=peak,
                metrics={"steps_per_s": round(sps, 2), "iters": n_iters},
                config=cfg, telemetry=_telem_dict(stats),
            )
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at batch={cfg['batch']} seq={cfg['seq_len']}, reducing…")
            cleanup(device)

    return TestResult(suite="bench", test_name="Mixed-Precision Training", gpu_index=gpu.index, gpu_name=gpu.name, status="oom")


def _conv_resnet(
    gpu: GpuDevice, device: torch.device, nvml_ok: bool, n_iters: int = 20,
) -> TestResult:
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
            with torch.no_grad(): _ = model(x)
            torch.cuda.synchronize(device)

            def _work():
                with cuda_timer(device) as elapsed:
                    with torch.no_grad():
                        for _ in range(n_iters): _ = model(x)
                return elapsed

            elapsed, stats = run_monitored(gpu.index, _work, nvml_ok=nvml_ok)
            ips = (cfg["batch"] * n_iters) / elapsed[0] if elapsed[0] > 0 else 0
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            return TestResult(
                suite="bench", test_name="Conv Stack (ResNet)",
                gpu_index=gpu.index, gpu_name=gpu.name,
                elapsed_s=elapsed[0], peak_vram_gb=peak,
                metrics={"img_per_s": round(ips), "iters": n_iters},
                config={"batch": cfg["batch"], "resolution": f"{cfg['res']}x{cfg['res']}"},
                telemetry=_telem_dict(stats),
            )
        except torch.cuda.OutOfMemoryError:
            note(f"OOM at batch={cfg['batch']} res={cfg['res']}, reducing…")
            cleanup(device)

    return TestResult(suite="bench", test_name="Conv Stack (ResNet)", gpu_index=gpu.index, gpu_name=gpu.name, status="oom")


def _vram_fill(
    gpu: GpuDevice, device: torch.device, nvml_ok: bool,
) -> TestResult:
    cleanup(device)
    torch.cuda.reset_peak_memory_stats(device)
    tensors: list[torch.Tensor] = []
    try:
        while True:
            tensors.append(torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=device))
    except torch.cuda.OutOfMemoryError:
        pass
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    total_gb = gpu.vram_gb
    pct = (peak_gb / total_gb) * 100 if total_gb > 0 else 0
    del tensors
    cleanup(device)
    return TestResult(
        suite="bench", test_name="VRAM Fill",
        gpu_index=gpu.index, gpu_name=gpu.name,
        peak_vram_gb=peak_gb,
        metrics={"filled_gb": round(peak_gb, 1), "total_gb": total_gb, "filled_pct": round(pct, 1)},
    )


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


def run() -> list[TestResult]:
    """Run all benchmarks on every detected GPU. Returns all results."""
    gpus = detect_gpus()
    nvml_ok = nvml_init()

    print_header("Smart GPU Benchmark", gpus)

    for gpu in gpus:
        print(f"\n  {gpu}")
        if nvml_ok:
            print(f"  {snapshot(gpu.index)}")

    results: list[TestResult] = []

    for label, fn in ALL_BENCHMARKS:
        banner(label)
        for gpu in gpus:
            dev = torch.device(f"cuda:{gpu.index}")
            torch.cuda.reset_peak_memory_stats(dev)
            try:
                result = fn(gpu, dev, nvml_ok)
                results.append(result)
                print(f"  {result.summary_line()}")
                if result.telemetry:
                    tl = result.telemetry
                    print(
                        f"    ⚡ GPU={tl.get('avg_gpu_util_pct', '?')}%→{tl.get('peak_gpu_util_pct', '?')}%  "
                        f"Power={tl.get('avg_power_w', '?')}→{tl.get('peak_power_w', '?')}W  "
                        f"Temp={tl.get('avg_temp_c', '?')}→{tl.get('peak_temp_c', '?')}°C  "
                        f"Clock↑{tl.get('peak_clock_mhz', '?')}MHz"
                    )
            except Exception as exc:
                results.append(TestResult(
                    suite="bench", test_name=label, gpu_index=gpu.index,
                    gpu_name=gpu.name, status="error",
                    metrics={"error": str(exc)},
                ))
                print(f"  GPU {gpu.index}: ERROR — {exc}")
            finally:
                cleanup(dev)

    if nvml_ok:
        nvml_shutdown()

    print(f"\n{'=' * 72}")
    print(f"  Benchmark complete — {len(results)} results collected.")
    print(f"{'=' * 72}")
    return results
