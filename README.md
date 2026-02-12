# gpu-test

Production-grade GPU benchmark & stress testing kit for CUDA 13.0+ NVIDIA GPUs.

Built with PyTorch 2.9+ and NVML telemetry — designed to validate new hardware (Blackwell RTX 6000 Pro and beyond) with modern AI workloads.

> [!NOTE]
> Tested on **NVIDIA RTX 6000 Pro** (Blackwell, 96 GB VRAM).

## Quick start (Docker)

```bash
# GPU info + telemetry snapshot
make info

# Smart benchmark (6 AI workloads, OOM-safe) → writes CSV
make bench

# Full stress-test suite (10 tests, default 30s sustained) → writes CSV
make stress

# Pass CLI args via ARGS=
make stress ARGS="--tests throughput kv_cache --duration 120"
make bench ARGS="--output results/bench.csv"
```

## CLI

```bash
gpu-test info                                           # NVML snapshot
gpu-test bench                                          # 6 workloads → CSV
gpu-test bench --output my_report.csv                   # custom CSV path
gpu-test stress --tests all --duration 60               # full suite → CSV
gpu-test stress --tests throughput compute -o report.csv # pick tests
```

Every `bench` and `stress` run automatically generates a timestamped CSV report.

## CSV report

The CSV contains one row per test per GPU with these columns:

| Column group | Examples |
|---|---|
| **Fixed** | `timestamp`, `suite`, `test_name`, `status`, `gpu_index`, `gpu_name`, `elapsed_s`, `peak_vram_gb` |
| **Metrics** (`metric_*`) | `tok_per_s`, `tflops`, `img_per_s`, `steps_per_s`, `bandwidth_gb_s`, `filled_gb`, `max_context_tokens`, `ms_per_call`, `efficiency_pct` |
| **Config** (`config_*`) | `batch`, `seq_len`, `d_model`, `n_heads`, `matrix_size`, `duration_s` |
| **Telemetry** (`telem_*`) | `avg_gpu_util_pct`, `peak_gpu_util_pct`, `avg_power_w`, `peak_power_w`, `avg_temp_c`, `peak_temp_c`, `peak_clock_mhz` |

Dynamic columns expand automatically — only keys present in results appear.

## Telemetry

Every run reports via NVML:
- **GPU/mem utilization** (%, avg + peak during workloads)
- **VRAM** usage (GB, peak)
- **Power** draw (W, avg + peak vs TDP limit)
- **Temperature** (°C, avg + peak)
- **Clock speeds** (GPU + memory, current vs max)
- **Fan speed** (%) · **PCIe** gen, width, TX/RX throughput

## Benchmarks (`gpu-test bench`)

| Workload | Key metric | What it stresses |
|----------|-----------|-----------------|
| Transformer Block | `tok_per_s` | MHA + FFN inference, LLM/ViT core |
| SDPA / Flash Attention | `tok_per_s` | Long-context BF16 attention (up to 64K) |
| Large GEMM | `tflops` | BF16 matmul up to 32K×32K |
| Mixed-Precision Training | `steps_per_s` | Full AMP fwd + bwd + optimizer |
| Conv Stack (ResNet) | `img_per_s` | Vision, batch up to 128 @ 512px |
| VRAM Capacity Fill | `filled_gb` | Usable VRAM measurement |

## Stress tests (`gpu-test stress`)

| Test | Key metric | What it measures |
|------|-----------|-----------------|
| `throughput` | `tok_per_s` | Transformer inference |
| `memory` | `filled_gb` | 1 GB BF16 chunk fill |
| `sustained` | `steps_per_s` | Continuous AMP training |
| `p2p` | `bandwidth_gb_s` | GPU↔GPU 1 GB transfers |
| `scaling` | `efficiency_pct` | Multi-GPU scaling |
| `compute` | `tflops` | FP32/FP16/BF16 matmul |
| `bandwidth` | `bandwidth_gb_s` | Streaming memory r/w |
| `kv_cache` | `max_context_tokens` | 70B LLM KV-cache growth |
| `gradient` | `steps_per_s` | 6-layer gradient accumulation |
| `attention` | `ms_per_call` | Flash attention 4K → 128K |

## Local development

```bash
uv venv && uv pip install -e .
gpu-test info
gpu-test bench
gpu-test stress --tests all
```

## Project structure

```
gpu_test/
├── __init__.py      # v2.0.0
├── __main__.py      # CLI: bench | stress | info
├── utils.py         # GpuDevice, TestResult, cuda_timer, cleanup
├── monitor.py       # NVML snapshots + background Monitor
├── benchmark.py     # 6 AI workloads with OOM-safe fallback
├── stress.py        # 10 stress tests
└── report.py        # CSV report writer
```

## Requirements

- Python ≥ 3.13, PyTorch ≥ 2.9.0 (CUDA 13.0), nvidia-ml-py
- NVIDIA Container Toolkit (for Docker)

## License

MIT
