# gpu-test

Production-grade GPU benchmark & stress testing kit for CUDA 13.0+ NVIDIA GPUs.

Built with PyTorch 2.9+ and NVML telemetry — designed to validate new hardware (Blackwell RTX 6000 Pro and beyond) with modern AI workloads.

> [!NOTE]
> Tested on **NVIDIA RTX 6000 Pro** (Blackwell, 96 GB VRAM).

## Quick start (Docker)

```bash
# GPU info + telemetry snapshot
make info

# Smart benchmark (6 AI workloads, OOM-safe)
make bench

# Full stress-test suite (10 tests, 60s sustained)
make stress
```

## Subcommands

| Command | Description |
|---------|-------------|
| `gpu-test info` | NVML snapshot: power, temp, clocks, VRAM, PCIe, fan |
| `gpu-test bench` | 6 workloads: Transformer, Flash Attention, GEMM, AMP training, ConvNet, VRAM fill |
| `gpu-test stress --tests all` | 10 tests: throughput, memory, sustained load, P2P, scaling, compute, bandwidth, KV-cache, gradient, attention |
| `gpu-test stress --tests throughput kv_cache --duration 120` | Pick specific tests |

## Telemetry

Every run reports via NVML:
- **GPU/mem utilization** (%, avg + peak during workloads)
- **VRAM** usage (GB, peak)
- **Power** draw (W, avg + peak vs TDP limit)
- **Temperature** (°C, avg + peak)
- **Clock speeds** (GPU + memory, current vs max)
- **Fan speed** (%)
- **PCIe** generation, link width, TX/RX throughput

## Benchmarks (`gpu-test bench`)

| Workload | What it stresses |
|----------|-----------------|
| Transformer Block | MHA + FFN inference, LLM/ViT core |
| SDPA / Flash Attention | Long-context BF16 attention (up to 64K tokens) |
| Large GEMM | BF16 matmul up to 32K×32K — LLM bottleneck |
| Mixed-Precision Training | Full AMP forward + backward + optimizer step |
| Conv Stack (ResNet) | Vision workload, batch up to 128 @ 512px |
| VRAM Capacity Fill | Measures usable VRAM |

All workloads use **OOM-safe progressive fallback** — start aggressive, scale down automatically.

## Stress tests (`gpu-test stress`)

| Test | What it measures |
|------|-----------------|
| `throughput` | Transformer inference tok/s |
| `memory` | Fill VRAM with 1 GB BF16 chunks |
| `sustained` | Continuous AMP training for N seconds |
| `p2p` | GPU↔GPU 1 GB transfers |
| `scaling` | Multi-GPU scaling efficiency |
| `compute` | Matmul TFLOPS at FP32/FP16/BF16 |
| `bandwidth` | 2 GB streaming memory bandwidth |
| `kv_cache` | 70B-class LLM KV-cache growth |
| `gradient` | 6-layer transformer, 16-step gradient accumulation |
| `attention` | Flash attention 4K → 128K sequence lengths |

## Local development (no Docker)

```bash
# Install with uv
uv venv && uv pip install -e .

# Run
gpu-test info
gpu-test bench
gpu-test stress --tests all

# Or via python
python -m gpu_test bench
```

## Project structure

```
gpu_test/
├── __init__.py      # Package metadata
├── __main__.py      # CLI entry point (bench / stress / info)
├── utils.py         # Shared: GPU detection, timing, cleanup, formatting
├── monitor.py       # NVML telemetry: snapshots + background Monitor
├── benchmark.py     # 6 AI workloads with OOM-safe fallback
└── stress.py        # 10 stress tests
```

## Requirements

- Python ≥ 3.13
- PyTorch ≥ 2.9.0 (CUDA 13.0)
- NVIDIA GPU with CUDA 13.0 driver
- NVIDIA Container Toolkit (for Docker)

## Roadmap

- [ ] JSON / CSV report export
- [ ] Configurable CUDA arch targets via CLI
- [ ] AMD ROCm + Intel XPU backends
- [ ] Multi-node distributed benchmarking (NCCL)
- [ ] Automated regression testing

## License

MIT
