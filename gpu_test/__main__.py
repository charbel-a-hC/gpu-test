"""CLI entry point — ``python -m gpu_test`` or ``gpu-test`` after install.

Sub-commands:
    bench   — Run the smart benchmark suite (6 modern AI workloads).
    stress  — Run the full stress-test suite (10 tests).
    info    — Print GPU details + NVML telemetry snapshot.
"""

from __future__ import annotations

import argparse
import sys

import torch


def _cmd_bench(_args: argparse.Namespace) -> None:
    from gpu_test.benchmark import run
    run()


def _cmd_stress(args: argparse.Namespace) -> None:
    from gpu_test.stress import run, _build_parser

    # Forward relevant args
    stress_ns = argparse.Namespace(tests=args.tests, duration=args.duration)
    run(stress_ns)


def _cmd_info(_args: argparse.Namespace) -> None:
    from gpu_test.monitor import init as nvml_init, shutdown as nvml_shutdown, snapshot
    from gpu_test.utils import detect_gpus, print_header

    gpus = detect_gpus()
    nvml_ok = nvml_init()
    print_header("GPU Information", gpus)

    for gpu in gpus:
        print(f"\n  {gpu}")
        if nvml_ok:
            snap = snapshot(gpu.index)
            print(f"  {snap}")

    if nvml_ok:
        nvml_shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gpu-test",
        description="Production-grade GPU benchmark & stress testing kit — CUDA 13.0+",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── bench ──
    sub.add_parser("bench", help="Run smart benchmark suite (6 AI workloads)")

    # ── stress ──
    stress_p = sub.add_parser("stress", help="Run full stress-test suite (10 tests)")
    stress_p.add_argument(
        "--tests", nargs="+",
        choices=["throughput", "memory", "sustained", "p2p", "scaling",
                 "compute", "bandwidth", "kv_cache", "gradient", "attention", "all"],
        default=["all"],
        help="Tests to run (default: all)",
    )
    stress_p.add_argument("--duration", type=int, default=30, help="Sustained-test duration (s)")

    # ── info ──
    sub.add_parser("info", help="Print GPU details + telemetry snapshot")

    args = parser.parse_args()

    handlers = {"bench": _cmd_bench, "stress": _cmd_stress, "info": _cmd_info}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
