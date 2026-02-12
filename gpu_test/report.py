"""CSV report writer — collects TestResult objects and writes a flat CSV.

The CSV has fixed columns for common fields and dynamically expands for
per-test metrics, config, and telemetry key-value pairs.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from gpu_test.utils import TestResult


# ── Fixed column order ───────────────────────────────────────────────────────

_FIXED_COLS = [
    "timestamp",
    "suite",
    "test_name",
    "status",
    "gpu_index",
    "gpu_name",
    "elapsed_s",
    "peak_vram_gb",
]


def _collect_dynamic_keys(
    results: Sequence[TestResult], attr: str
) -> list[str]:
    """Gather all unique keys from a dict attribute across results, sorted."""
    keys: set[str] = set()
    for r in results:
        keys.update(getattr(r, attr, {}).keys())
    return sorted(keys)


def results_to_rows(results: Sequence[TestResult]) -> tuple[list[str], list[dict[str, str]]]:
    """Convert results to (headers, rows) suitable for csv.DictWriter."""
    metric_keys = _collect_dynamic_keys(results, "metrics")
    config_keys = _collect_dynamic_keys(results, "config")
    telem_keys = _collect_dynamic_keys(results, "telemetry")

    headers = list(_FIXED_COLS)
    headers += [f"metric_{k}" for k in metric_keys]
    headers += [f"config_{k}" for k in config_keys]
    headers += [f"telem_{k}" for k in telem_keys]

    rows: list[dict[str, str]] = []
    for r in results:
        row: dict[str, str] = {
            "timestamp": r.timestamp,
            "suite": r.suite,
            "test_name": r.test_name,
            "status": r.status,
            "gpu_index": str(r.gpu_index),
            "gpu_name": r.gpu_name,
            "elapsed_s": f"{r.elapsed_s:.4f}",
            "peak_vram_gb": f"{r.peak_vram_gb:.2f}",
        }
        for k in metric_keys:
            row[f"metric_{k}"] = str(r.metrics.get(k, ""))
        for k in config_keys:
            row[f"config_{k}"] = str(r.config.get(k, ""))
        for k in telem_keys:
            row[f"telem_{k}"] = str(r.telemetry.get(k, ""))
        rows.append(row)
    return headers, rows


def write_csv(results: Sequence[TestResult], path: str | Path) -> Path:
    """Write all results to a CSV file. Returns the resolved path."""
    path = Path(path)
    headers, rows = results_to_rows(results)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def to_csv_string(results: Sequence[TestResult]) -> str:
    """Return CSV content as a string (useful for stdout)."""
    headers, rows = results_to_rows(results)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def default_filename() -> str:
    """Generate a timestamped default filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"gpu_test_report_{ts}.csv"
