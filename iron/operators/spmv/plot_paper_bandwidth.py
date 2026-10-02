#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot four paper bandwidth figures from the warmup-5 measurement JSONL.

The x-axis of the combined figure uses the exact logical input/output buffer
bytes in ``run_test``. Sparse control buffers already contain copies of x and
the SELL row map, so those fields must not be added a second time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from iron.operators.spmv.paper_config import (
    PAPER_CVS, PAPER_DENSITIES, PAPER_DESIGNS, PAPER_TIMING_PROTOCOL_ID,
    PAPER_WEIGHTS, paper_conditions,
)
from iron.operators.spmv.plot_real_latency_input import STYLES, weight_label


SPMV_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SPMV_DIR / "paper_evaluation3"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "figures/bandwidth"
SPARSE_CONTROL_DESIGNS = {"slice_ell", "sell_dedicated_reorder"}
MARKERS = {"dense_k_tiled": "D", "ell": "o", "slice_ell": "s",
           "sell_dedicated_reorder": "^"}
BYTES_PER_MB = 1_000_000


@dataclass(frozen=True)
class BandwidthCase:
    """One matrix and one NPU implementation, averaged over five timed calls."""

    source: str
    key: str
    design: str
    rows: int
    columns: int
    row_cv: float
    latency_us: float
    io_bytes: int
    bandwidth_gbps: float

    @property
    def weight(self) -> str:
        """Expose the real-weight name to the shared axis-label formatter."""
        return self.key


def npu_io_bytes(record: dict) -> int:
    """Match run_test's sum of its input/output tensor buffer sizes."""
    design = record["requested_design"]
    matrix = int(record["packed_a_bytes"])
    output = int(record["y_transfer_bytes"])
    if design in SPARSE_CONTROL_DESIGNS:
        # Control includes x, and SELL control also includes the row map.
        additional_input = int(record["control_bytes"])
    else:
        additional_input = int(record["x_transfer_bytes"])
    total = matrix + additional_input + output
    if min(matrix, additional_input, output) < 0 or total == 0:
        raise ValueError(f"invalid I/O payload: {design}")
    return total


def read_cases(path: Path, source: str) -> list[BandwidthCase]:
    """Recover run_test's I/O byte count and check the recorded breakdown."""
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        design = record["requested_design"]
        if design not in PAPER_DESIGNS:
            raise ValueError(f"unknown design at line {line_number}: {design}")
        if record.get("timing_protocol_id") != PAPER_TIMING_PROTOCOL_ID:
            raise ValueError(f"wrong timing protocol at line {line_number}")
        if (record.get("status") not in ("ok", "ok_with_tolerance_exceptions")
                or len(record.get("timed_samples_us", [])) != 5):
            raise ValueError(f"incomplete measurement at line {line_number}")
        key = (record["tensor_name"] if source == "real" else
               f"{record['condition']}/{record['seed']}")
        if (key, design) in seen:
            raise ValueError(f"duplicate matrix/design at line {line_number}")
        seen.add((key, design))
        latency = float(record["npu_latency_us"])
        bandwidth = float(record["effective_bandwidth_gbps"])
        # run_test derives this rate from total_bytes and NPU latency. The
        # recorded rate implies 64 more bytes for both sparse-control designs
        # than the current per-field JSON breakdown; retain run_test's count
        # and reject any other unaccounted difference.
        measured_io = bandwidth * latency * 1000.0  # decimal GB/s and µs
        io_bytes = round(measured_io)
        if not math.isclose(measured_io, io_bytes, abs_tol=1e-5):
            raise ValueError(f"non-integral I/O byte count at line {line_number}")
        expected_difference = 64 if design in SPARSE_CONTROL_DESIGNS else 0
        if io_bytes - npu_io_bytes(record) != expected_difference:
            raise ValueError(f"I/O bytes disagree with buffer summary at line {line_number}")
        cases.append(BandwidthCase(
            source=source, key=key, design=design,
            rows=int(record["M"]), columns=int(record["K"]),
            row_cv=float(record["actual_row_cv"]), latency_us=latency,
            io_bytes=io_bytes, bandwidth_gbps=bandwidth,
        ))
    return cases


def validate_paper_cases(real: list[BandwidthCase], synthetic: list[BandwidthCase]) -> None:
    """Prevent a plausible-looking paper figure from silently dropping cases."""
    actual_real = {(case.key, case.design) for case in real}
    expected_real = {(name, design) for name in PAPER_WEIGHTS
                     for design in PAPER_DESIGNS}
    expected_synthetic = {(f"{condition}/{seed}", design)
                          for condition, _, _ in paper_conditions()
                          for seed in range(1000, 1010)
                          for design in PAPER_DESIGNS}
    actual_synthetic = {(case.key, case.design) for case in synthetic}
    if actual_real != expected_real or actual_synthetic != expected_synthetic:
        raise ValueError("the paper's 5 real or 110 synthetic matrices are incomplete")


def setup_matplotlib() -> None:
    """Use an editable, vector-quality PDF with consistent paper typography."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9,
        "axes.linewidth": 0.8, "pdf.fonttype": 42,
        "hatch.linewidth": 0.7,
    })


def finish_figure(fig, path: Path, *, bottom: float) -> None:
    """Apply the same legend and save a standalone manuscript panel."""
    import matplotlib.pyplot as plt

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), frameon=False,
               handlelength=2.1, columnspacing=2.0)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.77, bottom=bottom)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_real(real: list[BandwidthCase], path: Path) -> None:
    """Connect the five real-weight values in ascending row-CV order."""
    import matplotlib.pyplot as plt
    import numpy as np

    by_key = {(case.key, case.design): case for case in real}
    labels = [weight_label(by_key[name, "dense_k_tiled"]) for name in PAPER_WEIGHTS]
    x = np.arange(len(PAPER_WEIGHTS))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for design in PAPER_DESIGNS:
        style = STYLES[design]
        values = [by_key[name, design].bandwidth_gbps for name in PAPER_WEIGHTS]
        ax.plot(x, values, label=style.name, color=style.color,
                marker=MARKERS[design], markersize=5.5, linewidth=1.5,
                markeredgecolor="black", markeredgewidth=0.5)
    ax.set_xticks(x, labels, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylabel("Effective NPU I/O bandwidth (GB/s)")
    ax.set_ylim(0, 65)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D5D5D5", linestyle=":", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, bottom=0.30)


def sweep_statistics(cases: list[BandwidthCase], condition: str,
                     design: str) -> tuple[float, float]:
    """Mean and sample SD across ten independent generated matrices."""
    values = [case.bandwidth_gbps for case in cases
              if case.key.startswith(f"{condition}/") and case.design == design]
    if len(values) != 10:
        raise ValueError(f"expected ten matrices: {condition}/{design}")
    return statistics.mean(values), statistics.stdev(values)


def plot_sweep(synthetic: list[BandwidthCase], conditions: list[str],
               x_values: list[float], x_labels: list[str], xlabel: str,
               path: Path) -> None:
    """Show per-condition mean bandwidth with seed-to-seed standard deviation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for design in PAPER_DESIGNS:
        means, deviations = zip(*(sweep_statistics(synthetic, condition, design)
                                  for condition in conditions))
        style = STYLES[design]
        ax.errorbar(x_values, means, yerr=deviations, label=style.name,
                    color=style.color, marker=MARKERS[design], markersize=5,
                    linewidth=1.3, capsize=2, elinewidth=0.8,
                    markeredgecolor="black", markeredgewidth=0.5)
    ax.set_xticks(x_values, x_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean effective NPU I/O bandwidth (GB/s)")
    ax.set_ylim(0, 65)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D5D5D5", linestyle=":", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, bottom=0.18)


def plot_combined(cases: list[BandwidthCase], path: Path) -> None:
    """Place every matrix/design case on the I/O-size versus bandwidth plane."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for design in PAPER_DESIGNS:
        subset = [case for case in cases if case.design == design]
        style = STYLES[design]
        ax.scatter([case.io_bytes / BYTES_PER_MB for case in subset],
                   [case.bandwidth_gbps for case in subset],
                   label=style.name, marker=MARKERS[design],
                   color=style.color, edgecolors="black", linewidths=0.35,
                   s=25, alpha=0.55)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128], ["4", "8", "16", "32", "64", "128"])
    ax.set_xlim(4, 128)
    ax.set_ylim(0, 65)
    ax.set_xlabel("Total NPU input and output payload (MB)")
    ax.set_ylabel("Effective NPU I/O bandwidth (GB/s)")
    ax.set_axisbelow(True)
    ax.grid(axis="both", color="#D5D5D5", linestyle=":", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, bottom=0.18)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=DEFAULT_DATA_DIR /
                        "real_weights_w5_t5_idle0s_between0s.jsonl")
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_DATA_DIR /
                        "synthetic_paper_w5_t5_idle0s_between0s.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", choices=("all", "real", "density", "cv", "combined"),
                        default="all", help="regenerate one figure or the complete set")
    args = parser.parse_args()

    real = read_cases(args.real, "real")
    synthetic = read_cases(args.synthetic, "synthetic")
    validate_paper_cases(real, synthetic)
    setup_matplotlib()

    if args.only in ("all", "real"):
        plot_real(real, args.output_dir / "real_bandwidth.pdf")
    if args.only in ("all", "density"):
        plot_sweep(synthetic, [f"density_{int(d * 100):02d}" for d in PAPER_DENSITIES],
                   list(PAPER_DENSITIES), [f"{d:.0%}" for d in PAPER_DENSITIES],
                   "Nonzero fraction (target row-NNZ CV = 0.44)",
                   args.output_dir / "density_bandwidth.pdf")
    if args.only in ("all", "cv"):
        plot_sweep(synthetic,
                   ["density_10" if cv == 0.44 else f"cv_{cv:.2f}" for cv in PAPER_CVS],
                   list(range(len(PAPER_CVS))), [f"{cv:.2f}" for cv in PAPER_CVS],
                   "Target row-NNZ CV (nonzero fraction = 10%)",
                   args.output_dir / "cv_bandwidth.pdf")
    if args.only in ("all", "combined"):
        plot_combined(real + synthetic, args.output_dir / "combined_bandwidth_vs_io.pdf")
    print(f"saved {args.only} bandwidth figure(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
