#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw paper-ready latency and input-payload comparisons for real weights.

The input is the raw ``measure_paper real`` JSONL. No column position or
record order is assumed: measurements are matched by tensor name and design.
Input data counts packed A and control; the row map is already inside control.
Separate Dense/ELL vector transfers are not included in this requested metric.
Run ``python -m iron.operators.spmv.plot_real_latency_input --help``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from iron.operators.spmv.paper_config import PAPER_DESIGNS, PAPER_WEIGHTS


SPMV_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SPMV_DIR / "paper_evaluation3/real_weights_w5_t5_idle0s_between0s.jsonl"
DEFAULT_OUTPUT_DIR = SPMV_DIR / "paper_evaluation3/figures"
BYTES_PER_MB = 1_000_000
TARGET_SUFFIX = {"latex": ".pdf", "powerpoint": ".svg"}


@dataclass(frozen=True)
class MethodStyle:
    name: str
    color: str
    hatch: str


STYLES = {
    "dense_k_tiled": MethodStyle("Dense", "#666666", ""),
    "ell": MethodStyle("ELL", "#4496C7", "//"),
    "slice_ell": MethodStyle("Blocked Slice-ELL", "#F1A25A", "\\\\"),
    "sell_dedicated_reorder": MethodStyle("Blocked SELL-C-σ", "#69BA69", ".."),
}


@dataclass(frozen=True)
class Measurement:
    weight: str
    design: str
    rows: int
    columns: int
    row_cv: float
    latency_us: float
    packed_a_bytes: int
    control_bytes: int
    row_map_bytes: int

    @property
    def input_bytes(self) -> int:
        """Count A, other control, and row map exactly once.

        SELL's row map is physically inside ``control_bytes``. Splitting it
        here makes the requested A + control + row-map definition explicit
        without double-counting that map.
        """
        other_control = self.control_bytes - self.row_map_bytes
        return self.packed_a_bytes + other_control + self.row_map_bytes


def read_measurements(path: Path) -> dict[tuple[str, str], Measurement]:
    """Load and validate the five-weight, four-design paper measurement set."""
    measurements: dict[tuple[str, str], Measurement] = {}
    protocols: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        weight = record.get("tensor_name")
        design = record.get("requested_design")
        if weight not in PAPER_WEIGHTS or design not in PAPER_DESIGNS:
            continue
        key = (weight, design)
        if key in measurements:
            raise ValueError(f"duplicate weight/design at line {line_number}: {key}")
        if record.get("status") not in ("ok", "ok_with_tolerance_exceptions"):
            raise ValueError(f"unsuccessful measurement at line {line_number}: {key}")
        if len(record.get("timed_samples_us", [])) != 5:
            raise ValueError(f"expected five timing samples at line {line_number}: {key}")
        protocols.add(str(record.get("timing_protocol_id")))
        packed_a = int(record["packed_a_bytes"])
        control = int(record["control_bytes"])
        row_map = int(record["row_indices_bytes"])
        if min(packed_a, control, row_map) < 0 or row_map > control:
            raise ValueError(f"invalid payload decomposition at line {line_number}: {key}")
        measurements[key] = Measurement(
            weight=weight,
            design=design,
            rows=int(record["M"]),
            columns=int(record["K"]),
            row_cv=float(record["actual_row_cv"]),
            latency_us=float(record["npu_latency_us"]),
            packed_a_bytes=packed_a,
            control_bytes=control,
            row_map_bytes=row_map,
        )

    expected = {(weight, design) for weight in PAPER_WEIGHTS for design in PAPER_DESIGNS}
    missing = expected - measurements.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} weight/design pairs; first: {sorted(missing)[0]}")
    if len(protocols) != 1:
        raise ValueError(f"mixed timing protocols: {sorted(protocols)}")
    for weight in PAPER_WEIGHTS:
        shapes = {(measurements[weight, design].rows,
                   measurements[weight, design].columns)
                  for design in PAPER_DESIGNS}
        if len(shapes) != 1:
            raise ValueError(f"inconsistent shape across designs: {weight}")
    return measurements


def weight_label(measurement: Measurement) -> str:
    """Keep category labels short while retaining shape and row-CV context."""
    match = re.fullmatch(r"model\.layers\.(\d+)\.(?:mlp|self_attn)\.(.+)\.weight",
                         measurement.weight)
    short = f"L{match.group(1)} {match.group(2)}" if match else measurement.weight
    return (f"{short}\n{measurement.rows}×{measurement.columns}; "
            f"CV={measurement.row_cv:.2f}")


def metric_values(measurements: dict[tuple[str, str], Measurement], view: str):
    """Return both metrics in the same weight and method order."""
    latency, input_data = [], []
    for weight in PAPER_WEIGHTS:
        dense = measurements[weight, "dense_k_tiled"]
        latency_row, input_row = [], []
        for design in PAPER_DESIGNS:
            item = measurements[weight, design]
            if view == "absolute":
                latency_row.append(item.latency_us)
                input_row.append(item.input_bytes / BYTES_PER_MB)
            else:
                latency_row.append(dense.latency_us / item.latency_us)
                input_row.append(dense.input_bytes / item.input_bytes)
        latency.append(latency_row)
        input_data.append(input_row)
    return latency, input_data


def _draw_bars(ax, values, labels, view: str) -> None:
    """Draw the same grouped bars in either the slide or manuscript layout."""
    import numpy as np

    x = np.arange(len(labels))
    bar_width = 0.19
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D5D5D5", linestyle=":", linewidth=0.8)
    for index, design in enumerate(PAPER_DESIGNS):
        style = STYLES[design]
        offset = (index - 1.5) * bar_width
        ax.bar(x + offset, [row[index] for row in values], bar_width,
               label=style.name, color=style.color, edgecolor="black",
               linewidth=0.65, hatch=style.hatch)
    ax.set_xticks(x, labels, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    if view == "normalized":
        ax.axhline(1, color="#555555", linestyle="--", linewidth=0.9)


def _axis_labels(view: str) -> tuple[str, str]:
    """Keep axis terminology identical across both layouts."""
    if view == "absolute":
        return "Latency (µs)", "Input data (MB)"
    return "Dense latency / method latency (×)", "Dense input / method input (×)"


def draw_figure(measurements: dict[tuple[str, str], Measurement], view: str,
                target: str, output_prefix: Path, layout: str = "paired") -> list[Path]:
    """Save a paired slide figure or two independent manuscript figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 0.7,
        "pdf.fonttype": 42,
        # Outline SVG text so PowerPoint does not substitute the plot font.
        "svg.fonttype": "path",
    })
    latency, input_data = metric_values(measurements, view)
    labels = [weight_label(measurements[weight, "dense_k_tiled"])
              for weight in PAPER_WEIGHTS]
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    suffix = TARGET_SUFFIX[target]
    ylabels = _axis_labels(view)
    if layout == "paired":
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
        for ax, values, ylabel in zip(axes, (latency, input_data), ylabels):
            _draw_bars(ax, values, labels, view)
            ax.set_ylabel(ylabel)
        captions = (("(a) Latency", "(b) Input data") if view == "absolute"
                    else ("(a) Speedup over Dense", "(b) Input reduction vs Dense"))
        for ax, caption in zip(axes, captions):
            ax.set_title(caption, fontsize=10, fontweight="bold", pad=8)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 0.99), frameon=False,
                   handlelength=2.1, columnspacing=1.5)
        fig.subplots_adjust(left=0.065, right=0.99, top=0.78,
                            bottom=0.29, wspace=0.28)
        output_path = output_prefix.with_suffix(suffix)
        fig.savefig(output_path)
        plt.close(fig)
        return [output_path]

    if layout != "separate":
        raise ValueError(f"unknown layout: {layout}")
    output_paths = []
    for name, values, ylabel in zip(("latency", "input_data"),
                                    (latency, input_data), ylabels):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _draw_bars(ax, values, labels, view)
        ax.set_ylabel(ylabel)
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(handles, legend_labels, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, 0.99), frameon=False,
                   handlelength=2.1, columnspacing=2.0)
        fig.subplots_adjust(left=0.11, right=0.98, top=0.78, bottom=0.30)
        output_path = output_prefix.with_name(f"{output_prefix.name}_{name}").with_suffix(suffix)
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="real-weight JSONL from measure_paper")
    parser.add_argument("--view", choices=("absolute", "normalized"),
                        default="absolute",
                        help="absolute µs/MB, or Dense-divided benefit factors")
    parser.add_argument("--target", choices=tuple(TARGET_SUFFIX), default="latex",
                        help="LaTeX writes PDF; PowerPoint writes SVG")
    parser.add_argument("--layout", choices=("paired", "separate"), default="paired",
                        help="paired slide figure or separate manuscript figures")
    parser.add_argument("--output-prefix", type=Path,
                        help="path without extension; separate layout adds metric names")
    args = parser.parse_args()
    stem = "real_latency_input" if args.view == "absolute" else "real_dense_relative"
    output_prefix = args.output_prefix or DEFAULT_OUTPUT_DIR / stem
    output_paths = draw_figure(read_measurements(args.input), args.view,
                               args.target, output_prefix, args.layout)
    for output_path in output_paths:
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
