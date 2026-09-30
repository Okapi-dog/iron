#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw a publication-ready SVG of row-NNZ CV for all projection weights."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path


PROJECTIONS = (
    "down_proj", "o_proj", "v_proj", "up_proj",
    "gate_proj", "q_proj", "k_proj",
)


def quantile(values: list[float], fraction: float) -> float:
    """Linear-interpolation quantile, matching NumPy's default method."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[min(lower + 1, len(ordered) - 1)] * weight


def make_svg(rows: list[dict]) -> str:
    """Show the pooled CV histogram and every weight grouped by projection."""
    if not rows:
        raise ValueError("no rows to plot")
    cvs = [float(row["row_nnz_cv"]) for row in rows]
    limits = [quantile(cvs, p) for p in (0, .25, .5, .75, 1)]
    x0, x1 = 118, 852
    xmax = max(2.0, max(cvs) * 1.04)
    x = lambda value: x0 + (x1 - x0) * value / xmax
    bin_width = .1
    bins = Counter(min(int(value / bin_width), 19) for value in cvs)
    y_bottom, y_top = 332, 120
    max_count = max(bins.values())
    count_ceiling = ((max_count + 4) // 5) * 5
    y_hist = lambda count: y_bottom - (y_bottom - y_top) * count / count_ceiling

    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="670" viewBox="0 0 900 670" role="img" aria-labelledby="title desc">',
        '<title id="title">Distribution of row-NNZ coefficient of variation in 224 pruned Llama-2-7B projection weights</title>',
        '<desc id="desc">Histogram and all 224 individual weights grouped by projection type. All weights have approximately 10 percent nonzero elements.</desc>',
        '<rect x="0" y="0" width="900" height="670" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202530} .title{font-size:20px;font-weight:600} .subtitle{font-size:13px;fill:#4b5563} .label{font-size:13px} .tick{font-size:12px;fill:#4b5563} .grid{stroke:#dce1e8;stroke-width:1} .axis{stroke:#596273;stroke-width:1.2} .bar{fill:#7e9db6} .dot{fill:#225777;fill-opacity:.58} .median{stroke:#c35530;stroke-width:2.2} .quartile{stroke:#c35530;stroke-width:1.4;stroke-dasharray:5 5}</style>',
        '<text x="118" y="34" class="title">Row-NNZ variability of pruned Llama-2-7B weights</text>',
        '<text x="118" y="56" class="subtitle">224 projection matrices · ELSA 90% sparsity · population CV = std(row NNZ) / mean(row NNZ)</text>',
        '<text x="118" y="84" class="label">Min {:.3f}   Q1 {:.3f}   Median {:.3f}   Q3 {:.3f}   Max {:.3f}</text>'.format(*limits),
        '<text x="118" y="109" class="label">All projection weights</text>',
    ]
    for tick in range(0, count_ceiling + 1, 5):
        yy = y_hist(tick)
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{x0 - 10}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')
    out.append('<text x="28" y="226" transform="rotate(-90 28 226)" class="label">Number of matrices</text>')
    for index in range(20):
        left = x(index * bin_width) + 1.1
        right = x((index + 1) * bin_width) - 1.1
        top = y_hist(bins[index])
        out.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{right-left:.1f}" height="{y_bottom-top:.1f}" class="bar"/>')
    for value, kind in zip(limits[1:4], ("quartile", "median", "quartile")):
        xx = x(value)
        out.append(f'<line x1="{xx:.1f}" y1="{y_top}" x2="{xx:.1f}" y2="{y_bottom}" class="{kind}"/>')
    out.append(f'<line x1="{x0}" y1="{y_bottom}" x2="{x1}" y2="{y_bottom}" class="axis"/>')
    out.append('<text x="118" y="376" class="label">By projection (each dot is one layer)</text>')
    for group_index, projection in enumerate(PROJECTIONS):
        yy = 408 + group_index * 28
        group = [row for row in rows if row["projection"] == projection]
        out.append(f'<text x="{x0 - 11}" y="{yy + 4}" text-anchor="end" class="tick">{html.escape(projection)}</text>')
        out.append(f'<line x1="{x0}" y1="{yy}" x2="{x1}" y2="{yy}" class="grid"/>')
        for row in group:
            # Deterministic layer jitter keeps repeated values visible.
            jitter = ((int(row["layer"]) * 13) % 9) - 4
            out.append(f'<circle cx="{x(float(row["row_nnz_cv"])):.1f}" cy="{yy+jitter:.1f}" r="3.1" class="dot"/>')
        median_x = x(quantile([float(row["row_nnz_cv"]) for row in group], .5))
        out.append(f'<line x1="{median_x:.1f}" y1="{yy-10}" x2="{median_x:.1f}" y2="{yy+10}" class="median"/>')
    axis_y = 612
    out.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" class="axis"/>')
    tick = 0.0
    while tick <= xmax + 1e-9:
        xx = x(tick)
        out.append(f'<line x1="{xx:.1f}" y1="{axis_y}" x2="{xx:.1f}" y2="{axis_y+5}" class="axis"/>')
        out.append(f'<text x="{xx:.1f}" y="{axis_y+21}" text-anchor="middle" class="tick">{tick:.1f}</text>')
        tick += .25
    out.append('<text x="485" y="657" text-anchor="middle" class="label">Coefficient of variation of row NNZ</text>')
    out.append('</svg>')
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input_jsonl.read_text().splitlines() if line]
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(make_svg(rows))


if __name__ == "__main__":
    main()
