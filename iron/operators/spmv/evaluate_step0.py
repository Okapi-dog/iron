# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-0 storage/column-work sweep over one shared canonical matrix source.

This does not compile or run an NPU design.  Example::

    python -m iron.operators.spmv.evaluate_step0 \
      --synthetic 4096 4096 0.1 skewed 42 --format all

The design selector checks format compatibility and reports implementation
status.  Later steps can use the same MatrixSpec/FormatSpec/DesignSpec objects
for actual packing and NPU measurements.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from iron.operators.spmv.evaluation import (
    DesignSpec,
    FormatSpec,
    MatrixSpec,
    estimate_storage,
    safetensors_profile,
    synthetic_profile,
)

REPRESENTATIVE_WEIGHTS = (
    "model.layers.3.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.25.mlp.down_proj.weight",
)


def make_format_specs(args: argparse.Namespace) -> list[FormatSpec]:
    """Expand a small, explicit format sweep without hidden autotuning."""

    requested = (
        ("dense", "ell", "slice_ell", "sell_c_sigma", "global_sort_bound")
        if "all" in args.format
        else args.format
    )
    specs = []
    for name in requested:
        if name != "sell_c_sigma":
            specs.append(
                FormatSpec(
                    name=name,
                    block_height=args.block_height,
                    block_width=args.block_width,
                    columns=args.columns,
                )
            )
            continue
        for windows in args.windows:
            for boundary in args.boundary:
                assignments = (
                    ("contiguous",) if windows == args.columns else args.assignment
                )
                for assignment in assignments:
                    specs.append(
                        FormatSpec(
                            name=name,
                            block_height=args.block_height,
                            block_width=args.block_width,
                            columns=args.columns,
                            window_count=windows,
                            boundary_policy=boundary,
                            assignment_policy=assignment,
                        )
                    )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        nargs=5,
        action="append",
        metavar=("M", "K", "DENSITY", "PATTERN", "SEED"),
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--weight", action="append", help="safetensors tensor name (repeatable)"
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=(
            "all",
            "dense",
            "ell",
            "slice_ell",
            "sell_c_sigma",
            "global_sort_bound",
        ),
        default=[],
    )
    parser.add_argument(
        "--design",
        default="storage_only",
        choices=(
            "storage_only",
            "dense_k_tiled",
            "ell",
            "slice_ell",
            "sell_dedicated_reorder",
            "sell_time_multiplex_reorder",
        ),
    )
    parser.add_argument("--block-height", type=int, default=8)
    parser.add_argument("--block-width", type=int, default=256)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--windows", type=int, nargs="+", default=[8, 16])
    parser.add_argument(
        "--boundary",
        nargs="+",
        choices=("equal_rows", "equal_nnz"),
        default=["equal_rows", "equal_nnz"],
    )
    parser.add_argument(
        "--assignment",
        nargs="+",
        choices=("contiguous", "balanced"),
        default=["contiguous", "balanced"],
    )
    parser.add_argument("--output", choices=("jsonl", "markdown"), default="markdown")
    args = parser.parse_args()
    if not args.synthetic and args.model_dir is None:
        parser.error("provide --synthetic or --model-dir")
    if args.weight and args.model_dir is None:
        parser.error("--weight requires --model-dir")
    if not args.format:
        args.format = ["all"]
    return args


def main() -> None:
    args = parse_args()
    specs: list[MatrixSpec] = []
    for values in args.synthetic or []:
        M, K, density, pattern, seed = values
        specs.append(
            MatrixSpec(
                source="synthetic",
                M=int(M),
                K=int(K),
                density=float(density),
                row_pattern=pattern,
                seed=int(seed),
            )
        )
    if args.model_dir is not None:
        for name in args.weight or REPRESENTATIVE_WEIGHTS:
            specs.append(
                MatrixSpec(
                    source="safetensors",
                    model_dir=str(args.model_dir),
                    tensor_name=name,
                )
            )
    design = DesignSpec(args.design)
    formats = make_format_specs(args)
    if args.output == "markdown":
        print(
            "| Matrix | M×K | density | format | windows | boundary | assignment | A MiB | total/dense | dense/total | max col blocks | imbalance | reorder L1 KiB |"
        )
        print("|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for spec in specs:
        profile = (
            synthetic_profile(spec)
            if spec.source == "synthetic"
            else safetensors_profile(spec)
        )
        for fmt in formats:
            if design.format_name is not None and design.format_name != fmt.name:
                continue
            record = estimate_storage(profile, fmt, design)
            record["matrix_spec"] = asdict(spec)
            record["format_spec"] = asdict(fmt)
            if args.output == "jsonl":
                print(json.dumps(record, sort_keys=True))
            else:
                label = spec.tensor_name or f"synthetic:{spec.row_pattern}:{spec.seed}"
                print(
                    f"| {label} | {profile.M}×{profile.K} | {profile.density:.4f} | {fmt.name} | "
                    f"{fmt.window_count if fmt.name == 'sell_c_sigma' else '—'} | "
                    f"{fmt.boundary_policy if fmt.name == 'sell_c_sigma' else '—'} | "
                    f"{fmt.assignment_policy if fmt.name == 'sell_c_sigma' else '—'} | "
                    f"{record['packed_a_bytes'] / 2**20:.3f} | {record['storage_over_dense']:.3f} | "
                    f"{record['dense_over_storage'] or 0:.3f} | {record.get('max_column_blocks', '—')} | "
                    f"{record.get('column_imbalance', 0):.3f} | "
                    f"{record.get('reorder_l1_min_bytes', 0) / 1024:.2f} |"
                )


if __name__ == "__main__":
    main()
