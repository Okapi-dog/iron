# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pack and CPU-verify Slice-ELL / SELL-C-sigma without running the NPU."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from iron.operators.spmv.evaluate_step0 import REPRESENTATIVE_WEIGHTS
from iron.operators.spmv.evaluation import (
    CSRMatrix,
    DesignSpec,
    FormatSpec,
    MatrixInput,
    estimate_storage,
    load_or_generate_csr,
    pack_for_design,
)
from iron.operators.spmv.slice_ell import (
    cpu_spmv_csr,
    cpu_spmv_slice_ell,
    cpu_unpermute_windows,
)


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
        "--format", action="append", choices=("slice_ell", "sell_c_sigma"), default=[]
    )
    parser.add_argument(
        "--design",
        choices=(
            "storage_only",
            "slice_ell",
            "sell_dedicated_reorder",
            "sell_time_multiplex_reorder",
        ),
        default="storage_only",
    )
    parser.add_argument("--block-height", type=int, default=8)
    parser.add_argument("--block-width", type=int, default=256)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--windows", type=int, nargs="+", default=[8, 16])
    parser.add_argument(
        "--boundary",
        nargs="+",
        choices=("equal_rows", "equal_nnz"),
        default=["equal_rows"],
    )
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()
    if not args.synthetic and args.model_dir is None:
        parser.error("provide --synthetic or --model-dir")
    if args.weight and args.model_dir is None:
        parser.error("--weight requires --model-dir")
    if not args.format:
        args.format = ["slice_ell", "sell_c_sigma"]
    return args


def format_specs(args: argparse.Namespace) -> list[FormatSpec]:
    """Expand only packable contiguous-window formats (not the global bound)."""

    formats = []
    for name in args.format:
        if name == "slice_ell":
            formats.append(
                FormatSpec(
                    name=name,
                    block_height=args.block_height,
                    block_width=args.block_width,
                    columns=args.columns,
                )
            )
        else:
            for windows in args.windows:
                for boundary in args.boundary:
                    formats.append(
                        FormatSpec(
                            name=name,
                            block_height=args.block_height,
                            block_width=args.block_width,
                            columns=args.columns,
                            window_count=windows,
                            boundary_policy=boundary,
                        )
                    )
    return formats


def verify_one(
    matrix: CSRMatrix,
    reference: torch.Tensor,
    fmt: FormatSpec,
    design: DesignSpec,
    atol: float,
    rtol: float,
) -> dict:
    """Return a reproducible CPU correctness/storage record for one configuration."""

    spec = matrix.profile.spec
    packed = pack_for_design(matrix, fmt, design)
    physical = cpu_spmv_slice_ell(packed, matrix.vector)
    canonical = cpu_unpermute_windows(packed, physical)
    prediction = estimate_storage(matrix.profile, fmt, design)
    if packed.packed_a.nbytes != prediction["packed_a_bytes"]:
        raise AssertionError("actual packed A size differs from storage model")
    correct = bool(
        torch.allclose(canonical.float(), reference.float(), atol=atol, rtol=rtol)
    )
    return {
        "matrix_input": asdict(spec),
        "matrix_id": spec.matrix_id,
        "source_sha256": matrix.profile.source_sha256,
        "row_nnz_sha256": matrix.profile.row_nnz_sha256,
        "format_spec": asdict(fmt),
        "format_id": fmt.format_id,
        "design": design.name,
        "design_npu_status": design.npu_status,
        "cpu_correct": correct,
        "max_abs_error": float((canonical.float() - reference.float()).abs().max()),
        "packed_a_bytes": packed.packed_a.nbytes,
        "row_indices_bytes": (
            packed.row_indices.nbytes if packed.row_indices is not None else 0
        ),
        "window_count": packed.window_count,
        "window_slice_offsets": (
            packed.window_slice_offsets.tolist()
            if packed.window_slice_offsets is not None
            else None
        ),
        "packed_a_sha256": packed.manifest()["packed_a_sha256"],
    }


def main() -> None:
    args = parse_args()
    specs = []
    for M, K, density, pattern, seed in args.synthetic or []:
        specs.append(
            MatrixInput(
                source="synthetic",
                M=int(M),
                K=int(K),
                density=float(density),
                row_pattern=pattern,
                seed=int(seed),
            )
        )
    if args.model_dir is not None:
        specs.extend(
            MatrixInput(
                source="safetensors", model_dir=str(args.model_dir), tensor_name=name
            )
            for name in (args.weight or REPRESENTATIVE_WEIGHTS)
        )
    design = DesignSpec(args.design)
    formats = format_specs(args)
    for spec in specs:
        matrix = load_or_generate_csr(spec)
        reference = cpu_spmv_csr(
            matrix.indptr, matrix.indices, matrix.values, matrix.vector
        )
        for fmt in formats:
            if design.format_name is not None and design.format_name != fmt.name:
                continue
            record = verify_one(matrix, reference, fmt, design, args.atol, args.rtol)
            print(json.dumps(record, sort_keys=True), flush=True)
            if not record["cpu_correct"]:
                raise AssertionError(
                    "packed result differs from canonical CSR reference"
                )


if __name__ == "__main__":
    main()
