#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare horizontal and vertical SELL-C-sigma on the same CSR rows/x.

Start with one physical column.  ``--columns 8`` is an optional follow-up;
it is not implicitly run when the one-column experiment regresses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aie.utils as aie_utils
import torch
from aie.iron.device import NPU2

from iron.common.test_utils import run_test
from iron.operators.spmv.evaluation import MatrixInput, load_or_generate_csr
from iron.operators.spmv.measure_llama_slice_ell import REPRESENTATIVE_WEIGHTS
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated
from iron.operators.spmv.sell_c_sigma_runtime import make_dedicated_inputs, make_vertical16_inputs
from iron.operators.spmv.slice_ell import SliceELLConfig, cpu_spmv_csr, csr_to_slice_ell


def measure_one(matrix, *, columns: int, layout: str,
                full_matrix_one_column: bool = False) -> dict:
    """Pack, check, and time one layout; report payload and effective bandwidth."""
    vertical16 = layout == "vertical16"
    height, width = (8, 256) if layout == "horizontal" else (48, 128)
    M, K = matrix.profile.M, matrix.profile.K
    if columns == 1 and not full_matrix_one_column:
        M = (M + 7) // 8  # Window-sized probe; same source rows across layouts.
    end = int(matrix.indptr[M])
    packed = csr_to_slice_ell(
        matrix.indptr[:M + 1], matrix.indices[:end], matrix.values[:end], K=K,
        config=SliceELLConfig(core_rows=4 if layout == "horizontal" else 3,
                              block_height=height, block_width=width,
                              shim_columns=columns, window_count=columns),
    )
    if vertical16:
        A, control, counts = make_vertical16_inputs(packed, matrix.vector)
        rows_per_core = (16, 16, 16)
    elif layout == "horizontal16":
        from iron.operators.spmv.sell_c_sigma_runtime import make_window_inputs
        A, control, counts = make_window_inputs(
            packed, matrix.vector, (16, 16, 16), block_width=128,
        )
        rows_per_core = (16, 16, 16)
    else:
        A, control, counts = make_dedicated_inputs(packed, matrix.vector)
        rows_per_core = (2, 3, 3)
    expected = torch.zeros(packed.padded_rows, dtype=torch.bfloat16)
    expected[:M] = cpu_spmv_csr(
        matrix.indptr[:M + 1], matrix.indices[:end], matrix.values[:end], matrix.vector,
    )
    operator = SpMVSELLDedicated(
        packed.padded_rows, K, counts, columns, rows_per_core=rows_per_core,
        block_width=width, vertical16=vertical16,
    )
    errors, latency_us, effective_gbps, samples = run_test(
        operator, {"packed": A, "control": control}, {"output": expected},
        rel_tol=0.08, abs_tol=0.025, warmup_iters=2, timed_iters=5,
        return_timings=True,
    )
    if errors:
        raise AssertionError(f"{sum(map(len, errors.values()))} output mismatches")
    return {
        "layout": layout,
        "columns": columns, "source_rows": M, "padded_rows": packed.padded_rows,
        "full_matrix_one_column": full_matrix_one_column,
        "K": K, "block_height": height, "block_width": width,
        "blocks_per_column": counts, "a_bytes": A.numel() * 2,
        "control_bytes": control.numel() * 2, "output_bytes": expected.numel() * 2,
        "latency_us": latency_us, "samples_us": samples,
        "a_only_gbps": A.numel() * 2 / (latency_us * 1e3),
        "effective_gbps": effective_gbps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--weight", action="append", choices=REPRESENTATIVE_WEIGHTS)
    parser.add_argument("--columns", type=int, choices=(1, 8), default=1)
    parser.add_argument("--full-matrix-one-column", action="store_true",
                        help="when columns=1, process all M rows instead of the first window-sized segment")
    parser.add_argument("--layout", choices=("horizontal", "horizontal16", "vertical16", "both", "all"), default="both")
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()
    if args.full_matrix_one_column and args.columns != 1:
        parser.error("--full-matrix-one-column requires --columns 1")
    aie_utils.set_current_device(NPU2())
    for name in args.weight or REPRESENTATIVE_WEIGHTS:
        matrix = load_or_generate_csr(MatrixInput(
            "safetensors", model_dir=str(args.model_dir.resolve()), tensor_name=name,
            x_seed=3000 + REPRESENTATIVE_WEIGHTS.index(name),
        ))
        for layout in ("horizontal", "horizontal16", "vertical16"):
            if args.layout == "both" and layout == "horizontal16":
                continue
            if args.layout not in ("both", "all", layout):
                continue
            result = {"weight": name, **measure_one(matrix, columns=args.columns,
                                                     layout=layout,
                                                     full_matrix_one_column=args.full_matrix_one_column)}
            line = json.dumps(result, sort_keys=True)
            print(line, flush=True)
            if args.output_jsonl:
                with args.output_jsonl.open("a") as handle:
                    handle.write(line + "\n")


if __name__ == "__main__":
    main()
