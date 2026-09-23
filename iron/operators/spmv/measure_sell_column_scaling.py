#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare eight-column SELL with each *identical packed column* in isolation.

One-column cases slice the eight-column A/control payload. They do not repack
or globally re-sort the matrix, so the per-column workloads are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aie.utils as aie_utils
import torch
from aie.iron.device import NPU2

from iron.common.test_utils import run_test
from iron.operators.spmv.evaluation import (
    DesignSpec, FormatSpec, MatrixInput, load_or_generate_csr, pack_for_design,
)
from iron.operators.spmv.measure_llama_slice_ell import REPRESENTATIVE_WEIGHTS
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated
from iron.operators.spmv.sell_c_sigma_runtime import make_dedicated_inputs
from iron.operators.spmv.slice_ell import cpu_spmv_csr


def benchmark(operator, packed_a, control, expected):
    """Return canonical-output latency and two explicitly defined bandwidths."""
    errors, latency_us, effective_gbps, samples_us = run_test(
        operator, {"packed": packed_a, "control": control},
        {"output": expected}, rel_tol=0.08, abs_tol=0.025,
        warmup_iters=2, timed_iters=5, return_timings=True,
    )
    if errors:
        raise AssertionError(f"canonical output mismatched at {sum(map(len, errors.values()))} positions")
    return {
        "latency_us": latency_us,
        "samples_us": samples_us,
        "a_only_gbps": packed_a.numel() * 2 / (latency_us * 1e3),
        "effective_gbps": effective_gbps,
        "a_bytes": packed_a.numel() * 2,
        "control_bytes": control.numel() * 2,
        "output_bytes": expected.numel() * 2,
    }


def measure_weight(model_dir: Path, name: str, height: int) -> dict:
    """Run the same eight packed windows together, then one at a time."""
    spec = MatrixInput(
        "safetensors", model_dir=str(model_dir.resolve()), tensor_name=name,
        x_seed=3000 + REPRESENTATIVE_WEIGHTS.index(name),
    )
    matrix = load_or_generate_csr(spec)
    fmt = FormatSpec("sell_c_sigma", block_height=height, block_width=256,
                     columns=8, window_count=8)
    packed = pack_for_design(matrix, fmt, DesignSpec("sell_dedicated_reorder"))
    rows_per_core = (2, 3, 3) if height == 8 else (height // 3,) * 3
    A, control, block_counts = make_dedicated_inputs(packed, matrix.vector, rows_per_core)
    expected = cpu_spmv_csr(matrix.indptr, matrix.indices, matrix.values, matrix.vector)
    full = SpMVSELLDedicated(
        packed.padded_rows, packed.K, block_counts, 8,
        rows_per_core=rows_per_core,
    )
    full_result = benchmark(full, A, control, expected)

    rows_per_window = packed.padded_rows // 8
    control_words = control.numel() // 8
    words_per_block = height * 256 * 2
    block_offset = 0
    column_results = []
    for column, block_count in enumerate(block_counts):
        first = block_offset * words_per_block
        last = (block_offset + block_count) * words_per_block
        A_column = A[first:last]
        control_column = control[column * control_words:(column + 1) * control_words]
        row0 = column * rows_per_window
        reference = torch.zeros(rows_per_window, dtype=torch.bfloat16)
        real_rows = max(0, min(rows_per_window, matrix.profile.M - row0))
        reference[:real_rows] = expected[row0:row0 + real_rows]
        one_column = SpMVSELLDedicated(
            rows_per_window, packed.K, (block_count,), 1,
            rows_per_core=rows_per_core,
        )
        result = benchmark(one_column, A_column, control_column, reference)
        column_results.append({"column": column, "blocks": block_count,
                               "real_rows": real_rows, **result})
        block_offset += block_count

    # Bracket the single-column sweep to expose temporal drift.
    full_after = benchmark(full, A, control, expected)
    return {
        "weight": name, "M": matrix.profile.M, "K": matrix.profile.K,
        "nnz": matrix.profile.nnz, "height": height,
        "padded_rows": packed.padded_rows, "rows_per_window": rows_per_window,
        "rows_per_core": rows_per_core, "full_before": full_result,
        "columns_alone": column_results, "full_after": full_after,
        "same_packed_a": True, "same_control": True,
        "warmup_iters": 2, "timed_iters": 5, "dummy_between_runs": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--weight", action="append", choices=REPRESENTATIVE_WEIGHTS)
    parser.add_argument("--height", type=int, choices=(6, 8, 9, 18, 36), default=8)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()
    aie_utils.set_current_device(NPU2())
    for name in args.weight or REPRESENTATIVE_WEIGHTS:
        record = measure_weight(args.model_dir, name, args.height)
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if args.output_jsonl:
            with args.output_jsonl.open("a") as handle:
                handle.write(line + "\n")


if __name__ == "__main__":
    main()
