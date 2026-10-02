#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare equal-row and equal-NNZ SELL windows on one real weight.

This is an isolated experiment, not a change to the paper measurement path.
The original row order is retained across windows; only rows *inside* a window
are sorted. Unequal logical windows are padded to one fixed NPU FIFO shape.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from iron.operators.spmv.matrix_preparation import (
    CSRMatrix, FormatSpec, MatrixInput, estimate_storage, load_or_generate_csr,
    safetensors_profile, window_slice_bounds,
)
from iron.operators.spmv.slice_ell import (
    SliceELLConfig, cpu_spmv_csr, cpu_spmv_slice_ell,
    cpu_unpermute_windows, csr_to_slice_ell,
)


BLOCK_HEIGHT = 6
BLOCK_WIDTH = 256
COLUMNS = WINDOWS = 8
ROWS_PER_CORE = (2, 2, 2)
DEFAULT_TENSOR = "model.layers.0.self_attn.v_proj.weight"


@dataclass
class BalancedCase:
    """One NPU-compatible physical representation of NNZ-balanced windows."""

    packed: object
    valid_rows: tuple[int, ...]
    physical_rows: int
    logical_slice_bounds: tuple[int, ...]
    physical_csr_indptr: np.ndarray


def make_balanced_case(matrix: CSRMatrix) -> BalancedCase:
    """Pad each contiguous NNZ-balanced window to the same physical height."""

    fmt = FormatSpec(
        "sell_c_sigma", BLOCK_HEIGHT, BLOCK_WIDTH, COLUMNS,
        window_count=WINDOWS, boundary_policy="equal_nnz",
    )
    bounds = window_slice_bounds(matrix.profile, fmt)
    rows = matrix.profile.M
    valid_rows = tuple(
        max(0, min(int(last) * BLOCK_HEIGHT, rows) - int(first) * BLOCK_HEIGHT)
        for first, last in zip(bounds[:-1], bounds[1:])
    )
    # The fixed-length object FIFO needs the same number of whole slices in
    # each window. Keep the original rows first and append zero rows to each.
    slices_per_window = int(np.diff(bounds).max())
    if slices_per_window % 2 and BLOCK_HEIGHT % 2:
        slices_per_window += 1
    rows_per_window = slices_per_window * BLOCK_HEIGHT
    physical_counts = np.zeros(WINDOWS * rows_per_window, dtype=np.int64)
    original_counts = matrix.profile.row_nnz
    for window, (first, count) in enumerate(zip(bounds[:-1], valid_rows)):
        source = int(first) * BLOCK_HEIGHT
        physical_counts[window * rows_per_window:window * rows_per_window + count] = (
            original_counts[source:source + count]
        )
    physical_indptr = np.empty(physical_counts.size + 1, dtype=np.int64)
    physical_indptr[0] = 0
    np.cumsum(physical_counts, out=physical_indptr[1:])
    if int(physical_indptr[-1]) != int(matrix.indptr[-1]):
        raise AssertionError("padding changed the nonzero count")

    packed = csr_to_slice_ell(
        physical_indptr, matrix.indices, matrix.values, K=matrix.profile.K,
        config=SliceELLConfig(
            block_height=BLOCK_HEIGHT, block_width=BLOCK_WIDTH,
            core_rows=3, shim_columns=COLUMNS, window_count=WINDOWS,
        ),
    )
    return BalancedCase(
        packed, valid_rows, physical_counts.size, tuple(int(x) for x in bounds),
        physical_indptr,
    )


def analyze(profile) -> dict:
    """Report the exact post-sort A-block counts, not an NNZ proxy."""

    results = {}
    for policy in ("equal_rows", "equal_nnz"):
        fmt = FormatSpec(
            "sell_c_sigma", BLOCK_HEIGHT, BLOCK_WIDTH, COLUMNS,
            window_count=WINDOWS, boundary_policy=policy,
        )
        estimate = estimate_storage(profile, fmt)
        results[policy] = {
            key: estimate[key] for key in (
                "logical_window_slice_bounds", "window_rows", "column_blocks",
                "total_blocks", "max_column_blocks", "column_imbalance",
                "packed_a_bytes",
            )
        }
    return results


def check_packing(matrix: CSRMatrix, case: BalancedCase) -> None:
    """Confirm row compaction reproduces canonical BF16 output on the CPU."""

    expected = cpu_spmv_csr(matrix.indptr, matrix.indices, matrix.values, matrix.vector)
    physical = cpu_spmv_slice_ell(case.packed, matrix.vector)
    canonical_padded = cpu_unpermute_windows(case.packed, physical)
    rows_per_window = case.physical_rows // WINDOWS
    compact = torch.cat([
        canonical_padded[window * rows_per_window:window * rows_per_window + count]
        for window, count in enumerate(case.valid_rows)
    ])
    # The NPU has its own rounding order; the CPU packer must at least meet
    # the same tolerance used in the existing NPU measurements.
    torch.testing.assert_close(compact.float(), expected.float(), rtol=0.08, atol=0.025)


def measure_npu(matrix: CSRMatrix, case: BalancedCase, warmups: int, timed: int) -> dict:
    """Measure the current design with compact output, including FIFO padding."""

    import aie.utils as aie_utils
    from aie.iron.device import NPU2
    from iron.common import AIEContext
    from iron.common.test_utils import run_test
    from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated
    from iron.operators.spmv.sell_c_sigma_runtime import make_dedicated_inputs

    aie_utils.set_current_device(NPU2())
    A, control, blocks_per_column = make_dedicated_inputs(
        case.packed, matrix.vector, ROWS_PER_CORE,
    )
    operator = SpMVSELLDedicated(
        case.physical_rows, matrix.profile.K, blocks_per_column, WINDOWS,
        rows_per_core=ROWS_PER_CORE, block_width=BLOCK_WIDTH,
        valid_rows_per_window=case.valid_rows, context=AIEContext(),
    )
    expected = torch.zeros(case.physical_rows, dtype=torch.bfloat16)
    expected[:matrix.profile.M] = cpu_spmv_csr(
        matrix.indptr, matrix.indices, matrix.values, matrix.vector,
    )
    errors, latency_us, bandwidth_gbps, samples = run_test(
        operator, {"packed": A, "control": control}, {"output": expected},
        rel_tol=0.08, abs_tol=0.025, warmup_iters=warmups,
        timed_iters=timed, return_timings=True, idle_s=0.0,
    )
    return {
        "latency_us": latency_us,
        "samples_us": samples,
        "effective_bandwidth_gbps": bandwidth_gbps,
        "output_errors": sum(map(len, errors.values())),
        "physical_output_rows": case.physical_rows,
        "valid_output_rows": matrix.profile.M,
        "control_bytes": control.numel() * control.element_size(),
        "packed_a_bytes": case.packed.packed_a.nbytes,
        "column_blocks": list(blocks_per_column),
    }


def measure_baseline_npu(matrix: CSRMatrix, warmups: int, timed: int) -> dict:
    """Time the unchanged equal-row design with the same input and schedule."""

    from iron.operators.spmv.matrix_measure import measure_case

    expected = cpu_spmv_csr(
        matrix.indptr, matrix.indices, matrix.values, matrix.vector,
    )
    result = measure_case(
        matrix, "sell_dedicated_reorder", WINDOWS, expected,
        block_height=BLOCK_HEIGHT, warmup_iters=warmups,
        timed_iters=timed, idle_s=0.0,
    )
    return {
        "latency_us": result["npu_latency_us"],
        "samples_us": result["timed_samples_us"],
        "effective_bandwidth_gbps": result["effective_bandwidth_gbps"],
        "output_errors": result["cpu_error_count"],
        "physical_output_rows": (
            (matrix.profile.M + BLOCK_HEIGHT * WINDOWS - 1)
            // (BLOCK_HEIGHT * WINDOWS) * BLOCK_HEIGHT * WINDOWS
        ),
        "valid_output_rows": matrix.profile.M,
        "control_bytes": result["control_bytes"],
        "packed_a_bytes": result["packed_a_bytes"],
        "column_blocks": result["column_blocks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR)
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--timed", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.timed <= 0 or args.rounds <= 0:
        parser.error("warmups must be nonnegative; timed and rounds must be positive")
    source = MatrixInput(
        source="safetensors", model_dir=str(args.model_dir),
        tensor_name=args.tensor_name,
    )
    profile = safetensors_profile(source)
    report = {
        "tensor_name": args.tensor_name, "M": profile.M, "K": profile.K,
        "nnz": profile.nnz, "source_sha256": profile.source_sha256,
        "formats": analyze(profile),
    }
    if args.npu:
        import aie.utils as aie_utils
        from aie.iron.device import NPU2

        aie_utils.set_current_device(NPU2())
        matrix = load_or_generate_csr(source)
        case = make_balanced_case(matrix)
        check_packing(matrix, case)
        report["npu_rounds"] = []
        for round_number in range(args.rounds):
            # Reverse order every other round to expose order/thermal effects.
            if round_number % 2:
                balanced = measure_npu(matrix, case, args.warmups, args.timed)
                baseline = measure_baseline_npu(matrix, args.warmups, args.timed)
            else:
                baseline = measure_baseline_npu(matrix, args.warmups, args.timed)
                balanced = measure_npu(matrix, case, args.warmups, args.timed)
            report["npu_rounds"].append({
                "round": round_number + 1,
                "measurement_order": ["equal_nnz", "equal_rows"] if round_number % 2 else ["equal_rows", "equal_nnz"],
                "equal_rows": baseline, "equal_nnz": balanced,
            })
        report["valid_rows_per_window"] = list(case.valid_rows)
        report["physical_rows_per_window"] = case.physical_rows // WINDOWS
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
