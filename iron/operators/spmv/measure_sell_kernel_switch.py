# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 4.5: measure normal dispatch, optionally with a 32-core GEMV between runs.

By default, run only the target: two warmups followed by five timed calls.
``--with-dummy`` additionally measures a switched mode, running a different
32-core dense GEMV immediately before every target call.  Only the target's
``result.npu_time`` is sampled.  Both modes use the same packed A and seven
varying x vectors.  When requested, both operators are compiled/loaded before
either mode starts.

This is a dispatch-after-another-kernel experiment, not a measurement of
compilation or ``DefaultNPURuntime.load``.  The installed XRT runtime measures
from kernel submission through wait inside ``run``; it syncs BOs beforehand.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import aie.utils as aie_utils
import numpy as np
import torch
from ml_dtypes import bfloat16

from iron.common.test_utils import verify_buffer
from iron.operators.gemv.k_tiled_op import DenseGEMVKTile
from iron.operators.spmv.measure_sell_dedicated import make_32core_config
from iron.operators.spmv.op import SpMVSliceELLDynamicScalarMultiCol
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated, SpMVSELLTimeMultiplex
from iron.operators.spmv.sell_c_sigma_runtime import make_window_inputs
from iron.operators.spmv.slice_ell import SliceELLConfig, cpu_spmv_csr, csr_to_slice_ell


SHAPES = ((4096, 4096), (11008, 4096), (4096, 11008), (28672, 8192))
WARMUP_ITERS = 2
TIMED_ITERS = 5


def make_target_matrix(M: int, K: int, seed: int):
    """Use exactly the Step-4 synthetic CSR recipe, then pack only once."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(4, 300, size=M, dtype=np.int64)
    counts[::13] = 0
    indptr = np.zeros(M + 1, dtype=np.int64)
    indptr[1:] = counts.cumsum()
    indices = rng.integers(0, K, size=int(indptr[-1]), dtype=np.uint16)
    values = rng.uniform(-0.1, 0.1, size=int(indptr[-1])).astype(np.float32)
    packed = csr_to_slice_ell(
        indptr, indices, values, K=K,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=256,
            shim_columns=8, window_count=8,
        ),
    )
    return packed, indptr, indices, values


def make_target_operator(design: str, packed, x: torch.Tensor):
    """Return one selected NPU topology and its per-sample control encoder."""
    A, _, blocks = make_window_inputs(packed, x)
    if design == "dedicated":
        operator = SpMVSELLDedicated(packed.padded_rows, packed.K, blocks, 8)
        encode_control = lambda vector: make_window_inputs(packed, vector)[1]
    elif design == "time_multiplex":
        operator = SpMVSELLTimeMultiplex(packed.padded_rows, packed.K, blocks, 8)
        encode_control = lambda vector: make_window_inputs(packed, vector)[1]
    elif design == "slice_ell_physical":
        operator = SpMVSliceELLDynamicScalarMultiCol(
            M=packed.padded_rows, K=packed.K, blocks_per_column=blocks,
            block_height=8,
        )
        encode_control = lambda vector: make_32core_config(packed, vector)
    else:
        raise ValueError(design)
    return operator, A, encode_control


def make_dummy_samples(count: int, seed: int):
    """Build distinct inputs for a genuine 8-column/32-core dense GEMV."""
    samples = []
    for sample in range(count):
        generator = torch.Generator().manual_seed(seed + 10000 + sample)
        matrix = torch.rand((64, 4096), generator=generator).to(torch.bfloat16)
        vector = torch.rand(4096, generator=generator).to(torch.bfloat16)
        # One 8-row block per column, one K tile: row-major A is already
        # the exact column/block/K-tile order of DenseGEMVKTile.
        tiled_x = vector.repeat(8)
        expected = matrix.float() @ vector.float()
        samples.append((matrix.reshape(-1), tiled_x, expected.to(torch.bfloat16)))
    return samples


def measure_mode(mode: str, target, dummy, target_args, dummy_args, expected):
    """Measure only the target after two warmups; verify every varying input."""
    npu_us = []
    wall_us = []
    dummy_us = []
    for sample in range(WARMUP_ITERS + TIMED_ITERS):
        if mode == "switched":
            dummy_result = dummy(*dummy_args[sample])
            if sample >= WARMUP_ITERS:
                dummy_us.append(dummy_result.npu_time / 1000)
        start = time.perf_counter_ns()
        result = target(*target_args[sample])
        wall = time.perf_counter_ns() - start
        if sample >= WARMUP_ITERS:
            npu_us.append(result.npu_time / 1000)
            wall_us.append(wall / 1000)
    for sample, args in enumerate(target_args):
        errors = verify_buffer(
            args[2].to_torch(), f"{mode} output sample {sample}", expected[sample],
            rel_tol=0.08, abs_tol=0.025,
        )
        if errors:
            raise AssertionError(f"{mode} target sample {sample} mismatch: {errors[:10]}")
    return {
        "npu_us": npu_us,
        "mean_npu_us": statistics.mean(npu_us),
        "median_npu_us": statistics.median(npu_us),
        "mean_wall_us": statistics.mean(wall_us),
        "dummy_npu_us_excluded": dummy_us,
    }


def select_modes(with_dummy: bool, reverse_order: bool) -> tuple[str, ...]:
    """Keep the ordinary measurement free of dummy compilation and execution."""
    if reverse_order and not with_dummy:
        raise ValueError("--reverse-order requires --with-dummy")
    if not with_dummy:
        return ("consecutive",)
    return ("switched", "consecutive") if reverse_order else ("consecutive", "switched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=2, type=int, metavar=("M", "K"), required=True)
    parser.add_argument("--design", choices=("dedicated", "time_multiplex", "slice_ell_physical"),
                        required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--with-dummy", action="store_true",
                        help="also measure after a different 32-core GEMV before every target call")
    parser.add_argument("--reverse-order", action="store_true",
                        help="with --with-dummy, measure switched first to check order sensitivity")
    args = parser.parse_args()
    M, K = args.shape
    if (M, K) not in SHAPES:
        parser.error(f"Step-4 comparison supports only {SHAPES}")
    try:
        modes = select_modes(args.with_dummy, args.reverse_order)
    except ValueError as error:
        parser.error(str(error))

    packed, indptr, indices, values = make_target_matrix(M, K, args.seed)
    count = WARMUP_ITERS + TIMED_ITERS
    vectors = [
        torch.rand(K, generator=torch.Generator().manual_seed(args.seed + 1 + sample))
        .to(torch.bfloat16)
        for sample in range(count)
    ]
    operator, packed_a, encode_control = make_target_operator(args.design, packed, vectors[0])
    operator.compile()
    target = operator.get_callable()
    tensor = aie_utils.DEFAULT_TENSOR_CLASS
    packed_bo = tensor.from_torch(packed_a)
    expected = [cpu_spmv_csr(indptr, indices, values, x) for x in vectors]
    if args.design == "slice_ell_physical":
        rows_per_window = packed.padded_rows // packed.window_count
        local_rows = torch.from_numpy(packed.row_indices.astype(np.int64, copy=False))
        window_bases = torch.arange(packed.padded_rows) // rows_per_window * rows_per_window
        physical_to_canonical = window_bases + local_rows
        expected = [output[physical_to_canonical] for output in expected]
    controls = [tensor.from_torch(encode_control(x)) for x in vectors]
    dummy = None
    dummy_samples = []
    dummy_args = []
    if args.with_dummy:
        dummy_operator = DenseGEMVKTile(M=64, K=4096, cols=8, k_tile=4096)
        dummy_operator.compile()
        if operator.xclbin_artifact.filename == dummy_operator.xclbin_artifact.filename:
            raise AssertionError("target and dummy must be distinct xclbins")
        dummy = dummy_operator.get_callable()
        dummy_samples = make_dummy_samples(count, args.seed)
        dummy_args = [
            (tensor.from_torch(matrix), tensor.from_torch(x), tensor((64,), dtype=np.dtype(bfloat16)))
            for matrix, x, _ in dummy_samples
        ]
        # Check that the independent GEMV is functional before using it as a flush.
        dummy(*dummy_args[0])
        dummy_errors = verify_buffer(
            dummy_args[0][2].to_torch(), "dummy GEMV", dummy_samples[0][2],
            rel_tol=0.08, abs_tol=0.025,
        )
        if dummy_errors:
            raise AssertionError(f"dummy GEMV mismatch: {dummy_errors[:10]}")

    result = {}
    for mode in modes:
        target_args = [
            (packed_bo, controls[sample], tensor((packed.padded_rows,), dtype=np.dtype(bfloat16)))
            for sample in range(count)
        ]
        result[mode] = measure_mode(mode, target, dummy, target_args, dummy_args, expected)
    if args.with_dummy:
        for sample, (_, _, dummy_expected) in enumerate(dummy_samples):
            dummy_errors = verify_buffer(
                dummy_args[sample][2].to_torch(), f"dummy GEMV sample {sample}", dummy_expected,
                rel_tol=0.08, abs_tol=0.025,
            )
            if dummy_errors:
                raise AssertionError(f"dummy GEMV sample {sample} mismatch: {dummy_errors[:10]}")
    print(json.dumps({
        "shape": [M, K], "seed": args.seed, "design": args.design,
        "packed_a_bytes": packed.packed_a.nbytes,
        "dummy": "DenseGEMVKTile 64x4096, 8 columns x 4 cores, K tile 4096" if args.with_dummy else None,
        "warmup_iters": WARMUP_ITERS, "timed_iters": TIMED_ITERS,
        "order": list(modes), "results": result,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
