# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible Phase-4 dynamic scalar-state measurements.

Run from the IRON checkout after enabling XRT and the IRON environment:
``python operators/spmv/measure_dynamic_scalar.py``.
It measures the three fixed 12.5%-density shapes with all 32 cores and with
one column/four cores.  The reported time is device-only ``result.npu_time``.
"""

import argparse

import numpy as np
import torch

from iron.common.test_utils import run_test
from iron.operators.spmv.op import SpMVSliceELLDynamicScalarMultiCol
from iron.operators.spmv.slice_ell import SliceELLConfig, csr_to_slice_ell


CASES = ((4096, 4096), (4096, 11008), (28672, 8192))


def make_uniform_case(M: int, K: int, cols: int, seed: int, block_height: int = 32):
    """Pack a fixed-seed 12.5%-density matrix and per-column config objects."""
    logical_width = K // 8
    rng = np.random.default_rng(seed)
    indptr = np.arange(M + 1, dtype=np.int64) * logical_width
    indices = rng.integers(0, K, size=M * logical_width, dtype=np.uint16)
    values = rng.uniform(-1.0, 1.0, size=M * logical_width).astype(np.float32)
    packed = csr_to_slice_ell(
        indptr, indices, values, K=K,
        config=SliceELLConfig(core_rows=4, block_height=block_height, block_width=256, shim_columns=cols),
    )
    vector = torch.rand(K, generator=torch.Generator().manual_seed(seed + 1)).to(torch.bfloat16)
    slices_per_col = M // (block_height * cols)
    p_by_col = packed.blocks_per_slice.reshape(cols, slices_per_col)
    blocks_per_col = tuple(int(p.sum()) for p in p_by_col)
    config_words = 2 + K + slices_per_col
    config_words += config_words % 2
    config = torch.zeros(cols * config_words, dtype=torch.int16)
    x_words = vector.view(torch.uint16).view(torch.int16)
    for col in range(cols):
        base = col * config_words
        config[base] = block_height // 4
        config[base + 2 : base + 2 + K] = x_words
        config[base + 2 + K : base + 2 + K + slices_per_col] = torch.from_numpy(
            p_by_col[col].astype(np.int16, copy=False)
        )
    return packed, config, logical_width, blocks_per_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs=2, type=int, metavar=("M", "K"))
    parser.add_argument("--cols", type=int, choices=range(1, 9))
    parser.add_argument("--block-height", type=int, default=32)
    args = parser.parse_args()
    cases = (tuple(args.shape),) if args.shape else CASES
    columns = (args.cols,) if args.cols else (8, 1)
    for case_index, (M, K) in enumerate(cases):
        for cols in columns:
            packed, config, logical_width, blocks_per_col = make_uniform_case(
                M, K, cols, seed=1000 + case_index, block_height=args.block_height
            )
            operator = SpMVSliceELLDynamicScalarMultiCol(
                M=M, K=K, blocks_per_column=blocks_per_col, block_height=args.block_height
            )
            _, latency_us, bandwidth_gbps = run_test(
                operator,
                {"packed": packed.packed_a_as_bf16, "config": config},
                {"output": None},
                warmup_iters=2,
                timed_iters=5,
            )
            physical_width = int(packed.blocks_per_slice.max()) * 256
            print(
                f"M={M} K={K} cols={cols} cores={4 * cols} "
                f"block_height={args.block_height} rows_per_core={args.block_height // 4} "
                f"logical_width={logical_width} physical_width={physical_width} "
                f"blocks_per_col={blocks_per_col} latency_us={latency_us:.4f} "
                f"effective_bandwidth_gbps={bandwidth_gbps:.4f}"
            )


if __name__ == "__main__":
    main()
