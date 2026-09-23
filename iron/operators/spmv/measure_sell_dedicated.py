# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare 24-core SELL/reorder with 32-core physical-output Slice-ELL.

All three runs consume the *same packed A* and x.  The 24-core identity-map
run returns physical y' and isolates the cost of nontrivial scatter.  The
32-core old design also returns physical y', while 24-core SELL returns y.
The comparison is architectural, not a row-sort-vs-no-sort A-traffic study.
"""

import argparse

import numpy as np
import torch

from iron.common.test_utils import run_test
from iron.operators.spmv.op import SpMVSliceELLDynamicScalarMultiCol
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated
from iron.operators.spmv.sell_c_sigma_runtime import make_dedicated_inputs
from iron.operators.spmv.slice_ell import (
    SliceELLConfig, cpu_spmv_csr, cpu_spmv_slice_ell, csr_to_slice_ell,
)


def make_32core_config(packed, x):
    """Encode the legacy per-column config for the same SELL packed A."""
    slices_per_column = packed.slices_per_column
    config_words = 2 + packed.K + slices_per_column
    config_words += config_words % 2
    result = torch.zeros(8 * config_words, dtype=torch.int16)
    x_words = x.view(torch.uint16).view(torch.int16)
    for col in range(8):
        base = col * config_words
        result[base] = 2
        result[base + 2 : base + 2 + packed.K] = x_words
        result[base + 2 + packed.K : base + 2 + packed.K + slices_per_column] = torch.from_numpy(
            packed.column_blocks_per_slice(col).view(np.int16)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    M, K = args.M, args.K
    if M % 64 or K < 512:
        raise ValueError("this experiment needs M divisible by 64 and K >= 512")

    rng = np.random.default_rng(args.seed)
    counts = rng.integers(4, 300, size=M, dtype=np.int64)
    counts[::13] = 0
    pointers = np.zeros(M + 1, dtype=np.int64)
    pointers[1:] = counts.cumsum()
    indices = rng.integers(0, K, size=int(pointers[-1]), dtype=np.uint16)
    values = rng.uniform(-0.1, 0.1, size=int(pointers[-1])).astype(np.float32)
    packed = csr_to_slice_ell(
        pointers, indices, values, K=K,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=256,
            shim_columns=8, window_count=8,
        ),
    )
    x = torch.rand(K, generator=torch.Generator().manual_seed(args.seed + 1)).to(torch.bfloat16)
    A, control, block_counts = make_dedicated_inputs(packed, x)
    physical = cpu_spmv_slice_ell(packed, x)
    canonical = cpu_spmv_csr(pointers, indices, values, x)

    # The row map is the only difference between the two 24-core runs.
    identity_control = control.clone()
    window_rows = M // 8
    control_words = identity_control.numel() // 8
    config_words = control_words - window_rows
    for window in range(8):
        first = window * control_words + config_words
        identity_control[first : first + window_rows] = torch.arange(window_rows, dtype=torch.int16)

    experiments = (
        ("24-core physical", SpMVSELLDedicated(M, K, block_counts, 8),
         {"packed": A, "control": identity_control}, physical),
        ("24-core canonical", SpMVSELLDedicated(M, K, block_counts, 8),
         {"packed": A, "control": control}, canonical),
        ("32-core physical", SpMVSliceELLDynamicScalarMultiCol(
            M=M, K=K, blocks_per_column=block_counts, block_height=8,
        ), {"packed": A, "config": make_32core_config(packed, x)}, physical),
    )
    print(f"M={M} K={K} seed={args.seed} packed_A_bytes={packed.packed_a.nbytes}")
    for label, operator, inputs, expected in experiments:
        errors, latency_us, bandwidth = run_test(
            operator, inputs, {"output": expected}, rel_tol=0.08, abs_tol=0.025,
            warmup_iters=2, timed_iters=5,
        )
        if errors:
            raise AssertionError(f"{label}: {errors}")
        print(f"{label}: {latency_us:.2f} us, {bandwidth:.2f} GB/s")


if __name__ == "__main__":
    main()
