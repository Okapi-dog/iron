# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure selectable 3-core/column SELL layouts on the same input matrix.

Within one run, identity-map and canonical-output tests consume the same
packed A and x.  For B_h=8, the 32-core old design is also measured on the
same A, but returns physical y'.  Different B_h values have different packed
A sizes, so cross-run latency is not an isolated core-layout comparison.
"""

import argparse

import numpy as np
import torch

from iron.common.test_utils import run_test
from iron.operators.spmv.op import SpMVSliceELLDynamicScalarMultiCol
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated, SpMVSELLTimeMultiplex
from iron.operators.spmv.sell_c_sigma_runtime import make_dedicated_inputs
from iron.operators.spmv.sell_c_sigma_layout import SELLCoreLayout
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
    parser.add_argument("--rows-per-core", nargs=3, type=int, default=(2, 3, 3),
                        metavar=("CORE0", "CORE1", "CORE2"))
    parser.add_argument("--include-time-multiplex", action="store_true",
                        help="also measure the 4-compute-core Step-4 design on exactly the same packed A")
    parser.add_argument("--windows", type=int, default=8, choices=(8, 16))
    args = parser.parse_args()
    M, K = args.M, args.K
    layout = SELLCoreLayout(tuple(args.rows_per_core))
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
            core_rows=4 if layout.block_height == 8 else 3,
            block_height=layout.block_height, block_width=256,
            shim_columns=8, window_count=args.windows,
        ),
    )
    x = torch.rand(K, generator=torch.Generator().manual_seed(args.seed + 1)).to(torch.bfloat16)
    A, control, block_counts = make_dedicated_inputs(packed, x, layout.rows_per_core)
    physical = torch.zeros(packed.padded_rows, dtype=torch.bfloat16)
    physical[:M] = cpu_spmv_slice_ell(packed, x)
    canonical = torch.zeros_like(physical)
    canonical[:M] = cpu_spmv_csr(pointers, indices, values, x)

    # The row map is the only difference between the two 24-core runs.
    identity_control = control.clone()
    window_rows = packed.padded_rows // args.windows
    control_words = identity_control.numel() // args.windows
    config_words = control_words - window_rows
    for window in range(args.windows):
        first = window * control_words + config_words
        identity_control[first : first + window_rows] = torch.arange(window_rows, dtype=torch.int16)

    operator = SpMVSELLDedicated(
        packed.padded_rows, K, block_counts, args.windows, rows_per_core=layout.rows_per_core,
    )
    experiments = [
        ("24-core physical", operator,
         {"packed": A, "control": identity_control}, physical),
        ("24-core canonical", operator,
         {"packed": A, "control": control}, canonical),
    ]
    if layout.block_height == 8:
        if args.include_time_multiplex:
            mux = SpMVSELLTimeMultiplex(packed.padded_rows, K, block_counts, args.windows)
            experiments.extend([
                ("32-core time-multiplex physical", mux,
                 {"packed": A, "control": identity_control}, physical),
                ("32-core time-multiplex canonical", mux,
                 {"packed": A, "control": control}, canonical),
            ])
        experiments.append((
            "32-core physical", SpMVSliceELLDynamicScalarMultiCol(
                M=packed.padded_rows, K=K, blocks_per_column=block_counts, block_height=8,
            ), {"packed": A, "config": make_32core_config(packed, x)}, physical,
        ))
    print(f"M={M} K={K} padded_M={packed.padded_rows} seed={args.seed} "
          f"windows={args.windows} "
          f"rows_per_core={layout.rows_per_core} packed_A_bytes={packed.packed_a.nbytes}")
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
