# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 3: real SELL-C-sigma SpMV, MemTile join, and canonical output."""

import numpy as np
import pytest
import torch

from iron.common.test_utils import run_test
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated
from iron.operators.spmv.sell_c_sigma_runtime import make_dedicated_inputs
from iron.operators.spmv.slice_ell import SliceELLConfig, cpu_spmv_csr, csr_to_slice_ell


def make_matrix(M: int, K: int, pattern: str):
    """Make deterministic CSR with zero rows and p=0, 1, 2, or variable."""

    rng = np.random.default_rng(31)
    if pattern == "identity":
        counts = np.full(M, 30, dtype=np.int64)
    elif pattern == "reverse":
        counts = np.resize(np.array([0, 1, 257, 300, 4, 280, 0, 260]), M)
    elif pattern == "random":
        counts = rng.integers(0, 340, size=M, dtype=np.int64)
        counts[::11] = 0
    elif pattern == "zero_slice":
        counts = np.full(M, 20, dtype=np.int64)
        counts[:8] = 0  # Sorting moves these eight rows into a p=0 slice.
    else:
        raise ValueError(pattern)
    pointers = np.zeros(M + 1, dtype=np.int64)
    pointers[1:] = counts.cumsum()
    indices = rng.integers(0, K, size=int(pointers[-1]), dtype=np.uint16)
    values = rng.uniform(-0.1, 0.1, size=int(pointers[-1])).astype(np.float32)
    return pointers, indices, values


@pytest.mark.parametrize(
    "M,K,columns,windows,pattern",
    [
        (64, 512, 1, 1, "identity"),
        (64, 512, 1, 1, "reverse"),
        (64, 512, 1, 1, "zero_slice"),
        (1024, 512, 8, 8, "random"),
        (1024, 512, 8, 16, "random"),
        (1021, 512, 8, 8, "random"),
    ],
)
def test_sell_dedicated_matches_csr(aie_context, M, K, columns, windows, pattern):
    pointers, indices, values = make_matrix(M, K, pattern)
    packed = csr_to_slice_ell(
        pointers, indices, values, K=K,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=256,
            shim_columns=columns, window_count=windows,
        ),
    )
    x = torch.rand(K, generator=torch.Generator().manual_seed(61)).to(torch.bfloat16)
    A, control, block_counts = make_dedicated_inputs(packed, x)
    operator = SpMVSELLDedicated(
        M=packed.padded_rows, K=K, blocks_per_column=block_counts, windows=windows,
        context=aie_context,
    )
    expected = torch.zeros(packed.padded_rows, dtype=torch.bfloat16)
    expected[:M] = cpu_spmv_csr(pointers, indices, values, x)
    errors, latency_us, _ = run_test(
        operator, {"packed": A, "control": control}, {"output": expected},
        rel_tol=0.08, abs_tol=0.025, warmup_iters=2, timed_iters=5,
    )
    assert not errors, errors
    print(f"Step 3 canonical-output latency: {M}x{K}, {columns} columns, "
          f"{windows} windows, {pattern}: {latency_us:.2f} us")
