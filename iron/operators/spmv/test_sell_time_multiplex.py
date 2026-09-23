# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 4: the same four cores compute and then one reorders each window."""

import numpy as np
import pytest
import torch

from iron.common.test_utils import run_test
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLTimeMultiplex
from iron.operators.spmv.sell_c_sigma_runtime import (
    make_time_multiplex_inputs, prepare_sell_design,
)
from iron.operators.spmv.slice_ell import SliceELLConfig, cpu_spmv_csr, csr_to_slice_ell


def make_matrix(M: int, K: int, pattern: str):
    """Build deterministic CSR whose row permutation is easy to check."""
    rng = np.random.default_rng(31)
    if pattern == "identity":
        counts = np.full(M, 30, dtype=np.int64)
    elif pattern == "reverse":
        counts = np.arange(M, dtype=np.int64)
    elif pattern == "random":
        counts = rng.integers(0, 340, size=M, dtype=np.int64)
        counts[::11] = 0
    elif pattern == "zero_slice":
        counts = np.full(M, 20, dtype=np.int64)
        counts[:8] = 0
    else:
        raise ValueError(pattern)
    indptr = np.zeros(M + 1, dtype=np.int64)
    indptr[1:] = counts.cumsum()
    indices = rng.integers(0, K, size=int(indptr[-1]), dtype=np.uint16)
    values = rng.uniform(-0.1, 0.1, size=int(indptr[-1])).astype(np.float32)
    return indptr, indices, values


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
def test_sell_time_multiplex_matches_csr(aie_context, M, K, columns, windows, pattern):
    indptr, indices, values = make_matrix(M, K, pattern)
    packed = csr_to_slice_ell(
        indptr, indices, values, K=K,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=256,
            shim_columns=columns, window_count=windows,
        ),
    )
    if pattern == "identity":
        assert np.array_equal(packed.row_indices, np.arange(M))
    elif pattern == "reverse":
        assert np.array_equal(packed.row_indices, np.arange(M - 1, -1, -1))
    x = torch.rand(K, generator=torch.Generator().manual_seed(61)).to(torch.bfloat16)
    A, control, block_counts = make_time_multiplex_inputs(packed, x)
    operator, inputs = prepare_sell_design(
        packed, x, "sell_time_multiplex_reorder", context=aie_context,
    )
    assert isinstance(operator, SpMVSELLTimeMultiplex)
    assert operator.blocks_per_column == block_counts
    assert torch.equal(inputs["packed"], A)
    assert torch.equal(inputs["control"], control)
    expected = torch.zeros(packed.padded_rows, dtype=torch.bfloat16)
    expected[:M] = cpu_spmv_csr(indptr, indices, values, x)
    errors, latency_us, _ = run_test(
        operator, inputs, {"output": expected},
        rel_tol=0.08, abs_tol=0.025, warmup_iters=2, timed_iters=5,
    )
    assert not errors, errors
    print(f"Step 4 time-multiplex latency: {M}x{K}, {columns} columns, "
          f"{windows} windows, {pattern}: {latency_us:.2f} us")
