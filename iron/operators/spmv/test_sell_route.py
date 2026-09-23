# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 2 NPU routing tests; copy producers are deliberately not SpMV."""

import numpy as np
import pytest
import torch

from iron.common.test_utils import run_test
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLReorderRoute


def make_route_input(M: int, windows: int, pattern: str, seed: int = 81):
    """Construct 10-slot producer input, local row maps, and canonical output."""

    rng = np.random.default_rng(seed)
    physical = torch.from_numpy(rng.uniform(-2, 2, size=M).astype(np.float32)).to(torch.bfloat16)
    joined = torch.zeros((M // 8, 10), dtype=torch.bfloat16)
    joined[:, [0, 1, 2, 3, 4, 6, 7, 8]] = physical.reshape(-1, 8)
    joined[:, [5, 9]] = -100  # Dummy values must never reach canonical output.
    rows_per_window = M // windows
    row_map = np.empty(M, dtype=np.uint16)
    canonical = torch.empty_like(physical)
    for window in range(windows):
        base = window * rows_per_window
        if pattern == "identity":
            order = np.arange(rows_per_window)
        elif pattern == "reverse":
            order = np.arange(rows_per_window - 1, -1, -1)
        elif pattern == "random":
            order = rng.permutation(rows_per_window)
        else:
            raise ValueError(pattern)
        row_map[base : base + rows_per_window] = order
        canonical[base + torch.from_numpy(order.copy())] = physical[base : base + rows_per_window]
    return joined.reshape(-1), torch.from_numpy(row_map.view(np.int16)), canonical


@pytest.mark.parametrize(
    "M,columns,windows,pattern",
    [
        (1024, 8, 8, "identity"),
        (1024, 8, 8, "reverse"),
        (1024, 8, 8, "random"),
        (28672, 8, 8, "random"),
        (28672, 8, 16, "random"),
    ],
)
def test_sell_route_canonical_output(aie_context, M, columns, windows, pattern):
    physical, row_map, expected = make_route_input(M, windows, pattern)
    operator = SpMVSELLReorderRoute(M=M, columns=columns, windows=windows, context=aie_context)
    errors, _, _ = run_test(
        operator,
        {"physical": physical, "row_map": row_map},
        {"output": expected},
        rel_tol=0.0,
        abs_tol=0.0,
        warmup_iters=1,
        timed_iters=1,
    )
    assert not errors, errors
