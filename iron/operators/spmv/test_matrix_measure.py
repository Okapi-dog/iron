# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side checks for benchmark input layouts before NPU execution."""

import numpy as np
import torch

from iron.operators.spmv.matrix_preparation import MatrixInput, generate_synthetic_csr
from iron.operators.spmv.matrix_measure import make_case
from iron.operators.spmv.slice_ell import _bf16_bits


def test_ell_case_matches_previous_vertical_packing():
    matrix = generate_synthetic_csr(MatrixInput(
        source="synthetic", M=1024, K=128, density=0.125,
        row_pattern="cv", row_cv=0.25, seed=17,
    ))
    _, inputs, fmt, storage, extra = make_case(matrix, "ell", windows=0)
    width = extra["ell_width"]

    previous = np.zeros((matrix.profile.M // 32, width, 2, 32), dtype=np.uint16)
    bits = _bf16_bits(matrix.values)
    for row in range(matrix.profile.M):
        start, end = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
        count = end - start
        previous[row // 32, :count, 0, row % 32] = matrix.indices[start:end]
        previous[row // 32, :count, 1, row % 32] = bits[start:end]

    actual = inputs["packed"].view(torch.uint16).numpy().reshape(previous.shape)
    assert fmt.name == "ell"
    assert np.array_equal(actual, previous)
    assert extra["packed_a_bytes"] == storage["packed_a_bytes"]
    assert torch.equal(inputs["vector"], matrix.vector)
