# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the isolated equal-NNZ window experiment."""

import unittest

from iron.operators.spmv.matrix_preparation import MatrixInput, generate_synthetic_csr
from iron.operators.spmv.measure_nnz_windows import (
    BLOCK_HEIGHT, COLUMNS, analyze, check_packing, make_balanced_case,
)


class NNZWindowExperimentTests(unittest.TestCase):
    def test_padding_preserves_canonical_output_and_block_counts(self):
        matrix = generate_synthetic_csr(MatrixInput(
            source="synthetic", M=96, K=64, density=0.20,
            row_cv=0.8, row_pattern="cv", seed=11,
        ))
        case = make_balanced_case(matrix)
        self.assertEqual(sum(case.valid_rows), matrix.profile.M)
        self.assertEqual(case.physical_rows % (BLOCK_HEIGHT * COLUMNS), 0)
        self.assertEqual(case.physical_csr_indptr[-1], matrix.indptr[-1])
        self.assertEqual(
            [int(case.packed.column_blocks_per_slice(c).sum()) for c in range(COLUMNS)],
            analyze(matrix.profile)["equal_nnz"]["column_blocks"],
        )
        check_packing(matrix, case)


if __name__ == "__main__":
    unittest.main()
