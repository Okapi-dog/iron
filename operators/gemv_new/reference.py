# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16
import time


def generate_golden_reference(M=42, K=42,calc_c=True, seed=42):
    """
    Generate golden reference data for GEMV (General Matrix-Vector Multiplication).

    Parameters:
        M: Number of rows of matrix A
        K: Number of columns of matrix A (equals vector B length)
        seed: Random seed
        calc_c: Whether to calculate the output vector C

    Returns:
        dict: Contains 'A' (matrix), 'B' (vector), 'C' (output vector)
    """
    torch.manual_seed(seed)

    # Generate golden inputs
    val_range = 4
    A = torch.rand(M, K, dtype=torch.bfloat16) * val_range
    B = torch.rand(K, dtype=torch.bfloat16) * val_range

    if not calc_c:
        return {
            "A": A,
            "B": B,
            "C": None,
        }

    # Generate golden outputs
    start_time = time.perf_counter()
    C = A @ B
    end_time = time.perf_counter()
    elapsed_us = (end_time - start_time) * 1e6
    #print(f"[Reference] GEMV computation time: {elapsed_us:.2f} us")

    return {
        "A": A,
        "B": B,
        "C": C,
    }
