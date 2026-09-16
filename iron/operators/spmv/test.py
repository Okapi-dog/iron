# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from iron.common.test_utils import run_test
from iron.operators.spmv.op import SpMVELL
from iron.operators.spmv.reference import make_uniform_ell, reference_ell
from iron.operators.spmv.op import SpMVSELL32
from iron.operators.spmv.op import SpMVSELL32Block
from iron.operators.spmv.reference import make_uniform_sell32, reference_sell32, reference_sell32_block


def test_static_ell_1024x2048(aie_context):
    M, K, width = 1024, 2048, 256
    packed = make_uniform_ell(M, K, width, seed=17)
    vector = torch.rand(K, dtype=torch.float32).to(torch.bfloat16)
    expected = reference_ell(packed, vector, M, width)
    operator = SpMVELL(
        M=M,
        K=K,
        ell_width=width,
        rows=4,
        cols=8,
        rows_per_core=2,
        context=aie_context,
    )
    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"packed": packed, "vector": vector},
        {"output": expected},
        rel_tol=0.04,
        abs_tol=1e-4,
        warmup_iters=2,
    )
    print(f"SpMV ELL latency: {latency_us:.1f} us; effective BW: {bandwidth_gbps:.3f} GB/s")
    assert not errors, errors


def test_static_sell32_block_1024x2048(aie_context):
    M, K, width = 1024, 2048, 256
    packed = make_uniform_sell32(M, K, width, seed=31)
    vector = torch.rand(K, dtype=torch.float32).to(torch.bfloat16)
    expected = reference_sell32_block(packed, vector, M, width)
    operator = SpMVSELL32Block(M=M, K=K, ell_width=width, rows=4, cols=8, context=aie_context)
    errors, latency_us, bandwidth_gbps = run_test(operator, {"packed": packed, "vector": vector}, {"output": expected}, rel_tol=0.06, abs_tol=1e-4, warmup_iters=2)
    print(f"SpMV SELL-32 block latency: {latency_us:.1f} us; effective BW: {bandwidth_gbps:.3f} GB/s")
    assert not errors, errors


def test_static_sell32_1024x2048(aie_context):
    M, K, width = 1024, 2048, 256
    packed = make_uniform_sell32(M, K, width, seed=29)
    vector = torch.rand(K, dtype=torch.float32).to(torch.bfloat16)
    expected = reference_sell32(packed, vector, M, width)
    operator = SpMVSELL32(M=M, K=K, ell_width=width, rows=4, cols=8, context=aie_context)
    errors, latency_us, bandwidth_gbps = run_test(operator, {"packed": packed, "vector": vector}, {"output": expected}, rel_tol=0.04, abs_tol=1e-4, warmup_iters=2)
    print(f"SpMV SELL-32 latency: {latency_us:.1f} us; effective BW: {bandwidth_gbps:.3f} GB/s")
    assert not errors, errors
