# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from iron.common.test_utils import run_test
from iron.operators.spmv.op import SpMVELL
from iron.operators.spmv.reference import make_uniform_ell, reference_ell
from iron.operators.spmv.op import SpMVSELL32
from iron.operators.spmv.op import SpMVSELL32Block
from iron.operators.spmv.op import SpMVSliceELLStatic
from iron.operators.spmv.op import SpMVSliceELLDynamicScalar
from iron.operators.spmv.reference import make_uniform_sell32, reference_sell32, reference_sell32_block
from iron.operators.spmv.slice_ell import SliceELLConfig, cpu_spmv_slice_ell, csr_to_slice_ell

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


def _make_uniform_slice_ell(M: int, K: int, blocks_per_slice: int, seed: int):
    """Create a row-ordered CSR input whose every slice has the requested p."""
    width = blocks_per_slice * 256
    rng = np.random.default_rng(seed)
    indptr = np.arange(M + 1, dtype=np.int64) * width
    indices = rng.integers(0, K, size=M * width, dtype=np.uint16)
    values = rng.uniform(-1, 1, size=M * width).astype(np.float32)
    return csr_to_slice_ell(
        indptr,
        indices,
        values,
        K=K,
        config=SliceELLConfig(core_rows=4, block_height=32, block_width=256, shim_columns=8),
    )


def _make_dynamic_scalar_micro_input():
    """Four slices whose exact packed control values are p=[0, 1, 4, 2]."""
    M, K, block_width = 128, 2048, 256
    ps = [0, 1, 4, 2]
    row_counts = np.repeat(np.asarray(ps, dtype=np.int64) * block_width, 32)
    indptr = np.concatenate(([0], np.cumsum(row_counts, dtype=np.int64)))
    rng = np.random.default_rng(211)
    indices = rng.integers(0, K, size=int(indptr[-1]), dtype=np.uint16)
    values = rng.uniform(-1.0, 1.0, size=int(indptr[-1])).astype(np.float32)
    packed = csr_to_slice_ell(
        indptr,
        indices,
        values,
        K=K,
        config=SliceELLConfig(core_rows=4, block_height=32, block_width=256, shim_columns=1),
    )
    assert packed.blocks_per_slice.tolist() == ps
    return packed


def test_slice_ell_dynamic_scalar_state_p0142(aie_context):
    """NPU2 Phase-4 micro test: dynamic acquire count and scalar FP32 L1 state."""
    packed = _make_dynamic_scalar_micro_input()
    vector = torch.rand(2048, generator=torch.Generator().manual_seed(212)).to(torch.bfloat16)
    expected = cpu_spmv_slice_ell(packed, vector)
    config = torch.empty(2048 + 4, dtype=torch.int16)
    config[:2048] = vector.view(torch.uint16).view(torch.int16)
    config[2048:] = torch.from_numpy(packed.blocks_per_slice.astype(np.int16, copy=False))
    operator = SpMVSliceELLDynamicScalar(
        M=128, K=2048, total_blocks=int(packed.blocks_per_slice.sum()), context=aie_context
    )
    errors, _, _ = run_test(
        operator,
        {"packed": packed.packed_a_as_bf16, "config": config},
        {"output": expected},
        rel_tol=0.06,
        abs_tol=1e-3,
        warmup_iters=1,
    )
    assert not errors, errors


def test_slice_ell_horizontal_static_p1_1024x2048(aie_context):
    """Run the p=1 horizontal kernel with eight register-resident row accumulators."""
    packed = _make_uniform_slice_ell(1024, 2048, blocks_per_slice=1, seed=71)
    vector = torch.rand(2048, generator=torch.Generator().manual_seed(72)).to(torch.bfloat16)
    expected = cpu_spmv_slice_ell(packed, vector)
    operator = SpMVSliceELLStatic(1024, 2048, blocks_per_slice=1, context=aie_context)
    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"packed": packed.packed_a_as_bf16, "vector": vector},
        {"output": expected},
        rel_tol=0.06,
        abs_tol=1e-3,
        warmup_iters=2,
    )
    print(f"Slice-ELL horizontal p=1 latency: {latency_us:.1f} us; effective BW: {bandwidth_gbps:.3f} GB/s")
    assert not errors, errors


def test_slice_ell_horizontal_static_p2_1024x2048(aie_context):
    """Run the p=2 A-plan: two FIFO objects, one kernel call, eight FP32 accumulators."""
    packed = _make_uniform_slice_ell(1024, 2048, blocks_per_slice=2, seed=73)
    vector = torch.rand(2048, generator=torch.Generator().manual_seed(74)).to(torch.bfloat16)
    expected = cpu_spmv_slice_ell(packed, vector)
    operator = SpMVSliceELLStatic(1024, 2048, blocks_per_slice=2, context=aie_context)
    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        {"packed": packed.packed_a_as_bf16, "vector": vector},
        {"output": expected},
        rel_tol=0.06,
        abs_tol=1e-3,
        warmup_iters=2,
    )
    print(f"Slice-ELL horizontal p=2 latency: {latency_us:.1f} us; effective BW: {bandwidth_gbps:.3f} GB/s")
    assert not errors, errors
