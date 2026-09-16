# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import torch

from iron.operators.spmv.slice_ell import (
    SliceELLConfig,
    cpu_spmv_csr,
    cpu_spmv_slice_ell,
    csr_to_slice_ell,
    dense_to_slice_ell,
    make_runtime_config,
)


def _csr(row_counts, K=97):
    indptr = np.zeros(len(row_counts) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_counts)
    rng = np.random.default_rng(91)
    indices = rng.integers(0, K, size=int(indptr[-1]), dtype=np.int64)
    values = rng.uniform(-1.0, 1.0, size=int(indptr[-1])).astype(np.float32)
    return indptr, indices, values, K


def test_slice_ell_round_trip_preserves_rows_with_tail_and_empty_slice():
    # Slice 0 has p=2 and slice 1 has p=1; the final three rows are a tail.
    counts = [45] + [1] * 30 + [0] + [0] * 5 + [7, 2, 0]
    indptr, indices, values, K = _csr(counts)
    config = SliceELLConfig(core_rows=4, block_height=32, block_width=32, shim_columns=2)
    packed = csr_to_slice_ell(indptr, indices, values, K=K, config=config)
    x = torch.rand(K, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)

    assert packed.M == len(counts)
    assert packed.padded_rows == 64
    assert packed.blocks_per_slice.tolist() == [2, 1]
    assert torch.allclose(cpu_spmv_slice_ell(packed, x).float(), cpu_spmv_csr(indptr, indices, values, x).float(), atol=0.02, rtol=0.02)


def test_all_zero_matrix_has_no_a_payload_and_zero_output():
    indptr = np.zeros(34, dtype=np.int64)
    config = SliceELLConfig(shim_columns=2)
    packed = csr_to_slice_ell(indptr, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), K=64, config=config)
    assert packed.blocks_per_slice.tolist() == [0, 0]
    assert packed.packed_a.size == 0
    x = torch.ones(64, dtype=torch.bfloat16)
    assert torch.equal(cpu_spmv_slice_ell(packed, x), torch.zeros(33, dtype=torch.bfloat16))


def test_dense_to_slice_ell_is_a_row_order_preserving_csr_bridge():
    matrix = torch.zeros((35, 67), dtype=torch.bfloat16)
    matrix[0, [3, 9, 65]] = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
    matrix[34, 2] = 4.0
    packed = dense_to_slice_ell(matrix, config=SliceELLConfig(block_width=32, shim_columns=2))
    x = torch.rand(67, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
    expected = (matrix.float() @ x.float()).to(torch.bfloat16)
    assert torch.equal(cpu_spmv_slice_ell(packed, x), expected)


def test_packed_config_keeps_bf16_vector_bits_and_uint16_control_words():
    x = torch.tensor([1.0, -2.5, 0.25], dtype=torch.bfloat16)
    config = make_runtime_config(x, [0, 257], max_local_slices=4)
    words = config.view(torch.uint16).numpy()
    assert config.numel() == 32  # 64-byte alignment
    assert np.array_equal(words[:3], x.view(torch.uint16).numpy())
    assert words[3:7].tolist() == [0, 257, 0, 0]


def test_manifest_and_payloads_are_reproducible(tmp_path):
    indptr, indices, values, K = _csr([3, 0, 7, 1, 32, 33])
    packed = csr_to_slice_ell(indptr, indices, values, K=K, config=SliceELLConfig(shim_columns=1))
    paths = packed.save_cache(tmp_path, "tiny")
    manifest = json.loads(paths["manifest"].read_text())
    assert paths["packed_a"].exists() and paths["blocks_per_slice"].exists()
    assert manifest["format"] == "row-order-preserving-slice-ell"
    assert manifest["config"]["block_height"] == 32
    assert manifest["packed_a_words"] == np.fromfile(paths["packed_a"], dtype="<u2").size
