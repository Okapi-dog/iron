# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only Step-1 tests for windowed row sorting and inverse permutation."""

import json

import numpy as np
import pytest
import torch

from iron.operators.spmv.evaluation import (
    DesignSpec,
    FormatSpec,
    MatrixInput,
    estimate_storage,
    generate_synthetic_csr,
    pack_for_design,
)
from iron.operators.spmv.slice_ell import (
    SliceELLConfig,
    cpu_spmv_csr,
    cpu_spmv_slice_ell,
    cpu_unpermute_windows,
    csr_to_slice_ell,
    dense_to_slice_ell,
)


def _csr(counts, K=97):
    pointers = np.zeros(len(counts) + 1, dtype=np.int64)
    pointers[1:] = np.cumsum(counts)
    rng = np.random.default_rng(22)
    indices = rng.integers(0, K, size=int(pointers[-1]), dtype=np.int64)
    values = rng.uniform(-1, 1, size=int(pointers[-1])).astype(np.float32)
    return pointers, indices, values, K


def _check_reference(packed, pointers, indices, values, K):
    x = torch.rand(K, generator=torch.Generator().manual_seed(31)).to(torch.bfloat16)
    physical = cpu_spmv_slice_ell(packed, x)
    canonical = cpu_unpermute_windows(packed, physical)
    expected = cpu_spmv_csr(pointers, indices, values, x)
    assert torch.allclose(canonical.float(), expected.float(), atol=0.05, rtol=0.05)
    return physical, canonical


def test_disabled_sort_preserves_old_payload_bitwise():
    pointers, indices, values, K = _csr([33, 0, 5, 2, 8, 0, 1, 40, 3])
    old = csr_to_slice_ell(
        pointers,
        indices,
        values,
        K=K,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=32, shim_columns=2
        ),
    )
    explicit = csr_to_slice_ell(
        pointers,
        indices,
        values,
        K=K,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=32, shim_columns=2, window_count=0
        ),
    )
    assert np.array_equal(old.packed_a, explicit.packed_a)
    assert np.array_equal(old.blocks_per_slice, explicit.blocks_per_slice)
    assert np.array_equal(old.slice_word_offsets, explicit.slice_word_offsets)
    assert np.array_equal(old.column_slice_offsets, explicit.column_slice_offsets)
    assert old.manifest() == explicit.manifest()
    # Independent hash from the unmodified spmv/slice-ell branch.
    assert (
        old.manifest()["packed_a_sha256"]
        == "da67f1efe4156f46a7b7fe56b28095b9564cbe38a4d7efe5ff71bd7df88ea291"
    )
    physical, canonical = _check_reference(old, pointers, indices, values, K)
    assert torch.equal(physical, canonical)


@pytest.mark.parametrize("windows", [1, 8, 16])
def test_sorted_output_unpermutes_and_matches_csr(windows):
    counts = np.random.default_rng(9).integers(0, 73, size=129, dtype=np.int64)
    counts[16:24] = 0  # One entire zero-block slice.
    counts[128] = 5  # Final incomplete slice.
    pointers, indices, values, K = _csr(counts)
    packed = csr_to_slice_ell(
        pointers,
        indices,
        values,
        K=K,
        config=SliceELLConfig(
            core_rows=4,
            block_height=8,
            block_width=32,
            shim_columns=8,
            window_count=windows,
        ),
    )
    assert packed.window_count == windows
    assert packed.row_indices is not None
    assert packed.window_slice_offsets.size == windows + 1
    for first, last in zip(
        packed.window_slice_offsets[:-1], packed.window_slice_offsets[1:]
    ):
        base = int(first) * 8
        end = min(int(last) * 8, packed.M)
        assert np.array_equal(
            np.sort(packed.row_indices[base:end]), np.arange(end - base)
        )
    assert (
        packed.row_indices[packed.M :].min() == np.iinfo(packed.row_indices.dtype).max
    )
    _check_reference(packed, pointers, indices, values, K)


def test_stable_tie_break_and_global_sort_reference():
    counts = [3, 7, 7, 1, 7, 0, 3, 1, 10, 2, 10, 0, 2, 2, 2, 10]
    pointers, indices, values, K = _csr(counts)
    packed = csr_to_slice_ell(
        pointers,
        indices,
        values,
        K=K,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=32, shim_columns=2, window_count=1
        ),
    )
    expected_order = np.argsort(-np.asarray(counts), kind="stable")
    assert packed.row_indices[: len(counts)].tolist() == expected_order.tolist()
    _check_reference(packed, pointers, indices, values, K)


def test_sorted_zero_block_slice_keeps_zero_control_entry():
    pointers, indices, values, K = _csr([0] * 8 + [3, 0, 2, 1, 0, 4, 0, 1])
    packed = csr_to_slice_ell(
        pointers,
        indices,
        values,
        K=K,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=32, shim_columns=2, window_count=2
        ),
    )
    assert packed.blocks_per_slice.tolist() == [0, 1]
    _check_reference(packed, pointers, indices, values, K)


def test_custom_boundaries_and_multicolumn_ownership():
    counts = np.arange(1, 65, dtype=np.int64)[::-1]
    pointers, indices, values, K = _csr(counts)
    config = SliceELLConfig(
        core_rows=4,
        block_height=8,
        block_width=32,
        shim_columns=2,
        window_count=4,
        window_slice_boundaries=(0, 1, 3, 6, 8),
    )
    packed = csr_to_slice_ell(pointers, indices, values, K=K, config=config)
    assert packed.window_slice_offsets.tolist() == [0, 1, 3, 6, 8]
    assert packed.column_slice_offsets.tolist() == [0, 3, 8]
    with pytest.raises(ValueError, match="unequal slice counts"):
        _ = packed.slices_per_column
    assert packed.column_blocks_per_slice(0).size == 3
    assert packed.column_blocks_per_slice(1).size == 5
    _check_reference(packed, pointers, indices, values, K)


def test_global_reference_uses_uint32_for_large_window():
    pointers = np.zeros(65537, dtype=np.int64)
    packed = csr_to_slice_ell(
        pointers,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        K=1,
        config=SliceELLConfig(
            block_height=8, block_width=32, shim_columns=1, window_count=1
        ),
    )
    assert packed.row_indices.dtype == np.uint32
    assert packed.row_indices.size == 65536
    assert packed.packed_a.size == 0


def test_dense_bridge_sorted_and_cache_map(tmp_path):
    weight = torch.zeros((19, 41), dtype=torch.bfloat16)
    weight[0, [1, 3, 5, 7]] = 2
    weight[5, 2] = -1
    weight[18, [2, 8, 10]] = 3
    packed = dense_to_slice_ell(
        weight,
        config=SliceELLConfig(
            core_rows=4, block_height=8, block_width=32, shim_columns=2, window_count=2
        ),
    )
    x = torch.rand(41, generator=torch.Generator().manual_seed(8)).to(torch.bfloat16)
    actual = cpu_unpermute_windows(packed, cpu_spmv_slice_ell(packed, x))
    expected = (weight.float() @ x.float()).to(torch.bfloat16)
    assert torch.equal(actual, expected)
    paths = packed.save_cache(tmp_path, "sorted")
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["format"] == "windowed-sell-c-sigma"
    assert manifest["window_count"] == 2
    assert np.array_equal(
        np.fromfile(paths["row_indices"], dtype="<u2"), packed.row_indices
    )


def test_designs_share_payload_and_balanced_assignment_is_not_silently_accepted():
    matrix = generate_synthetic_csr(
        MatrixInput(
            source="synthetic",
            M=64,
            K=128,
            density=0.1,
            row_pattern="skewed",
            seed=4,
        )
    )
    fmt = FormatSpec(
        name="sell_c_sigma", block_height=8, block_width=32, columns=2, window_count=2
    )
    a = pack_for_design(matrix, fmt, DesignSpec("sell_dedicated_reorder"))
    b = pack_for_design(matrix, fmt, DesignSpec("sell_time_multiplex_reorder"))
    assert np.array_equal(a.packed_a, b.packed_a)
    assert np.array_equal(a.row_indices, b.row_indices)
    assert a.manifest()["packed_a_sha256"] == b.manifest()["packed_a_sha256"]
    with pytest.raises(NotImplementedError, match="balanced"):
        pack_for_design(
            matrix,
            FormatSpec(
                name="sell_c_sigma",
                block_height=8,
                block_width=32,
                columns=2,
                window_count=4,
                assignment_policy="balanced",
            ),
            DesignSpec("sell_dedicated_reorder"),
        )


@pytest.mark.parametrize("windows", [8, 16])
@pytest.mark.parametrize("boundary", ["equal_rows", "equal_nnz"])
def test_packed_size_matches_step0_storage_model(windows, boundary):
    matrix = generate_synthetic_csr(
        MatrixInput(
            source="synthetic",
            M=136,
            K=128,
            density=0.1,
            row_pattern="skewed",
            seed=44,
        )
    )
    fmt = FormatSpec(
        name="sell_c_sigma",
        block_height=8,
        block_width=32,
        columns=8,
        window_count=windows,
        boundary_policy=boundary,
    )
    packed = pack_for_design(matrix, fmt, DesignSpec("sell_dedicated_reorder"))
    predicted = estimate_storage(matrix.profile, fmt)
    assert packed.packed_a.nbytes == predicted["packed_a_bytes"]
    assert packed.row_indices.nbytes == predicted["row_indices_bytes"]
    assert int(packed.blocks_per_slice.sum()) == predicted["total_blocks"]
    actual_column_blocks = [
        int(packed.column_blocks_per_slice(c).sum()) for c in range(8)
    ]
    assert actual_column_blocks == predicted["column_blocks"]
    _check_reference(
        packed, matrix.indptr, matrix.indices, matrix.values, matrix.profile.K
    )
