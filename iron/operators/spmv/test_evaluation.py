# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only Step-0 tests for the shared SpMV evaluation inputs/model."""

import numpy as np
import pytest
import torch

from iron.operators.spmv.evaluation import (
    DesignSpec,
    FormatSpec,
    MatrixProfile,
    MatrixInput,
    estimate_storage,
    generate_synthetic_csr,
    load_or_generate_csr,
    pack_existing_format,
    synthetic_profile,
    window_slice_bounds,
)
from iron.operators.spmv.reference import reference_ell
from iron.operators.spmv.slice_ell import (
    cpu_spmv_csr,
    cpu_spmv_slice_ell,
    cpu_unpermute_windows,
)


@pytest.mark.parametrize("pattern", ["uniform", "skewed"])
def test_synthetic_source_is_reproducible_and_has_exact_density(pattern):
    spec = MatrixInput(
        source="synthetic", M=128, K=512, density=0.125, row_pattern=pattern, seed=13
    )
    first = synthetic_profile(spec)
    second = synthetic_profile(spec)
    assert np.array_equal(first.row_nnz, second.row_nnz)
    assert first.nnz == round(128 * 512 * 0.125)
    assert first.source_sha256 == second.source_sha256
    assert first.row_nnz.min() >= 0 and first.row_nnz.max() <= 512
    if pattern == "skewed":
        assert first.row_nnz.max() > 2 * first.row_nnz.mean()


def test_window_models_preserve_rows_and_report_column_work():
    spec = MatrixInput(source="synthetic", M=136, K=512, density=0.1, seed=1)
    counts = np.array([250] + [12] * 134 + [0], dtype=np.int64)
    profile = MatrixProfile(spec, 136, 512, counts, "manual")
    baseline = estimate_storage(profile, FormatSpec(name="slice_ell", columns=8))
    sell8 = estimate_storage(
        profile, FormatSpec(name="sell_c_sigma", columns=8, window_count=8)
    )
    sell16 = estimate_storage(
        profile,
        FormatSpec(
            name="sell_c_sigma",
            columns=8,
            window_count=16,
            assignment_policy="balanced",
            boundary_policy="equal_nnz",
        ),
    )
    assert sell8["packed_a_bytes"] <= baseline["packed_a_bytes"]
    assert sell8["window_slice_bounds"][0] == 0
    assert sell8["logical_window_slice_bounds"][-1] == 17
    assert sell8["window_slice_bounds"][-1] == 24
    assert sum(sell8["window_rows"]) == 136
    assert sum(sell8["column_blocks"]) == sell8["total_blocks"]
    assert sell16["row_indices_bytes"] == sell16["padded_rows"] * 2
    assert sell16["max_column_blocks"] <= sell16["total_blocks"]
    assert (
        sell8["total_storage_bytes"]
        == sell8["packed_a_bytes"] + sell8["row_indices_bytes"]
    )
    global_bound = estimate_storage(profile, FormatSpec(name="global_sort_bound"))
    assert global_bound["reference_only"]
    assert global_bound["total_blocks"] <= sell8["total_blocks"]
    assert "max_column_blocks" not in global_bound


def test_equal_nnz_boundaries_are_nonempty_even_with_empty_rows():
    spec = MatrixInput(source="synthetic", M=128, K=512, density=0.1)
    counts = np.zeros(128, dtype=np.int64)
    counts[:8] = 400
    profile = MatrixProfile(spec, 128, 512, counts, "manual")
    fmt = FormatSpec(
        name="sell_c_sigma", columns=8, window_count=16, boundary_policy="equal_nnz"
    )
    bounds = window_slice_bounds(profile, fmt)
    assert bounds[0] == 0 and bounds[-1] == 16
    assert np.all(np.diff(bounds) == 1)


@pytest.mark.parametrize("height", [6, 9, 18])
def test_equal_rows_packer_uses_fixed_length_physical_windows(height):
    """A padded final window must not desynchronize the NPU's control FIFO."""
    matrix = generate_synthetic_csr(MatrixInput(
        source="synthetic", M=130, K=512, density=0.1,
        row_pattern="skewed", seed=41,
    ))
    fmt = FormatSpec("sell_c_sigma", block_height=height, window_count=8)
    packed = pack_existing_format(matrix, fmt)
    storage = estimate_storage(matrix.profile, fmt)
    assert len(set(np.diff(packed.window_slice_offsets))) == 1
    assert packed.padded_rows % (height * 8) == 0
    assert packed.packed_a.nbytes == storage["packed_a_bytes"]
    physical = cpu_spmv_slice_ell(packed, matrix.vector)
    canonical = cpu_unpermute_windows(packed, physical)
    reference = cpu_spmv_csr(matrix.indptr, matrix.indices, matrix.values, matrix.vector)
    assert torch.allclose(canonical.float(), reference.float(), rtol=0.08, atol=0.025)


def test_one_csr_and_vector_feed_existing_format_adapters():
    matrix_input = MatrixInput(
        source="synthetic",
        M=64,
        K=128,
        density=0.1,
        row_pattern="skewed",
        seed=18,
        x_seed=43,
    )
    matrix = load_or_generate_csr(matrix_input)
    direct = generate_synthetic_csr(matrix_input)
    assert np.array_equal(matrix.indptr, direct.indptr)
    assert np.array_equal(matrix.indices, direct.indices)
    assert np.array_equal(matrix.values, direct.values)
    assert torch.equal(matrix.vector, direct.vector)
    expected = cpu_spmv_csr(matrix.indptr, matrix.indices, matrix.values, matrix.vector)
    dense = pack_existing_format(matrix, FormatSpec(name="dense"))
    ell = pack_existing_format(matrix, FormatSpec(name="ell"))
    slice_ell = pack_existing_format(
        matrix,
        FormatSpec(
            name="slice_ell",
            block_height=8,
            block_width=32,
            columns=8,
        ),
    )
    width = ell.numel() // (2 * matrix.profile.M)
    assert torch.allclose(
        (dense.float() @ matrix.vector.float()).to(torch.bfloat16).float(),
        expected.float(),
        atol=0.05,
        rtol=0.05,
    )
    assert torch.allclose(
        reference_ell(ell, matrix.vector, matrix.profile.M, width).float(),
        expected.float(),
        atol=0.05,
        rtol=0.05,
    )
    assert torch.allclose(
        cpu_spmv_slice_ell(slice_ell, matrix.vector).float(),
        expected.float(),
        atol=0.05,
        rtol=0.05,
    )
    assert (
        slice_ell.packed_a.nbytes
        == estimate_storage(
            matrix.profile,
            FormatSpec(name="slice_ell", block_height=8, block_width=32, columns=8),
        )["packed_a_bytes"]
    )


def test_design_selector_rejects_wrong_format_and_packs_sell():
    matrix = generate_synthetic_csr(
        MatrixInput(source="synthetic", M=64, K=64, density=0.1)
    )
    with pytest.raises(ValueError, match="requires sell_c_sigma"):
        estimate_storage(
            matrix.profile, FormatSpec(name="ell"), DesignSpec("sell_dedicated_reorder")
        )
    packed = pack_existing_format(
        matrix, FormatSpec(name="sell_c_sigma", window_count=8)
    )
    assert packed.window_count == 8
    actual = cpu_unpermute_windows(packed, cpu_spmv_slice_ell(packed, matrix.vector))
    expected = cpu_spmv_csr(matrix.indptr, matrix.indices, matrix.values, matrix.vector)
    assert torch.allclose(actual.float(), expected.float(), atol=0.05, rtol=0.05)


def test_safetensors_uses_canonical_csr_and_content_hash(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    weight = torch.tensor([[1.0, 0.0, 2.0], [0.0, -3.0, 0.0]], dtype=torch.bfloat16)
    path = tmp_path / "model.safetensors"
    safetensors.save_file({"weight": weight}, path)
    spec = MatrixInput(
        source="safetensors", model_dir=str(tmp_path), tensor_name="weight", x_seed=7
    )
    matrix = load_or_generate_csr(spec)
    assert matrix.profile.row_nnz.tolist() == [2, 1]
    assert matrix.indices.tolist() == [0, 2, 1]
    assert matrix.vector.dtype == torch.bfloat16
    assert torch.equal(pack_existing_format(matrix, FormatSpec(name="dense")), weight)
    safetensors.save_file({"weight": weight * 2}, path)
    changed = load_or_generate_csr(spec)
    assert matrix.profile.source_sha256 != changed.profile.source_sha256
