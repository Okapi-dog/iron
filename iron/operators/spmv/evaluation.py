# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NPU-independent matrix, format, and design selection for SpMV experiments.

The storage model and Step-1 SELL packer are CPU-only; neither implies that a
SELL-C-sigma NPU kernel or reorder data path has been implemented.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Literal

import numpy as np

FormatName = Literal["dense", "ell", "slice_ell", "sell_c_sigma", "global_sort_bound"]
BoundaryPolicy = Literal["equal_rows", "equal_nnz"]
AssignmentPolicy = Literal["contiguous", "balanced"]


@dataclass(frozen=True)
class MatrixInput:
    """Reproducible source of one canonical-row-order matrix and input vector."""

    source: Literal["synthetic", "safetensors"]
    M: int | None = None
    K: int | None = None
    density: float | None = None
    row_pattern: Literal["uniform", "skewed"] = "uniform"
    seed: int = 0
    x_seed: int = 3000
    model_dir: str | None = None
    tensor_name: str | None = None

    def __post_init__(self) -> None:
        if self.source == "synthetic":
            if self.M is None or self.K is None or self.M <= 0 or self.K <= 0:
                raise ValueError("synthetic MatrixInput requires positive M and K")
            if self.density is None or not 0 <= self.density <= 1:
                raise ValueError("synthetic MatrixInput requires density in [0, 1]")
        elif self.source == "safetensors":
            if not self.model_dir or not self.tensor_name:
                raise ValueError(
                    "safetensors MatrixInput requires model_dir and tensor_name"
                )
        else:
            raise ValueError(f"unsupported matrix source: {self.source}")

    @property
    def matrix_id(self) -> str:
        """Identify the generation recipe, not the weight-file contents."""
        recipe = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(recipe.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FormatSpec:
    """One storage format and its pack-time geometry."""

    name: FormatName
    block_height: int = 8
    block_width: int = 256
    columns: int = 8
    window_count: int = 8
    boundary_policy: BoundaryPolicy = "equal_rows"
    assignment_policy: AssignmentPolicy = "contiguous"
    ell_alignment: int = 32

    def __post_init__(self) -> None:
        if self.block_height <= 0 or self.block_width <= 0 or self.columns <= 0:
            raise ValueError("block_height, block_width, and columns must be positive")
        if self.ell_alignment <= 0:
            raise ValueError("ell_alignment must be positive")
        if self.name not in (
            "dense",
            "ell",
            "slice_ell",
            "sell_c_sigma",
            "global_sort_bound",
        ):
            raise ValueError(f"unsupported format: {self.name}")
        if self.name != "sell_c_sigma":
            object.__setattr__(self, "window_count", 0)
        elif self.window_count <= 0:
            raise ValueError("SELL window_count must be positive")
        if self.name == "sell_c_sigma" and self.window_count < self.columns:
            raise ValueError("SELL needs at least one window per active column")

    @property
    def format_id(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]


@dataclass(frozen=True)
class DesignSpec:
    """Select a planned/available NPU design without silently substituting one."""

    name: str

    @property
    def format_name(self) -> FormatName | None:
        return {
            "storage_only": None,
            "dense_k_tiled": "dense",
            "ell": "ell",
            "slice_ell": "slice_ell",
            "sell_dedicated_reorder": "sell_c_sigma",
            "sell_time_multiplex_reorder": "sell_c_sigma",
        }[self.name]

    @property
    def npu_status(self) -> str:
        return {
            "storage_only": "no_npu",
            "dense_k_tiled": "existing",
            "ell": "existing",
            "slice_ell": "existing",
            "sell_dedicated_reorder": "planned",
            "sell_time_multiplex_reorder": "unproven",
        }[self.name]

    def validate_format(self, fmt: FormatSpec) -> None:
        if self.format_name is not None and self.format_name != fmt.name:
            raise ValueError(f"{self.name} requires {self.format_name}, not {fmt.name}")


@dataclass(frozen=True)
class MatrixProfile:
    """Rows and content fingerprint needed by the Step-0 storage model."""

    spec: MatrixInput
    M: int
    K: int
    row_nnz: np.ndarray
    source_sha256: str

    def __post_init__(self) -> None:
        counts = np.asarray(self.row_nnz, dtype=np.int64)
        if self.M <= 0 or self.K <= 0 or counts.shape != (self.M,):
            raise ValueError("profile shape does not match M and K")
        if np.any(counts < 0) or np.any(counts > self.K):
            raise ValueError("row_nnz must be within [0, K]")
        object.__setattr__(self, "row_nnz", counts)

    @property
    def nnz(self) -> int:
        return int(self.row_nnz.sum(dtype=np.int64))

    @property
    def density(self) -> float:
        return self.nnz / (self.M * self.K)

    @property
    def row_nnz_sha256(self) -> str:
        return hashlib.sha256(
            self.row_nnz.astype("<i8", copy=False).tobytes()
        ).hexdigest()


@dataclass(frozen=True)
class CSRMatrix:
    """One materialized CSR and BF16 x, shared across all format adapters."""

    profile: MatrixProfile
    indptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    vector: object

    def __post_init__(self) -> None:
        pointers = np.asarray(self.indptr, dtype=np.int64)
        indices = np.asarray(self.indices, dtype=np.int64)
        values = np.asarray(self.values, dtype=np.float32)
        if pointers.shape != (self.profile.M + 1,) or pointers[0] != 0:
            raise ValueError("CSR pointer shape/start does not match the profile")
        if (
            np.any(pointers[1:] < pointers[:-1])
            or pointers[-1] != indices.size
            or indices.size != values.size
        ):
            raise ValueError("CSR arrays have inconsistent nnz")
        if np.any(np.diff(pointers) != self.profile.row_nnz):
            raise ValueError("CSR row lengths do not match the matrix profile")
        if np.any(indices < 0) or np.any(indices >= self.profile.K):
            raise ValueError("CSR column index lies outside K")
        object.__setattr__(self, "indptr", pointers)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "values", values)


def generate_synthetic_csr(spec: MatrixInput) -> CSRMatrix:
    """Generate one reusable CSR/x pair; storage sweeps need not call this."""

    import torch

    profile = synthetic_profile(spec)
    assert spec.K is not None
    pointers = np.zeros(profile.M + 1, dtype=np.int64)
    pointers[1:] = profile.row_nnz.cumsum(dtype=np.int64)
    indices = np.empty(profile.nnz, dtype=np.int64)
    values = np.empty(profile.nnz, dtype=np.float32)
    rng = np.random.default_rng(spec.seed + 1)
    for row, count in enumerate(profile.row_nnz):
        begin, end = int(pointers[row]), int(pointers[row + 1])
        if count:
            indices[begin:end] = np.sort(
                rng.choice(profile.K, size=int(count), replace=False)
            )
            values[begin:end] = rng.uniform(-1.0, 1.0, size=int(count)).astype(
                np.float32
            )
    vector = torch.rand(
        profile.K, generator=torch.Generator().manual_seed(spec.x_seed)
    ).to(torch.bfloat16)
    return CSRMatrix(profile, pointers, indices, values, vector)


def safetensors_profile(spec: MatrixInput) -> MatrixProfile:
    """Read row lengths and tensor-content hash without materializing CSR."""

    import torch
    from safetensors.torch import safe_open

    if spec.source != "safetensors":
        raise ValueError("safetensors_profile requires a safetensors MatrixInput")
    assert spec.model_dir is not None and spec.tensor_name is not None
    tensor_path = None
    for path_string in sorted(glob(str(Path(spec.model_dir) / "*.safetensors"))):
        path = Path(path_string)
        with safe_open(path, framework="pt", device="cpu") as handle:
            if spec.tensor_name in handle.keys():
                if tensor_path is not None:
                    raise ValueError(f"duplicate tensor name: {spec.tensor_name}")
                tensor_path = path
    if tensor_path is None:
        raise FileNotFoundError(
            f"tensor {spec.tensor_name} not found in {spec.model_dir}"
        )
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(spec.tensor_name).contiguous()
    if weight.ndim != 2:
        raise ValueError(f"tensor {spec.tensor_name} is not a matrix")
    M, K = (int(x) for x in weight.shape)
    counts = weight.count_nonzero(dim=1).cpu().numpy().astype(np.int64, copy=False)
    digest = hashlib.sha256()
    digest.update(f"{weight.dtype}:{M}:{K}".encode())
    digest.update(weight.view(-1).view(torch.uint8).numpy().tobytes())
    return MatrixProfile(spec, M, K, counts, digest.hexdigest())


def load_or_generate_csr(spec: MatrixInput) -> CSRMatrix:
    """Create canonical CSR/x once, independent of the selected packed format."""

    if spec.source == "synthetic":
        return generate_synthetic_csr(spec)
    import torch
    from safetensors.torch import safe_open

    profile = safetensors_profile(spec)
    assert spec.model_dir is not None and spec.tensor_name is not None
    for path_string in sorted(glob(str(Path(spec.model_dir) / "*.safetensors"))):
        with safe_open(path_string, framework="pt", device="cpu") as handle:
            if spec.tensor_name in handle.keys():
                weight = handle.get_tensor(spec.tensor_name).to(torch.float32)
                break
    else:
        raise FileNotFoundError(spec.tensor_name)
    csr = weight.to_sparse_csr()
    vector = torch.rand(
        profile.K, generator=torch.Generator().manual_seed(spec.x_seed)
    ).to(torch.bfloat16)
    return CSRMatrix(
        profile,
        csr.crow_indices().numpy(),
        csr.col_indices().numpy(),
        csr.values().numpy(),
        vector,
    )


def pack_existing_format(matrix: CSRMatrix, fmt: FormatSpec):
    """Pack the same CSR/x into a selected format, without NPU execution."""

    import torch

    from iron.operators.spmv.slice_ell import (
        SliceELLConfig,
        _bf16_bits,
        _words_to_bf16,
        csr_to_slice_ell,
    )

    M, K = matrix.profile.M, matrix.profile.K
    if fmt.name == "dense":
        dense = torch.zeros((M, K), dtype=torch.bfloat16)
        values = torch.from_numpy(matrix.values).to(torch.bfloat16)
        for row in range(M):
            begin, end = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
            if end > begin:
                dense[row, torch.from_numpy(matrix.indices[begin:end])] = values[
                    begin:end
                ]
        return dense
    if fmt.name == "ell":
        if K > np.iinfo(np.uint16).max:
            raise ValueError("ELL uint16 index contract requires K <= 65535")
        width = (
            (int(matrix.profile.row_nnz.max()) + fmt.ell_alignment - 1)
            // fmt.ell_alignment
            * fmt.ell_alignment
        )
        if width == 0:
            raise ValueError("current NPU ELL kernel requires a nonzero width")
        words = np.zeros((M, 2, width), dtype=np.uint16)
        bits = _bf16_bits(matrix.values)
        for row in range(M):
            begin, end = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
            count = end - begin
            words[row, 0, :count] = matrix.indices[begin:end]
            words[row, 1, :count] = bits[begin:end]
        return _words_to_bf16(words.reshape(-1))
    if fmt.name in ("slice_ell", "sell_c_sigma"):
        if fmt.name == "sell_c_sigma" and fmt.assignment_policy != "contiguous":
            raise NotImplementedError(
                "balanced window assignment needs a future NPU/payload contract"
            )
        window_bounds = None
        if fmt.name == "sell_c_sigma":
            window_bounds = tuple(
                int(x) for x in window_slice_bounds(matrix.profile, fmt)
            )
        return csr_to_slice_ell(
            matrix.indptr,
            matrix.indices,
            matrix.values,
            K=K,
            config=SliceELLConfig(
                core_rows=4,
                block_height=fmt.block_height,
                block_width=fmt.block_width,
                shim_columns=fmt.columns,
                window_count=fmt.window_count if fmt.name == "sell_c_sigma" else 0,
                window_slice_boundaries=window_bounds,
            ),
        )
    raise NotImplementedError(f"{fmt.name} is an offline storage bound, not a packer")


def pack_for_design(matrix: CSRMatrix, fmt: FormatSpec, design: DesignSpec):
    """Check format/design compatibility before choosing the shared packer.

    The two planned SELL execution designs consume identical packed A only
    when they use the same FormatSpec and input CSR; runtime wiring is separate.
    """

    design.validate_format(fmt)
    return pack_existing_format(matrix, fmt)


def synthetic_row_counts(spec: MatrixInput) -> np.ndarray:
    """Make exact-total row NNZ with uniform or heavy-tailed row lengths."""

    if spec.source != "synthetic":
        raise ValueError("synthetic_row_counts requires a synthetic MatrixInput")
    assert spec.M is not None and spec.K is not None and spec.density is not None
    target = round(spec.M * spec.K * spec.density)
    if target == 0 or target == spec.M * spec.K:
        return np.full(spec.M, target // spec.M, dtype=np.int64)
    rng = np.random.default_rng(spec.seed)
    weights = np.ones(spec.M, dtype=np.float64)
    if spec.row_pattern == "skewed":
        weights = rng.lognormal(mean=0.0, sigma=1.25, size=spec.M)
    elif spec.row_pattern != "uniform":
        raise ValueError(f"unsupported row pattern: {spec.row_pattern}")

    low, high = 0.0, float(spec.K / weights.min())
    for _ in range(70):
        middle = (low + high) / 2
        if np.minimum(weights * middle, spec.K).sum() < target:
            low = middle
        else:
            high = middle
    quotas = np.minimum(weights * high, spec.K)
    counts = np.floor(quotas).astype(np.int64)
    remainder = target - int(counts.sum())
    if remainder:
        fractions = quotas - counts
        order = np.argsort(-fractions, kind="stable")
        eligible = order[counts[order] < spec.K]
        if remainder < 0 or remainder > eligible.size:
            raise RuntimeError("cannot round synthetic row counts to exact NNZ")
        counts[eligible[:remainder]] += 1
    if int(counts.sum()) != target:
        raise RuntimeError("synthetic NNZ did not match requested density")
    return counts


def synthetic_profile(spec: MatrixInput) -> MatrixProfile:
    """Produce a profile without materializing a potentially huge CSR payload."""

    counts = synthetic_row_counts(spec)
    source_hash = hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True).encode()
        + counts.astype("<i8").tobytes()
    ).hexdigest()
    assert spec.M is not None and spec.K is not None
    return MatrixProfile(spec, spec.M, spec.K, counts, source_hash)


def window_slice_bounds(profile: MatrixProfile, fmt: FormatSpec) -> np.ndarray:
    """Partition original-order slices into nonempty contiguous windows."""

    total_slices = (profile.M + fmt.block_height - 1) // fmt.block_height
    windows = fmt.window_count
    if windows > total_slices:
        raise ValueError("window_count exceeds the number of real slices")
    if fmt.boundary_policy == "equal_rows":
        sizes = np.full(windows, total_slices // windows, dtype=np.int64)
        sizes[: total_slices % windows] += 1
        return np.concatenate(([0], sizes.cumsum()))
    if fmt.boundary_policy != "equal_nnz":
        raise ValueError(f"unsupported boundary policy: {fmt.boundary_policy}")

    per_slice = np.add.reduceat(
        profile.row_nnz, np.arange(0, profile.M, fmt.block_height, dtype=np.int64)
    )
    cumulative = per_slice.cumsum(dtype=np.int64)
    if cumulative[-1] == 0:
        return window_slice_bounds(
            profile,
            FormatSpec(
                name=fmt.name,
                block_height=fmt.block_height,
                block_width=fmt.block_width,
                columns=fmt.columns,
                window_count=fmt.window_count,
                boundary_policy="equal_rows",
                assignment_policy=fmt.assignment_policy,
                ell_alignment=fmt.ell_alignment,
            ),
        )
    bounds = [0]
    for boundary in range(1, windows):
        target = cumulative[-1] * boundary / windows
        proposed = int(np.searchsorted(cumulative, target, side="left")) + 1
        bounds.append(
            max(bounds[-1] + 1, min(proposed, total_slices - (windows - boundary)))
        )
    return np.asarray([*bounds, total_slices], dtype=np.int64)


def assign_windows(
    window_blocks: list[int], columns: int, policy: AssignmentPolicy
) -> list[int]:
    """Map windows to columns without changing their canonical output ranges."""

    windows = len(window_blocks)
    if windows < columns:
        raise ValueError("every column must own at least one window")
    if policy == "contiguous":
        return [
            min(window * columns // windows, columns - 1) for window in range(windows)
        ]
    if policy != "balanced":
        raise ValueError(f"unsupported assignment policy: {policy}")
    capacity = (windows + columns - 1) // columns
    loads = [0] * columns
    owned = [0] * columns
    assignment = [-1] * windows
    for window in sorted(range(windows), key=lambda w: (-window_blocks[w], w)):
        column = min(
            (c for c in range(columns) if owned[c] < capacity),
            key=lambda c: (loads[c], owned[c], c),
        )
        assignment[window] = column
        loads[column] += window_blocks[window]
        owned[column] += 1
    return assignment


def _slice_blocks(
    counts: np.ndarray, block_height: int, block_width: int
) -> np.ndarray:
    padded = ((counts.size + block_height - 1) // block_height) * block_height
    rows = np.zeros(padded, dtype=np.int64)
    rows[: counts.size] = counts
    maxima = rows.reshape(-1, block_height).max(axis=1)
    return (maxima + block_width - 1) // block_width


def estimate_storage(
    profile: MatrixProfile,
    fmt: FormatSpec,
    design: DesignSpec = DesignSpec("storage_only"),
) -> dict:
    """Estimate fixed-size matrix payload and per-column work from row lengths."""

    design.validate_format(fmt)
    counts = profile.row_nnz
    dense_bytes = 2 * profile.M * profile.K
    common = {
        "matrix_id": profile.spec.matrix_id,
        "source_sha256": profile.source_sha256,
        "row_nnz_sha256": profile.row_nnz_sha256,
        "format_id": fmt.format_id,
        "format": fmt.name,
        "design": design.name,
        "design_npu_status": design.npu_status,
        "M": profile.M,
        "K": profile.K,
        "nnz": profile.nnz,
        "density": profile.density,
        "dense_bf16_bytes": dense_bytes,
    }
    if fmt.name == "dense":
        detail = {"packed_a_bytes": dense_bytes, "padded_slots": profile.M * profile.K}
    elif fmt.name == "ell":
        width = (
            (int(counts.max()) + fmt.ell_alignment - 1)
            // fmt.ell_alignment
            * fmt.ell_alignment
        )
        slots = profile.M * width
        detail = {
            "ell_width": width,
            "packed_a_bytes": 4 * slots,
            "padded_slots": slots,
        }
    elif fmt.name == "slice_ell":
        blocks = _slice_blocks(counts, fmt.block_height, fmt.block_width)
        slices_per_column = (blocks.size + fmt.columns - 1) // fmt.columns
        padded_blocks = np.zeros(slices_per_column * fmt.columns, dtype=np.int64)
        padded_blocks[: blocks.size] = blocks
        column_blocks = padded_blocks.reshape(fmt.columns, slices_per_column).sum(
            axis=1
        )
        total = int(column_blocks.sum())
        slots = total * fmt.block_height * fmt.block_width
        detail = {
            "packed_a_bytes": 4 * slots,
            "padded_slots": slots,
            "total_blocks": total,
            "max_blocks_per_slice": int(padded_blocks.max(initial=0)),
            "column_blocks": column_blocks.tolist(),
            "max_column_blocks": int(column_blocks.max(initial=0)),
            "column_imbalance": (
                float(column_blocks.max() / column_blocks.mean()) if total else 0.0
            ),
            "padded_rows": int(padded_blocks.size * fmt.block_height),
        }
    elif fmt.name == "global_sort_bound":
        # Offline lower bound only: no column assignment or feasible NPU design is asserted.
        blocks = _slice_blocks(np.sort(counts)[::-1], fmt.block_height, fmt.block_width)
        slots = int(blocks.sum()) * fmt.block_height * fmt.block_width
        padded_rows = int(blocks.size * fmt.block_height)
        index_bytes = 2 if profile.M <= 65535 else 4
        detail = {
            "reference_only": True,
            "sigma_max_rows": profile.M,
            "max_blocks_per_slice": int(blocks.max(initial=0)),
            "total_blocks": int(blocks.sum()),
            "padded_rows": padded_rows,
            "padded_slots": slots,
            "packed_a_bytes": 4 * slots,
            "row_indices_bytes": padded_rows * index_bytes,
            "row_index_dtype": "uint16" if index_bytes == 2 else "uint32",
            "reorder_l1_min_bytes": profile.M * (2 + index_bytes),
        }
    else:
        logical_bounds = window_slice_bounds(profile, fmt)
        bounds = logical_bounds.copy()
        total_slices = (profile.M + fmt.block_height - 1) // fmt.block_height
        padded_slices = ((total_slices + fmt.columns - 1) // fmt.columns) * fmt.columns
        bounds[-1] = padded_slices
        window_blocks: list[int] = []
        window_rows: list[int] = []
        max_block = 0
        for first, last in zip(logical_bounds[:-1], logical_bounds[1:]):
            begin = int(first) * fmt.block_height
            end = min(int(last) * fmt.block_height, profile.M)
            sorted_counts = counts[begin:end][
                np.argsort(-counts[begin:end], kind="stable")
            ]
            blocks = _slice_blocks(sorted_counts, fmt.block_height, fmt.block_width)
            window_blocks.append(int(blocks.sum()))
            window_rows.append(end - begin)
            max_block = max(max_block, int(blocks.max(initial=0)))
        assignment = assign_windows(window_blocks, fmt.columns, fmt.assignment_policy)
        column_blocks = [
            sum(work for work, owner in zip(window_blocks, assignment) if owner == c)
            for c in range(fmt.columns)
        ]
        total = sum(window_blocks)
        slots = total * fmt.block_height * fmt.block_width
        sigma = int(max(np.diff(bounds)) * fmt.block_height)
        index_bytes = 2 if sigma <= 65535 else 4
        padded_rows = int(bounds[-1] * fmt.block_height)
        detail = {
            "window_count": fmt.window_count,
            "boundary_policy": fmt.boundary_policy,
            "assignment_policy": fmt.assignment_policy,
            "window_slice_bounds": bounds.tolist(),
            "logical_window_slice_bounds": logical_bounds.tolist(),
            "window_rows": window_rows,
            "sigma_max_rows": sigma,
            "window_blocks": window_blocks,
            "window_to_column": assignment,
            "column_blocks": column_blocks,
            "max_column_blocks": max(column_blocks),
            "column_imbalance": (
                max(column_blocks) / (total / fmt.columns) if total else 0.0
            ),
            "max_blocks_per_slice": max_block,
            "total_blocks": total,
            "padded_rows": padded_rows,
            "padded_slots": slots,
            "packed_a_bytes": 4 * slots,
            "row_indices_bytes": padded_rows * index_bytes,
            "row_index_dtype": "uint16" if index_bytes == 2 else "uint32",
            "reorder_l1_min_bytes": sigma * (2 + index_bytes),
        }
    packed = int(detail["packed_a_bytes"])
    total_storage = packed + int(detail.get("row_indices_bytes", 0))
    detail["total_storage_bytes"] = total_storage
    detail["a_over_dense"] = packed / dense_bytes
    detail["dense_over_a"] = dense_bytes / packed if packed else None
    detail["storage_over_dense"] = total_storage / dense_bytes
    detail["dense_over_storage"] = (
        dense_bytes / total_storage if total_storage else None
    )
    if "reorder_l1_min_bytes" in detail:
        detail["reorder_l1_lower_bound_fits_64k"] = (
            detail["reorder_l1_min_bytes"] <= 64 * 1024
        )
    return common | detail
