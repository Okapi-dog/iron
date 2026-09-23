# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline format contract for Slice-ELL and window-sorted SELL-C-sigma.

This module deliberately has no MLIR, XRT, or NPU dependency.  Phase 2 uses
it to make the weight format, CPU reference, and host config object testable
before dynamic ObjectFIFO lowering is introduced.

The linear A-word order is::

    slice -> horizontal block -> core-row -> local row
          -> [uint16 indices[B_w], BF16 values[B_w]]

``blocks_per_slice[s]`` gives the number of horizontal blocks in slice ``s``.
The first implementation defaults to R=4, B_h=32, B_w=256, but validation and
the CLI intentionally allow an explicit legal geometry for later experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


LANES = 32
FORMAT_VERSION = 1
CONFIG_ALIGNMENT_BYTES = 64


# Hold and validate the Slice-ELL geometry shared by packer and future NPU code.
@dataclass(frozen=True)
class SliceELLConfig:
    """Geometry shared by packer, reference, and the future NPU design."""

    core_rows: int = 4
    block_height: int = 32
    block_width: int = 256
    lanes: int = LANES
    shim_columns: int = 8
    window_count: int = 0
    window_slice_boundaries: tuple[int, ...] | None = None

    # Reject a geometry that cannot map to the first NPU2 Slice-ELL design.
    def __post_init__(self) -> None:
        if not 1 <= self.core_rows <= 4:
            raise ValueError("core_rows must be in [1, 4] for one NPU2 MemTile")
        if self.block_height <= 0 or self.block_height % self.core_rows:
            raise ValueError("block_height must be positive and divisible by core_rows")
        if self.lanes != LANES:
            raise ValueError(f"the first Slice-ELL format uses exactly {LANES} lanes")
        if self.block_width <= 0 or self.block_width % self.lanes:
            raise ValueError("block_width must be a positive multiple of the lane count")
        if self.shim_columns <= 0:
            raise ValueError("shim_columns must be positive")
        if self.window_count < 0:
            raise ValueError("window_count must be nonnegative (0 disables row sort)")
        if self.window_count not in (0, 1) and self.window_count % self.shim_columns:
            raise ValueError("window_count must be 1 or a multiple of shim_columns")
        if self.window_slice_boundaries is not None:
            if self.window_count == 0 or len(self.window_slice_boundaries) != self.window_count + 1:
                raise ValueError("window boundaries require one entry per window plus the endpoint")
            if self.window_slice_boundaries[0] != 0 or any(
                right <= left for left, right in zip(
                    self.window_slice_boundaries[:-1], self.window_slice_boundaries[1:]
                )
            ):
                raise ValueError("window slice boundaries must start at zero and increase")

    @property
    # Derive the number of rows owned by one core.
    def core_height(self) -> int:
        return self.block_height // self.core_rows

    @property
    # Return the uint16 payload length of one core's horizontal A block.
    def words_per_core_block(self) -> int:
        # Each local row is [indices B_w][values B_w].
        return self.core_height * 2 * self.block_width

    @property
    # Return the uint16 payload length of one complete slice horizontal A block.
    def words_per_slice_block(self) -> int:
        return self.block_height * 2 * self.block_width


# Convert numerical values to their BF16 bit pattern for mixed index/value storage.
def _bf16_bits(values: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    """Round numeric values to BF16 and return their raw uint16 words."""

    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu().contiguous().to(torch.bfloat16).reshape(-1)
    else:
        tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32).contiguous().to(torch.bfloat16).reshape(-1)
    return tensor.view(torch.uint16).numpy().copy()


# Reinterpret raw uint16 storage as a BF16 tensor without changing bits.
def _words_to_bf16(words: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(words, dtype=np.uint16)).view(torch.int16).view(torch.bfloat16)


# Hash a raw uint16 payload so a cache and manifest can be matched safely.
def _sha256_words(words: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(words, dtype=np.uint16).tobytes()).hexdigest()


# Keep one in-memory packed Slice-ELL matrix and its interpretation metadata.
@dataclass
class PackedSliceELL:
    """A complete packed matrix plus the metadata required to consume it."""

    M: int
    K: int
    nnz: int
    config: SliceELLConfig
    padded_rows: int
    packed_a: np.ndarray
    blocks_per_slice: np.ndarray
    slice_word_offsets: np.ndarray
    column_slice_offsets: np.ndarray
    window_count: int = 0
    window_slice_offsets: np.ndarray | None = None
    row_indices: np.ndarray | None = None

    @property
    # Return the common number of contiguous slices assigned to each Shim column.
    def slices_per_column(self) -> int:
        counts = np.diff(self.column_slice_offsets)
        if not np.all(counts == counts[0]):
            raise ValueError("columns have unequal slice counts; fixed-length NPU ABI is unavailable")
        return int(counts[0])

    @property
    # Return the number of slices including whole zero slices added for column balance.
    def total_slices(self) -> int:
        return int(self.blocks_per_slice.size)

    @property
    # View packed A as BF16 for APIs whose buffer object is BF16-typed.
    def packed_a_as_bf16(self) -> torch.Tensor:
        return _words_to_bf16(self.packed_a)

    # Return only the control entries owned by one Shim column.
    def column_blocks_per_slice(self, column: int) -> np.ndarray:
        if not 0 <= column < self.config.shim_columns:
            raise IndexError("column is outside this packed matrix")
        begin, end = self.column_slice_offsets[column : column + 2]
        return self.blocks_per_slice[begin:end]

    # Build the JSON sidecar that describes, but does not contain, packed A.
    def manifest(self) -> dict:
        """Return the static sidecar for optional raw-binary cache files."""

        column_counts = np.diff(self.column_slice_offsets)
        result = {
            "format": "windowed-sell-c-sigma" if self.window_count else "row-order-preserving-slice-ell",
            "format_version": 2 if self.window_count else FORMAT_VERSION,
            "M": self.M,
            "K": self.K,
            "nnz": self.nnz,
            "padded_rows": self.padded_rows,
            "config": asdict(self.config),
            "core_height": self.config.core_height,
            "total_slices": self.total_slices,
            "slices_per_column": int(column_counts[0]) if np.all(column_counts == column_counts[0]) else None,
            "column_slice_offsets": self.column_slice_offsets.tolist(),
            "slice_word_offsets": self.slice_word_offsets.tolist(),
            "blocks_per_slice_dtype": "uint16",
            "packed_a_dtype": "uint16",
            "index_dtype": "uint16",
            "padding_index": 0,
            "padding_value": "BF16 +0",
            "packed_a_words": int(self.packed_a.size),
            "packed_a_sha256": _sha256_words(self.packed_a),
            "blocks_per_slice_sha256": _sha256_words(self.blocks_per_slice),
        }
        if self.window_count:
            assert self.window_slice_offsets is not None and self.row_indices is not None
            result.update({
                "window_count": self.window_count,
                "window_slice_offsets": self.window_slice_offsets.tolist(),
                "row_indices_dtype": str(self.row_indices.dtype),
                "row_indices_sha256": hashlib.sha256(self.row_indices.tobytes()).hexdigest(),
                "row_index_sentinel": int(np.iinfo(self.row_indices.dtype).max),
            })
        return result

    # Optionally save raw-binary cache files; normal tests use this object in memory.
    def save_cache(self, directory: str | Path, stem: str = "slice_ell") -> dict[str, Path]:
        """Write optional ``.bin`` payloads and a manifest, then return their paths."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "packed_a": directory / f"{stem}_packed_a.bin",
            "blocks_per_slice": directory / f"{stem}_blocks_per_slice.bin",
            "manifest": directory / f"{stem}_manifest.json",
        }
        if self.window_count:
            paths["row_indices"] = directory / f"{stem}_row_indices.bin"
        np.ascontiguousarray(self.packed_a, dtype="<u2").tofile(paths["packed_a"])
        np.ascontiguousarray(self.blocks_per_slice, dtype="<u2").tofile(paths["blocks_per_slice"])
        if self.window_count:
            assert self.row_indices is not None
            self.row_indices.astype(self.row_indices.dtype.newbyteorder("<"), copy=False).tofile(paths["row_indices"])
        manifest = self.manifest() | {
            "storage_byte_order": "little",
            "payload_files": {name: path.name for name, path in paths.items() if name != "manifest"},
        }
        paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return paths


# Determine fixed slice boundaries before sorting; all offsets include padding slices.
def _window_boundaries(logical_slices: int, total_slices: int, config: SliceELLConfig) -> np.ndarray:
    if config.window_count == 0:
        return np.asarray([0, total_slices], dtype=np.int64)
    if config.window_slice_boundaries is not None:
        bounds = np.asarray(config.window_slice_boundaries, dtype=np.int64).copy()
        if bounds[-1] == logical_slices:
            bounds[-1] = total_slices
        elif bounds[-1] != total_slices:
            raise ValueError("window boundaries must end at logical or padded slice count")
    elif config.window_count == 1:
        bounds = np.asarray([0, total_slices], dtype=np.int64)
    else:
        if config.window_count > total_slices:
            raise ValueError("window_count exceeds padded slice count")
        sizes = np.full(config.window_count, total_slices // config.window_count, dtype=np.int64)
        sizes[: total_slices % config.window_count] += 1
        bounds = np.concatenate(([0], sizes.cumsum()))
    if np.any(np.diff(bounds) <= 0) or bounds[-1] != total_slices:
        raise ValueError("every window needs at least one slice")
    return bounds


# Convert a CSR matrix to row-preserving Slice-ELL or window-sorted SELL-C-sigma.
def csr_to_slice_ell(
    indptr: np.ndarray | Sequence[int],
    indices: np.ndarray | Sequence[int],
    values: np.ndarray | torch.Tensor | Sequence[float],
    *,
    K: int,
    config: SliceELLConfig = SliceELLConfig(),
) -> PackedSliceELL:
    """Pack CSR; optionally sort rows by descending NNZ inside each window.

    The final incomplete slice and enough whole zero slices to distribute a
    contiguous slice range to every Shim column are appended.  Padding slots
    always have both ``index=0`` and ``value=0`` so an eager gather is safe.
    """

    indptr = np.asarray(indptr, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    value_bits = _bf16_bits(values)
    if indptr.size < 2 or indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("indptr must be a nondecreasing CSR pointer beginning at zero")
    M = int(indptr.size - 1)
    nnz = int(indptr[-1])
    if nnz != indices.size or nnz != value_bits.size:
        raise ValueError("indptr[-1], indices, and values must have the same nnz")
    if K <= 0 or K > np.iinfo(np.uint16).max:
        raise ValueError("Slice-ELL uint16 indices require 0 < K <= 65535")
    if np.any(indices < 0) or np.any(indices >= K):
        raise ValueError("CSR column indices must lie in [0, K)")

    logical_slices = (M + config.block_height - 1) // config.block_height
    slices_per_column = (logical_slices + config.shim_columns - 1) // config.shim_columns
    if config.window_count:
        # Equal-size physical windows are required by the fixed-length NPU
        # control/output ABI. Odd B_h also needs an even number of slices per
        # window for four-byte-aligned BF16 and uint16 transfers.
        windows_per_column = max(1, config.window_count // config.shim_columns)
        alignment = windows_per_column * (2 if config.block_height % 2 else 1)
        slices_per_column = (slices_per_column + alignment - 1) // alignment * alignment
    total_slices = slices_per_column * config.shim_columns
    padded_rows = total_slices * config.block_height
    blocks_per_slice = np.zeros(total_slices, dtype=np.uint16)
    window_bounds = _window_boundaries(logical_slices, total_slices, config)
    physical_rows = np.full(padded_rows, -1, dtype=np.int64)
    row_indices = None
    row_lengths = np.diff(indptr)
    if config.window_count:
        max_window_rows = int(np.diff(window_bounds).max()) * config.block_height
        index_dtype = np.uint16 if max_window_rows <= np.iinfo(np.uint16).max else np.uint32
        row_indices = np.full(padded_rows, np.iinfo(index_dtype).max, dtype=index_dtype)
        for first, last in zip(window_bounds[:-1], window_bounds[1:]):
            base = int(first) * config.block_height
            end = min(int(last) * config.block_height, M)
            if end <= base:
                continue
            order = np.argsort(-row_lengths[base:end], kind="stable")
            physical_rows[base : base + order.size] = base + order
            row_indices[base : base + order.size] = order.astype(index_dtype)
    else:
        physical_rows[:M] = np.arange(M)

    for slice_id in range(total_slices):
        row_begin = slice_id * config.block_height
        source_rows = physical_rows[row_begin : row_begin + config.block_height]
        source_rows = source_rows[source_rows >= 0]
        max_nnz = int(row_lengths[source_rows].max(initial=0))
        p = (max_nnz + config.block_width - 1) // config.block_width
        if p > np.iinfo(np.uint16).max:
            raise ValueError("blocks_per_slice does not fit uint16")
        blocks_per_slice[slice_id] = p
    # Allocate exactly once.  A list of tiny row arrays would add a large Python
    # memory overhead for model-scale matrices and would transiently duplicate
    # the full packed payload during concatenate().
    slice_word_offsets = np.zeros(total_slices + 1, dtype=np.uint64)
    slice_word_offsets[1:] = np.cumsum(
        blocks_per_slice.astype(np.uint64) * config.words_per_slice_block, dtype=np.uint64
    )
    packed_a = np.zeros(int(slice_word_offsets[-1]), dtype=np.uint16)
    for slice_id, p in enumerate(blocks_per_slice.tolist()):
        row_begin = slice_id * config.block_height
        for block_id in range(p):
            block_base = int(slice_word_offsets[slice_id]) + block_id * config.words_per_slice_block
            slot_begin = block_id * config.block_width
            for core_row in range(config.core_rows):
                for local_row in range(config.core_height):
                    physical_row = row_begin + core_row * config.core_height + local_row
                    row_base = block_base + (core_row * config.core_height + local_row) * 2 * config.block_width
                    row = int(physical_rows[physical_row])
                    if row >= 0:
                        start, stop = int(indptr[row]), int(indptr[row + 1])
                        take_begin = min(start + slot_begin, stop)
                        take_end = min(take_begin + config.block_width, stop)
                        count = take_end - take_begin
                        if count:
                            packed_a[row_base : row_base + count] = indices[take_begin:take_end].astype(np.uint16, copy=False)
                            value_base = row_base + config.block_width
                            packed_a[value_base : value_base + count] = value_bits[take_begin:take_end]
    if config.window_count > 1:
        per_column = config.window_count // config.shim_columns
        column_slice_offsets = window_bounds[::per_column].astype(np.uint32)
    else:
        column_slice_offsets = np.arange(config.shim_columns + 1, dtype=np.uint32) * slices_per_column
    return PackedSliceELL(
        M=M,
        K=K,
        nnz=nnz,
        config=config,
        padded_rows=padded_rows,
        packed_a=packed_a,
        blocks_per_slice=blocks_per_slice,
        slice_word_offsets=slice_word_offsets,
        column_slice_offsets=column_slice_offsets,
        window_count=config.window_count,
        window_slice_offsets=window_bounds if config.window_count else None,
        row_indices=row_indices,
    )


# Convert one dense pruning weight tensor to in-memory Slice-ELL via CSR.
def dense_to_slice_ell(matrix: torch.Tensor, *, config: SliceELLConfig = SliceELLConfig()) -> PackedSliceELL:
    """Convert one 2-D pruned weight tensor to CSR and pack it.

    This is intentionally a convenience bridge for safetensors checkpoints,
    not a dense SpMV implementation.  ``nonzero`` preserves the original row
    order and returns columns in increasing order within each row.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    matrix = matrix.detach().cpu().contiguous()
    M, K = (int(dim) for dim in matrix.shape)
    rows, columns = torch.nonzero(matrix, as_tuple=True)
    counts = torch.bincount(rows, minlength=M).to(torch.int64)
    indptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).numpy()
    return csr_to_slice_ell(
        indptr,
        columns.numpy(),
        matrix[rows, columns],
        K=K,
        config=config,
    )


# Compute y = A*x on CPU directly from CSR; used as the packing correctness oracle.
def cpu_spmv_csr(
    indptr: np.ndarray | Sequence[int],
    indices: np.ndarray | Sequence[int],
    values: np.ndarray | torch.Tensor | Sequence[float],
    vector: torch.Tensor,
) -> torch.Tensor:
    """BF16-input, FP32-accumulate CSR reference for packer property tests."""

    indptr = np.asarray(indptr, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    value_bits = _bf16_bits(values)
    x = vector.detach().cpu().contiguous().to(torch.bfloat16).float()
    values_f32 = _words_to_bf16(value_bits).float()
    out = torch.zeros(indptr.size - 1, dtype=torch.float32)
    for row in range(out.numel()):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if stop > start:
            out[row] = (values_f32[start:stop] * x[torch.from_numpy(indices[start:stop])]).sum()
    return out.to(torch.bfloat16)


# Compute y = A*x on CPU by interpreting the packed Slice-ELL layout.
def cpu_spmv_slice_ell(packed: PackedSliceELL, vector: torch.Tensor) -> torch.Tensor:
    """Interpret packed A and return physical y' (sorted if windows are enabled)."""

    if vector.numel() != packed.K:
        raise ValueError(f"vector has {vector.numel()} elements; expected K={packed.K}")
    x = vector.detach().cpu().contiguous().to(torch.bfloat16).float()
    words = np.asarray(packed.packed_a, dtype=np.uint16)
    out = torch.zeros(packed.padded_rows, dtype=torch.float32)
    cursor = 0
    width, core_height = packed.config.block_width, packed.config.core_height
    for slice_id, p in enumerate(packed.blocks_per_slice.tolist()):
        row_base = slice_id * packed.config.block_height
        for _ in range(p):
            for core_row in range(packed.config.core_rows):
                for local_row in range(core_height):
                    object_row = words[cursor : cursor + 2 * width]
                    cursor += 2 * width
                    row = row_base + core_row * core_height + local_row
                    if row >= packed.M:
                        continue
                    indices = torch.from_numpy(object_row[:width].astype(np.int64, copy=False))
                    values = _words_to_bf16(object_row[width:]).float()
                    out[row] += (values * x[indices]).sum()
    if cursor != words.size:
        raise RuntimeError("packed_a length disagrees with blocks_per_slice")
    return out[: packed.M].to(torch.bfloat16)


# Restore canonical row order from physical SELL output using each window's local map.
def cpu_unpermute_windows(packed: PackedSliceELL, physical_y: torch.Tensor) -> torch.Tensor:
    """Scatter window-local physical y' into canonical y without touching x."""

    if physical_y.numel() != packed.M:
        raise ValueError(f"physical output has {physical_y.numel()} elements; expected M={packed.M}")
    if packed.window_count == 0:
        return physical_y.clone()
    if packed.window_slice_offsets is None or packed.row_indices is None:
        raise ValueError("sorted packed matrix is missing its window row map")
    canonical = torch.empty_like(physical_y)
    for first, last in zip(packed.window_slice_offsets[:-1], packed.window_slice_offsets[1:]):
        base = int(first) * packed.config.block_height
        end = min(int(last) * packed.config.block_height, packed.M)
        if end <= base:
            continue
        local = packed.row_indices[base:end].astype(np.int64)
        if np.any(local >= end - base) or np.unique(local).size != end - base:
            raise ValueError("row_indices is not a permutation within its window")
        canonical[base + torch.from_numpy(local)] = physical_y[base:end]
    return canonical


# Create the fixed-length host config object [x bits | blocks_per_slice | alignment zeros].
def make_runtime_config(
    vector: torch.Tensor,
    local_blocks_per_slice: np.ndarray | Sequence[int],
    *,
    max_local_slices: int | None = None,
) -> torch.Tensor:
    """Pack ``[BF16 x bits | uint16 blocks_per_slice | alignment zeros]``.

    All Shim columns use the same object length by passing their common
    ``max_local_slices``.  This function is host-side only; the future kernel
    reinterprets the first part as BF16 and the tail as uint16 control words.
    """

    x_bits = _bf16_bits(vector)
    controls = np.asarray(local_blocks_per_slice, dtype=np.uint64).reshape(-1)
    if np.any(controls > np.iinfo(np.uint16).max):
        raise ValueError("blocks_per_slice must fit uint16")
    if max_local_slices is None:
        max_local_slices = int(controls.size)
    if max_local_slices < controls.size:
        raise ValueError("max_local_slices is smaller than the supplied control table")
    words = x_bits.size + max_local_slices
    alignment_words = CONFIG_ALIGNMENT_BYTES // np.dtype(np.uint16).itemsize
    padded_words = ((words + alignment_words - 1) // alignment_words) * alignment_words
    packed = np.zeros(padded_words, dtype=np.uint16)
    packed[: x_bits.size] = x_bits
    packed[x_bits.size : x_bits.size + controls.size] = controls.astype(np.uint16)
    return _words_to_bf16(packed)


# Load a conventional CSR NPZ for the optional standalone packing CLI.
def _load_csr_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(path)
    required = {"indptr", "indices", "values"}
    if not required.issubset(data.files):
        raise ValueError(f"{path} must contain {sorted(required)}")
    if "shape" in data.files:
        shape = np.asarray(data["shape"], dtype=np.int64).reshape(-1)
        if shape.size != 2:
            raise ValueError("shape must have exactly two entries")
        K = int(shape[1])
    elif "K" in data.files:
        K = int(np.asarray(data["K"]).item())
    else:
        raise ValueError("CSR npz must contain shape=[M,K] or K")
    return data["indptr"], data["indices"], data["values"], K


# Run the optional standalone packer; imported use is the normal test path.
def main() -> None:
    parser = argparse.ArgumentParser(description="Pack CSR into Slice-ELL or windowed SELL-C-sigma")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csr-npz", type=Path, help="npz with indptr, indices, values, and shape")
    source.add_argument("--safetensors", type=Path, help="one pruned-model safetensors shard")
    parser.add_argument("--tensor", help="2-D tensor name when --safetensors is used")
    parser.add_argument("--save-dir", type=Path, help="optional directory for raw-binary cache files")
    parser.add_argument("--stem", default="slice_ell")
    parser.add_argument("--core-rows", type=int, default=4)
    parser.add_argument("--block-height", type=int, default=32)
    parser.add_argument("--block-width", type=int, default=256)
    parser.add_argument("--shim-columns", type=int, default=8)
    parser.add_argument("--window-count", type=int, default=0, help="0: no row sort; 1: offline global sort; 8/16: windowed sort")
    parser.add_argument("--window-slice-boundaries", type=int, nargs="+", help="optional boundaries in B_h-row slice units")
    args = parser.parse_args()
    config = SliceELLConfig(
        core_rows=args.core_rows,
        block_height=args.block_height,
        block_width=args.block_width,
        shim_columns=args.shim_columns,
        window_count=args.window_count,
        window_slice_boundaries=tuple(args.window_slice_boundaries) if args.window_slice_boundaries else None,
    )
    if args.csr_npz:
        indptr, indices, values, K = _load_csr_npz(args.csr_npz)
        packed = csr_to_slice_ell(indptr, indices, values, K=K, config=config)
    else:
        if not args.tensor:
            parser.error("--tensor is required with --safetensors")
        try:
            from safetensors.torch import safe_open
        except ImportError as error:
            parser.error(f"--safetensors requires safetensors: {error}")
        with safe_open(args.safetensors, framework="pt", device="cpu") as handle:
            if args.tensor not in handle.keys():
                parser.error(f"tensor {args.tensor!r} was not found in {args.safetensors}")
            packed = dense_to_slice_ell(handle.get_tensor(args.tensor), config=config)
    if args.save_dir:
        paths = packed.save_cache(args.save_dir, args.stem)
        print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    print(json.dumps(packed.manifest(), sort_keys=True))


if __name__ == "__main__":
    main()
