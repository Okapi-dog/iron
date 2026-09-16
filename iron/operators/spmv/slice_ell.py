# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline format contract for row-order-preserving Slice-ELL.

This module deliberately has no MLIR, XRT, or NPU dependency.  Phase 2 uses
it to make the weight format, CPU reference, and host config object testable
before dynamic ObjectFIFO lowering is introduced.

The linear A-word order is::

    slice -> horizontal block -> core-row -> local row
          -> [uint16 indices[B_w], BF16 values[B_w]]

``slice_blocks[s]`` gives the number of horizontal blocks in slice ``s``.
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


@dataclass(frozen=True)
class SliceELLConfig:
    """Geometry shared by packer, reference, and the future NPU design."""

    core_rows: int = 4
    block_height: int = 32
    block_width: int = 256
    lanes: int = LANES
    shim_columns: int = 8

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

    @property
    def core_height(self) -> int:
        return self.block_height // self.core_rows

    @property
    def words_per_core_block(self) -> int:
        # Each local row is [indices B_w][values B_w].
        return self.core_height * 2 * self.block_width

    @property
    def words_per_slice_block(self) -> int:
        return self.block_height * 2 * self.block_width


def _bf16_bits(values: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    """Round numeric values to BF16 and return their raw uint16 words."""

    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu().contiguous().to(torch.bfloat16).reshape(-1)
    else:
        tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32).contiguous().to(torch.bfloat16).reshape(-1)
    return tensor.view(torch.uint16).numpy().copy()


def _words_to_bf16(words: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(words, dtype=np.uint16)).view(torch.int16).view(torch.bfloat16)


def _sha256_words(words: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(words, dtype=np.uint16).tobytes()).hexdigest()


@dataclass
class SliceELLPacked:
    """A complete packed matrix plus the metadata required to consume it."""

    M: int
    K: int
    nnz: int
    config: SliceELLConfig
    padded_rows: int
    packed_words: np.ndarray
    slice_blocks: np.ndarray
    slice_word_offsets: np.ndarray
    column_slice_offsets: np.ndarray

    @property
    def slices_per_column(self) -> int:
        return int(self.column_slice_offsets[1] - self.column_slice_offsets[0])

    @property
    def total_slices(self) -> int:
        return int(self.slice_blocks.size)

    @property
    def packed_bf16(self) -> torch.Tensor:
        return _words_to_bf16(self.packed_words)

    def column_slice_blocks(self, column: int) -> np.ndarray:
        if not 0 <= column < self.config.shim_columns:
            raise IndexError("column is outside this packed matrix")
        begin, end = self.column_slice_offsets[column : column + 2]
        return self.slice_blocks[begin:end]

    def manifest(self) -> dict:
        """Return the static sidecar; payloads themselves stay in .npy files."""

        return {
            "format": "row-order-preserving-slice-ell",
            "format_version": FORMAT_VERSION,
            "M": self.M,
            "K": self.K,
            "nnz": self.nnz,
            "padded_rows": self.padded_rows,
            "config": asdict(self.config),
            "core_height": self.config.core_height,
            "total_slices": self.total_slices,
            "slices_per_column": self.slices_per_column,
            "column_slice_offsets": self.column_slice_offsets.tolist(),
            "slice_word_offsets": self.slice_word_offsets.tolist(),
            "slice_blocks_dtype": "uint16",
            "packed_words_dtype": "uint16",
            "index_dtype": "uint16",
            "padding_index": 0,
            "padding_value": "BF16 +0",
            "packed_words": int(self.packed_words.size),
            "packed_A_sha256": _sha256_words(self.packed_words),
            "slice_blocks_sha256": _sha256_words(self.slice_blocks),
        }

    def save(self, directory: str | Path, stem: str = "slice_ell") -> dict[str, Path]:
        """Write a reproducible payload triplet and return its paths."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "packed_A": directory / f"{stem}_packed_A.npy",
            "slice_blocks": directory / f"{stem}_slice_blocks.npy",
            "manifest": directory / f"{stem}_manifest.json",
        }
        np.save(paths["packed_A"], self.packed_words)
        np.save(paths["slice_blocks"], self.slice_blocks)
        paths["manifest"].write_text(json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n")
        return paths


def pack_csr(
    indptr: np.ndarray | Sequence[int],
    indices: np.ndarray | Sequence[int],
    values: np.ndarray | torch.Tensor | Sequence[float],
    *,
    K: int,
    config: SliceELLConfig = SliceELLConfig(),
) -> SliceELLPacked:
    """Pack a CSR matrix without reordering rows.

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
    total_slices = slices_per_column * config.shim_columns
    padded_rows = total_slices * config.block_height
    slice_blocks = np.zeros(total_slices, dtype=np.uint16)

    for slice_id in range(total_slices):
        row_begin = slice_id * config.block_height
        row_end = min(row_begin + config.block_height, M)
        if row_begin >= M:
            continue
        max_nnz = int(max(indptr[row + 1] - indptr[row] for row in range(row_begin, row_end)))
        p = (max_nnz + config.block_width - 1) // config.block_width
        if p > np.iinfo(np.uint16).max:
            raise ValueError("slice_blocks does not fit uint16")
        slice_blocks[slice_id] = p
    # Allocate exactly once.  A list of tiny row arrays would add a large Python
    # memory overhead for model-scale matrices and would transiently duplicate
    # the full packed payload during concatenate().
    slice_word_offsets = np.zeros(total_slices + 1, dtype=np.uint64)
    slice_word_offsets[1:] = np.cumsum(
        slice_blocks.astype(np.uint64) * config.words_per_slice_block, dtype=np.uint64
    )
    packed_words = np.zeros(int(slice_word_offsets[-1]), dtype=np.uint16)
    for slice_id, p in enumerate(slice_blocks.tolist()):
        row_begin = slice_id * config.block_height
        for block_id in range(p):
            block_base = int(slice_word_offsets[slice_id]) + block_id * config.words_per_slice_block
            slot_begin = block_id * config.block_width
            for core_row in range(config.core_rows):
                for local_row in range(config.core_height):
                    row = row_begin + core_row * config.core_height + local_row
                    row_base = block_base + (core_row * config.core_height + local_row) * 2 * config.block_width
                    if row < M:
                        start, stop = int(indptr[row]), int(indptr[row + 1])
                        take_begin = min(start + slot_begin, stop)
                        take_end = min(take_begin + config.block_width, stop)
                        count = take_end - take_begin
                        if count:
                            packed_words[row_base : row_base + count] = indices[take_begin:take_end].astype(np.uint16, copy=False)
                            value_base = row_base + config.block_width
                            packed_words[value_base : value_base + count] = value_bits[take_begin:take_end]
    column_slice_offsets = np.arange(config.shim_columns + 1, dtype=np.uint32) * slices_per_column
    return SliceELLPacked(
        M=M,
        K=K,
        nnz=nnz,
        config=config,
        padded_rows=padded_rows,
        packed_words=packed_words,
        slice_blocks=slice_blocks,
        slice_word_offsets=slice_word_offsets,
        column_slice_offsets=column_slice_offsets,
    )


def reference_csr(
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


def reference_slice_ell(packed: SliceELLPacked, vector: torch.Tensor) -> torch.Tensor:
    """Interpret ``packed_A`` exactly in the documented block/core-row order."""

    if vector.numel() != packed.K:
        raise ValueError(f"vector has {vector.numel()} elements; expected K={packed.K}")
    x = vector.detach().cpu().contiguous().to(torch.bfloat16).float()
    words = np.asarray(packed.packed_words, dtype=np.uint16)
    out = torch.zeros(packed.padded_rows, dtype=torch.float32)
    cursor = 0
    width, core_height = packed.config.block_width, packed.config.core_height
    for slice_id, p in enumerate(packed.slice_blocks.tolist()):
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
        raise RuntimeError("packed_A length disagrees with slice_blocks")
    return out[: packed.M].to(torch.bfloat16)


def make_packed_config(
    vector: torch.Tensor,
    local_slice_blocks: np.ndarray | Sequence[int],
    *,
    max_local_slices: int | None = None,
) -> torch.Tensor:
    """Pack ``[BF16 x bits | uint16 slice_blocks | 64-B zero padding]``.

    All Shim columns use the same object length by passing their common
    ``max_local_slices``.  This function is host-side only; the future kernel
    reinterprets the first part as BF16 and the tail as uint16 control words.
    """

    x_bits = _bf16_bits(vector)
    controls = np.asarray(local_slice_blocks, dtype=np.uint64).reshape(-1)
    if np.any(controls > np.iinfo(np.uint16).max):
        raise ValueError("slice_blocks must fit uint16")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a CSR matrix into row-order-preserving Slice-ELL")
    parser.add_argument("--csr-npz", type=Path, required=True, help="npz with indptr, indices, values, and shape")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="slice_ell")
    parser.add_argument("--core-rows", type=int, default=4)
    parser.add_argument("--block-height", type=int, default=32)
    parser.add_argument("--block-width", type=int, default=256)
    parser.add_argument("--shim-columns", type=int, default=8)
    args = parser.parse_args()
    indptr, indices, values, K = _load_csr_npz(args.csr_npz)
    config = SliceELLConfig(args.core_rows, args.block_height, args.block_width, LANES, args.shim_columns)
    packed = pack_csr(indptr, indices, values, K=K, config=config)
    paths = packed.save(args.output_dir, args.stem)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    print(json.dumps(packed.manifest(), sort_keys=True))


if __name__ == "__main__":
    main()
