# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static, row-order-preserving ELL SpMV operator.

This is the Phase-1 compatibility baseline.  It intentionally has no dynamic
ObjectFIFO or Slice-ELL control path: one packed ELL matrix, one vector, and
one contiguous output vector are transferred on every invocation.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import aie.utils as aie_utils
import numpy as np
import torch

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)


@dataclass
class SpMVELL(MLIROperator):
    M: int
    K: int
    ell_width: int
    rows: int = 4
    cols: int = 8
    rows_per_core: int = 2
    trace_size: int = 0
    context: object | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "ell_width": "w",
        "rows_per_core": "rpc",
        "trace_size": "trace",
    }

    def __post_init__(self) -> None:
        if self.M <= 0 or self.K <= 0 or self.ell_width <= 0:
            raise ValueError("M, K, and ell_width must be positive")
        if self.rows <= 0 or self.cols <= 0 or self.rows_per_core <= 0:
            raise ValueError("rows, cols, and rows_per_core must be positive")
        if self.trace_size < 0:
            raise ValueError("trace_size must be non-negative")
        if self.ell_width % 32:
            raise ValueError("ell_width must be a multiple of the 32-lane kernel")
        block_rows = self.rows * self.cols * self.rows_per_core
        if self.M % block_rows:
            raise ValueError(
                f"M ({self.M}) must be divisible by rows*cols*rows_per_core ({block_rows})"
            )
        super().__init__(context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "spmv_ell",
                (
                    aie_utils.get_current_device(),
                    self.M,
                    self.K,
                    self.ell_width,
                    self.rows,
                    self.cols,
                    self.rows_per_core,
                    self.trace_size,
                ),
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                "spmv_ell.o",
                dependencies=[SourceArtifact(self.operator_dir / "spmv.cc")],
            )
        ]

    def get_arg_spec(self):
        # Each row carries ``[ell_width uint16 indices][ell_width BF16 values]``
        # as BF16 words.  The kernel reinterprets the first half as uint16.
        return [
            AIERuntimeArgSpec("in", (self.M * self.ell_width * 2,)),
            AIERuntimeArgSpec("in", (self.K,)),
            AIERuntimeArgSpec("out", (self.M,)),
        ]

    def reference(self, packed, vector):
        from .reference import reference_ell

        return reference_ell(packed, vector, self.M, self.ell_width)


@dataclass
class SpMVSELL32(MLIROperator):
    """Static SELL-C with C=32 and row-order-preserving block order."""

    M: int
    K: int
    ell_width: int
    rows: int = 4
    cols: int = 8
    trace_size: int = 0
    context: object | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "ell_width": "w",
        "trace_size": "trace",
    }

    def __post_init__(self) -> None:
        if self.M % (32 * self.rows * self.cols):
            raise ValueError("M must be divisible by 32*rows*cols")
        if self.K % 32 or self.ell_width % 32:
            raise ValueError("K and ell_width must be multiples of 32")
        if self.trace_size < 0:
            raise ValueError("trace_size must be non-negative")
        super().__init__(context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py", "spmv_sell32",
                (
                    aie_utils.get_current_device(),
                    self.M,
                    self.K,
                    self.ell_width,
                    self.rows,
                    self.cols,
                    self.trace_size,
                ),
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact("spmv_ell.o", dependencies=[SourceArtifact(self.operator_dir / "spmv.cc")])]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.M * self.ell_width * 2,)),
            AIERuntimeArgSpec("in", (self.K,)),
            AIERuntimeArgSpec("out", (self.M,)),
        ]


@dataclass
class SpMVSELL32Block(SpMVSELL32):
    """SELL-32 with a fixed 16-slot horizontal input object."""

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py", "spmv_sell32_block",
                (
                    aie_utils.get_current_device(),
                    self.M,
                    self.K,
                    self.ell_width,
                    self.rows,
                    self.cols,
                    self.trace_size,
                ),
            ),
        )


@dataclass
class SpMVSliceELLStatic(MLIROperator):
    """Phase-3 horizontal Slice-ELL with one static p=1 or p=2 ABI.

    This is intentionally not the final ragged operator.  It is the smallest
    device experiment that can hold eight FP32 32-lane accumulators across all
    horizontal A blocks belonging to one slice.
    """

    M: int
    K: int
    blocks_per_slice: int
    rows: int = 4
    cols: int = 8
    block_height: int = 32
    block_width: int = 256
    trace_size: int = 0
    context: object | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "blocks_per_slice": "p",
        "block_height": "bh",
        "block_width": "bw",
        "trace_size": "trace",
    }

    def __post_init__(self) -> None:
        if self.blocks_per_slice not in (1, 2):
            raise ValueError("Phase-3 static Slice-ELL supports blocks_per_slice=1 or 2")
        if not 1 <= self.rows <= 4 or self.block_height % self.rows:
            raise ValueError("block_height must be divisible by 1..4 core rows")
        if self.block_height // self.rows != 8:
            raise ValueError("the first horizontal kernel is specialized for C_h=8")
        if self.block_width != 256:
            raise ValueError("the first horizontal kernel is specialized for B_w=256")
        if self.M <= 0 or self.K <= 0 or self.M % (self.block_height * self.cols):
            raise ValueError("M must be positive and divisible by block_height*cols")
        if self.trace_size < 0:
            raise ValueError("trace_size must be non-negative")
        super().__init__(context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "spmv_slice_ell_static",
                (
                    aie_utils.get_current_device(),
                    self.M,
                    self.K,
                    self.blocks_per_slice,
                    self.rows,
                    self.cols,
                    self.block_height,
                    self.block_width,
                    self.trace_size,
                ),
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact("spmv_ell.o", dependencies=[SourceArtifact(self.operator_dir / "spmv.cc")])]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.M * self.blocks_per_slice * self.block_width * 2,)),
            AIERuntimeArgSpec("in", (self.K,)),
            AIERuntimeArgSpec("out", (self.M,)),
        ]

    def reference(self, packed, vector):
        from .slice_ell import PackedSliceELL, SliceELLConfig, cpu_spmv_slice_ell

        config = SliceELLConfig(
            core_rows=self.rows,
            block_height=self.block_height,
            block_width=self.block_width,
            shim_columns=self.cols,
        )
        slices = self.M // self.block_height
        words_per_slice = config.words_per_slice_block
        raw = packed.detach().cpu().contiguous().view(torch.uint16).numpy()
        return cpu_spmv_slice_ell(
            PackedSliceELL(
                M=self.M,
                K=self.K,
                nnz=0,
                config=config,
                padded_rows=self.M,
                packed_a=raw,
                blocks_per_slice=np.full(slices, self.blocks_per_slice, dtype=np.uint16),
                slice_word_offsets=np.arange(slices + 1, dtype=np.uint64)
                * self.blocks_per_slice * words_per_slice,
                column_slice_offsets=np.arange(self.cols + 1, dtype=np.uint32) * (slices // self.cols),
            ),
            vector,
        )


@dataclass
class SpMVSliceELLDynamicScalar(MLIROperator):
    """Phase-4 one-column dynamic Slice-ELL with FP32 scalar L1 state."""

    M: int
    K: int
    total_blocks: int
    trace_size: int = 0
    context: object | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "total_blocks": "blocks",
        "trace_size": "trace",
    }

    def __post_init__(self) -> None:
        if self.M <= 0 or self.M % 32 or self.K <= 0 or self.total_blocks < 0:
            raise ValueError("M must be divisible by 32; K and total_blocks must be valid")
        if self.trace_size < 0:
            raise ValueError("trace_size must be non-negative")
        super().__init__(context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "spmv_slice_ell_dynamic_scalar",
                (aie_utils.get_current_device(), self.M, self.K, self.total_blocks, self.trace_size),
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact("spmv_ell.o", dependencies=[SourceArtifact(self.operator_dir / "spmv.cc")])]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.total_blocks * 32 * 256 * 2,)),
            AIERuntimeArgSpec("in", (self.K + self.M // 32,), dtype=np.dtype(np.int16)),
            AIERuntimeArgSpec("out", (self.M,)),
        ]


@dataclass
class SpMVSliceELLDynamicScalarMultiCol(MLIROperator):
    """Phase-4 dynamic scalar-state Slice-ELL across one to eight columns."""

    M: int
    K: int
    blocks_per_column: tuple[int, ...]
    block_height: int = 32
    trace_size: int = 0
    context: object | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "blocks_per_column": "bpc",
        "block_height": "bh",
        "trace_size": "trace",
    }

    def __post_init__(self) -> None:
        self.blocks_per_column = tuple(int(n) for n in self.blocks_per_column)
        cols = len(self.blocks_per_column)
        if (not 1 <= cols <= 8 or self.block_height <= 0 or self.block_height % 8
                or self.M <= 0 or self.M % (self.block_height * cols) or self.K <= 0):
            raise ValueError("block_height must be a positive multiple of eight and divide M")
        if any(n <= 0 for n in self.blocks_per_column):
            raise ValueError("the first multicolumn implementation requires non-empty A columns")
        super().__init__(context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "spmv_slice_ell_dynamic_scalar_multicol",
                (aie_utils.get_current_device(), self.M, self.K, self.blocks_per_column,
                 self.block_height, self.trace_size),
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact("spmv_ell.o", dependencies=[SourceArtifact(self.operator_dir / "spmv.cc")])]

    def get_arg_spec(self):
        cols = len(self.blocks_per_column)
        config_words = 2 + self.K + self.M // (self.block_height * cols)
        config_words += config_words % 2
        return [
            AIERuntimeArgSpec("in", (sum(self.blocks_per_column) * self.block_height * 256 * 2,)),
            AIERuntimeArgSpec("in", (cols * config_words,), dtype=np.dtype(np.int16)),
            AIERuntimeArgSpec("out", (self.M,)),
        ]
