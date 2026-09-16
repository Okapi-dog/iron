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
    context: object | None = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "ell_width": "w",
        "rows_per_core": "rpc",
    }

    def __post_init__(self) -> None:
        if self.M <= 0 or self.K <= 0 or self.ell_width <= 0:
            raise ValueError("M, K, and ell_width must be positive")
        if self.rows <= 0 or self.cols <= 0 or self.rows_per_core <= 0:
            raise ValueError("rows, cols, and rows_per_core must be positive")
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
