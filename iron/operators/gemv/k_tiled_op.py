# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator wrapper for the experimental 4x8 K-tiled dense GEMV baseline."""

from dataclasses import dataclass, field

import aie.utils as aie_utils
import numpy as np

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)


@dataclass
class DenseGEMVKTile(MLIROperator):
    """Dense BF16 GEMV across 4*cols cores with a fixed horizontal K tile."""

    M: int
    K: int
    cols: int = 8
    k_tile: int = 256
    context: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.cols <= 8:
            raise ValueError("cols must be in [1, 8]")
        if self.M <= 0 or self.M % (8 * self.cols):
            raise ValueError("M must be divisible by 8*cols")
        if self.K <= 0 or self.k_tile not in (1376, 4096):
            raise ValueError("the current performance kernel supports k_tile=1376 or 4096")
        super().__init__(context=self.context)

    @property
    def output_blocks_per_column(self) -> int:
        return self.M // (8 * self.cols)

    @property
    def k_blocks(self) -> int:
        return (self.K + self.k_tile - 1) // self.k_tile

    @property
    def packed_a_elements(self) -> int:
        return self.cols * self.output_blocks_per_column * self.k_blocks * 8 * self.k_tile

    @property
    def tiled_x_elements(self) -> int:
        return self.cols * self.output_blocks_per_column * self.k_blocks * self.k_tile

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "k_tiled_design.py", "dense_gemv_k_tiled",
                (aie_utils.get_current_device(), self.M, self.K, self.cols, self.k_tile),
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact(
            "dense_gemv_k_tiled.o",
            dependencies=[SourceArtifact(self.operator_dir / "k_tiled.cc")],
        )]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.packed_a_elements,)),
            AIERuntimeArgSpec("in", (self.tiled_x_elements,)),
            AIERuntimeArgSpec("out", (self.M,)),
        ]
