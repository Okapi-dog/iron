# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NPU operators for the SELL-C-sigma route, dedicated, and time-multiplex designs."""

from dataclasses import dataclass, field

import aie.utils as aie_utils
import numpy as np
from iron.operators.spmv.sell_c_sigma_layout import SELLCoreLayout

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)


@dataclass
class SpMVSELLReorderRoute(MLIROperator):
    """Step-2 route test: three copy producers plus one reorder core/column."""

    M: int
    columns: int = 8
    windows: int = 8
    context: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.columns not in (1, 8) or self.windows not in (self.columns, 2 * self.columns):
            raise ValueError("route test supports 1 or 8 columns and 1 or 2 windows/column")
        if self.M <= 0 or self.M % (8 * self.windows):
            raise ValueError("M must be divisible by 8 * windows")
        if self.M // self.windows > 65535:
            raise ValueError("uint16 row map requires at most 65535 rows/window")
        super().__init__(context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "sell_c_sigma_design.py",
                "sell_reorder_route",
                (aie_utils.get_current_device(), self.M, self.columns, self.windows),
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact(
            "sell_c_sigma.o",
            dependencies=[SourceArtifact(self.operator_dir / "sell_c_sigma.cc")],
        )]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (10 * (self.M // 8),)),
            AIERuntimeArgSpec("in", (self.M,), dtype=np.dtype(np.int16)),
            AIERuntimeArgSpec("out", (self.M,)),
        ]


@dataclass
class SpMVSELLDedicated(MLIROperator):
    """Step-3 SELL-C-sigma with 3 compute and 1 reorder core per column."""

    M: int
    K: int
    blocks_per_column: tuple[int, ...]
    windows: int
    rows_per_core: tuple[int, int, int] = (2, 3, 3)
    block_width: int = 256
    vertical16: bool = False
    valid_rows_per_window: tuple[int, ...] | None = None
    context: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.blocks_per_column = tuple(int(n) for n in self.blocks_per_column)
        self.rows_per_core = tuple(int(n) for n in self.rows_per_core)
        if self.rows_per_core == (16, 16, 16) and self.block_width == 128:
            pass  # Exact-geometry horizontal/vertical experiment.
        else:
            if self.vertical16:
                raise ValueError("vertical16 requires rows_per_core=(16,16,16), B_w=128")
            SELLCoreLayout(self.rows_per_core)
            if self.block_width != 256:
                raise ValueError("horizontal SELL requires B_w=256")
        block_height = sum(self.rows_per_core)
        columns = len(self.blocks_per_column)
        if columns not in (1, 8) or self.windows not in (columns, 2 * columns):
            raise ValueError("dedicated design supports 1 or 8 columns and 1 or 2 windows/column")
        if (self.M <= 0 or self.M % (block_height * self.windows)
                or self.M // self.windows % 2 or not 0 < self.K <= 65535):
            raise ValueError("M needs even-length fixed windows of complete slices, and K must fit uint16")
        if self.M // self.windows > 65535 or any(n <= 0 for n in self.blocks_per_column):
            raise ValueError("window map must fit uint16 and each column needs A blocks")
        if self.valid_rows_per_window is not None:
            self.valid_rows_per_window = tuple(int(n) for n in self.valid_rows_per_window)
            rows_per_window = self.M // self.windows
            if (self.windows != columns
                    or len(self.valid_rows_per_window) != self.windows
                    or any(not 0 < n <= rows_per_window
                           for n in self.valid_rows_per_window)):
                raise ValueError("compact output needs one nonempty window per column")
        super().__init__(context=self.context)

    @property
    def config_words(self) -> int:
        """Size of the Slice-ELL config in one window, rounded to 4 bytes."""
        words = 2 + self.K + self.M // (sum(self.rows_per_core) * self.windows)
        return words + words % 2

    @property
    def control_words(self) -> int:
        """Size of the combined config and local row map in one window."""
        return self.config_words + self.M // self.windows

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "sell_c_sigma_design.py",
                "sell_spmv_dedicated",
                (aie_utils.get_current_device(), self.M, self.K,
                 self.blocks_per_column, self.windows, self.rows_per_core,
                 self.block_width, self.vertical16, self.valid_rows_per_window),
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact(
            "sell_c_sigma.o",
            dependencies=[SourceArtifact(self.operator_dir / "sell_c_sigma.cc")],
        )]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec(
                "in", (sum(self.blocks_per_column) * sum(self.rows_per_core) * self.block_width * 2,)
            ),
            AIERuntimeArgSpec(
                "in", (self.windows * self.control_words,), dtype=np.dtype(np.int16),
            ),
            AIERuntimeArgSpec("out", (self.M,)),
        ]


@dataclass
class SpMVSELLTimeMultiplex(MLIROperator):
    """Step-4 SELL-C-sigma: 4 compute cores/column, one reorders afterward."""

    M: int
    K: int
    blocks_per_column: tuple[int, ...]
    windows: int
    context: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.blocks_per_column = tuple(int(n) for n in self.blocks_per_column)
        columns = len(self.blocks_per_column)
        if columns not in (1, 8) or self.windows not in (columns, 2 * columns):
            raise ValueError("time-multiplex design supports 1 or 8 columns and 1 or 2 windows/column")
        if (self.M <= 0 or self.M % (8 * self.windows)
                or self.M // self.windows % 2 or not 0 < self.K <= 65535):
            raise ValueError("M needs even-length complete windows, and K must fit uint16")
        if self.M // self.windows > 65535 or any(n <= 0 for n in self.blocks_per_column):
            raise ValueError("window map must fit uint16 and each column needs A blocks")
        super().__init__(context=self.context)

    @property
    def config_words(self) -> int:
        words = 2 + self.K + self.M // (8 * self.windows)
        return words + words % 2

    @property
    def control_words(self) -> int:
        return self.config_words + self.M // self.windows

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "sell_c_sigma_design.py",
                "sell_spmv_time_multiplex",
                (aie_utils.get_current_device(), self.M, self.K,
                 self.blocks_per_column, self.windows),
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact(
            "sell_c_sigma.o",
            dependencies=[SourceArtifact(self.operator_dir / "sell_c_sigma.cc")],
        )]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (sum(self.blocks_per_column) * 8 * 256 * 2,)),
            AIERuntimeArgSpec("in", (self.windows * self.control_words,), dtype=np.dtype(np.int16)),
            AIERuntimeArgSpec("out", (self.M,)),
        ]
