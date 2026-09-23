# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One source of truth for SELL-C-sigma compute-core row ownership.

The fourth core in each column performs reordering.  The three compute cores
each transfer an even number of BF16 outputs so every FIFO object is at least
four-byte aligned.  Padding slots are ignored by the reorder kernel.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SELLCoreLayout:
    rows_per_core: tuple[int, int, int] = (2, 3, 3)

    def __post_init__(self):
        supported = (1, 2, 3, 4, 6, 12, 24)
        if len(self.rows_per_core) != 3 or any(rows not in supported for rows in self.rows_per_core):
            raise ValueError(f"exactly three compute cores with rows in {supported} are supported")

    @property
    def block_height(self) -> int:
        return sum(self.rows_per_core)

    @property
    def output_slots(self) -> tuple[int, int, int]:
        """Pad each core's BF16 output to a 4-byte DMA length."""
        return tuple(rows + rows % 2 for rows in self.rows_per_core)

    @property
    def a_offsets(self) -> tuple[int, int, int]:
        """BF16-word offsets in a packed horizontal A block (B_w=256)."""
        first, second, _ = self.rows_per_core
        return (0, first * 512, (first + second) * 512)

    @property
    def output_offsets(self) -> tuple[int, int, int]:
        """BF16-word offsets in one MemTile joined output object."""
        first, second, _ = self.output_slots
        return (0, first, first + second)

    @property
    def joined_slots(self) -> int:
        return sum(self.output_slots)
