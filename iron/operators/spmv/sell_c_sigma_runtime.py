# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side ABI for the dedicated SELL-C-sigma NPU design."""

import numpy as np
import torch

from iron.operators.spmv.slice_ell import PackedSliceELL


def make_dedicated_inputs(packed: PackedSliceELL, x: torch.Tensor):
    """Return packed A, fixed-length control stream, and A blocks/column.

    Each control window is ``[2 header words | BF16 x | p per slice | pad |
    local row indices]``.  The MemTile splits the config and row-map portions.
    This keeps the Shim at two input DMA streams per column.
    """

    config = packed.config
    if (config.core_rows, config.block_height, config.block_width) != (4, 8, 256):
        raise ValueError("dedicated design requires R=4, B_h=8, B_w=256")
    if packed.window_count not in (config.shim_columns, 2 * config.shim_columns):
        raise ValueError("one or two windows per column are required")
    if packed.row_indices is None or packed.row_indices.dtype != np.uint16:
        raise ValueError("dedicated design requires a uint16 window-local row map")
    if packed.padded_rows % (8 * packed.window_count):
        raise ValueError("NPU ABI requires equal-size padded windows")
    if x.numel() != packed.K:
        raise ValueError("x length must equal K")

    windows = packed.window_count
    rows_per_window = packed.padded_rows // windows
    slices_per_window = rows_per_window // 8
    config_words = 2 + packed.K + slices_per_window
    config_words += config_words % 2
    control_words = config_words + rows_per_window
    controls = torch.zeros(windows * control_words, dtype=torch.int16)
    x_words = x.detach().cpu().contiguous().to(torch.bfloat16).view(torch.uint16).view(torch.int16)
    for window in range(windows):
        base = window * control_words
        first_slice = window * slices_per_window
        controls[base + 2 : base + 2 + packed.K] = x_words
        controls[base + 2 + packed.K : base + 2 + packed.K + slices_per_window] = torch.from_numpy(
            packed.blocks_per_slice[first_slice : first_slice + slices_per_window].view(np.int16)
        )
        controls[base + config_words : base + control_words] = torch.from_numpy(
            packed.row_indices[window * rows_per_window : (window + 1) * rows_per_window].view(np.int16)
        )

    blocks_per_column = tuple(
        int(packed.column_blocks_per_slice(col).sum())
        for col in range(config.shim_columns)
    )
    if any(n == 0 for n in blocks_per_column):
        raise ValueError("an entirely zero A column needs a zero-output design path")
    return packed.packed_a_as_bf16, controls, blocks_per_column
