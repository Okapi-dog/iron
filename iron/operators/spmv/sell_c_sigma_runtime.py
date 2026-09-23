# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side ABI shared by the two SELL-C-sigma NPU designs."""

import numpy as np
import torch

from iron.operators.spmv.slice_ell import PackedSliceELL
from iron.operators.spmv.sell_c_sigma_layout import SELLCoreLayout


def make_window_inputs(packed: PackedSliceELL, x: torch.Tensor,
                       rows_per_core=(2, 3, 3)):
    """Return packed A, fixed-length control stream, and A blocks/column.

    Each control window is ``[2 header words | BF16 x | p per slice | pad |
    local row indices]``.  Dedicated mode splits config/map in the MemTile;
    time-multiplex mode broadcasts the complete object to its four cores.
    Both keep the Shim at two input DMA streams per column.
    """

    config = packed.config
    rows_per_core = tuple(int(n) for n in rows_per_core)
    if any(n <= 0 for n in rows_per_core):
        raise ValueError("every compute core must own at least one row")
    block_height = sum(rows_per_core)
    if (config.block_height, config.block_width) != (block_height, 256):
        raise ValueError("packed B_h must match the core layout and B_w must be 256")
    if packed.window_count not in (config.shim_columns, 2 * config.shim_columns):
        raise ValueError("one or two windows per column are required")
    if packed.row_indices is None or packed.row_indices.dtype != np.uint16:
        raise ValueError("SELL NPU designs require a uint16 window-local row map")
    if packed.padded_rows % (block_height * packed.window_count):
        raise ValueError("NPU ABI requires equal-size padded windows")
    if x.numel() != packed.K:
        raise ValueError("x length must equal K")

    windows = packed.window_count
    rows_per_window = packed.padded_rows // windows
    if rows_per_window % 2:
        raise ValueError("BF16 output and uint16 row-map DMA need an even-length window")
    slices_per_window = rows_per_window // block_height
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


def make_dedicated_inputs(packed: PackedSliceELL, x: torch.Tensor,
                          rows_per_core=(2, 3, 3)):
    """Prepare the Step-3 dedicated-core ABI."""
    layout = SELLCoreLayout(tuple(int(n) for n in rows_per_core))
    return make_window_inputs(packed, x, layout.rows_per_core)


def make_time_multiplex_inputs(packed: PackedSliceELL, x: torch.Tensor):
    """Prepare the Step-4 four-compute-core ABI with two rows per core."""
    return make_window_inputs(packed, x, (2, 2, 2, 2))


def prepare_sell_design(packed: PackedSliceELL, x: torch.Tensor,
                        design_name: str, rows_per_core=(2, 3, 3), context=None):
    """Choose a SELL NPU topology while keeping the packed matrix unchanged.

    The returned operator and input mapping can be handed directly to
    ``run_test``.  Unsupported topology names fail instead of silently
    substituting a different kernel.
    """
    from iron.operators.spmv.evaluation import DesignSpec, FormatSpec
    from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated, SpMVSELLTimeMultiplex

    design = DesignSpec(design_name)
    design.validate_format(FormatSpec(
        "sell_c_sigma", block_height=packed.config.block_height,
        block_width=packed.config.block_width,
        columns=packed.config.shim_columns, window_count=packed.window_count,
    ))
    if design_name == "sell_dedicated_reorder":
        A, control, counts = make_dedicated_inputs(packed, x, rows_per_core)
        operator = SpMVSELLDedicated(
            packed.padded_rows, packed.K, counts, packed.window_count,
            rows_per_core=tuple(rows_per_core), context=context,
        )
    elif design_name == "sell_time_multiplex_reorder":
        A, control, counts = make_time_multiplex_inputs(packed, x)
        operator = SpMVSELLTimeMultiplex(
            packed.padded_rows, packed.K, counts, packed.window_count, context=context,
        )
    else:
        raise ValueError(f"{design_name} is not a SELL NPU design")
    return operator, {"packed": A, "control": control}
