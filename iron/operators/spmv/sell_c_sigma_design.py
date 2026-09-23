# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NPU designs for SELL-C-sigma, following the existing Slice-ELL FIFO style."""

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aie import T
import aie.dialects.index as index
from aie.dialects import arith, memref
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile


def _layout(M: int, columns: int, windows: int) -> tuple[int, int, int]:
    """Return rows/window, slices/column, and windows/column for B_h=8."""

    if columns not in (1, 8) or windows not in (columns, 2 * columns):
        raise ValueError("Step 2 supports 1 or 8 columns and 1 or 2 windows per column")
    if M <= 0 or M % (8 * windows):
        raise ValueError("M must be divisible by 8 * window_count")
    rows_per_window = M // windows
    if rows_per_window > 65535:
        raise ValueError("Step 2 uint16 row map needs at most 65535 rows/window")
    return rows_per_window, M // (8 * columns), windows // columns


def sell_reorder_route(dev, M: int, columns: int, windows: int):
    """Step 2: three copy producers, MemTile join, and one reorder core/column.

    Input is a 10-BF16 object per 8 physical rows.  The last element in each
    four-element producer segment is a dummy.  Row maps and final outputs are
    fixed-length window objects.  No intermediate DRAM output is used.
    """

    rows_per_window, slices_per_column, windows_per_column = _layout(M, columns, windows)
    slices_per_window = rows_per_window // 8
    bf16 = np.dtype[bfloat16]
    i16 = np.dtype[np.int16]
    l1_pair = np.ndarray[(2,), bf16]
    l1_quad = np.ndarray[(4,), bf16]
    l2_joined = np.ndarray[(10,), bf16]
    l1_map = np.ndarray[(rows_per_window,), i16]
    l1_canonical = np.ndarray[(rows_per_window,), bf16]
    l3_physical = np.ndarray[(10 * (M // 8),), bf16]
    l3_map = np.ndarray[(M,), i16]
    l3_canonical = np.ndarray[(M,), bf16]

    copy2 = Kernel("sell_route_copy2", "sell_c_sigma.o", [l1_pair, l1_pair])
    copy4 = Kernel("sell_route_copy4", "sell_c_sigma.o", [l1_quad, l1_quad])
    scatter = Kernel(
        "sell_reorder_scatter8", "sell_c_sigma.o",
        [l2_joined, l1_map, l1_canonical, np.int32],
    )

    physical_fifos, map_fifos, canonical_fifos, workers = [], [], [], []
    for col in range(columns):
        mem = Tile(col, 1)
        physical = ObjectFifo(l2_joined, name=f"route_physical_{col}", depth=2)
        joined = ObjectFifo(l2_joined, name=f"route_joined_{col}", depth=2)
        row_map = ObjectFifo(l1_map, name=f"route_map_{col}", depth=1)
        canonical = ObjectFifo(l1_canonical, name=f"route_canonical_{col}", depth=1)
        physical_fifos.append(physical)
        map_fifos.append(row_map)
        canonical_fifos.append(canonical)

        producer_input = physical.cons().split(
            [0, 2, 6], obj_types=[l1_pair, l1_quad, l1_quad],
            tile=mem, names=[f"route_input_{col}_{row}" for row in range(3)],
        )
        producer_output = joined.prod().join(
            [0, 2, 6], obj_types=[l1_pair, l1_quad, l1_quad],
            tile=mem, names=[f"route_output_{col}_{row}" for row in range(3)],
        )

        def producer_body(input_fifo, output_fifo, copy_kernel):
            for _ in range_(slices_per_column):
                src = input_fifo.acquire(1)
                dst = output_fifo.acquire(1)
                copy_kernel(src, dst)
                input_fifo.release(1)
                output_fifo.release(1)

        for row in range(3):
            workers.append(Worker(
                producer_body,
                [producer_input[row].cons(), producer_output[row].prod(), copy2 if row == 0 else copy4],
                tile=Tile(col, 2 + row),
            ))

        def reorder_body(joined_fifo, map_fifo, canonical_fifo, reorder_kernel):
            for _ in range_(windows_per_column):
                mapping = map_fifo.acquire(1)
                output = canonical_fifo.acquire(1)
                for local_slice in range_(slices_per_window):
                    source = joined_fifo.acquire(1)
                    reorder_kernel(source, mapping, output, local_slice)
                    joined_fifo.release(1)
                map_fifo.release(1)
                canonical_fifo.release(1)

        workers.append(Worker(
            reorder_body,
            [joined.cons(), row_map.cons(), canonical.prod(), scatter],
            tile=Tile(col, 5), stack_size=2048,
        ))

    physical_taps = []
    map_taps = []
    canonical_taps = []
    for col in range(columns):
        physical_taps.append(TensorAccessPattern(
            [10 * (M // 8)], col * slices_per_column * 10,
            [1, 1, 1, slices_per_column * 10], [0, 0, 0, 1],
        ))
        window_offset = col * windows_per_column * rows_per_window
        window_tap = TensorAccessPattern(
            [M], window_offset,
            [windows_per_column, 1, 1, rows_per_window],
            [rows_per_window, 0, 0, 1],
        )
        map_taps.append(window_tap)
        canonical_taps.append(window_tap)

    def sequence(physical_input, row_map, output, physical_prods, map_prods, output_conss):
        tasks = TaskGroup()
        for col in range(columns):
            physical_prods[col].fill(physical_input, physical_taps[col], group=tasks)
            map_prods[col].fill(row_map, map_taps[col], group=tasks)
        for col in range(columns):
            output_conss[col].drain(output, canonical_taps[col], group=tasks, wait=True)
        tasks.finish()

    runtime = Runtime(
        sequence,
        [l3_physical, l3_map, l3_canonical,
         [fifo.prod() for fifo in physical_fifos],
         [fifo.prod() for fifo in map_fifos],
         [fifo.cons() for fifo in canonical_fifos]],
    )
    return Program(dev, runtime, workers=workers).resolve_program()


def sell_spmv_dedicated(dev, M: int, K: int, blocks_per_column, windows: int):
    """Step 3: three SpMV cores and one dedicated reorder core in each column.

    The two shim inputs are packed A and a control stream.  Each control
    window contains the usual Slice-ELL config followed by its local row map;
    the MemTile splits these into independent fixed-length core FIFOs.
    """

    blocks_per_column = tuple(int(n) for n in blocks_per_column)
    columns = len(blocks_per_column)
    rows_per_window, slices_per_column, windows_per_column = _layout(M, columns, windows)
    if K <= 0 or K > 65535 or any(n < 0 for n in blocks_per_column):
        raise ValueError("K must fit uint16 and block counts must be nonnegative")
    slices_per_window = rows_per_window // 8
    config_words = 2 + K + slices_per_window
    config_words += config_words % 2  # 4-byte ObjectFIFO DMA alignment.
    control_words = config_words + rows_per_window
    words_per_block = 8 * 256 * 2
    total_blocks = sum(blocks_per_column)

    bf16 = np.dtype[bfloat16]
    i16 = np.dtype[np.int16]
    a_types = [np.ndarray[(rows * 512,), bf16] for rows in (2, 3, 3)]
    y_types = [np.ndarray[(rows,), bf16] for rows in (2, 4, 4)]
    l2_a = np.ndarray[(words_per_block,), bf16]
    l2_y = np.ndarray[(10,), bf16]
    l2_control = np.ndarray[(control_words,), i16]
    l1_config = np.ndarray[(config_words,), i16]
    l1_map = np.ndarray[(rows_per_window,), i16]
    l1_output = np.ndarray[(rows_per_window,), bf16]
    l1_state = np.ndarray[(4,), np.dtype[np.float32]]
    l3_a = np.ndarray[(total_blocks * words_per_block,), bf16]
    l3_control = np.ndarray[(columns * windows_per_column * control_words,), i16]
    l3_output = np.ndarray[(M,), bf16]

    init = Kernel("sell_state_init", "sell_c_sigma.o", [l1_state])
    accumulate2 = Kernel("sell_accumulate2", "sell_c_sigma.o", [a_types[0], l1_config, l1_state])
    accumulate3 = Kernel("sell_accumulate3", "sell_c_sigma.o", [a_types[1], l1_config, l1_state])
    finalize2 = Kernel("sell_finalize2", "sell_c_sigma.o", [l1_state, y_types[0]])
    finalize3 = Kernel("sell_finalize3", "sell_c_sigma.o", [l1_state, y_types[1]])
    accumulate = (accumulate2, accumulate3, accumulate3)
    finalize = (finalize2, finalize3, finalize3)
    scatter = Kernel("sell_reorder_scatter8", "sell_c_sigma.o", [l2_y, l1_map, l1_output, np.int32])
    zero_output = Kernel("sell_zero_output", "sell_c_sigma.o", [l1_output, np.int32])

    a_fifos, control_fifos, output_fifos, workers = [], [], [], []
    for col in range(columns):
        mem = Tile(col, 1)
        a_col = ObjectFifo(l2_a, name=f"sell_a_col_{col}", depth=2)
        control_col = ObjectFifo(l2_control, name=f"sell_control_col_{col}", depth=1)
        joined = ObjectFifo(l2_y, name=f"sell_joined_{col}", depth=2)
        output = ObjectFifo(l1_output, name=f"sell_output_col_{col}", depth=1)
        a_fifos.append(a_col)
        control_fifos.append(control_col)
        output_fifos.append(output)

        a_cores = a_col.cons().split(
            [0, 2 * 512, 5 * 512], obj_types=a_types, tile=mem,
            depths=[2, 2, 2], names=[f"sell_a_{col}_{row}" for row in range(3)],
        )
        y_cores = joined.prod().join(
            [0, 2, 6], obj_types=y_types, tile=mem,
            names=[f"sell_y_{col}_{row}" for row in range(3)],
        )
        config_fifo, map_fifo = control_col.cons().split(
            [0, config_words], obj_types=[l1_config, l1_map], tile=mem,
            names=[f"sell_config_{col}", f"sell_map_{col}"],
        )

        def compute_body(a_fifo, config_input, y_fifo, state, init_kernel, accumulate_kernel, finalize_kernel):
            for _ in range_(windows_per_column):
                config = config_input.acquire(1)
                for local_slice in range_(slices_per_window):
                    init_kernel(state)
                    p_word = memref.load(config, [arith.addi(index.constant(K + 2), local_slice)])
                    p = index.casts(T.index(), p_word)
                    for _ in range_(p):
                        a = a_fifo.acquire(1)
                        accumulate_kernel(a, config, state)
                        a_fifo.release(1)
                    y = y_fifo.acquire(1)
                    finalize_kernel(state, y)
                    y_fifo.release(1)
                config_input.release(1)

        for row in range(3):
            state = Buffer(l1_state, name=f"sell_state_{col}_{row}")
            workers.append(Worker(
                compute_body,
                [a_cores[row].cons(), config_fifo.cons(), y_cores[row].prod(), state,
                 init, accumulate[row], finalize[row]],
                tile=Tile(col, 2 + row), stack_size=2048, dynamic_objfifo_lowering=True,
            ))

        def reorder_body(joined_input, map_input, canonical_output, clear_kernel, reorder_kernel):
            for _ in range_(windows_per_column):
                mapping = map_input.acquire(1)
                canonical = canonical_output.acquire(1)
                clear_kernel(canonical, rows_per_window)
                for local_slice in range_(slices_per_window):
                    physical = joined_input.acquire(1)
                    reorder_kernel(physical, mapping, canonical, local_slice)
                    joined_input.release(1)
                map_input.release(1)
                canonical_output.release(1)

        workers.append(Worker(
            reorder_body, [joined.cons(), map_fifo.cons(), output.prod(), zero_output, scatter],
            tile=Tile(col, 5), stack_size=2048,
        ))

    a_taps, control_taps, output_taps = [], [], []
    a_offset = 0
    for col, count in enumerate(blocks_per_column):
        if count == 0:
            raise ValueError("a completely empty column needs a separate zero-output path")
        a_words = count * words_per_block
        a_taps.append(TensorAccessPattern(
            [total_blocks * words_per_block], a_offset, [1, 1, 1, a_words], [0, 0, 0, 1],
        ))
        control_taps.append(TensorAccessPattern(
            [columns * windows_per_column * control_words],
            col * windows_per_column * control_words,
            [windows_per_column, 1, 1, control_words], [control_words, 0, 0, 1],
        ))
        output_taps.append(TensorAccessPattern(
            [M], col * windows_per_column * rows_per_window,
            [windows_per_column, 1, 1, rows_per_window], [rows_per_window, 0, 0, 1],
        ))
        a_offset += a_words

    def sequence(A, control, Y, a_prods, control_prods, output_conss):
        tasks = TaskGroup()
        for col in range(columns):
            a_prods[col].fill(A, a_taps[col], group=tasks)
            control_prods[col].fill(control, control_taps[col], group=tasks)
        for col in range(columns):
            output_conss[col].drain(Y, output_taps[col], group=tasks, wait=True)
        tasks.finish()

    runtime = Runtime(
        sequence,
        [l3_a, l3_control, l3_output,
         [fifo.prod() for fifo in a_fifos],
         [fifo.prod() for fifo in control_fifos],
         [fifo.cons() for fifo in output_fifos]],
    )
    return Program(dev, runtime, workers=workers).resolve_program()
