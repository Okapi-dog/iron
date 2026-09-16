# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aie import T
import aie.dialects.index as index
from aie.dialects import memref
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile
from iron.operators._trace import maybe_enable_trace


def spmv_ell(dev, M, K, ell_width, rows, cols, rows_per_core, trace_size=0, func_prefix=""):
    """Build the static ELL baseline with contiguous row ownership per column."""
    cores = rows * cols
    assert M % (cores * rows_per_core) == 0
    assert ell_width % 32 == 0

    dtype = np.dtype[bfloat16]
    l1_a_ty = np.ndarray[(rows_per_core * ell_width * 2,), dtype]
    l1_x_ty = np.ndarray[(K,), dtype]
    l1_y_ty = np.ndarray[(rows_per_core,), dtype]
    l2_a_ty = np.ndarray[(rows * rows_per_core * ell_width * 2,), dtype]
    l2_y_ty = np.ndarray[(rows * rows_per_core,), dtype]
    l3_a_ty = np.ndarray[(M * ell_width * 2,), dtype]
    l3_x_ty = np.ndarray[(K,), dtype]
    l3_y_ty = np.ndarray[(M,), dtype]

    kernel = Kernel(
        f"{func_prefix}sparse_matvec_vectorized_bf16_bf16",
        f"{func_prefix}spmv_ell.o",
        [np.int32, np.int32, np.int32, np.int32, l1_a_ty, l1_x_ty, l1_y_ty],
    )

    a_cols, x_cols, y_cols, workers = [], [], [], []
    rows_per_column = M // cols
    iterations = rows_per_column // (rows * rows_per_core)

    for col in range(cols):
        mem_tile = Tile(col, 1)
        a_col = ObjectFifo(l2_a_ty, name=f"a_col_{col}", depth=2)
        x_col = ObjectFifo(l1_x_ty, name=f"x_col_{col}", depth=1)
        y_col = ObjectFifo(l2_y_ty, name=f"y_col_{col}", depth=2)
        a_cols.append(a_col)
        x_cols.append(x_col)
        y_cols.append(y_col)

        a_words = rows_per_core * ell_width * 2
        a_cores = a_col.cons().split(
            [r * a_words for r in range(rows)],
            obj_types=[l1_a_ty] * rows,
            tile=mem_tile,
            names=[f"a_{col}_{r}" for r in range(rows)],
        )
        y_cores = y_col.prod().join(
            [r * rows_per_core for r in range(rows)],
            obj_types=[l1_y_ty] * rows,
            tile=mem_tile,
            names=[f"y_{col}_{r}" for r in range(rows)],
        )

        def core_body(a_fifo, x_fifo, y_fifo, spmv_kernel):
            x = x_fifo.acquire(1)
            for _ in range_(iterations):
                a = a_fifo.acquire(1)
                y = y_fifo.acquire(1)
                zero = index.casts(T.i32(), index.constant(0))
                # Keep the legacy ABI as well as its kernel body.  ``K`` and
                # ``zero`` are unused by the implementation, but changing the
                # call ABI can change AIE code generation.
                spmv_kernel(rows_per_core, K, ell_width, zero, a, x, y)
                a_fifo.release(1)
                y_fifo.release(1)
            x_fifo.release(1)

        for row in range(rows):
            workers.append(
                Worker(
                    core_body,
                    [a_cores[row].cons(), x_col.cons(), y_cores[row].prod(), kernel],
                    tile=Tile(col, 2 + row),
                )
            )

    a_taps = [
        TensorAccessPattern(
            tensor_dims=l3_a_ty.__args__[0],
            offset=col * rows_per_column * ell_width * 2,
            sizes=[1, 1, 1, rows_per_column * ell_width * 2],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    x_tap = TensorAccessPattern(
        tensor_dims=l3_x_ty.__args__[0], offset=0, sizes=[1, 1, 1, K], strides=[0, 0, 0, 1]
    )
    y_taps = [
        TensorAccessPattern(
            tensor_dims=l3_y_ty.__args__[0],
            offset=col * rows_per_column,
            sizes=[1, 1, 1, rows_per_column],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    def sequence(A, X, Y, a_prods, x_prods, y_conss):
        # Phase 0 submitted all column A/x fills before any output drain.  Do
        # not fuse drain into the first loop: task construction order affects
        # the DMA command stream even though the calls share a TaskGroup.
        ta = TaskGroup()
        for col in range(cols):
            a_prods[col].fill(A, a_taps[col], group=ta)
            x_prods[col].fill(X, x_tap, group=ta)
        for col in range(cols):
            y_conss[col].drain(Y, y_taps[col], group=ta, wait=True)
        ta.finish()

    runtime = Runtime(
        sequence,
        [
            l3_a_ty,
            l3_x_ty,
            l3_y_ty,
            [fifo.prod() for fifo in a_cols],
            [fifo.prod() for fifo in x_cols],
            [fifo.cons() for fifo in y_cols],
        ],
    )
    prog = Program(dev, runtime, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()


def spmv_sell32(dev, M, K, ell_width, rows, cols, trace_size=0, func_prefix=""):
    """Static SELL-32 baseline: one 32-row block per core and FIFO object."""
    assert M % (32 * rows * cols) == 0 and K % 32 == 0 and ell_width % 32 == 0
    dtype = np.dtype[bfloat16]
    l1_a_ty = np.ndarray[(32 * ell_width * 2,), dtype]
    l1_x_ty = np.ndarray[(K,), dtype]
    l1_y_ty = np.ndarray[(32,), dtype]
    l2_a_ty = np.ndarray[(rows * 32 * ell_width * 2,), dtype]
    l2_y_ty = np.ndarray[(rows * 32,), dtype]
    l3_a_ty = np.ndarray[(M * ell_width * 2,), dtype]
    l3_x_ty = np.ndarray[(K,), dtype]
    l3_y_ty = np.ndarray[(M,), dtype]
    kernel = Kernel(
        f"{func_prefix}sell32_spmv_bf16",
        f"{func_prefix}spmv_ell.o",
        [np.int32, l1_a_ty, l1_x_ty, l1_y_ty],
    )
    blocks_per_column = M // (32 * cols)
    iterations = blocks_per_column // rows
    a_cols, x_cols, y_cols, workers = [], [], [], []
    for col in range(cols):
        mem = Tile(col, 1)
        a_col = ObjectFifo(l2_a_ty, name=f"sell_a_col_{col}", depth=2)
        x_col = ObjectFifo(l1_x_ty, name=f"sell_x_col_{col}", depth=1)
        y_col = ObjectFifo(l2_y_ty, name=f"sell_y_col_{col}", depth=2)
        a_cols.append(a_col); x_cols.append(x_col); y_cols.append(y_col)
        a_cores = a_col.cons().split(
            [r * 32 * ell_width * 2 for r in range(rows)], tile=mem,
            depths=[1] * rows, obj_types=[l1_a_ty] * rows,
            names=[f"sell_a_{col}_{r}" for r in range(rows)],
        )
        y_cores = y_col.prod().join(
            [r * 32 for r in range(rows)], tile=mem,
            obj_types=[l1_y_ty] * rows, names=[f"sell_y_{col}_{r}" for r in range(rows)],
        )
        def core_body(a_fifo, x_fifo, y_fifo, sell_kernel):
            x = x_fifo.acquire(1)
            for _ in range_(iterations):
                a = a_fifo.acquire(1); y = y_fifo.acquire(1)
                sell_kernel(ell_width, a, x, y)
                a_fifo.release(1); y_fifo.release(1)
            x_fifo.release(1)
        for row in range(rows):
            workers.append(Worker(core_body, [a_cores[row].cons(), x_col.cons(), y_cores[row].prod(), kernel], tile=Tile(col, 2 + row)))
    a_taps = [TensorAccessPattern(l3_a_ty.__args__[0], col * blocks_per_column * 32 * ell_width * 2,
              [1, 1, 1, blocks_per_column * 32 * ell_width * 2], [0, 0, 0, 1]) for col in range(cols)]
    x_tap = TensorAccessPattern(l3_x_ty.__args__[0], 0, [1, 1, 1, K], [0, 0, 0, 1])
    y_taps = [TensorAccessPattern(l3_y_ty.__args__[0], col * blocks_per_column * 32,
              [1, 1, 1, blocks_per_column * 32], [0, 0, 0, 1]) for col in range(cols)]
    def sequence(A, X, Y, a_prods, x_prods, y_conss):
        ta = TaskGroup()
        for col in range(cols):
            a_prods[col].fill(A, a_taps[col], group=ta)
            x_prods[col].fill(X, x_tap, group=ta)
        for col in range(cols):
            y_conss[col].drain(Y, y_taps[col], group=ta, wait=True)
        ta.finish()
    runtime = Runtime(sequence, [l3_a_ty, l3_x_ty, l3_y_ty, [f.prod() for f in a_cols], [f.prod() for f in x_cols], [f.cons() for f in y_cols]])
    prog = Program(dev, runtime, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()


def spmv_sell32_block(dev, M, K, ell_width, rows, cols, trace_size=0, func_prefix=""):
    """SELL-32, streamed as 16-slot horizontal blocks (2 KiB/core A object)."""
    block_width = 16
    assert M % (32 * rows * cols) == 0 and K % 32 == 0 and ell_width % block_width == 0
    dtype = np.dtype[bfloat16]
    l1_a_ty = np.ndarray[(32 * block_width * 2,), dtype]
    l1_x_ty, l1_y_ty = np.ndarray[(K,), dtype], np.ndarray[(32,), dtype]
    l2_a_ty, l2_y_ty = np.ndarray[(rows * 32 * block_width * 2,), dtype], np.ndarray[(rows * 32,), dtype]
    l3_a_ty, l3_x_ty, l3_y_ty = np.ndarray[(M * ell_width * 2,), dtype], np.ndarray[(K,), dtype], np.ndarray[(M,), dtype]
    kernel = Kernel(
        f"{func_prefix}sell32_block_spmv_vectorized_bf16_bf16",
        f"{func_prefix}spmv_ell.o",
        [np.int32, np.int32, l1_a_ty, l1_x_ty, l1_y_ty],
    )
    blocks_per_col = M // (32 * cols); iterations = blocks_per_col // rows; horizontal_blocks = ell_width // block_width
    a_cols, x_cols, y_cols, workers = [], [], [], []
    for col in range(cols):
        mem = Tile(col, 1)
        a_col = ObjectFifo(l2_a_ty, name=f"block_a_col_{col}", depth=2)
        x_col = ObjectFifo(l1_x_ty, name=f"block_x_col_{col}", depth=1)
        y_col = ObjectFifo(l2_y_ty, name=f"block_y_col_{col}", depth=2)
        a_cols.append(a_col); x_cols.append(x_col); y_cols.append(y_col)
        a_cores = a_col.cons().split([r * 32 * block_width * 2 for r in range(rows)], tile=mem, obj_types=[l1_a_ty] * rows, names=[f"block_a_{col}_{r}" for r in range(rows)])
        y_cores = y_col.prod().join([r * 32 for r in range(rows)], tile=mem, obj_types=[l1_y_ty] * rows, names=[f"block_y_{col}_{r}" for r in range(rows)])
        def core_body(a_fifo, x_fifo, y_fifo, block_kernel):
            x = x_fifo.acquire(1)
            for _ in range_(iterations):
                y = y_fifo.acquire(1)
                for h in range_(horizontal_blocks):
                    a = a_fifo.acquire(1)
                    block_kernel(block_width, index.casts(T.i32(), h), a, x, y)
                    a_fifo.release(1)
                y_fifo.release(1)
            x_fifo.release(1)
        for row in range(rows): workers.append(Worker(core_body, [a_cores[row].cons(), x_col.cons(), y_cores[row].prod(), kernel], tile=Tile(col, 2 + row)))
    # Source is [row-block][slot][index/value][lane].  This 4-D TAP emits one
    # L2 object per horizontal block in the order expected by split(): slot block,
    # then its four core-row payloads.
    words_per_payload = 32 * block_width * 2
    words_per_row_block = 32 * ell_width * 2
    words_per_time = rows * words_per_row_block
    # A tap's fourth DMA iteration dimension is limited to 64.  Preserve the
    # Phase-0 workaround: split the time dimension into <=64-iteration taps
    # and dispatch four chunks per TaskGroup.
    max_dma_iterations = 64
    a_taps = []
    for col in range(cols):
        base = col * blocks_per_col * words_per_row_block
        col_taps = []
        for first in range(0, iterations, max_dma_iterations):
            chunk_iterations = min(max_dma_iterations, iterations - first)
            col_taps.append(
                TensorAccessPattern(
                    l3_a_ty.__args__[0],
                    base + first * words_per_time,
                    [chunk_iterations, horizontal_blocks, rows, words_per_payload],
                    [words_per_time, words_per_payload, words_per_row_block, 1],
                )
            )
        a_taps.append(col_taps)
    x_tap = TensorAccessPattern(l3_x_ty.__args__[0], 0, [1, 1, 1, K], [0, 0, 0, 1])
    y_taps = [TensorAccessPattern(l3_y_ty.__args__[0], col * blocks_per_col * 32, [1, 1, 1, blocks_per_col * 32], [0, 0, 0, 1]) for col in range(cols)]
    def sequence(A, X, Y, a_prods, x_prods, y_conss):
        # The old block design deliberately issued the one-shot x broadcasts
        # and output drains first, then fed A in bounded batches.  Besides
        # avoiding a d3>64 BD, this overlaps x/y with the streamed blocks.
        tg_xy = TaskGroup()
        for col in range(cols):
            x_prods[col].fill(X, x_tap, group=tg_xy, wait=False)
            y_conss[col].drain(Y, y_taps[col], group=tg_xy, wait=True)
        chunks_per_group = 4
        for chunk in range(len(a_taps[0])):
            if chunk % chunks_per_group == 0:
                tg_a = TaskGroup()
            for col in range(cols):
                a_prods[col].fill(A, a_taps[col][chunk], group=tg_a, wait=True)
            if (chunk + 1) % chunks_per_group == 0 or chunk + 1 == len(a_taps[0]):
                tg_a.finish()
        tg_xy.finish()
    runtime = Runtime(sequence, [l3_a_ty, l3_x_ty, l3_y_ty, [f.prod() for f in a_cols], [f.prod() for f in x_cols], [f.cons() for f in y_cols]])
    prog = Program(dev, runtime, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()


def spmv_slice_ell_static(
    dev,
    M,
    K,
    blocks_per_slice,
    rows=4,
    cols=8,
    block_height=32,
    block_width=256,
    trace_size=0,
    func_prefix="",
):
    """Build the Phase-3 horizontal Slice-ELL kernel for a uniform block count.

    ``blocks_per_slice`` is deliberately static and limited to one or two in
    this first device design.  The worker acquires all of a slice's A objects
    before calling one kernel, so the compiler can keep its eight FP32 vector
    accumulators live across the complete slice.  Phase 4 replaces this fixed ABI with
    the ragged ``blocks_per_slice[s]`` control path.
    """
    if rows < 1 or rows > 4 or block_height % rows:
        raise ValueError("block_height must be divisible by 1..4 core rows")
    if blocks_per_slice not in (1, 2):
        raise ValueError("Phase-3 static Slice-ELL supports blocks_per_slice=1 or 2")
    if block_width != 256 or block_width % 32:
        raise ValueError("the first horizontal kernel uses block_width=256")
    core_height = block_height // rows
    if core_height != 8:
        raise ValueError("the first horizontal kernel is specialized for C_h=8")
    if M % (block_height * cols):
        raise ValueError("M must be divisible by block_height * cols")
    if K <= 0:
        raise ValueError("K must be positive")

    dtype = np.dtype[bfloat16]
    words_per_core_block = core_height * block_width * 2
    words_per_slice_block = block_height * block_width * 2
    l1_a_ty = np.ndarray[(words_per_core_block,), dtype]
    l1_x_ty = np.ndarray[(K,), dtype]
    l1_y_ty = np.ndarray[(core_height,), dtype]
    l2_a_ty = np.ndarray[(words_per_slice_block,), dtype]
    l2_y_ty = np.ndarray[(block_height,), dtype]
    slices_per_column = M // (block_height * cols)
    l3_a_ty = np.ndarray[(M * blocks_per_slice * block_width * 2,), dtype]
    l3_x_ty = np.ndarray[(K,), dtype]
    l3_y_ty = np.ndarray[(M,), dtype]

    kernel_args = [l1_a_ty, l1_x_ty, l1_y_ty]
    if blocks_per_slice == 2:
        # p=2 deliberately passes both distinct FIFO objects to one function:
        # this is the A-plan register-residency experiment, not a BF16 y
        # read/modify/write between horizontal blocks.
        kernel_args = [l1_a_ty, l1_a_ty, l1_x_ty, l1_y_ty]
    kernel = Kernel(
        f"{func_prefix}slice_ell_horizontal_p{blocks_per_slice}_bf16",
        f"{func_prefix}spmv_ell.o",
        kernel_args,
    )

    a_cols, x_cols, y_cols, workers = [], [], [], []
    for col in range(cols):
        mem = Tile(col, 1)
        a_col = ObjectFifo(l2_a_ty, name=f"slice_a_col_{col}", depth=blocks_per_slice)
        x_col = ObjectFifo(l1_x_ty, name=f"slice_x_col_{col}", depth=1)
        y_col = ObjectFifo(l2_y_ty, name=f"slice_y_col_{col}", depth=2)
        a_cols.append(a_col)
        x_cols.append(x_col)
        y_cols.append(y_col)
        a_cores = a_col.cons().split(
            [r * words_per_core_block for r in range(rows)],
            tile=mem,
            depths=[blocks_per_slice] * rows,
            obj_types=[l1_a_ty] * rows,
            names=[f"slice_a_{col}_{r}" for r in range(rows)],
        )
        y_cores = y_col.prod().join(
            [r * core_height for r in range(rows)],
            tile=mem,
            obj_types=[l1_y_ty] * rows,
            names=[f"slice_y_{col}_{r}" for r in range(rows)],
        )

        def core_body(a_fifo, x_fifo, y_fifo, slice_kernel):
            x = x_fifo.acquire(1)
            for _ in range_(slices_per_column):
                a = a_fifo.acquire(blocks_per_slice)
                y = y_fifo.acquire(1)
                if blocks_per_slice == 1:
                    slice_kernel(a, x, y)
                else:
                    slice_kernel(a[0], a[1], x, y)
                a_fifo.release(blocks_per_slice)
                y_fifo.release(1)
            x_fifo.release(1)

        for row in range(rows):
            workers.append(
                Worker(
                    core_body,
                    [a_cores[row].cons(), x_col.cons(), y_cores[row].prod(), kernel],
                    tile=Tile(col, 2 + row),
                    # The horizontal A-plan uses eight live 1024-bit FP32
                    # accumulators.  Let aiecc reserve the measured 1664 B
                    # frame instead of rejecting the 1 KiB device default.
                    stack_size=2048,
                )
            )

    a_taps = [
        TensorAccessPattern(
            l3_a_ty.__args__[0],
            col * slices_per_column * blocks_per_slice * words_per_slice_block,
            [slices_per_column, blocks_per_slice, rows, words_per_core_block],
            [
                blocks_per_slice * words_per_slice_block,
                words_per_slice_block,
                words_per_core_block,
                1,
            ],
        )
        for col in range(cols)
    ]
    x_tap = TensorAccessPattern(l3_x_ty.__args__[0], 0, [1, 1, 1, K], [0, 0, 0, 1])
    y_taps = [
        TensorAccessPattern(
            l3_y_ty.__args__[0],
            col * slices_per_column * block_height,
            [1, 1, 1, slices_per_column * block_height],
            [0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    def sequence(A, X, Y, a_prods, x_prods, y_conss):
        task_group = TaskGroup()
        for col in range(cols):
            a_prods[col].fill(A, a_taps[col], group=task_group)
            x_prods[col].fill(X, x_tap, group=task_group)
        for col in range(cols):
            y_conss[col].drain(Y, y_taps[col], group=task_group, wait=True)
        task_group.finish()

    runtime = Runtime(
        sequence,
        [
            l3_a_ty,
            l3_x_ty,
            l3_y_ty,
            [fifo.prod() for fifo in a_cols],
            [fifo.prod() for fifo in x_cols],
            [fifo.cons() for fifo in y_cols],
        ],
    )
    prog = Program(dev, runtime, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()


def spmv_slice_ell_dynamic_scalar(dev, M, K, total_blocks, trace_size=0, func_prefix=""):
    """Phase-4 micro design: one column, four cores, runtime p and FP32 scalar state.

    ``total_blocks`` is the sum of the per-slice ``blocks_per_slice`` values.
    The runtime config object is ``[BF16 x bits | uint16 p]`` represented as
    int16 words.  This intentionally stays at one column until the dynamic
    acquire/release and fixed y drain contract is proven on hardware.
    """
    rows, cols, block_height, block_width = 4, 1, 32, 256
    core_height = block_height // rows
    if M <= 0 or M % block_height or K <= 0 or total_blocks < 0:
        raise ValueError("M must be divisible by 32; K and total_blocks must be non-negative")

    slices = M // block_height
    dtype = np.dtype[bfloat16]
    config_dtype = np.dtype[np.int16]
    l1_a_ty = np.ndarray[(core_height * block_width * 2,), dtype]
    l2_a_ty = np.ndarray[(block_height * block_width * 2,), dtype]
    l1_config_ty = np.ndarray[(K + slices,), config_dtype]
    l1_state_ty = np.ndarray[(core_height,), np.dtype[np.float32]]
    l1_y_ty = np.ndarray[(core_height,), dtype]
    l3_a_ty = np.ndarray[(total_blocks * block_height * block_width * 2,), dtype]
    l3_config_ty = np.ndarray[(K + slices,), config_dtype]
    l3_y_ty = np.ndarray[(M,), dtype]

    init_kernel = Kernel(
        f"{func_prefix}slice_ell_scalar_state_init",
        f"{func_prefix}spmv_ell.o",
        [l1_state_ty],
    )
    accumulate_kernel = Kernel(
        f"{func_prefix}slice_ell_scalar_state_accumulate_bf16",
        f"{func_prefix}spmv_ell.o",
        [l1_a_ty, l1_config_ty, l1_state_ty],
    )
    finalize_kernel = Kernel(
        f"{func_prefix}slice_ell_scalar_state_finalize_bf16",
        f"{func_prefix}spmv_ell.o",
        [l1_state_ty, l1_y_ty],
    )

    mem = Tile(0, 1)
    a_col = ObjectFifo(l2_a_ty, name="dynamic_scalar_a", depth=2)
    config_col = ObjectFifo(l1_config_ty, name="dynamic_scalar_config", depth=1)
    y_col = ObjectFifo(np.ndarray[(block_height,), dtype], name="dynamic_scalar_y", depth=2)
    a_cores = a_col.cons().split(
        [r * core_height * block_width * 2 for r in range(rows)],
        tile=mem,
        depths=[2] * rows,
        obj_types=[l1_a_ty] * rows,
        names=[f"dynamic_scalar_a_{r}" for r in range(rows)],
    )
    y_cores = y_col.prod().join(
        [r * core_height for r in range(rows)],
        tile=mem,
        obj_types=[l1_y_ty] * rows,
        names=[f"dynamic_scalar_y_{r}" for r in range(rows)],
    )

    workers = []
    for row in range(rows):
        state = Buffer(l1_state_ty, name=f"dynamic_scalar_state_{row}")

        def core_body(a_fifo, config_fifo, y_fifo, state_buf, init, accumulate, finalize):
            config = config_fifo.acquire(1)
            # The surrounding slice count is static, while p is data-dependent.
            # Emit one dynamic ObjectFIFO loop per slice so the config offset is
            # a compile-time constant and p=0 naturally acquires no A object.
            for local_slice in range(slices):
                init(state_buf)
                p_word = memref.load(config, [index.constant(K + local_slice)])
                p = index.casts(T.index(), p_word)
                for _ in range_(p):
                    a = a_fifo.acquire(1)
                    accumulate(a, config, state_buf)
                    a_fifo.release(1)
                y = y_fifo.acquire(1)
                finalize(state_buf, y)
                y_fifo.release(1)
            config_fifo.release(1)

        workers.append(
            Worker(
                core_body,
                [
                    a_cores[row].cons(),
                    config_col.cons(),
                    y_cores[row].prod(),
                    state,
                    init_kernel,
                    accumulate_kernel,
                    finalize_kernel,
                ],
                tile=Tile(0, 2 + row),
                stack_size=2048,
                dynamic_objfifo_lowering=True,
            )
        )

    a_words = total_blocks * block_height * block_width * 2
    config_words = K + slices
    a_tap = TensorAccessPattern([a_words], 0, [1, 1, 1, a_words], [0, 0, 0, 1])
    config_tap = TensorAccessPattern(
        [config_words], 0, [1, 1, 1, config_words], [0, 0, 0, 1]
    )
    y_tap = TensorAccessPattern([M], 0, [1, 1, 1, M], [0, 0, 0, 1])

    def sequence(A, config, Y, a_prod, config_prod, y_cons):
        tasks = TaskGroup()
        a_prod.fill(A, a_tap, group=tasks)
        config_prod.fill(config, config_tap, group=tasks)
        y_cons.drain(Y, y_tap, group=tasks, wait=True)
        tasks.finish()

    runtime = Runtime(
        sequence,
        [l3_a_ty, l3_config_ty, l3_y_ty, a_col.prod(), config_col.prod(), y_col.cons()],
    )
    prog = Program(dev, runtime, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()


def spmv_slice_ell_dynamic_scalar_multicol(
    dev, M, K, blocks_per_column, trace_size=0, func_prefix=""
):
    """Phase-4 scalar-state Slice-ELL with four core rows in every active column.

    ``blocks_per_column[c]`` is the exact number of fixed-size A objects sent
    to column ``c``.  A counts may differ, while the number of y slice objects
    is identical across columns; this preserves the static join/drain contract.
    """
    rows, block_height, block_width = 4, 32, 256
    core_height = block_height // rows
    blocks_per_column = tuple(int(n) for n in blocks_per_column)
    cols = len(blocks_per_column)
    if not 1 <= cols <= 8 or M <= 0 or M % (block_height * cols) or K <= 0:
        raise ValueError("cols must be 1..8 and M divisible by 32*cols")
    if any(n < 0 for n in blocks_per_column):
        raise ValueError("blocks_per_column entries must be non-negative")

    slices_per_column = M // (block_height * cols)
    config_words = K + slices_per_column
    total_blocks = sum(blocks_per_column)
    dtype = np.dtype[bfloat16]
    config_dtype = np.dtype[np.int16]
    l1_a_ty = np.ndarray[(core_height * block_width * 2,), dtype]
    l2_a_ty = np.ndarray[(block_height * block_width * 2,), dtype]
    l1_config_ty = np.ndarray[(config_words,), config_dtype]
    l1_state_ty = np.ndarray[(core_height,), np.dtype[np.float32]]
    l1_y_ty = np.ndarray[(core_height,), dtype]
    l3_a_ty = np.ndarray[(total_blocks * block_height * block_width * 2,), dtype]
    l3_config_ty = np.ndarray[(cols * config_words,), config_dtype]
    l3_y_ty = np.ndarray[(M,), dtype]

    init_kernel = Kernel(
        f"{func_prefix}slice_ell_scalar_state_init", f"{func_prefix}spmv_ell.o", [l1_state_ty]
    )
    accumulate_kernel = Kernel(
        f"{func_prefix}slice_ell_scalar_state_accumulate_bf16",
        f"{func_prefix}spmv_ell.o",
        [l1_a_ty, l1_config_ty, l1_state_ty],
    )
    finalize_kernel = Kernel(
        f"{func_prefix}slice_ell_scalar_state_finalize_bf16", f"{func_prefix}spmv_ell.o", [l1_state_ty, l1_y_ty]
    )

    a_cols, config_cols, y_cols, workers = [], [], [], []
    for col in range(cols):
        mem = Tile(col, 1)
        a_col = ObjectFifo(l2_a_ty, name=f"dynamic_scalar_a_col_{col}", depth=2)
        config_col = ObjectFifo(l1_config_ty, name=f"dynamic_scalar_config_col_{col}", depth=1)
        y_col = ObjectFifo(np.ndarray[(block_height,), dtype], name=f"dynamic_scalar_y_col_{col}", depth=2)
        a_cols.append(a_col)
        config_cols.append(config_col)
        y_cols.append(y_col)
        a_cores = a_col.cons().split(
            [r * core_height * block_width * 2 for r in range(rows)], tile=mem,
            depths=[2] * rows, obj_types=[l1_a_ty] * rows,
            names=[f"dynamic_scalar_a_{col}_{r}" for r in range(rows)],
        )
        y_cores = y_col.prod().join(
            [r * core_height for r in range(rows)], tile=mem, obj_types=[l1_y_ty] * rows,
            names=[f"dynamic_scalar_y_{col}_{r}" for r in range(rows)],
        )
        for row in range(rows):
            state = Buffer(l1_state_ty, name=f"dynamic_scalar_state_{col}_{row}")

            def core_body(a_fifo, config_fifo, y_fifo, state_buf, init, accumulate, finalize):
                config = config_fifo.acquire(1)
                for local_slice in range(slices_per_column):
                    init(state_buf)
                    p_word = memref.load(config, [index.constant(K + local_slice)])
                    p = index.casts(T.index(), p_word)
                    for _ in range_(p):
                        a = a_fifo.acquire(1)
                        accumulate(a, config, state_buf)
                        a_fifo.release(1)
                    y = y_fifo.acquire(1)
                    finalize(state_buf, y)
                    y_fifo.release(1)
                config_fifo.release(1)

            workers.append(Worker(
                core_body,
                [a_cores[row].cons(), config_col.cons(), y_cores[row].prod(), state,
                 init_kernel, accumulate_kernel, finalize_kernel],
                tile=Tile(col, 2 + row), stack_size=2048, dynamic_objfifo_lowering=True,
            ))

    words_per_block = block_height * block_width * 2
    a_taps, config_taps, y_taps = [], [], []
    a_offset = 0
    for col, count in enumerate(blocks_per_column):
        if count == 0:
            raise ValueError("the first multicolumn implementation does not support an empty A column")
        a_words = count * words_per_block
        a_taps.append(TensorAccessPattern([total_blocks * words_per_block], a_offset, [1, 1, 1, a_words], [0, 0, 0, 1]))
        config_taps.append(TensorAccessPattern([cols * config_words], col * config_words, [1, 1, 1, config_words], [0, 0, 0, 1]))
        y_taps.append(TensorAccessPattern([M], col * slices_per_column * block_height, [1, 1, 1, slices_per_column * block_height], [0, 0, 0, 1]))
        a_offset += a_words

    def sequence(A, config, Y, a_prods, config_prods, y_conss):
        tasks = TaskGroup()
        for col in range(cols):
            a_prods[col].fill(A, a_taps[col], group=tasks)
            config_prods[col].fill(config, config_taps[col], group=tasks)
        for col in range(cols):
            y_conss[col].drain(Y, y_taps[col], group=tasks, wait=True)
        tasks.finish()

    runtime = Runtime(
        sequence,
        [l3_a_ty, l3_config_ty, l3_y_ty,
         [fifo.prod() for fifo in a_cols], [fifo.prod() for fifo in config_cols], [fifo.cons() for fifo in y_cols]],
    )
    prog = Program(dev, runtime, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
