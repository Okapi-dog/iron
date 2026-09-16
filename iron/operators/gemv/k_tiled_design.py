# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-row, K-tiled dense GEMV for a 1x4 or 4x8 NPU2 core rectangle.

This is deliberately separate from the upstream one-row GEMV operator.  The
upstream operator keeps the full input vector in every core's L1; this design
streams a small K tile and therefore supports Llama projection widths such as
K=11008.  A core owns two output rows, and reduces each K tile into two
FP32 scalar states held in L1 until the row is complete.
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile


def dense_gemv_k_tiled(dev, M, K, cols, k_tile=4096, func_prefix=""):
    """Build a row-partitioned GEMV with K-tiled A/x ObjectFIFO objects.

    The host provides ``A`` in ``[column][8 output rows][K tile]`` order and
    repeats every x K-tile once per output-row block.  Repetition keeps the
    state small (two FP32 values/core) and lets every core consume x in the
    same static order without requiring a full-K L1 resident vector.
    """
    rows, rows_per_core = 4, 2
    block_height = rows * rows_per_core
    if not 1 <= cols <= 8:
        raise ValueError("cols must be in [1, 8]")
    if M <= 0 or K <= 0 or M % (block_height * cols):
        raise ValueError("M must be divisible by 8*cols")
    if k_tile not in (1376, 4096):
        raise ValueError("the performance kernels support k_tile=1376 or 4096")

    output_blocks_per_col = M // (block_height * cols)
    # The final K object is zero-padded if needed.  This permits K=11008 with
    # a practical 4096-wide tile (three kernel calls), rather than forcing a
    # much smaller exact divisor such as 256 (43 calls).
    k_blocks = (K + k_tile - 1) // k_tile
    dtype = np.dtype[bfloat16]
    l1_a_ty = np.ndarray[(rows_per_core * k_tile,), dtype]
    l2_a_ty = np.ndarray[(block_height * k_tile,), dtype]
    l1_x_ty = np.ndarray[(k_tile,), dtype]
    l1_state_ty = np.ndarray[(rows_per_core,), np.dtype[np.float32]]
    l1_y_ty = np.ndarray[(rows_per_core,), dtype]
    l3_a_ty = np.ndarray[(cols * output_blocks_per_col * k_blocks * block_height * k_tile,), dtype]
    # x is repeated for every output block in every column (rather than
    # retaining all K elements in L1).  It adds only 1/8 of A's BF16 traffic.
    l3_x_ty = np.ndarray[(cols * output_blocks_per_col * k_blocks * k_tile,), dtype]
    l3_y_ty = np.ndarray[(M,), dtype]

    init = Kernel(
        f"{func_prefix}dense_gemv_k_tiled_init_2", f"{func_prefix}dense_gemv_k_tiled.o",
        [l1_state_ty],
    )
    accumulate_name = {
        1376: "dense_gemv_k_tiled_accumulate_2x1376_bf16",
        4096: "dense_gemv_k_tiled_accumulate_2x4096_bf16",
    }[k_tile]
    accumulate = Kernel(
        f"{func_prefix}{accumulate_name}",
        f"{func_prefix}dense_gemv_k_tiled.o", [l1_a_ty, l1_x_ty, l1_state_ty],
    )
    finalize = Kernel(
        f"{func_prefix}dense_gemv_k_tiled_finalize_2", f"{func_prefix}dense_gemv_k_tiled.o",
        [l1_state_ty, l1_y_ty],
    )

    a_cols, x_cols, y_cols, workers = [], [], [], []
    for col in range(cols):
        mem = Tile(col, 1)
        a_col = ObjectFifo(l2_a_ty, name=f"dense_k_tiled_a_col_{col}", depth=2)
        x_col = ObjectFifo(l1_x_ty, name=f"dense_k_tiled_x_col_{col}", depth=1)
        y_col = ObjectFifo(np.ndarray[(block_height,), dtype], name=f"dense_k_tiled_y_col_{col}", depth=2)
        a_cols.append(a_col)
        x_cols.append(x_col)
        y_cols.append(y_col)
        a_cores = a_col.cons().split(
            [r * rows_per_core * k_tile for r in range(rows)], tile=mem,
            depths=[2] * rows, obj_types=[l1_a_ty] * rows,
            names=[f"dense_k_tiled_a_{col}_{r}" for r in range(rows)],
        )
        y_cores = y_col.prod().join(
            [r * rows_per_core for r in range(rows)], tile=mem,
            obj_types=[l1_y_ty] * rows,
            names=[f"dense_k_tiled_y_{col}_{r}" for r in range(rows)],
        )

        for row in range(rows):
            state = Buffer(l1_state_ty, name=f"dense_k_tiled_state_{col}_{row}")

            def core_body(a_fifo, x_fifo, y_fifo, state_buf, init_kernel, accumulate_kernel, finalize_kernel):
                for _ in range_(output_blocks_per_col):
                    init_kernel(state_buf)
                    for _ in range_(k_blocks):
                        a = a_fifo.acquire(1)
                        x = x_fifo.acquire(1)
                        accumulate_kernel(a, x, state_buf)
                        a_fifo.release(1)
                        x_fifo.release(1)
                    y = y_fifo.acquire(1)
                    finalize_kernel(state_buf, y)
                    y_fifo.release(1)

            workers.append(Worker(
                core_body,
                [a_cores[row].cons(), x_col.cons(), y_cores[row].prod(), state,
                 init, accumulate, finalize],
                tile=Tile(col, 2 + row), stack_size=2048, dynamic_objfifo_lowering=True,
            ))

    a_words_per_col = output_blocks_per_col * k_blocks * block_height * k_tile
    x_words_per_col = output_blocks_per_col * k_blocks * k_tile
    y_words_per_col = output_blocks_per_col * block_height
    a_taps = [TensorAccessPattern([cols * a_words_per_col], col * a_words_per_col,
                                  [1, 1, 1, a_words_per_col], [0, 0, 0, 1])
              for col in range(cols)]
    x_taps = [TensorAccessPattern([cols * x_words_per_col], col * x_words_per_col,
                                  [1, 1, 1, x_words_per_col], [0, 0, 0, 1])
              for col in range(cols)]
    y_taps = [TensorAccessPattern([M], col * y_words_per_col,
                                  [1, 1, 1, y_words_per_col], [0, 0, 0, 1])
              for col in range(cols)]

    def sequence(A, X, Y, a_prods, x_prods, y_conss):
        tasks = TaskGroup()
        for col in range(cols):
            a_prods[col].fill(A, a_taps[col], group=tasks)
            x_prods[col].fill(X, x_taps[col], group=tasks)
        for col in range(cols):
            y_conss[col].drain(Y, y_taps[col], group=tasks, wait=True)
        tasks.finish()

    runtime = Runtime(
        sequence,
        [l3_a_ty, l3_x_ty, l3_y_ty,
         [fifo.prod() for fifo in a_cols], [fifo.prod() for fifo in x_cols],
         [fifo.cons() for fifo in y_cols]],
    )
    return Program(dev, runtime, workers=workers).resolve_program()
