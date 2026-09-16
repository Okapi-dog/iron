# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aie import T
import aie.dialects.index as index
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile


def spmv_ell(dev, M, K, ell_width, rows, cols, rows_per_core):
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
        "sparse_matvec_ell_bf16",
        "spmv_ell.o",
        [np.int32, np.int32, l1_a_ty, l1_x_ty, l1_y_ty],
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
                spmv_kernel(rows_per_core, ell_width, a, x, y)
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
        tx = TaskGroup()
        for col in range(cols):
            x_prods[col].fill(X, x_tap, group=tx)
        tx.finish()
        ta = TaskGroup()
        for col in range(cols):
            a_prods[col].fill(A, a_taps[col], group=ta)
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
    return Program(dev, runtime, workers=workers).resolve_program()


def spmv_sell32(dev, M, K, ell_width, rows, cols):
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
    kernel = Kernel("sell32_spmv_bf16", "spmv_ell.o", [np.int32, l1_a_ty, l1_x_ty, l1_y_ty])
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
        tx = TaskGroup()
        for col in range(cols): x_prods[col].fill(X, x_tap, group=tx)
        tx.finish()
        ta = TaskGroup()
        for col in range(cols):
            a_prods[col].fill(A, a_taps[col], group=ta)
            y_conss[col].drain(Y, y_taps[col], group=ta, wait=True)
        ta.finish()
    runtime = Runtime(sequence, [l3_a_ty, l3_x_ty, l3_y_ty, [f.prod() for f in a_cols], [f.prod() for f in x_cols], [f.cons() for f in y_cols]])
    return Program(dev, runtime, workers=workers).resolve_program()
