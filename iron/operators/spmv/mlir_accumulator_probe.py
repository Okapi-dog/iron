# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small IRON proof that an AIE vector accumulator can cross an SCF loop.

This is deliberately not a Slice-ELL kernel.  It has no indexed gather and no
MemTile split/join.  Its only purpose is to make the lowest-level question
testable on NPU2: can a Worker emit a vector BF16 MAC whose FP32 accumulator is
an ``scf.for`` loop-carried SSA value, rather than a C++ kernel-local value?
"""

import numpy as np
from ml_dtypes import bfloat16

from aie import ir
from aie.dialects import aievec, arith, vector
from aie.dialects.aie import T
import aie.dialects.index as index
from aie.helpers.dialects.scf import _for as range_, yield_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile


def mlir_accumulator_probe(dev):
    """Build two vector MACs with the FP32 vector carried by an AIE core loop."""
    lanes = 32
    input_ty = np.ndarray[(lanes,), np.dtype[bfloat16]]
    output_ty = np.ndarray[(lanes,), np.dtype[np.float32]]
    input_l3_ty = np.ndarray[(2 * lanes,), np.dtype[bfloat16]]

    a_fifo = ObjectFifo(input_ty, name="probe_a", depth=2)
    x_fifo = ObjectFifo(input_ty, name="probe_x", depth=1)
    y_fifo = ObjectFifo(output_ty, name="probe_y", depth=2)

    def core_body(a_cons, x_cons, y_prod):
        # Worker.resolve creates the core in its own MLIR context, so these
        # operation types must be materialized inside the Worker body.
        bf16x32 = T.vector(lanes, T.bf16())
        f32x32 = T.vector(lanes, T.f32())
        x = x_cons.acquire(1)
        zero_index = index.constant(0)
        # AIE2P currently does not lower aievec.broadcast_scalar for a 1024-bit
        # FP32 vector.  Materialize the all-zero vector as an SSA constant
        # instead; it is still a register candidate, not an L1 buffer.
        zero_attr = ir.DenseElementsAttr.get_splat(
            f32x32, ir.FloatAttr.get(T.f32(), 0.0)
        )
        zero_acc = arith.constant(f32x32, zero_attr)

        # ``acc`` is an SSA value of vector<32xf32>, not an L1 pointer.  The
        # explicit yield makes it the value used by the next loop iteration.
        for _, acc, final_acc in range_(2, iter_args=[zero_acc], insert_yield=False):
            a = a_cons.acquire(1)
            a_vec = vector.load(bf16x32, a, [zero_index])
            x_vec = vector.load(bf16x32, x, [zero_index])
            next_acc = aievec.mac_elem(f32x32, a_vec, x_vec, acc)
            a_cons.release(1)
            yield_([next_acc])

        y = y_prod.acquire(1)
        vector.store(final_acc, y, [zero_index])
        y_prod.release(1)
        x_cons.release(1)

    worker = Worker(
        core_body,
        [a_fifo.cons(), x_fifo.cons(), y_fifo.prod()],
        tile=Tile(0, 2),
        stack_size=2048,
    )

    a_tap = TensorAccessPattern([2 * lanes], 0, [1, 1, 1, 2 * lanes], [0, 0, 0, 1])
    x_tap = TensorAccessPattern([lanes], 0, [1, 1, 1, lanes], [0, 0, 0, 1])
    y_tap = TensorAccessPattern([lanes], 0, [1, 1, 1, lanes], [0, 0, 0, 1])

    def sequence(A, X, Y, a_prod, x_prod, y_cons):
        tasks = TaskGroup()
        a_prod.fill(A, a_tap, group=tasks)
        x_prod.fill(X, x_tap, group=tasks)
        y_cons.drain(Y, y_tap, group=tasks, wait=True)
        tasks.finish()

    runtime = Runtime(
        sequence,
        [input_l3_ty, input_ty, output_ty, a_fifo.prod(), x_fifo.prod(), y_fifo.cons()],
    )
    return Program(dev, runtime, workers=[worker]).resolve_program()
