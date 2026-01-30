# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16
import argparse

from aie.extras.context import mlir_mod_ctx
from aie.ir import StridedLayoutAttr, ShapedType
import aie.dialects.index as index
import aie.dialects.memref as memref
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.util import try_convert_np_type_to_mlir_type
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile
from aie.utils import trace as trace_utils
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent

def my_matvec(dev, num_cols, M, K, ell_width, m, trace_ddr_id=None, trace_size=65536):
    vectorized = True 
    dtype_in = np.dtype[bfloat16]
    dtype_in_str = "bf16"
    dtype_out = np.dtype[bfloat16]
    dtype_out_str = "bf16"

    # 設定
    active_cols = num_cols 
    cores_per_col = 4
    num_total_cores = active_cols * cores_per_col
    
    assert M % num_total_cores == 0, "M must be divisible by total number of cores"
    assert (M // num_total_cores) % m == 0, "Rows per core must be divisible by m"

    if dev == "npu" or isinstance(dev, NPU1):
        dev_ty = NPU1()
    elif dev == "npu2" or isinstance(dev, NPU2):
        dev_ty = NPU2()
    else:
        raise AssertionError(f"Unsupported device type: {dev}")
    
    rows_per_col_total = M // active_cols
    rows_per_step_col = m * cores_per_col
    iter_count = rows_per_col_total // rows_per_step_col
    AGG_FACTOR = 8
    assert iter_count % AGG_FACTOR == 0, "iter_count must be divisible by AGG_FACTOR"

    # --- 型定義 ---
    L1_A_ty = np.ndarray[(m * ell_width * 2,), dtype_in]
    L1_B_ty = np.ndarray[(K,), dtype_in]
    L1_C_ty = np.ndarray[(m*AGG_FACTOR,), dtype_out]

    L2_A_ty = np.ndarray[(m * cores_per_col * ell_width * 2,), dtype_in]
    L2_B_ty = np.ndarray[(K,), dtype_in]
    L2_C_ty = np.ndarray[(m * cores_per_col * AGG_FACTOR,), dtype_out]

    L3_A_ty = np.ndarray[(M * ell_width * 2,), dtype_in]
    L3_B_ty = np.ndarray[(K,), dtype_in] 
    L3_C_ty = np.ndarray[(M,), dtype_out]

    func_type = "vectorized" if vectorized else "scalar"
    matvec = Kernel(
        f"sparse_matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
        "mv.o",
        [np.int32, np.int32, np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )

    workers = []
    col_A_fifos = [] 
    col_B_fifos = [] 
    col_C_fifos = [] 
    all_A_taps = []
    # all_C_taps = [] # ← 削除: ループ内で動的に生成します

    for col_idx in range(active_cols):
        shim_tile = Tile(col_idx, 0)
        mem_tile = Tile(col_idx, 1)

        of_B_col = ObjectFifo(L2_B_ty, name=f"B_col_{col_idx}", depth=1)
        col_B_fifos.append(of_B_col)
        
        of_A_col = ObjectFifo(L2_A_ty, name=f"A_col_{col_idx}", depth=2)
        col_A_fifos.append(of_A_col)

        A_split_offsets = []
        A_split_types = []
        A_core_size = (m * ell_width * 2) 
        for i in range(cores_per_col):
            A_split_offsets.append(i * A_core_size)
            A_split_types.append(L1_A_ty)

        of_A_cores = of_A_col.cons().split(
            A_split_offsets,
            obj_types=A_split_types,
            placement=mem_tile,
            names=[f"A_core_{col_idx}_{r}" for r in range(cores_per_col)],
        )

        of_C_col = ObjectFifo(L2_C_ty, name=f"C_col_{col_idx}", depth=2)
        col_C_fifos.append(of_C_col)
        
        C_split_offsets = [i * m * AGG_FACTOR for i in range(cores_per_col)]
        C_split_types = [L1_C_ty for _ in range(cores_per_col)]
        
        of_C_cores = of_C_col.prod().join(
            C_split_offsets,
            obj_types=C_split_types,
            placement=mem_tile,
            names=[f"C_core_{col_idx}_{r}" for r in range(cores_per_col)]
        )

        for r in range(cores_per_col):
            target_tile = Tile(col_idx, 2 + r)
            
            def core_body(A_fifo, B_fifo, C_fifo, matvec_kernel):
                b_local = B_fifo.acquire(1)
                for _ in range_(iter_count // AGG_FACTOR):
                    c_local = C_fifo.acquire(1)
                    for i in range_(AGG_FACTOR):
                        a_local = A_fifo.acquire(1)
                        row_offset = index.casts(T.i32(), i)
                        matvec_kernel(m, K, ell_width, row_offset, a_local, b_local, c_local)
                        A_fifo.release(1)
                    C_fifo.release(1)
                B_fifo.release(1)

            workers.append(
                Worker(
                    core_body,
                    [
                        of_A_cores[r].cons(),
                        of_B_col.cons(),
                        of_C_cores[r].prod(),
                        matvec,
                    ],
                    placement=target_tile
                )
            )

        A_tap = TensorAccessPattern(
            (M, ell_width * 2),
            col_idx * rows_per_col_total * ell_width * 2,
            [1, 1, 1, rows_per_col_total * ell_width * 2],
            [0, 0, 0, 1]
        )
        all_A_taps.append(A_tap)
        
        # C_tapの静的生成は削除

    rt = Runtime()
    # (Event定義は省略 - 元のコードと同じ)
    my_shim_events_mix = [
        ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
        ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
        ShimTileEvent.DMA_S2MM_0_MEMORY_BACKPRESSURE,
        ShimTileEvent.DMA_S2MM_1_MEMORY_BACKPRESSURE,
        ShimTileEvent.DMA_MM2S_0_MEMORY_STARVATION,
        ShimTileEvent.DMA_MM2S_1_MEMORY_STARVATION,
        ShimTileEvent.DMA_MM2S_0_STREAM_BACKPRESSURE,
        ShimTileEvent.DMA_MM2S_1_STREAM_BACKPRESSURE,
    ]

    with rt.sequence(L3_A_ty, L3_B_ty, L3_C_ty) as (A, B, C):
        if trace_ddr_id is not None:
             rt.enable_trace(
                trace_size=trace_size,
                workers=[],
                shimtile_events=my_shim_events_mix,
                ddr_id=trace_ddr_id
            )
            
        rt.start(*workers)
        tg = rt.task_group()
        
        for col_idx in range(active_cols):
            shim_tile = Tile(col_idx, 0)
            rt.fill(
                col_A_fifos[col_idx].prod(),
                A,
                all_A_taps[col_idx],
                task_group=tg,
                placement=shim_tile
            )
            rt.fill(
                col_B_fifos[col_idx].prod(),
                B,
                task_group=tg,
                placement=shim_tile
            )

        # ★修正箇所: CのDrainを64反復ごとに分割して実行
        rows_per_step_total = m * cores_per_col * AGG_FACTOR
        total_outer_iters = rows_per_col_total // rows_per_step_total
        MAX_BD_ITER = 64 # ハードウェア制限

        for col_idx in range(active_cols):
            shim_tile = Tile(col_idx, 0)
            
            remaining_iters = total_outer_iters
            current_iter_offset = 0
            
            while remaining_iters > 0:
                # 64回以下になるようにチャンクサイズを決定
                chunk_size = min(remaining_iters, MAX_BD_ITER)
                
                # オフセット計算: (列の開始位置) + (現在のステップ数 * 1ステップのサイズ)
                chunk_offset = (col_idx * rows_per_col_total) + \
                               (current_iter_offset * rows_per_step_total)
                
                # 分割用のTAPを作成
                chunk_C_tap = TensorAccessPattern(
                    (M,),
                    chunk_offset,
                    [chunk_size, cores_per_col, AGG_FACTOR, m],
                    [rows_per_step_total, m, m * cores_per_col, 1]
                )

                rt.drain(
                    col_C_fifos[col_idx].cons(),
                    C,
                    chunk_C_tap,
                    task_group=tg,
                    wait=False,
                    placement=shim_tile
                )
                
                remaining_iters -= chunk_size
                current_iter_offset += chunk_size
            
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Vector Multiplication MLIR Design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=10000)
    argparser.add_argument("-K", type=int, default=10000)
    argparser.add_argument("-ell_width", type=int, default=64)
    argparser.add_argument("-m", type=int, default=32)
    argparser.add_argument("--cols", type=int, default=4)
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        default="aie.mlir",
        help="Output file path for the generated MLIR module",
    )
    args = argparser.parse_args()
    
    module = my_matvec(args.dev, args.cols, args.M, args.K, args.ell_width, args.m)

    output_file_path = Path(args.output_file_path)
    with open(output_file_path, "w") as f:
        f.write(str(module))

if __name__ == "__main__":
    main()