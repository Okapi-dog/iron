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
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile
from aie.utils import trace as trace_utils
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent



def my_matvec(dev, num_cores, M, K, m, trace_ddr_id=None, trace_size=65536):
    vectorized = True
    dtype_in = np.dtype[bfloat16]
    dtype_in_str = "bf16"
    dtype_out = np.dtype[bfloat16]
    dtype_out_str = "bf16"

    assert M % num_cores == 0

    if dev == "npu" or isinstance(dev, NPU1):
        dev_ty = NPU1()
        device_cols=4
    elif dev == "npu2" or isinstance(dev, NPU2):
        dev_ty = NPU2()
        device_cols=8
    else:
        raise AssertionError(f"Unsupported device type: {dev}")
    L1_A_ty = np.ndarray[(m * K,), dtype_in]
    L1_B_ty = np.ndarray[(K,), dtype_in]
    L1_C_ty = np.ndarray[(M // num_cores,), dtype_out]
    L3_A_ty = np.ndarray[(M * K,), dtype_in]
    L3_B_ty = np.ndarray[(K,), dtype_in]
    L3_C_ty = np.ndarray[(M,), dtype_out]

    func_type = "vectorized" if vectorized else "scalar"
    matvec = Kernel(
        f"matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
        "mv.o",
        [np.int32, np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )

    A_L3L1_fifos = [ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}") for i in range(num_cores)]
    B_L3L1_fifos = [
        ObjectFifo(L1_B_ty, name=f"B_L3L1_{i}", depth=1) for i in range(num_cores)
    ]
    C_L1L3_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=1) for i in range(num_cores)
    ]

    def core_body(A_L3L1_fifo, B_L3L1_fifo, C_L1L3_fifo, matvec):
        one_idx = index.constant(1)
        m_idx = index.constant(m)
        for _ in range_(0xFFFFFFFF):
            b = B_L3L1_fifo.acquire(1)
            c = C_L1L3_fifo.acquire(1)
            for i_idx in range_(M // m // num_cores):
                a = A_L3L1_fifo.acquire(1)
                i_i32 = index.casts(T.i32(), i_idx)
                matvec(m, K, i_i32, a, b, c)
                A_L3L1_fifo.release(1)
            C_L1L3_fifo.release(1)
            B_L3L1_fifo.release(1)

    workers = []
    for num_core in range(num_cores):
        core_row_offset = 2
        
        # NPU2(8列)で i=9 の場合 -> Tile(1, 3)
        tile_row = num_core // device_cols + core_row_offset
        tile_col = num_core % device_cols
        
        if tile_row > 5:
             raise AssertionError(f"Requested num_cores {num_cores} exceeds device capacity. tile_row should be <= 5 ")

        workers.append(
            Worker(
                core_body,
                [
                    A_L3L1_fifos[num_core].cons(),
                    B_L3L1_fifos[num_core].cons(),
                    C_L1L3_fifos[num_core].prod(),
                    matvec,
                ],
                # GEMMと同じように物理配置を明示的に指定
                placement=Tile(tile_col, tile_row)
            )
        )

    A_taps = [
        TensorAccessPattern(
            (M, K),
            col * (M // num_cores) * K,
            [1, 1, 1, (M // num_cores) * K],
            [0, 0, 0, 1],
        )
        for col in range(num_cores)
    ]
    # Every column gets the whole of B, no TAP needed.
    C_taps = [
        TensorAccessPattern(
            (1, M), col * (M // num_cores), [1, 1, 1, (M // num_cores)], [0, 0, 0, 1]
        )
        for col in range(num_cores)
    ]

    rt = Runtime()
    my_core_events = [
        trace_utils.CoreEvent.INSTR_EVENT_0,  # event0()
        trace_utils.CoreEvent.INSTR_EVENT_1,  # event1()
        trace_utils.CoreEvent.INSTR_VECTOR,   # ベクトル命令実行
        trace_utils.CoreEvent.MEMORY_STALL,   # メモリストール (重要)
        trace_utils.CoreEvent.STREAM_STALL,   # ストリームストール
        trace_utils.CoreEvent.LOCK_STALL,     # ロックストール
        trace_utils.CoreEvent.ACTIVE,         # アクティブ状態
        trace_utils.CoreEvent.DISABLED        # 無効状態
    ]
    my_core_events = [
    # --- 既存の重要な項目 ---
    trace_utils.CoreEvent.INSTR_VECTOR,       # ベクトル演算 (主役)
    #trace_utils.CoreEvent.MEMORY_STALL,       # メモリ待ち
    trace_utils.CoreEvent.LOCK_STALL,         # ロック待ち
    
    
    # 1. 関数のオーバーヘッドを見る
    trace_utils.CoreEvent.INSTR_EVENT_0,       # EVENT0
    trace_utils.CoreEvent.INSTR_EVENT_1,       # EVENT1
    
    # 2. スカラ/スタック処理を見る (雑用の主犯)
    trace_utils.CoreEvent.INSTR_LOAD,         # スカラデータのロード (スタック操作など)
    trace_utils.CoreEvent.INSTR_STORE,        # スカラデータのストア (レジスタ退避など)
    
    # 3. データ転送命令を見る (Stallではなく命令実行時間)
    trace_utils.CoreEvent.INSTR_LOCK_ACQUIRE_REQ, # ロック取得命令そのもの
    trace_utils.CoreEvent.INSTR_LOCK_RELEASE_REQ, # ロック解放命令そのもの
    ]
    my_core_noevents = [
        CoreEvent.NONE
    ]
    my_shim_events = [

        ShimTileEvent.DMA_MM2S_0_START_TASK,
        ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
        ShimTileEvent.DMA_MM2S_1_START_TASK,
        ShimTileEvent.DMA_MM2S_1_FINISHED_TASK,
        ShimTileEvent.DMA_MM2S_0_MEMORY_STARVATION,
        ShimTileEvent.DMA_MM2S_1_MEMORY_STARVATION,
        ShimTileEvent.PORT_RUNNING_0, 
        ShimTileEvent.PORT_IDLE_0,
    ]
    my_coremem_events=[
        MemEvent.DMA_S2MM_0_START_TASK,
        MemEvent.DMA_S2MM_0_FINISHED_BD,
        MemEvent.DMA_S2MM_1_FINISHED_BD,
        MemEvent.DMA_S2MM_0_FINISHED_TASK,
        MemEvent.DMA_S2MM_0_STALLED_LOCK,
        MemEvent.DMA_S2MM_0_STREAM_STARVATION,
        MemEvent.DMA_S2MM_0_MEMORY_BACKPRESSURE,
    ]

    with rt.sequence(L3_A_ty, L3_B_ty, L3_C_ty) as (A, B, C):
        if trace_ddr_id is not None:
            rt.enable_trace(
                trace_size=trace_size,
                #workers=[w for w in [workers[0]] for _ in range(2)],
                workers=[workers[0],workers[0]],
                coretile_events=my_core_events,
                shimtile_events=my_shim_events,
                coremem_events=my_coremem_events,
                ddr_id=trace_ddr_id  # 指定されたIDを使う
            )
        rt.start(*workers)
        tg = rt.task_group()
        for num_core in range(num_cores):
            
            tile_col = num_core % device_cols
            shim_tile = Tile(tile_col, 0)

            rt.fill(
                A_L3L1_fifos[num_core].prod(), 
                A, 
                A_taps[num_core], 
                task_group=tg,
                placement=shim_tile
            )
            rt.fill(
                B_L3L1_fifos[num_core].prod(), 
                B, 
                task_group=tg,
                placement=shim_tile
            )

        for num_core in range(num_cores):
            tile_col = num_core % device_cols
            shim_tile = Tile(tile_col, 0)

            rt.drain(
                C_L1L3_fifos[num_core].cons(), 
                C, 
                C_taps[num_core], 
                task_group=tg, 
                wait=True,
                placement=shim_tile  # 物理配置を指定
            )
        rt.finish_task_group(tg)

    return Program(dev_ty, rt).resolve_program(SequentialPlacer())  #SequentalPlacer don't replace already placed workers


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Vector Multiplication MLIR Design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int)
    argparser.add_argument("-K", type=int)
    argparser.add_argument("-m", type=int)
    argparser.add_argument("--cols", type=int)
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        help="Output file path for the generated MLIR module",
    )
    args = argparser.parse_args()
    module = my_matvec(args.dev, args.cols, args.M, args.K, args.m)

    output_file_path = Path(args.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(module))


if __name__ == "__main__":
    main()
