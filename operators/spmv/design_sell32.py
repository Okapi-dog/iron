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

def my_matvec(dev, M, K, ell_width, m, num_core_rows, num_core_cols, trace_ddr_id=None, trace_size=65536):
    vectorized = True 
    dtype_in = np.dtype[bfloat16]
    dtype_in_str = "bf16"
    dtype_out = np.dtype[bfloat16]
    dtype_out_str = "bf16"
    # 設定
    BLOCK_SIZE = 32 # SELL-32の固定値 
    num_total_cores = num_core_cols * num_core_rows
    
    # 全体のブロック数
    total_blocks = M // BLOCK_SIZE
    
    # チェック: 全ブロック数がコア数とmで割り切れること
    assert M % BLOCK_SIZE == 0, f"M must be a multiple of {BLOCK_SIZE}"
    assert total_blocks % (num_total_cores * m) == 0, "Total blocks must be divisible by (cores * m)"

    #Kが32の倍数ないと、アライメントでエラーになる。原因は不明。
    assert K % 32 == 0, "K must be a multiple of 32. if not, this causes output miscalculation. Cause of this problem is unknown."

    if dev == "npu" or isinstance(dev, NPU1):
        dev_ty = NPU1()
    elif dev == "npu2" or isinstance(dev, NPU2):
        dev_ty = NPU2()
    else:
        raise AssertionError(f"Unsupported device type: {dev}")

    # --- 型定義 (mはブロック数なので、要素数は m * BLOCK_SIZE で計算) ---
    
    # L1 (Core Local): mブロック分
    # 行列Aの1ブロックあたりの要素数 = BLOCK_SIZE * ell_width * 2 (indices + values)
    L1_A_ty = np.ndarray[(m * BLOCK_SIZE * ell_width * 2,), dtype_in]
    L1_B_ty = np.ndarray[(K,), dtype_in]
    L1_C_ty = np.ndarray[(m * BLOCK_SIZE,), dtype_out]

    # L2 (MemTile): 1列の全コアが1ステップで持つ量
    L2_A_ty = np.ndarray[(m * num_core_rows * BLOCK_SIZE * ell_width * 2,), dtype_in]
    L2_B_ty = np.ndarray[(K,), dtype_in]
    L2_C_ty = np.ndarray[(m * num_core_rows * BLOCK_SIZE,), dtype_out]

    # L3 (Global): 全体
    L3_A_ty = np.ndarray[(M * ell_width * 2,), dtype_in]
    L3_B_ty = np.ndarray[(K,), dtype_in] 
    L3_C_ty = np.ndarray[(M,), dtype_out]

    # カーネル定義 (sell32_spmv_kernel)
    # 引数: [num_blocks(m), ell_width, A_ptr, B_ptr, C_ptr]
    matvec = Kernel(
        "sell32_spmv_vectorized_bf16_bf16",
        "mv.o",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )

    workers = []
    col_A_fifos = [] 
    col_B_fifos = [] 
    col_C_fifos = []
    all_A_taps = []
    all_C_taps = []

    # 各コアが担当する反復回数
    # 反復回数 = 全ブロック / (全コア数 * 1回あたりのブロック数m)
    iter_count = total_blocks // (num_total_cores * m)
    assert total_blocks % (num_total_cores * m) == 0, "Total blocks must be divisible by (total cores * m)"

    for col_idx in range(num_core_cols):
        shim_tile = Tile(col_idx, 0)
        mem_tile = Tile(col_idx, 1)

        # Vector B (Broadcast)
        of_B_col = ObjectFifo(L2_B_ty, name=f"B_col_{col_idx}", depth=1)
        col_B_fifos.append(of_B_col)
        
        # Matrix A (Split)
        of_A_col = ObjectFifo(L2_A_ty, name=f"A_col_{col_idx}", depth=2)
        col_A_fifos.append(of_A_col)

        A_core_size = m * BLOCK_SIZE * ell_width * 2
        A_split_offsets = [i * A_core_size for i in range(num_core_rows)]
        A_split_types = [L1_A_ty for _ in range(num_core_rows)]

        of_A_cores = of_A_col.cons().split(
            A_split_offsets,
            obj_types=A_split_types,
            placement=mem_tile,
            names=[f"A_core_{col_idx}_{r}" for r in range(num_core_rows)],
        )
        
        # Output C (Join)
        of_C_col = ObjectFifo(L2_C_ty, name=f"C_col_{col_idx}", depth=2)
        col_C_fifos.append(of_C_col)
        
        C_core_size = m * BLOCK_SIZE
        C_split_offsets = [i * C_core_size for i in range(num_core_rows)]
        C_split_types = [L1_C_ty for _ in range(num_core_rows)]
        
        of_C_cores = of_C_col.prod().join(
            C_split_offsets,
            obj_types=C_split_types,
            placement=mem_tile,
            names=[f"C_core_{col_idx}_{r}" for r in range(num_core_rows)]
        )

        for r in range(num_core_rows):
            target_tile = Tile(col_idx, 2 + r)
            
            def core_body(A_fifo, B_fifo, C_fifo, matvec_kernel):
                b_local = B_fifo.acquire(1)
                
                for _ in range_(iter_count):
                    a_local = A_fifo.acquire(1)
                    c_local = C_fifo.acquire(1)
                    
                    # カーネル実行: mはそのまま「処理するブロック数」として渡される
                    matvec_kernel(m, ell_width, a_local, b_local, c_local)
                    
                    A_fifo.release(1)
                    C_fifo.release(1)
                
                B_fifo.release(1)

            workers.append(
                Worker(
                    core_body,
                    [of_A_cores[r].cons(), of_B_col.cons(), of_C_cores[r].prod(), matvec],
                    placement=target_tile
                )
            )

        # --- TAP定義 ---
        # A: この列が担当するブロック分をスライス
        # Aは (M, ell_width*2) の要素を持つ
        blocks_per_col = total_blocks // num_core_cols
        assert total_blocks % num_core_cols == 0, "Total blocks must be divisible by active columns"
        A_tap = TensorAccessPattern(
            (M, ell_width * 2),                                         #Aの全体形状
            col_idx * blocks_per_col * BLOCK_SIZE * ell_width * 2,      #offset
            [1, 1, 1, blocks_per_col * BLOCK_SIZE * ell_width * 2],     #size
            [0, 0, 0, 1]                                                #stride
        )
        all_A_taps.append(A_tap)
        # C: この列が担当する結果分をスライス
        C_tap = TensorAccessPattern(
            (1, M),
            col_idx * blocks_per_col * BLOCK_SIZE,
            [1, 1, 1, blocks_per_col * BLOCK_SIZE],
            [0, 0, 0, 1]
        )
        all_C_taps.append(C_tap)


    rt = Runtime()
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
    my_coremem_events=[
        MemEvent.DMA_S2MM_0_START_TASK,
        MemEvent.DMA_S2MM_0_FINISHED_BD,
        MemEvent.DMA_S2MM_1_FINISHED_BD,
        MemEvent.DMA_S2MM_0_FINISHED_TASK,
        MemEvent.DMA_S2MM_0_STALLED_LOCK,
        MemEvent.DMA_S2MM_0_STREAM_STARVATION,
        MemEvent.DMA_S2MM_0_MEMORY_BACKPRESSURE,
    ]
    my_shim_events_mm2s = [
        #下は全部データ送信の際のイベント
        #計算が遅くbufferが空いてない場合に発生する
        ShimTileEvent.DMA_MM2S_0_STALLED_LOCK,
        ShimTileEvent.DMA_MM2S_1_STALLED_LOCK,
        #shimdmaがデータを供給できずに待機している状態
        ShimTileEvent.DMA_MM2S_0_MEMORY_STARVATION,
        ShimTileEvent.DMA_MM2S_1_MEMORY_STARVATION,
        #送り先がbusy状態
        ShimTileEvent.DMA_MM2S_0_STREAM_BACKPRESSURE,
        ShimTileEvent.DMA_MM2S_1_STREAM_BACKPRESSURE,
        #データ送信の定期確認
        ShimTileEvent.DMA_S2MM_0_FINISHED_BD,
        ShimTileEvent.DMA_S2MM_1_FINISHED_BD,
    ]
    my_shim_events_s2mm = [
        #下は全部データ送信の際のイベント
        #計算が遅くbufferが空いてない場合に発生する
        ShimTileEvent.DMA_S2MM_0_STALLED_LOCK,
        ShimTileEvent.DMA_S2MM_1_STALLED_LOCK,
        #shimdmaがデータを供給できずに待機している状態
        ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
        ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
        #送り先がbusy状態
        ShimTileEvent.DMA_S2MM_0_MEMORY_BACKPRESSURE,
        ShimTileEvent.DMA_S2MM_1_MEMORY_BACKPRESSURE,
        #データ送信の定期確認
        ShimTileEvent.DMA_S2MM_0_FINISHED_BD,
        ShimTileEvent.DMA_S2MM_1_FINISHED_BD,
    ]
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
                workers=[workers[0]],
                coretile_events=my_core_events,
                shimtile_events=my_shim_events_mix,
                coremem_events=my_coremem_events,
                ddr_id=trace_ddr_id
            )
            
        rt.start(*workers)
        tg = rt.task_group()
        
        for col_idx in range(num_core_cols):
            shim_tile = Tile(col_idx, 0)
            
            # A: ストリーミング転送 (iter_count回分のデータが自動で流れる)
            rt.fill(
                col_A_fifos[col_idx].prod(),
                A,
                all_A_taps[col_idx],
                task_group=tg,
                wait=False,
                placement=shim_tile
            )
            
            # B: 1回だけ転送 (全コアがこれを保持する)
            rt.fill(
                col_B_fifos[col_idx].prod(),
                B,
                task_group=tg,
                wait=False,
                placement=shim_tile
            )

        for col_idx in range(num_core_cols):
            shim_tile = Tile(col_idx, 0)
            rt.drain(
                col_C_fifos[col_idx].cons(),
                C,
                all_C_taps[col_idx],
                task_group=tg,
                wait=True,
                placement=shim_tile
            )
            
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
    argparser.add_argument("--rows", type=int, default=4)
    argparser.add_argument("--cols", type=int, default=8)
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        default="aie.mlir",
        help="Output file path for the generated MLIR module",
    )
    args = argparser.parse_args()
    
    module = my_matvec(args.dev, args.M, args.K, args.ell_width, args.m, args.rows, args.cols)

    output_file_path = Path(args.output_file_path)
    with open(output_file_path, "w") as f:
        f.write(str(module))

if __name__ == "__main__":
    main()