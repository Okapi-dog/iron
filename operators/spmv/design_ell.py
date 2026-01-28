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
    cores_per_col = 4  # 1列あたりの並列コア数
    num_total_cores = active_cols * cores_per_col
    
    # 割り切れるかチェック
    assert M % num_total_cores == 0, "M must be divisible by total number of cores"
    # m (マイクロカーネルサイズ) でも割り切れるか確認
    assert (M // num_total_cores) % m == 0, "Rows per core must be divisible by m"

    if dev == "npu" or isinstance(dev, NPU1):
        dev_ty = NPU1()
    elif dev == "npu2" or isinstance(dev, NPU2):
        dev_ty = NPU2()
    else:
        raise AssertionError(f"Unsupported device type: {dev}")

    # --- 型定義 ---
    # ここが修正の肝です。
    # L1サイズを「全体」ではなく「1ステップの処理単位(m)」に小さくします。
    
    # L1 (Core Local): m行分 (約8KB @ m=32, ell=64)
    L1_A_ty = np.ndarray[(m * ell_width * 2,), dtype_in]
    # B: ベクトル全体 (K=10000 -> 20KB) ※NPU1(32KB)だとA,B,Stackでギリギリ
    L1_B_ty = np.ndarray[(K,), dtype_in]
    # C: 結果の一部 (m行分)
    L1_C_ty = np.ndarray[(m,), dtype_out]

    # L2 (MemTile): 1列の全コアが1ステップで処理する量 (m * 8行分)
    L2_A_ty = np.ndarray[(m * cores_per_col * ell_width * 2,), dtype_in]
    L2_B_ty = np.ndarray[(K,), dtype_in]
    L2_C_ty = np.ndarray[(m * cores_per_col,), dtype_out]

    # L3 (Global)
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
    all_C_taps = []

    # 各コアが何回ループすれば担当分(M/32)を処理しきれるか
    # 全行Mを、(列数 * 1列あたりの並列数8 * 1回の処理m) で割った回数
    # これにより Block-Cyclic な処理になります
    rows_per_col_total = M // active_cols
    rows_per_step_col = m * cores_per_col
    iter_count = rows_per_col_total // rows_per_step_col

    for col_idx in range(active_cols):
        shim_tile = Tile(col_idx, 0)
        mem_tile = Tile(col_idx, 1)

        # Vector B (Broadcast: K要素)
        # Bは大きいので depth=1 にしてメモリ節約（Broadcast+Reuseなので1でOKな場合が多い）
        of_B_col = ObjectFifo(L2_B_ty, name=f"B_col_{col_idx}", depth=1)
        col_B_fifos.append(of_B_col)
        
        #Matrix A (Split Pattern: m行 x 8コア)
        of_A_col = ObjectFifo(L2_A_ty, name=f"A_col_{col_idx}", depth=2)
        col_A_fifos.append(of_A_col)

        # MemTileのバッファ(m*8行)を、8個のFIFO(m行)に分割
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

        
        # 3. Output C (Join Pattern: m行 x 8コア)
        of_C_col = ObjectFifo(L2_C_ty, name=f"C_col_{col_idx}", depth=2)
        col_C_fifos.append(of_C_col)
        
        C_split_offsets = [i * m for i in range(cores_per_col)]
        C_split_types = [L1_C_ty for _ in range(cores_per_col)]
        
        of_C_cores = of_C_col.prod().join(
            C_split_offsets,
            obj_types=C_split_types,
            placement=mem_tile,
            names=[f"C_core_{col_idx}_{r}" for r in range(cores_per_col)]
        )

        # Workers
        for r in range(cores_per_col):
            target_tile = Tile(col_idx, 2 + r)
            
            def core_body(A_fifo, B_fifo, C_fifo, matvec_kernel):
                # ★修正: Bはループの外で1回だけ取得し、ずっと保持する (Reuseパターン)
                b_local = B_fifo.acquire(1)
                
                # AとCは小さく切って何度も流す
                for _ in range_(iter_count):
                    a_local = A_fifo.acquire(1)
                    c_local = C_fifo.acquire(1) # 出力バッファ取得
                    
                    # オフセットは常に0 (Splitされているため)
                    i_i32 = index.casts(T.i32(), index.constant(0))
                    
                    # カーネル実行 (m行分)
                    matvec_kernel(m, K, ell_width, i_i32, a_local, b_local, c_local)
                    
                    A_fifo.release(1) # Aは使い終わったので解放（次のデータが来る）
                    C_fifo.release(1) # Cは書き終わったので解放（MemTileへ送られる）
                
                # 全ての計算が終わったらBを解放
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

        # TAPs (Tensor Access Patterns)
        # A: 列ごとに担当するデータを切り出す (M/4 行分)
        # Runtimeが自動的にストリームしてくれるので、ここでは「列全体の範囲」を指定すればOK
        rows_per_col_total = M // active_cols
        A_tap = TensorAccessPattern(
            (M, ell_width * 2),
            col_idx * rows_per_col_total * ell_width * 2,
            [1, 1, 1, rows_per_col_total * ell_width * 2],
            [0, 0, 0, 1]
        )
        all_A_taps.append(A_tap)

        C_tap = TensorAccessPattern(
            (1, M),
            col_idx * rows_per_col_total,
            [1, 1, 1, rows_per_col_total],
            [0, 0, 0, 1]
        )
        all_C_taps.append(C_tap)


    rt = Runtime()
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
                workers=[],
                shimtile_events=my_shim_events_mix,
                ddr_id=trace_ddr_id
            )
            
        rt.start(*workers)
        tg = rt.task_group()
        
        for col_idx in range(active_cols):
            shim_tile = Tile(col_idx, 0)
            
            # A: ストリーミング転送 (iter_count回分のデータが自動で流れる)
            rt.fill(
                col_A_fifos[col_idx].prod(),
                A,
                all_A_taps[col_idx],
                task_group=tg,
                placement=shim_tile
            )
            
            # B: 1回だけ転送 (全コアがこれを保持する)
            rt.fill(
                col_B_fifos[col_idx].prod(),
                B,
                task_group=tg,
                placement=shim_tile
            )

        for col_idx in range(active_cols):
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