#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import numpy as np
import subprocess
import json
import torch
import os
import csv
import gc
import time
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.spmv.op import AIESPMV
from operators.gemv.op import AIEGEMV
from operators.spmv.reference import generate_reference_from_mtx
from operators.common.test_utils import run_test
from operators.spmv.save_sparse_matrix import save

# ==========================================
# 1. テスト設定
# ==========================================
design_name = "ell" # "ell" or "sell32" or "sell32_block"
tile_size = 1 #ellの場合64ぐらい。sellは1必須。


SRAM_LIMIT = 64 * 1024          #L1のサイズ64KB
PROG_RESERVED = 2 * 1024        #プログラム領域予約2KB    
DATA_LIMIT = SRAM_LIMIT - PROG_RESERVED     #データ領域制限62KB

# 計測したい構成リスト(行列名, タイルサイズ, コア行数, コア列数)
REGULAR_TEST_CONFIGS =[]
for M in [1024,2048,4096,8192,16384,28672,32768,65536,131072]:
    for K in [256, 512,1024,2048,4096,8192,16384,32768,65536,131072]:
        if M*K >32768*32768:# メモリ制限回避
            continue 
        # --- 1. メモリ制限に基づく最大 tile_size の計算 ---
        
        # Xベクトル (Kx1, bf16) は固定で乗る
        mem_x = K * 2
        
        # Xだけでメモリ不足ならスキップ
        if mem_x >= DATA_LIMIT:
            continue
            
        remaining_mem = DATA_LIMIT - mem_x
        ell_width = K // 8
        
        # 1行あたりの消費メモリ: A(double buffer) + Y
        # A: (width * 4bytes) * 2(buffer)
        # Y: 2bytes
        cost_per_row = (4 * ell_width * 2) + 2
        
        # メモリ的に許容される最大の行数
        max_possible_tile = remaining_mem // cost_per_row
        
        if max_possible_tile < 1:
            continue

        # --- 2. 制約 "M / tile_size / 32 が整数" を満たす tile_size の決定 ---
        
        # 式変形: tile_size = M / (32 * n)
        # 条件: tile_size <= max_possible_tile
        # よって: M / (32 * n) <= max_possible_tile
        #       n >= M / (32 * max_possible_tile)
        
        # 探索開始する最小の n を計算
        min_n = math.ceil(M / (32 * max_possible_tile))
        
        final_tile_size = 0
        
        # min_n から順に探索し、最初に割り切れる n を採用（＝最大のtile_size）
        # Mは最大でも131072程度なのでループ回数は知れている
        for n in range(min_n, M + 1):
            denominator = 32 * n
            
            # 割り切れるか確認 (M / tile_size / 32 が整数になるか)
            if M % denominator == 0:
                candidate_tile = M // denominator
                
                # 念のためメモリ制限チェック（min_nの計算上、基本は満たすはず）
                if candidate_tile <= max_possible_tile:
                    final_tile_size = candidate_tile
                    break
        
        if final_tile_size < 1:
            continue
        ell_width = K // 8
        matrix_name = f"random_M{M}_K{K}_ELL{ell_width}"
        REGULAR_TEST_CONFIGS.append((matrix_name, final_tile_size, 4, 8))
        save(
            output_dir="./npu_data",
            auto_padding=False,
            use_random=True,
            rand_m=M,
            rand_k=K,
            rand_nnz=ell_width,
        )



RESULT_CSV = "spmv_results.csv"

# ==========================================
# 2. ヘルパー関数 (test.pyより移植)
# ==========================================

def load_matrix_metadata(matrix_dir: Path, ell_format):
    if not matrix_dir.is_dir():
        return None
    json_files = list(matrix_dir.glob(f"*_{ell_format}_meta.json"))
    if not json_files:
        return None
    try:
        with open(json_files[0], 'r') as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON in {matrix_dir}: {e}")
        return None

    matrix_name = meta_data.get("name", matrix_dir.name)
    npy_path = matrix_dir / f"{matrix_name}_xdna_{ell_format}.npy"
    if not npy_path.exists():
        print(f"[WARNING] NPY file not found for {matrix_name}: {npy_path}")
        return None

    return {
        "name": matrix_name,
        "rows": meta_data["physical_layout"]["aligned_rows"],
        "cols": meta_data["logical_shape"]["cols"],
        "ell_width": meta_data["physical_layout"]["aligned_ell_width"],
        "npy_path": str(npy_path)
    }

def scan_available_matrices(base_dir: Path, ell_format):
    matrix_map = {}
    if not base_dir.exists():
        print(f"[WARNING] Directory {base_dir} not found.")
        return matrix_map
    for item in base_dir.iterdir():
        meta = load_matrix_metadata(item, ell_format)
        if meta:
            matrix_map[meta["name"]] = meta
    return matrix_map

def generate_test_params(test_configs):
    base_dir = Path("npu_data")
    ell_format = "sell32" if "sell32" in design_name else "ell"
    available_matrices = scan_available_matrices(base_dir, ell_format=ell_format)
    
    params = []
    names = []

    for matrix_name, tile_size, num_core_rows, num_core_cols in test_configs:
        if matrix_name not in available_matrices:
            print(f"[SKIP] Matrix '{matrix_name}' not found in {base_dir}. Skipping.")
            continue
            
        meta = available_matrices[matrix_name]
        
        # param: (matrix_name, npy_path, M, K, ell_width, tile_size, num_core_rows, num_core_cols)
        # matrix_nameをパラメータに追加してCSV出力に利用
        params.append((
            matrix_name,
            meta["npy_path"],
            meta["rows"],
            meta["cols"],
            meta["ell_width"],
            tile_size,
            num_core_rows,
            num_core_cols,
        ))
        
        names.append(
            f"{matrix_name}_{meta['rows']}x{meta['cols']}_ell{meta['ell_width']}_tile{tile_size}_{num_core_rows}x{num_core_cols}"
        )
        
    return params, names

# パラメータ生成の実行
all_params_list, all_names = generate_test_params(REGULAR_TEST_CONFIGS)

# pytestパラメータ作成
all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(all_params_list, all_names)
]

# ==========================================
# 3. CSV初期化
# ==========================================
if not os.path.exists(RESULT_CSV):
    with open(RESULT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "design", "MatrixName", "M", "K", "EllWidth", "TileSize", "CoreRows", "CoreCols", "TotalCores",
            "Mean(us)", "Min(us)", "Max(us)", "Std_Dev(us)", 
            "W1_Mean(us)", "W1_Min(us)", "W1_Max(us)", "W1_Std(us)",
            "Total_Bytes", "Bandwidth(GB/s)", "Status"
        ])

# ==========================================
# 4. 計測実行関数
# ==========================================
@pytest.mark.parametrize("matrix_name,npy_path,M,K,ell_width,tile_size,num_core_rows,num_core_cols", all_params)
def test_measure_spmv(matrix_name, npy_path, M, K, ell_width, tile_size, num_core_rows, num_core_cols, aie_context):
    print(f"\n--- [Testing] {matrix_name}: M={M}, K={K}, Ell={ell_width}, Tile={tile_size} ---")

    MEASURE_LOOPS = 5
    
    # 統計用リスト
    latencies = []
    w1_latencies = []
    total_errors = 0
    last_total_bytes = 0

    # 1. 行列データのロード (ループ外で一度だけ行う)
    print(f"Loading matrix from: {npy_path}")
    npu_data = np.load(npy_path)
    
    # Check dimensions
    if npu_data.shape[0] // (ell_width * 2) != M:
        raise AssertionError("NPU data shape does not match expected matrix dimensions.")

    # === 本計測用オペレータ ===
    operator = AIESPMV(
        M=M,
        K=K,
        ell_width=ell_width,
        tile_size=tile_size,
        num_core_rows=num_core_rows,
        num_core_cols=num_core_cols,
        design_name=design_name,
        context=aie_context,
        trace_ddr_id=None, # 計測時はトレースOFF推奨
        trace_size=0,
    )
    
    # === リセット用オペレータ ===
    # 命令キャッシュを追い出すため、ダミー構成で再度実行
    operator_reset = AIEGEMV(
        M=1024,
        K=1024,
        num_aie_columns=8,
        tile_size= 2,
        context=aie_context,
        trace_ddr_id=None,
        trace_size=0,
    )

    # === 10回のループ ===
    for i in range(MEASURE_LOOPS):
        
        # 2. データジェネレータ (ベクトルBを毎回ランダムに生成)
        def data_generator(calc_c=True):
            # generate_reference_from_mtx は通常ファイルから読むが、
            # ここでは既にロード済みの npu_data とランダムなBを使ってリファレンスを作る必要がある。
            # しかし test.py の実装では generate_reference_from_mtx が内部でファイルを読んでしまう仕様に見える。
            # 効率のため、一度 generate_reference_from_mtx を呼んでBだけ差し替えるアプローチを取るか、
            # 簡易的に既存関数を使う。ここでは既存関数を呼び出す形にする。
            
            # Note: generate_reference_from_mtxの実装詳細によるが、
            # 毎回呼ぶと遅い場合は、golden_refをループ外で作っておき、BとCだけ更新するのが良い。
            # 今回は安全策として test.py と同じフローにする。
            
            golden_ref = generate_reference_from_mtx(npy_path=npy_path, calc_c=calc_c, seed=42 + i)
            # ベクトルBをランダム化したい場合、ここでgolden_ref['B']を書き換えて
            # golden_ref['C']を再計算する必要があるが、
            # SpMVのリファレンス計算(CPU)は重いため、計測スクリプトでは
            # 「データキャッシュの影響」を見るために、あえて同じデータを使うか、
            # あるいは measure.py のように毎回変えるかの方針による。
            # measure.py は毎回変えているため、ここでも変えるべきだが、
            # Reference実装(Python)が遅いと計測全体の時間が長くなる。
            # ここでは test.py の動作を尊重し、test.pyと同じ「固定ファイルのデータ」を使う。
            # ※厳密にキャッシュフラッシュ効果を見たい場合は、measure.pyのようにinit_inを渡す。
            
            # SpMVの場合、行列Aが巨大で支配的なため、ベクトルBの変更有無はGEMVほど敏感ではない。
            
            input_buffers = {"sparse_matrix": npu_data, "vector": golden_ref["B"].to(torch.bfloat16)}
            output_buffers = {"output": golden_ref["C"]}
            return input_buffers, output_buffers

        # 最初のループでデータを用意（今回は固定データ）
        init_in, init_out = data_generator()

        # 3. 本計測 (Warmup1 -> Warmup2 -> Measure)
        # measure_mode=True を指定して stats を受け取る

        gc.collect()
        gc.disable()
        errors, latency_us, bandwidth_gbps, run_stats = run_test(
            operator, 
            init_in, 
            init_out, 
            rel_tol=0.04, 
            abs_tol=1e-3, # test.py は 1e-4 だが、measure.py は 1e-3。状況に合わせて調整
            verify=True,
            measure_mode=True, 
            data_generator=data_generator
        )
        gc.enable()

        # 4. 結果保存
        latencies.append(run_stats["latency"])
        w1_latencies.append(run_stats["w1_latency"])
        last_total_bytes = run_stats["total_bytes"]

        if errors:
            total_errors += 1
            print(f"  [Loop {i+1}] Verification Failed!")

        print(f"  Loop {i+1}/{MEASURE_LOOPS}: W1={run_stats['w1_latency']:.1f}us, Exec={run_stats['latency']:.1f}us")

        # 5. === NPU State Flush (Reset実行) ===
        try:
            A = torch.rand(1024, 1024, dtype=torch.bfloat16)
            B = torch.rand(1024, dtype=torch.bfloat16)
            reset_in = {"matrix": A, "vector": B}
            reset_out = {"output": A @ B}
            run_test(
                operator_reset, 
                reset_in, 
                reset_out, 
                rel_tol=0.04, 
                abs_tol=1e-3, 
                verify=False,    # 検証しない
                measure_mode=False,
                warmup_iters=0,  # ウォームアップなし
                timed_iters=1
            )
        except Exception as e:
            print(f"  [Warning] Reset run failed: {e}")

        # 6. 後片付け
        del run_stats
        time.sleep(4)

    # === オペレータ破棄 ===
    del operator
    del operator_reset
    del npu_data
    gc.collect()

    # === 統計計算 ===
    latencies_np = np.array(latencies)
    w1_latencies_np = np.array(w1_latencies)

    stats = {
        "mean": np.mean(latencies_np),
        "min": np.min(latencies_np),
        "max": np.max(latencies_np),
        "std": np.std(latencies_np),
        "w1_mean": np.mean(w1_latencies_np),
        "w1_min": np.min(w1_latencies_np),
        "w1_max": np.max(w1_latencies_np),
        "w1_std": np.std(w1_latencies_np),
    }

    # GFLOPS計算: (2 * M * K * nnz_ratio...) 
    # SpMVの場合、オペレーション数は 2 * nnz だが、
    # test.pyの計算式 `(2.0 * M * K)` (dense換算?) に合わせるか、
    # または `effective bandwidth` に注目するか。
    # ここでは test.py のロジック `(2.0 * M * K)` を踏襲します。

    status = "PASS" if total_errors == 0 else "FAIL"

    print(f"\n[Final Result {design_name} - {matrix_name}]")
    print(f"  Mean Latency: {stats['mean']:.2f} us")
    print(f"  Bandwidth  : {last_total_bytes / (stats['mean'] * 1e-6) / 1e9:.2f} GB/s")

    # CSV書き込み
    with open(RESULT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            design_name, matrix_name, M, K, ell_width, tile_size, num_core_rows, num_core_cols, num_core_rows*num_core_cols,
            f"{stats['mean']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
            f"{stats['std']:.4f}",
            f"{stats['w1_mean']:.4f}",
            f"{stats['w1_min']:.4f}",
            f"{stats['w1_max']:.4f}",
            f"{stats['w1_std']:.4f}",
            f"{last_total_bytes}",
            f"{last_total_bytes / (stats['mean'] * 1e-6) / 1e9:.4f}",
            status
        ])

    print("[Finished] Parameters done.\n")