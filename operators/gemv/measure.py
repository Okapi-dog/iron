#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import numpy as np
import subprocess
import os
import csv
import gc
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.gemv.op import AIEGEMV
from operators.gemv.reference import generate_golden_reference
from operators.common.test_utils import run_test


def generate_test_params(extensive=False):
    params = [
        (9600, 2048, 1, 4),
        (9600, 2048, 2, 4),
        (9600, 2048, 4, 4),
        (9600, 2048, 8, 4),
        (9600, 2048, 12, 4),
        (9600, 2048, 15, 4),
        (38400, 2048, 8, 4),
        (38400, 2048, 12, 4), 
        (38400, 2048, 15, 4),
    ]
    names = [
        f"matrix_vector_mul_{M}x{K}_{tile_size}_{num_aie_columns}col"
        for M, K, num_aie_columns, tile_size in params
    ]
    return params, names

regular_params, regular_names = generate_test_params(extensive=False)
extensive_params, extensive_names = generate_test_params(extensive=True)

# Combine params with marks - extensive params get pytest.mark.extensive
all_params = [
    pytest.param(*params, id=name)
    for params, name in zip(regular_params, regular_names)
]
def save_trace(operator, filename_suffix=""):
    if operator.trace_ddr_id is not None:
        try:
            trace_data = operator.read_buffer("trace", (operator.trace_size,), dtype=np.uint32)
            
            filename = f"trace_gemv_{filename_suffix}.txt"
            with open(filename, "w") as f:
                for val in trace_data.flatten():
                    f.write(f"{val:08x}\n")
            print(f"[AIEGEMV] Trace saved to {filename}")
            #トレースが何行まであるかを表示(空白行を除く)
            non_empty_lines = [line for line in trace_data.flatten() if line != 0]
            print(f"[AIEGEMV] Trace contains {len(non_empty_lines)} non-empty lines.")
            if len(non_empty_lines) != 0:
                PARSE_TRACE_SCRIPT = "../../applications/llama_3.2_1b/parse_trace.py"
                build_dir = Path("build")
                if not build_dir.exists():
                    print(f"[ERROR] 'build' directory not found at {build_dir.resolve()}")
                    return

                # ファイル名の生成ルール (ユーザー提示のコードに基づく)
                # prefix は外部から不明な場合があるためワイルドカード '*' で吸収します
                prefix="gemv_"
                trace_suffix = f"_traceddr{operator.trace_ddr_id}"
                
                # 検索パターン: *{cols}c_{M}x{K}_{tile}t{trace_suffix}.mlir
                file_pattern = (
                    f"{prefix}"
                    f"{operator.num_aie_columns}c_"
                    f"{operator.M}x{operator.K}_"
                    f"{operator.tile_size}t"
                    f"{trace_suffix}.mlir"
                )

                # buildフォルダ内を検索
                found_mlir_files = list(build_dir.glob(file_pattern))

                if not found_mlir_files:
                    print(f"[ERROR] Could not find MLIR file in '{build_dir}' matching pattern: {file_pattern}")
                    print(f"       Please check if the build finished successfully.")
                    return

                # 複数見つかった場合は、とりあえず最初の1つを使用 (通常は1つのはず)
                target_mlir_file = found_mlir_files[0]
                print(f"[AIEGEMV] Using MLIR file: {target_mlir_file}")

                # ---------------------------------------------------------
                # 4. parse_trace.py の実行
                # ---------------------------------------------------------
                cmd = [
                    "python3", 
                    PARSE_TRACE_SCRIPT,
                    "--input", filename,
                    "--mlir", str(target_mlir_file),
                    "--output", f"trace_gemv_{filename_suffix}.json"
                ]
                
                print(f"[AIEGEMV] Parsing trace command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True,timeout=10)
                
                if result.returncode == 0:
                    print(f"[AIEGEMV] Trace parsed successfully. Output: trace_gemv_{filename_suffix}.json")
                else:
                    print(f"[AIEGEMV] Parse failed. Stderr:\n{result.stderr}")
        except Exception as e:
            print(f"[AIEGEMV] Trace save failed: {e}")

RESULT_CSV = "gemv_results.csv"

if not os.path.exists(RESULT_CSV):
    with open(RESULT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "M", "K", "Cols", "TileSize", 
            "Mean(us)", "Min(us)", "Max(us)", "Std_Dev(us)", 
            "W1_Mean(us)", "W1_Min(us)", "W1_Max(us)", "W1_Std(us)",
            "Total_Bytes", "Bandwidth_W1(GB/s)", "Bandwidth(GB/s)" ,"Status"
        ])

@pytest.mark.parametrize("M,K,num_aie_columns,tile_size", all_params)
def test_gemv(M, K, num_aie_columns, tile_size, aie_context):
    print(f"\n--- [Testing] M={M}, K={K}, Cols={num_aie_columns}, Tile={tile_size} ---")

    MEASURE_LOOPS = 10
    
    # 統計用リスト
    latencies = []
    w1_latencies = []
    total_errors = 0
    last_total_bytes = 0


    # === 本計測用オペレータ ===
    operator = AIEGEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size=tile_size,
        context=aie_context,
        trace_ddr_id=None,
        trace_size=0,
    )
    
    operator_reset = AIEGEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size=2,  # 異なる構成にして命令キャッシュを変える
        context=aie_context,
        trace_ddr_id=None,
        trace_size=0,
    )
    np.random.seed(1104)

    # === 10回のループ ===
    for i in range(MEASURE_LOOPS):

        # 1. データ生成
        def data_generator():
            random_seed = np.random.randint(0, 100000)
            ref = generate_golden_reference(M=M, K=K, seed=random_seed)
            in_bufs = {"matrix": ref["A"].flatten(), "vector": ref["B"]}
            out_bufs = {"output": ref["C"]}
            return in_bufs, out_bufs

        init_in, init_out = data_generator()

        # 2. 本計測 (Warmup1 -> Warmup2 -> Measure)
        errors, latency_us, bandwidth_gbps, run_stats = run_test(
            operator, 
            init_in, 
            init_out, 
            rel_tol=0.04, 
            abs_tol=1e-3, 
            verify=True,
            measure_mode=True, 
            data_generator=data_generator
        )

        # 3. 結果保存
        latencies.append(run_stats["latency"])
        w1_latencies.append(run_stats["w1_latency"])
        last_total_bytes = run_stats["total_bytes"]

        if errors:
            total_errors += 1
            print(f"  [Loop {i+1}] Verification Failed!")

        print(f"  Loop {i+1}/{MEASURE_LOOPS}: W1={run_stats['w1_latency']:.1f}us, Exec={run_stats['latency']:.1f}us")

        # 4. === NPU State Flush (Reset実行) ===
        # 違うオペレータを実行して、前のオペレータのキャッシュ等を追い出す
        # verify=False, measure_mode=False でただ流すだけ
        # ※ init_in は M,K が同じなので使い回せる
        try:
            run_test(
                operator_reset, 
                init_in, 
                init_out, 
                rel_tol=0.04, 
                abs_tol=1e-3, 
                verify=False,
                measure_mode=False,
                warmup_iters=0, # ウォームアップなしで1回だけ実行
                timed_iters=1
            )
            # print("  [Info] NPU state flushed.") # うるさければコメントアウト
        except Exception as e:
            print(f"  [Warning] Reset run failed: {e}")

        # 5. 後片付け
        del init_in
        del init_out
        del run_stats
        gc.collect()
        
        # 次のループまで休憩
        time.sleep(1.0)

    # === オペレータ破棄 ===
    del operator
    del operator_reset
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

    status = "PASS" if total_errors == 0 else "FAIL"

    print(f"\n[Final Result M={M}, K={K}]")
    print(f"  Mean Latency: {stats['mean']:.2f} us")
    print(f"  W1 Mean     : {stats['w1_mean']:.2f} us")

    # CSV書き込み
    with open(RESULT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            M, K, num_aie_columns, tile_size,
            f"{stats['mean']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
            f"{stats['std']:.4f}",
            f"{stats['w1_mean']:.4f}",
            f"{stats['w1_min']:.4f}",
            f"{stats['w1_max']:.4f}",
            f"{stats['w1_std']:.4f}",
            f"{last_total_bytes}",
            f"{last_total_bytes / (stats['w1_mean'] * 1e-6) / 1e9:.4f}",
            f"{last_total_bytes / (stats['mean'] * 1e-6) / 1e9:.4f}",
            status
        ])

    print("[Finished] Parameters done.\n")