#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import numpy as np
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.gemv.op import AIEGEMV
from operators.gemv.reference import generate_golden_reference
from operators.common.test_utils import run_test


def generate_test_params(extensive=False):
    params = [
        (10240,2048,8,4)
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
] + [
    pytest.param(*params, marks=pytest.mark.extensive, id=name)
    for params, name in zip(extensive_params, extensive_names)
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

def inspect_kernel_memory_banks(operator, kernel_name="gemv"):
    print(f"\n{'='*60}")
    print(f"🔍 Kernel '{kernel_name}' Argument Inspector")
    print(f"{'='*60}")
    print(f"{'Index':<6} | {'Argument Name':<20} | {'Type':<12} | {'DDR ID (Bank)':<10}")
    print(f"{'-'*60}")

    # 1. カーネルオブジェクトの取得
    if kernel_name not in operator.xrt_kernels:
        print(f"❌ Error: Kernel '{kernel_name}' not found.")
        return
    
    xrt_kernel = operator.xrt_kernels[kernel_name][1]

    # 2. システム固定引数の確認 (AIEOperatorBaseの仕様に基づく)
    # Index 0: opcode (Scalar) -> IDなし
    print(f"{0:<6} | {'(Opcode)':<20} | {'Scalar':<12} | {'-'}")
    
    # Index 1: Instructions (Buffer)
    try:
        gid = xrt_kernel.group_id(1)
        print(f"{1:<6} | {'(Instructions)':<20} | {'Buffer':<12} | {gid:<10}")
    except:
        print(f"{1:<6} | {'(Instructions)':<20} | {'Buffer':<12} | {'Error'}")

    # Index 2: Instruction Length (Scalar) -> IDなし
    print(f"{2:<6} | {'(Instr Length)':<20} | {'Scalar':<12} | {'-'}")

    # 3. ユーザーバッファの確認 (runlistから名前を引いてくる)
    # runlistの中から、指定したkernel_nameのエントリを探す
    # entryの構造: ('gemv', 'matrix', 'vector', 'output', 'trace')
    target_entry = None
    for entry in operator.runlist:
        if entry[0] == kernel_name:
            target_entry = entry
            break
    
    if target_entry:
        # 先頭のカーネル名を除いたものがバッファ名リスト
        buffer_names = target_entry[1:]
        
        # システム引数が3つあるので、ユーザーバッファは Index 3 から始まる
        base_index = 3
        
        for i, name in enumerate(buffer_names):
            current_idx = base_index + i
            try:
                gid = xrt_kernel.group_id(current_idx)
                print(f"{current_idx:<6} | {name:<20} | {'Buffer':<12} | {gid:<10}")
                
                # コンフリクト警告
                if name == "trace" and operator.trace_ddr_id is not None:
                     if gid != operator.trace_ddr_id:
                         print(f"       ⚠️  Warning: Trace is physically at ID {gid}, but you set trace_ddr_id={operator.trace_ddr_id}")
            except Exception as e:
                print(f"{current_idx:<6} | {name:<20} | {'Buffer':<12} | {'Error/Scalar'}")
    else:
        print("❌ Error: Kernel entry not found in runlist.")

    print(f"{'='*60}\n")



@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize("M,K,num_aie_columns,tile_size", all_params)
def test_gemv(M, K, num_aie_columns, tile_size, aie_context):

    operator = AIEGEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size=tile_size,
        context=aie_context,
        trace_ddr_id=None,
        trace_size=8192*4,
    )

    golden_ref = generate_golden_reference(M=M, K=K)
    input_buffers = {"matrix": golden_ref["A"].flatten(), "vector": golden_ref["B"]}
    output_buffers = {"output": golden_ref["C"]}
    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-3, warmup_iters=2
    )
    save_trace(operator, filename_suffix=f"{M}_{K}_{tile_size}_{num_aie_columns}col_1st")
    print(f"\nLatency: {latency_us:.1f} us")
    gflops = (2.0 * M * K) / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    """
    golden_ref = generate_golden_reference(M=M, K=K,seed=100)
    input_buffers = {"matrix": golden_ref["A"].flatten(), "vector": golden_ref["B"]}
    output_buffers = {"output": golden_ref["C"]}
    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-3, warmup_iters=0
    )
    save_trace(operator, filename_suffix=f"{M}_{K}_{tile_size}_{num_aie_columns}col_2nd")
    print(f"\nLatency: {latency_us:.1f} us")
    gflops = (2.0 * M * K) / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    """
    assert not errors, f"Test failed with errors: {errors}"
