#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.gemv_trace.op import AIEGEMV
from operators.gemv_trace.reference import generate_golden_reference
from operators.common.test_utils import run_test


def generate_test_params(extensive=False):
    params = [
        (1920, 2048, 1, 1),
        (1920, 2048, 1, 2),
        (1920, 2048, 1, 3),
        (1920, 2048, 1, 4),
        (1920, 2048, 1, 6),
        (1920, 2048, 2, 1),
        (1920, 2048, 2, 2),
        (1920, 2048, 2, 3),
        (1920, 2048, 2, 4),
        (1920, 2048, 2, 6),
        (1920, 2048, 4, 1),
        (1920, 2048, 4, 2),
        (1920, 2048, 4, 3),
        (1920, 2048, 4, 4),
        (1920, 2048, 4, 6),
        (1920, 2048, 8, 1),
        (1920, 2048, 8, 2),
        (1920, 2048, 8, 3),
        (1920, 2048, 8, 4),
        (1920, 2048, 8, 6),
        (1920, 2048, 12, 1),
        (1920, 2048, 15, 1),
        (2048, 8192, 1, 4),
        (2048, 2048, 2, 4),
        (2048, 2048, 4, 4),
        (2048, 2048, 8, 4),
        (38400, 2048, 1, 4),
        (38400, 2048, 2, 4),
        (38400, 2048, 4, 4),
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
] + [
    pytest.param(*params, marks=pytest.mark.extensive, id=name)
    for params, name in zip(extensive_params, extensive_names)
]
def save_trace(operator, filename_suffix=""):
    if operator.trace_ddr_id is not None:
        try:
            # 1. まず、Trace部分の長さ（bfloat16換算の個数）を計算
            tracesize_bf16 = operator.trace_size * 2
            # 2. 出力バッファ全体を「M + trace」のサイズでbf16で読む
            total_len = operator.M + tracesize_bf16
            full_data = operator.read_buffer("output", (total_len,), dtype=np.uint16)
            # 3. 後ろのTrace部分だけをスライス
            trace_raw_u16 = full_data[operator.M:]
            # 4. uint16 (2byte) x 2個 を uint32 (4byte) x 1個 に変換
            trace_data = trace_raw_u16.view(np.uint32)

            # これで trace_data は trace_size 個の uint32 配列になります
            filename = "trace_gemv.txt"
            with open(filename, "w") as f:
                for val in trace_data.flatten():
                    f.write(f"{val:08x}\n")
            print(f"[AIEGEMV] Trace saved to {filename}")
            #トレースが何行まであるかを表示(空白行を除く)
            non_empty_lines = [line for line in trace_data.flatten() if line != 0]
            print(f"[AIEGEMV] Trace contains {len(non_empty_lines)} non-empty lines.")
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
    golden_ref = generate_golden_reference(M=M, K=K)

    operator = AIEGEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size=tile_size,
        context=aie_context,
        trace_ddr_id=2,
    )

    input_buffers = {"matrix": golden_ref["A"].flatten(), "vector": golden_ref["B"]}
    output_buffers = {"output": golden_ref["C"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-3, warmup_iters=2 ,timed_iters=5, is_traceuse_same_ddr_id=True
    )
    save_trace(operator, filename_suffix=f"{M}_{K}_{tile_size}_{num_aie_columns}col_1st")
    print(f"\nLatency: {latency_us:.1f} us")
    gflops = (2.0 * M * K) / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
