# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
from ml_dtypes import bfloat16
from .utils import torch_to_numpy
import logging


def nearly_equal(
    a, b, rel_tol=128 * np.finfo(np.float32).eps, abs_tol=np.finfo(np.float32).tiny
):
    """
    Compare two floating point numbers for approximate equality.

    Adapted from Stack Overflow, License CC BY-SA 4.0
    Original author: P-Gn
    Source: https://stackoverflow.com/a/32334103
    """
    assert np.finfo(np.float32).eps <= rel_tol
    assert rel_tol < 1.0

    if a == b:
        return True

    diff = abs(float(a) - float(b))
    norm = min(abs(float(a)) + abs(float(b)), np.finfo(np.float32).max)
    return diff < max(abs_tol, rel_tol * norm)


def verify_buffer(operator, buf_name, reference, rel_tol=0.04, abs_tol=1e-6, is_traceuse_same_ddr_id=False):
    errors = []
    expected_np = torch_to_numpy(reference).reshape((-1,))
    if operator.trace_ddr_id is not None and buf_name == "output" and is_traceuse_same_ddr_id:
        buf_size =operator.buffers[buf_name]// 2 - operator.trace_size * 2
    else:
        buf_size = operator.buffers[buf_name] // 2
    print(f"buf_name: {buf_name}, buf_size: {buf_size}, expected size: {len(expected_np)}")
    output = operator.read_buffer(buf_name, (buf_size,))
    if len(output) != len(expected_np):
        print(
            f"Buffer size mismatch for {buf_name}: expected {len(expected_np)}, got {len(output)}"
        )
        errors.extend(i for i in range(abs(len(output) - len(expected_np))))
    compare_len = min(len(output), len(expected_np))
    for i in range(compare_len):
        if not nearly_equal(float(output[i]), float(expected_np[i]), rel_tol, abs_tol):
            errors.append(i)
            if len(errors) <= 10:
                print(
                    f"Mismatch in {buf_name}[{i}]: expected {float(expected_np[i]):.6f}, got {float(output[i]):.6f}"
                )
    return errors


def run_test(
    operator,
    input_buffers,
    output_buffers,
    intermediate_buffers=None,
    rel_tol=0.04,
    abs_tol=1e-6,
    warmup_iters=1,
    timed_iters=1,
    is_traceuse_same_ddr_id=False,
    verify=True,
    measure_mode=False,     # 計測モードスイッチ
    data_generator=None     # 毎回データを変えるためのジェネレータ関数
):
    """
    Run operator test with specified input/output/intermediate buffers.

    Args:
        operator: AIE operator instance with registered buffers
        input_buffers: Dict mapping buffer names to input data arrays
        output_buffers: Dict mapping buffer names to reference output arrays
        intermediate_buffers: Optional dict mapping buffer names to reference arrays for validation
        rel_tol: Relative tolerance for comparison of output and intermediate buffers
        abs_tol: Absolute tolerance for comparison of output and intermediate buffers
        is_traceuse_same_ddr_id: Boolean flag indicating if trace use the same DDR ID as other buffers
    Returns:
        (errors: list, latency_us: float, bandwidth_gbps: float)
    """
    if intermediate_buffers is None:
        intermediate_buffers = {}

    # Build operator and prepare runtime
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    operator.context.compile_all()
    operator.context.prepare_runtime()

    
    # ==========================================
    # 計測モード (1セット分のみ実行)
    # ==========================================
    if measure_mode:
        if data_generator is None:
            raise ValueError("data_generator must be provided in measure_mode")

        # ---------------------------------------------------
        # 1. データ生成フェーズ (3回分まとめて作る)
        # ---------------------------------------------------
        # datasets[0]: Warmup1
        # datasets[1]: Warmup2
        # datasets[2]: Measurement
        datasets = [data_generator() for _ in range(3)]

        # ヘルパー: 書き込み関数
        def _write_data(inputs, outputs):
            for buf_name in outputs:
                buf_size = operator.buffers[buf_name]
                operator.write_buffer(buf_name, np.zeros(buf_size, dtype=np.uint8))
            for buf_name, data in inputs.items():
                if buf_name == "sparse_matrix":
                    data_np = data
                else:
                    data_np = torch_to_numpy(data)
                operator.write_buffer(buf_name, data_np)

        # ---------------------------------------------------
        # 2. 実行フェーズ
        # ---------------------------------------------------

        # --- Step 1: Warmup 1 (計測あり) ---
        input_buffers,output_buffers = datasets[0]
        _write_data(input_buffers, output_buffers) 
        w1_latency_s=operator.run_runlist()
        
        w1_latency_us = w1_latency_s * 1e6

        # --- Step 2: Warmup 2 (捨て) ---
        input_buffers,output_buffers = datasets[1]
        _write_data(input_buffers, output_buffers)
        operator.run_runlist()

        # --- Step 3: 本番計測 (計測 + 検証) ---
        input_buffers,output_buffers = datasets[2]
        _write_data(input_buffers, output_buffers)
        

        latency_s=operator.run_runlist()
        latency_us = latency_s * 1e6

        # 検証 (3回目のデータを使用)
        errors = {}
        if verify:
            for buf_name, expected in output_buffers.items():
                buf_errors = verify_buffer(operator, buf_name, expected, rel_tol, abs_tol, is_traceuse_same_ddr_id)
                if buf_errors:
                    errors[buf_name] = buf_errors

        # データサイズ計算
        input_bytes = sum(operator.buffers[buf_name] for buf_name in input_buffers)
        output_bytes = sum(operator.buffers[buf_name] for buf_name in output_buffers)
        total_bytes = input_bytes + output_bytes
        
        bandwidth_gbps = total_bytes / (latency_us * 1e-6) / 1e9

        # 1回分の結果を返す
        single_run_stats = {
            "latency": latency_us,
            "w1_latency": w1_latency_us,
            "total_bytes": total_bytes
        }
        
        # メモリ解放
        del datasets

        return errors, latency_us, bandwidth_gbps, single_run_stats
    # ==========================================
    # 通常モード
    # ==========================================

    # Run warmup iterations before writing to buffers (warmup iters might corrupt the buffers)
    for _ in range(warmup_iters):
        operator.run_runlist()  # warmup run to configure

    # Write input buffers and zero outputs
    for buf_name in output_buffers:
        buf_size = operator.buffers[buf_name]
        operator.write_buffer(buf_name, np.zeros(buf_size, dtype=np.uint8))
    # Operator may share the same buffer object for inputs and outputs; hence, write input after outputs
    for buf_name, data in input_buffers.items():
        if buf_name=="sparse_matrix":
            #sparse matrix is already numpy
            data_np = data
        else:
            data_np = torch_to_numpy(data)
        operator.write_buffer(buf_name, data_np)

    # Run operator
    elapsed_total = 0
    for _ in range(timed_iters):
        elapsed_total += operator.run_runlist()
    elapsed = elapsed_total / timed_iters
    latency_us = elapsed * 1e6

    # Verify outputs
    errors = {}
    if verify:

        for buf_name, expected in output_buffers.items():
            buf_errors = verify_buffer(operator, buf_name, expected, rel_tol, abs_tol, is_traceuse_same_ddr_id)
            if buf_errors:
                errors[buf_name] = buf_errors

        for buf_name, expected in intermediate_buffers.items():
            buf_errors = verify_buffer(operator, buf_name, expected, rel_tol, abs_tol)
            if buf_errors:
                errors[buf_name] = buf_errors

    # Calculate bandwidth
    input_bytes = sum(operator.buffers[buf_name] for buf_name in input_buffers)
    #bufとそのサイズをprintする
    for buf_name in input_buffers:
        print(f"Input Buffer: {buf_name}, Size (bytes): {operator.buffers[buf_name]}")
    for buf_name in output_buffers:
        print(f"Output Buffer: {buf_name}, Size (bytes): {operator.buffers[buf_name]}")
    output_bytes = sum(operator.buffers[buf_name] for buf_name in output_buffers)
    total_bytes = input_bytes + output_bytes
    bandwidth_gbps = total_bytes / (latency_us * 1e-6) / 1e9
    if "sparse_matrix" in input_buffers:
        #sparseではなく、密行列が転送されたとして計算
        M = operator.M
        K = operator.K
        matrix_bytes = M * K * 2  # bfloat16
        total_bytes = matrix_bytes + operator.buffers['vector'] + output_bytes
        bandwidth_gbps_dense = total_bytes / (latency_us * 1e-6) / 1e9
        print(f"Sparse matrix treated as dense for bandwidth calculation: {bandwidth_gbps_dense:.2f} GB/s")


    return errors, latency_us, bandwidth_gbps
