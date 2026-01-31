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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operators.spmv.op import AIESPMV
from operators.spmv.reference import generate_reference_from_mtx
from operators.common.test_utils import run_test


# ==========================================
# 1. テスト設定 (ここを編集してテストケースを追加・変更)
# ==========================================
# フォーマット: (matrix_name, tile_size, num_core_rows, num_core_cols)
design_name = "ell" # "ell" or "sell32" or "sell32_block"
tile_size = 80

REGULAR_TEST_CONFIGS = [
    ("random_M368640_K2048_ELL32", tile_size, 1, 1),
]

EXTENSIVE_TEST_CONFIGS = [
    # 長時間テストや詳細テスト用
]

# ==========================================
# 2. ヘルパー関数 (JSON読み込み・スキャン)
# ==========================================

def load_matrix_metadata(matrix_dir: Path, ell_format):
    """
    指定されたディレクトリ内の *_meta.json を読み込み、情報を辞書で返す。
    必要なファイルが存在しない場合は None を返す。
    ell_format: "ell" or "sell32"
    """
    if not matrix_dir.is_dir():
        return None

    # メタデータ(JSON)を探す
    json_files = list(matrix_dir.glob(f"*_{ell_format}_meta.json"))
    if not json_files:
        return None
    
    try:
        with open(json_files[0], 'r') as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON in {matrix_dir}: {e}")
        return None

    # 行列名を取得 (JSONにない場合はフォルダ名)
    matrix_name = meta_data.get("name", matrix_dir.name)
    
    # npyファイルのパスを確認
    npy_path = matrix_dir / f"{matrix_name}_xdna_{ell_format}.npy"
    if not npy_path.exists():
        print(f"[WARNING] NPY file not found for {matrix_name}: {npy_path}")
        return None

    # 必要な情報を辞書にまとめる
    return {
        "name": matrix_name,
        "rows": meta_data["physical_layout"]["aligned_rows"],
        "cols": meta_data["logical_shape"]["cols"],
        "ell_width": meta_data["physical_layout"]["aligned_ell_width"],
        "npy_path": str(npy_path)
    }

def scan_available_matrices(base_dir: Path, ell_format):
    """
    base_dir 以下の全ディレクトリを走査し、利用可能な行列データを辞書化して返す。
    Returns:
        dict: { "matrix_name": matrix_metadata_dict, ... }
    """
    matrix_map = {}
    if not base_dir.exists():
        print(f"[WARNING] Directory {base_dir} not found.")
        return matrix_map

    for item in base_dir.iterdir():
        meta = load_matrix_metadata(item, ell_format)
        if meta:
            matrix_map[meta["name"]] = meta
            
    return matrix_map

# ==========================================
# 3. パラメータ生成ロジック
# ==========================================

def generate_test_params(test_configs):
    """
    テスト設定リストと、ディスク上のデータを照合してpytest用のパラメータを生成する。
    """
    base_dir = Path("npu_data")
    
    # 1. 利用可能な行列データをスキャン
    ell_format = "sell32" if "sell32" in design_name else "ell"
    available_matrices = scan_available_matrices(base_dir, ell_format=ell_format)
    
    params = []
    names = []

    # 2. 設定リストに基づいてパラメータを構築
    for matrix_name, tile_size, num_core_rows, num_core_cols in test_configs:
        
        # 設定にある名前が、実際のデータフォルダに存在するか確認
        if matrix_name not in available_matrices:
            print(f"[SKIP] Matrix '{matrix_name}' not found in {base_dir}. Skipping.")
            continue
            
        meta = available_matrices[matrix_name]
        
        # param: (npy_path, M, K, ell_width, tile_size, num_core_rows, num_core_cols)
        params.append((
            meta["npy_path"],
            meta["rows"],
            meta["cols"],
            meta["ell_width"],
            tile_size,
            num_core_rows,
            num_core_cols,
        ))
        
        # テストケース名
        names.append(
            f"{matrix_name}_{meta['rows']}x{meta['cols']}_ellwidth{meta['ell_width']}_tile{tile_size}_core{num_core_rows}x{num_core_cols}"
        )
        
    return params, names


# パラメータ生成の実行
regular_params, regular_names = generate_test_params(REGULAR_TEST_CONFIGS)
extensive_params, extensive_names = generate_test_params(EXTENSIVE_TEST_CONFIGS)

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
            
            filename = f"trace_spmv_{filename_suffix}.txt"
            with open(filename, "w") as f:
                for val in trace_data.flatten():
                    f.write(f"{val:08x}\n")
            print(f"[AIESPMV] Trace saved to {filename}")
            #トレースが何行まであるかを表示(空白行を除く)
            non_empty_lines = [line for line in trace_data.flatten() if line != 0]
            print(f"[AIESPMV] Trace contains {len(non_empty_lines)} non-empty lines.")
            if len(non_empty_lines) != 0:
                PARSE_TRACE_SCRIPT = "../../applications/llama_3.2_1b/parse_trace.py"
                build_dir = Path("build")
                if not build_dir.exists():
                    print(f"[ERROR] 'build' directory not found at {build_dir.resolve()}")
                    return

                # ファイル名の生成ルール (ユーザー提示のコードに基づく)
                # prefix は外部から不明な場合があるためワイルドカード '*' で吸収します
                prefix="spmv_"
                trace_suffix = f"_traceddr{operator.trace_ddr_id}" if operator.trace_ddr_id is not None else ""
                
                # 検索パターン: f"{prefix}{self.M}x{self.K}_ellwidth{self.ell_width}_tile{self.tile_size}_core{self.num_core_rows}x{self.num_core_cols}{trace_suffix}.mlir"
                file_pattern = (
                    f"{prefix}"
                    f"{operator.M}x{operator.K}_"
                    f"ellwidth{operator.ell_width}_"
                    f"tile{operator.tile_size}_"
                    f"core{operator.num_core_rows}x{operator.num_core_cols}"
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
                print(f"[AIESPMV] Using MLIR file: {target_mlir_file}")

                # ---------------------------------------------------------
                # 4. parse_trace.py の実行
                # ---------------------------------------------------------
                cmd = [
                    "python3", 
                    PARSE_TRACE_SCRIPT,
                    "--input", filename,
                    "--mlir", str(target_mlir_file),
                    "--output", f"trace_spmv_{filename_suffix}.json"
                ]
                
                print(f"[AIESPMV] Parsing trace command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True,timeout=10)
                
                if result.returncode == 0:
                    print(f"[AIESPMV] Trace parsed successfully. Output: trace_spmv_{filename_suffix}.json")
                else:
                    print(f"[AIESPMV] Parse failed. Stderr:\n{result.stderr}")
        except Exception as e:
            print(f"[AIESPMV] Trace save failed: {e}")


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize("npy_path,M,K,ell_width,tile_size,num_core_rows,num_core_cols", all_params)
def test_spmv(npy_path, M, K, ell_width, tile_size, num_core_rows, num_core_cols, aie_context):

    operator = AIESPMV(
        M=M,
        K=K,
        ell_width=ell_width,
        tile_size=tile_size,
        num_core_rows=num_core_rows,
        num_core_cols=num_core_cols,
        design_name=design_name,
        context=aie_context,
        trace_ddr_id=None,#3
        trace_size=8192*4,
    )
    golden_ref = generate_reference_from_mtx(npy_path=npy_path)
    print(f"Loading matrix from: {npy_path}")
    npu_data = np.load(npy_path)
    #ellデータの形状を確認(ell_widthが合っているか)
    print(f"NPU Data Shape: {npu_data.shape}")
    print(f"ell_width: {ell_width}")
    print(f"Matrix Rows(cal by npu data and ell_width): {npu_data.shape[0] // (ell_width*2)}")
    if npu_data.shape[0]/(ell_width*2) != M:
        raise AssertionError("NPU data shape does not match expected matrix dimensions.")

    input_buffers = {"sparse_matrix": npu_data, "vector": golden_ref["B"].to(torch.bfloat16)}
    output_buffers = {"output": golden_ref["C"]}
    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-4, warmup_iters=2, verify=True
    )
    save_trace(operator, filename_suffix=f"{M}_{K}_{tile_size}_{num_core_cols}col")
    print(f"\nLatency: {latency_us:.1f} us")
    gflops = (2.0 * M * K) / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
