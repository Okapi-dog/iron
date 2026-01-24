import os
import glob
import numpy as np
import scipy.io
import torch
from pathlib import Path
import time

def load_and_truncate_mtx(mtx_path):
    """
    MTXファイルを読み込み、NPUデータ生成時と同じロジック(ビット切り捨て)を適用するが、
    データ型は Float32 のまま返す。
    """
    # 1. MTX読み込み
    sparse_mtx = scipy.io.mmread(mtx_path)
    csr = sparse_mtx.tocsr()
    
    # 2. 値の切り捨て処理 (NPUデータ生成コードと一致させる)
    # float32 -> bits(uint32) -> shift right 16 -> shift left 16 -> float32
    # これで "値" はBF16精度になりますが、"型" はFloat32として扱います。
    data_f32 = csr.data.astype(np.float32)
    u32_view = data_f32.view(np.uint32)
    truncated_u32 = (u32_view >> 16) << 16
    truncated_f32 = truncated_u32.view(np.float32)
    
    # 3. PyTorchのFloat32テンソルへ変換 (ここをbf16にせずfp32にする)
    values = torch.tensor(truncated_f32, dtype=torch.float32)
    crow_indices = torch.tensor(csr.indptr, dtype=torch.int64) # PyTorch CSRは int64 推奨
    col_indices = torch.tensor(csr.indices, dtype=torch.int64)
    
    shape = csr.shape
    
    return values, crow_indices, col_indices, shape

def generate_reference_from_mtx(npy_path, seed=42):
    """
    指定された npy_path と同じディレクトリにある .mtx を読み込み、
    Float32 (CSR) でリファレンス計算を行う。
    """
    npy_path_obj = Path(npy_path)
    matrix_dir = npy_path_obj.parent
    matrix_name = npy_path_obj.stem.replace('_xdna_int16', '') 
    
    # .mtx ファイルを探す
    mtx_files = list(matrix_dir.glob("*.mtx"))
    if not mtx_files:
        mtx_files = list(matrix_dir.glob(f"{matrix_name}.mtx"))
    if not mtx_files:
        raise FileNotFoundError(f"No .mtx file found in {matrix_dir}")
    
    mtx_path = mtx_files[0]
    print(f"[Reference] Loading matrix from: {mtx_path}")

    # 1. 行列 A の構築 (Float32, 値はBF16相当)
    values, crow_indices, col_indices, shape = load_and_truncate_mtx(mtx_path)
    
    # PyTorchのSparse CSR Tensor (Float32)
    A = torch.sparse_csr_tensor(
        crow_indices, 
        col_indices, 
        values, 
        size=shape, 
        dtype=torch.float32
    )
    
    M, K = shape

    # 2. ベクトル B の生成 (Float32)
    torch.manual_seed(seed)
    val_range = 4
    
    # ここが重要: NPUにはBF16で転送されるため、Reference側も一度BF16にキャストして精度を落とし、
    # その後計算のためにFloat32に戻します。
    # これをやらないと「Python側だけ精度が高すぎる」ことによる不一致が起きます。
    B_raw = torch.rand(K, dtype=torch.float32) * val_range
    B = B_raw.to(torch.bfloat16).to(torch.float32)

    # 3. 行列演算 A @ B -> C (全てFloat32で計算)
    # PyTorch CPUは Float32 の Sparse CSR matmul をサポートしています。
    start_time = time.perf_counter()
    C = torch.matmul(A, B)
    end_time = time.perf_counter()
    elapsed_us = (end_time - start_time) * 1e6
    print(f"[Reference] SpMV computation time: {elapsed_us:.2f} us")
    return {
        "A": A,
        "B": B, # run_test内でNPUに送られるベクトル
        "C": C, # 検証に使われる正解データ (Float32)
        "M": M,
        "K": K
    }