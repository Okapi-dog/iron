import os
import glob
import numpy as np
import scipy.io
import torch
from pathlib import Path
import time
import json

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

def run_simple_ref(matrix_name, vector_b_numpy):
    """
    型変換なし。NumPyだけで ELL SpMV を計算する。
    
    Args:
        matrix_name (str): フォルダ名 (例: "random_M5000_K1024")
        vector_b_numpy (np.array): ベクトルB (NumPy配列, float32)
        
    Returns:
        np.array: 計算結果 C
    """
    # 1. ファイルパス (save_ell_npu.py で保存したもの)
    base_dir = os.path.join("npu_data", matrix_name)
    path_data = os.path.join(base_dir, f"{matrix_name}_raw_ell_data.npy")
    path_idx  = os.path.join(base_dir, f"{matrix_name}_raw_ell_indices.npy")

    # 2. ロード (ロードした時点で ell_indices は int32 ですが、NumPyならそのまま使えます)
    ell_data = np.load(path_data)       # shape: (Rows, Width)
    ell_indices = np.load(path_idx)     # shape: (Rows, Width)

    # 3. 計算 (Gather -> Mul -> Sum)
    # B[ell_indices] で、インデックスに対応するBの値を一気に取ってきます
    b_values = vector_b_numpy[ell_indices]  
    
    # 掛け算 (要素ごと)
    products = ell_data * b_values
    
    # 足し算 (横方向に合計)
    result = np.sum(products, axis=1)
    
    return result

def run_sell32_ref_from_npy(npy_path, vector_b):
    """
    NPU用SELL-32形式(.npy)を直接読み込み、BF16精度をエミュレートしてSpMV計算を行う。
    
    Args:
        npy_path (str or Path): _xdna_sell32.npy ファイルへのパス
        vector_b (torch.Tensor): 入力ベクトル B (float32)
        
    Returns:
        torch.Tensor: 計算結果 C (float32)
    """
    npy_path = Path(npy_path)
    matrix_dir = npy_path.parent
    
    # 1. メタデータから形状情報を取得
    # ファイル名ルール: {name}_xdna_sell32.npy -> {name}_sell32_meta.json
    name_prefix = npy_path.name.replace('_xdna_sell32.npy', '')
    meta_path = matrix_dir / f"{name_prefix}_sell32_meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    # JSONから必要な情報を取得
    # padded_rows: アライメント後の行数 (ブロック数 * 32)
    # ell_width: アライメント後のELL幅
    padded_rows = meta["physical_layout"]["aligned_rows"]
    ell_width = meta["physical_layout"]["aligned_ell_width"]
    
    # SELL-32固有定数
    BLOCK_SIZE = 32
    num_blocks = padded_rows // BLOCK_SIZE

    # 2. NPYデータ読み込み (uint16の1次元配列)
    raw_data = np.load(npy_path) # dtype=uint16
    
    # 3. 形状の復元 (NPUメモリレイアウト通りにReshape)
    # Layout: [Block][Column(Width)][2: Index/Value][Row(32)]
    # save_ell_npu.py の combined = np.stack(...) と reshape(-1) の逆操作
    try:
        packed_tensor = torch.from_numpy(raw_data)
        packed_tensor = packed_tensor.view(num_blocks, ell_width, 2, BLOCK_SIZE)
    except RuntimeError as e:
        raise RuntimeError(f"Data shape mismatch. Expected {num_blocks}*{ell_width}*2*32 elements, but got {raw_data.size}. Check meta json vs npy.") from e

    # 4. Indices と Values の分離
    # dim=2 の 0番目が Indices, 1番目が Values
    # shape: (num_blocks, ell_width, 32) -> Permute to (num_blocks, 32, ell_width) for easy calculation
    
    # Indices: uint16 -> long (int64) for PyTorch gathering
    indices_raw = packed_tensor[:, :, 0, :] # (NB, W, 32)
    indices = indices_raw.permute(0, 2, 1).contiguous().long() # (NB, 32, W)
    
    # Values: uint16 (bf16 bits) -> bfloat16 -> float32
    values_raw = packed_tensor[:, :, 1, :] # (NB, W, 32)
    # ビット列として解釈するために int16 ビューを経由して bfloat16 に変換
    values_bf16 = values_raw.view(torch.int16).view(torch.bfloat16)
    values = values_bf16.permute(0, 2, 1).contiguous().float() # (NB, 32, W) 計算用にfloat32に戻す

    # 5. ベクトル B の精度エミュレーション (BF16化)
    # NPU入力時にBF16にキャストされる挙動を再現
    vector_b_bf16 = vector_b.to(torch.bfloat16).float()

    # 6. SpMV計算 (Gather -> Mul -> Sum)
    # Indicesを使ってBから値を収集
    # ※パディング部分のindexが0であることを前提 (save_ell_npu.pyでzeros初期化されているため安全)
    # ただし、Bのサイズ外を参照しないよう安全策をとる
    K = vector_b.shape[0]
    
    # 範囲外インデックスをマスク (ゴミデータ対策)
    mask = (indices < K)
    safe_indices = indices.clone()
    safe_indices[~mask] = 0
    
    # Gather
    # b_gathered shape: (NB, 32, W)
    b_gathered = vector_b_bf16[safe_indices]
    
    # 無効なインデックスから取った値を0にする
    b_gathered[~mask] = 0.0
    
    # 積和演算 (MAC)
    # values (BF16由来のFP32) * b_gathered (BF16由来のFP32) -> FP32 Accumulate
    products = values * b_gathered
    block_sums = products.sum(dim=2) # (NB, 32) -> 行ごとの和
    
    # 7. フラット化して結果を返す
    # (NB, 32) -> (padded_rows)
    result_padded = block_sums.view(-1)
    
    # 元のMサイズに切り詰める必要があればここで行うが、
    # NPU出力もpadded_rowsで返ってくるはずなので、比較時はpaddedのままで良いことが多い。
    # 必要なら meta["logical_shape"]["rows"] でスライスする。
    
    return result_padded

def generate_reference_from_mtx(npy_path, calc_c=True, seed=42):
    """
    指定された npy_path と同じディレクトリにある .mtx を読み込み、
    Float32 (CSR) でリファレンス計算を行う。
    """
    npy_path_obj = Path(npy_path)
    matrix_dir = npy_path_obj.parent
    matrix_name = npy_path_obj.stem.replace('_xdna_ell', '').replace('_xdna_sell32', '') 
    ell_format = 'sell32' if 'sell32' in npy_path_obj.stem else 'ell'
    
    # .mtx ファイルを探す
    
    mtx_files = list(matrix_dir.glob(f"{matrix_name}.mtx"))
    if not mtx_files:
        raise FileNotFoundError(f"No {matrix_name}.mtx file found in {matrix_dir}")
    
    mtx_path = mtx_files[0]
    print(f"[Reference] Loading matrix from: {mtx_path}")

    #jsonからメタデータを読んで、padding後のサイズを取得
    meta_json_file = list(matrix_dir.glob(f"{matrix_name}_{ell_format}_meta.json"))[0]
    with open(meta_json_file, 'r') as f:
        meta_data = json.load(f)
        row_after_padding=meta_data['physical_layout']['aligned_rows'] #padding後の行数
    

    # 1. 行列 A の構築 (Float32, 値はBF16相当)
    values, crow_indices, col_indices, shape = load_and_truncate_mtx(mtx_path)

    # Padding処理
    M, K = shape
    if row_after_padding > M:
        # パディングが必要な行数を計算
        diff = row_after_padding - M
        
        # crow_indices (indptr) の末尾を拡張する
        # 追加される行はすべてゼロ要素なので、現在の総非ゼロ数(最後の値)を繰り返して追加します
        last_nnz = crow_indices[-1]
        padding = torch.full((diff,), last_nnz, dtype=crow_indices.dtype)
        crow_indices = torch.cat((crow_indices, padding))
        
        shape = (row_after_padding, K)
    
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
    #print(run_simple_ref(matrix_name, B.numpy())[258:275])
    #print(run_sell32_ref_from_npy(npy_path, B)[258:275])
    if not calc_c:

        return {
            "A": A,
            "B": B,
            "C": None,
            "M": M,
            "K": K
        }
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