import ssgetpy
import scipy.io
import numpy as np
import os
import glob
import json

# --- 1. Float32 を BFloat16 のビットパターン(uint16)に変換する関数 ---
def float32_to_bf16_bits_as_uint16(arr_float32):
    """
    Float32配列を、ビットシフトしてBFloat16相当のビット列にし、
    それをuint16のコンテナに入れて返す。
    """
    u32_view = arr_float32.view(np.uint32)
    bf16_bits_u32 = u32_view >> 16
    return bf16_bits_u32.astype(np.uint16)

# --- 2. ELLデータをNPU転送用にインターリーブする関数 ---
def pack_for_xdna(ell_data, ell_indices):
    """
    Output:
      xdna_buffer: (Rows * 2 * Width) の1次元配列 (int16)
      構造: [Row0_Index..., Row0_Val..., Row1_Index..., Row1_Val..., ...]
    """
    rows, width = ell_data.shape
    
    indices_uint16 = ell_indices.astype(np.uint16)
    values_bf16_as_uint16 = float32_to_bf16_bits_as_uint16(ell_data.astype(np.float32))
    
    combined = np.empty((rows, 2, width), dtype=np.uint16)
    combined[:, 0, :] = indices_uint16
    combined[:, 1, :] = values_bf16_as_uint16
    
    packed_buffer = combined.reshape(-1)
    
    return packed_buffer

# --- 3. メイン処理 ---
def process_matrix_for_npu(group, name, output_dir="npu_data", col_alignment=1, row_alignment=1):
    """
    col_alignment: ELLの幅(width)をこの倍数に合わせる
    row_alignment: 行数(rows)をこの倍数に合わせる (0埋めパディング)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Processing {group}/{name} ---")
    
    # ダウンロード
    results = ssgetpy.search(group=group, name=name, limit=1)
    if not results:
        raise AssertionError("Matrix not found.")
    item = results[0]
    print(f"Downloading {item.name}...")
    item.download(format='MM', destpath=output_dir, extract=True)
    
    # .mtx特定
    mtx_pattern = os.path.join(output_dir, "**", f"{item.name}.mtx")
    files = glob.glob(mtx_pattern, recursive=True)
    if not files:
        raise AssertionError("Downloaded .mtx file not found.")
    
    mtx_file_path = files[0]
    save_dir = os.path.dirname(mtx_file_path)

    # 読み込み
    print("Converting to ELL...")
    sparse_mtx = scipy.io.mmread(mtx_file_path)
    csr = sparse_mtx.tocsr()
    
    n_rows_orig = csr.shape[0]
    
    # --- 行数のパディング計算 ---
    if n_rows_orig % row_alignment == 0:
        n_rows_padded = n_rows_orig
    else:
        n_rows_padded = ((n_rows_orig // row_alignment) + 1) * row_alignment
    
    print(f"Rows: {n_rows_orig} -> Padded to: {n_rows_padded} (Alignment: {row_alignment})")

    # --- 列幅(ELL Width)のパディング計算 ---
    row_nnz = np.diff(csr.indptr)
    max_nnz = row_nnz.max() if n_rows_orig > 0 else 0
    
    if max_nnz % col_alignment == 0:
        aligned_width = int(max_nnz)
    else:
        aligned_width = int(((max_nnz // col_alignment) + 1) * col_alignment)
        
    print(f"max_nnz: {max_nnz} -> Aligned Width: {aligned_width} (Pad: {aligned_width - max_nnz})")
    
    # 配列確保 (パディング後の行数で確保)
    # np.zeros なので、パディングされた行はすべて 0 (Index=0, Value=0.0) になり、計算に影響しない
    ell_data = np.zeros((n_rows_padded, aligned_width), dtype=np.float32)
    ell_indices = np.zeros((n_rows_padded, aligned_width), dtype=np.int32)
    
    # データ埋め込み (元の行数分だけループ)
    for i in range(n_rows_orig):
        start = csr.indptr[i]
        end = csr.indptr[i+1]
        n = end - start
        if n > 0:
            ell_data[i, :n] = csr.data[start:end]
            ell_indices[i, :n] = csr.indices[start:end]

    print(f"ELL Shape: {ell_data.shape} (Rows: {n_rows_padded}, Width: {aligned_width})")

    # NPUフォーマットへ変換
    print("Packing for XDNA NPU (bf16/uint16 interleaved)...")
    npu_buffer = pack_for_xdna(ell_data, ell_indices)
    
    # 保存
    save_path = os.path.join(save_dir, f"{name}_xdna_ell.npy")
    np.save(save_path, npu_buffer)

    # メタデータ計算用
    csr_values_size = csr.nnz * 2        # bf16 (2bytes)
    csr_indices_size = csr.nnz * 2       # uint16 (2bytes)
    csr_indptr_size = (n_rows_orig + 1) * 4   # int32 (4bytes)
    total_csr_size = csr_values_size + csr_indices_size + csr_indptr_size

    # メタデータ保存
    meta_path = os.path.join(save_dir, f"{name}_ell_meta.json")
    meta_info = {
        "name": name,
        "rows": int(n_rows_padded),           # ★更新後のデータ (パディング済み)
        "original_rows": int(n_rows_orig),    # ★元のデータ
        "cols": int(csr.shape[1]),
        "ell_width": int(aligned_width),
        "original_ell_width": int(max_nnz),
        "buffer_size(bytes)": int(npu_buffer.size) * 2,
        "buffer_size(bytes in bf16 dense matrix)": int(n_rows_padded) * int(csr.shape[1]) * 2,
        "csr_estimate_bytes": {
            "total": int(total_csr_size),
            "values_bf16": int(csr_values_size),
            "indices_uint16": int(csr_indices_size),
            "indptr_int32": int(csr_indptr_size)
        },
        # 密度計算は元の行数ベースの方が直感的なので、ここでは元の行数を使っていますが、
        # 必要であれば n_rows_padded に変更してください。
        "nonzero%": (csr.nnz / (n_rows_orig * csr.shape[1])) * 100,
        "dtype": "uint16 (uint16 indices and bf16 values interleaved)"
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    
    print(f"Saved NPU buffer to: {save_path}")
    print(f"Saved Metadata to: {meta_path}")

def print_npu_matrix(matrix_npy_path, cols):
    npu_data = np.load(matrix_npy_path)
    print(f"Loaded data shape: {npu_data.shape}, dtype: {npu_data.dtype}")
    # 1行目のIndexとValueを分離して表示
    row0_indices = npu_data[0:cols]
    row0_values = npu_data[cols:cols*2]
    print("\n--- First Row of NPU Matrix ---")
    print("Indices (uint16):", row0_indices)
    print("Values (bf16 as uint16):", row0_values)

if __name__ == "__main__":
    # 使用例
    # col_alignment=32: ELL幅を32の倍数にする
    # row_alignment=32: 行数を32の倍数にする (10000 -> 10016 など)
    process_matrix_for_npu(
        "ML_Graph", 
        "mnist_test_norm_10NN", 
        col_alignment=32, 
        row_alignment=2560
    ) 
    
    # 確認表示
    print_npu_matrix("npu_data/mnist_test_norm_10NN/mnist_test_norm_10NN_xdna_ell.npy", 10)