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
    # 1. 32bit整数としてメモリを見る (ビットパターンを維持)
    u32_view = arr_float32.view(np.uint32)
    
    # 2. 16bit右シフト（BFloat16はFloat32の上位16bitと互換性があるため）
    #    ※丸め処理は省略し、切り捨て(Truncate)としています。
    bf16_bits_u32 = u32_view >> 16
    
    # 3. uint16型にキャストして返す
    #    (NPU転送関数が uint16 を要求しているため)
    return bf16_bits_u32.astype(np.uint16)

# --- 2. ELLデータをNPU転送用にインターリーブする関数 ---
def pack_for_xdna(ell_data, ell_indices):
    """
    Input:
      ell_data: (Rows, Width) float32
      ell_indices: (Rows, Width) int32/int16
    
    Output:
      xdna_buffer: (Rows * 2 * Width) の1次元配列 (int16)
      構造: [Row0_Index..., Row0_Val..., Row1_Index..., Row1_Val..., ...]
    """
    rows, width = ell_data.shape
    
    # 1. Indexを uint16 にキャスト
    #    (列数が32767を超えるとオーバーフローして負になりますが、bit的には合っているのでuint16解釈ならOK)
    indices_uint16 = ell_indices.astype(np.uint16)
    
    # 2. Value(float32) を bf16ビットパターン(uint16) に変換
    values_bf16_as_uint16 = float32_to_bf16_bits_as_uint16(ell_data.astype(np.float32))
    
    # 3. インターリーブ（交互配置）の作成
    #    Shapeを (Rows, 2, Width) にして、dim1=0にIndex, dim1=1にValueを置く
    combined = np.empty((rows, 2, width), dtype=np.uint16)
    combined[:, 0, :] = indices_uint16
    combined[:, 1, :] = values_bf16_as_uint16
    
    # 4. 平坦化 (C-contiguous)
    #    これにより [Row0_Idx, Row0_Val, Row1_Idx, Row1_Val...] の順にメモリが並ぶ
    packed_buffer = combined.reshape(-1)
    
    return packed_buffer

# --- 3. メイン処理（ダウンロード -> ELL -> NPU Pack -> 保存） ---
def process_matrix_for_npu(group, name, output_dir="npu_data", alignment=1):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Processing {group}/{name} ---")
    
    #ダウンロード (ssgetpy)
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

    
    #読み込み & ELL変換 (手動実装版)
    print("Converting to ELL...")
    sparse_mtx = scipy.io.mmread(mtx_file_path)
    csr = sparse_mtx.tocsr()
    
    n_rows = csr.shape[0]
    row_nnz = np.diff(csr.indptr)
    max_nnz = row_nnz.max() if n_rows > 0 else 0
    if max_nnz % alignment == 0:
        aligned_width = int(max_nnz)
    else:
        # 倍数になるように切り上げ
        aligned_width = int(((max_nnz // alignment) + 1) * alignment)
        
    print(f"max_nnz: {max_nnz} -> Aligned: {aligned_width} (Pad: {aligned_width - max_nnz})")
    
    # NPU用なので Width は 16 の倍数などにパディングした方が良い場合もありますが、
    # ここでは単純なMax幅にします。
    
    # 配列確保
    ell_data = np.zeros((n_rows, aligned_width), dtype=np.float32)
    ell_indices = np.zeros((n_rows, aligned_width), dtype=np.int32) # 一旦int32で
    
    for i in range(n_rows):
        start = csr.indptr[i]
        end = csr.indptr[i+1]
        n = end - start
        if n > 0:
            ell_data[i, :n] = csr.data[start:end]
            ell_indices[i, :n] = csr.indices[start:end]

    print(f"ELL Shape: {ell_data.shape} (Width: {aligned_width})")
    #ellを確認
    print("\n--- ELL Data ---")
    print(f"Data:\n{ell_data}")
    print(f"Indices:\n{ell_indices}")

    #NPUフォーマットへ変換
    print("Packing for XDNA NPU (bf16/uint16 interleaved)...")
    npu_buffer = pack_for_xdna(ell_data, ell_indices)
    
    #保存 (NPU用バイナリデータとして .npy で保存)
    save_path = os.path.join(save_dir, f"{name}_xdna_uint16.npy")
    np.save(save_path, npu_buffer)

    csr_values_size = csr.nnz * 2        # bf16 (2bytes)
    csr_indices_size = csr.nnz * 2       # uint16 (2bytes)
    csr_indptr_size = (n_rows + 1) * 4   # int32 (4bytes)
    total_csr_size = csr_values_size + csr_indices_size + csr_indptr_size

    #メタデータ(形状情報)を保存
    meta_path = os.path.join(save_dir, f"{name}_meta.json")
    meta_info = {
        "name": name,
        "rows": int(n_rows),
        "cols": int(csr.shape[1]),
        "ell_width": int(aligned_width),
        "original_ell_width": int(max_nnz),
        "buffer_size(bytes)": int(npu_buffer.size)*2,
        "buffer_size(bytes in bf16 dense matrix)": int(n_rows)*int(csr.shape[1])*2,
        "csr_estimate_bytes": {
            "total": int(total_csr_size),
            "values_bf16": int(csr_values_size),
            "indices_uint16": int(csr_indices_size),
            "indptr_int32": int(csr_indptr_size)
        },
        "nonzero%": csr.nnz / (n_rows * csr.shape[1])*100,
        "dtype": "uint16 (uint16 indices and bf16 values interleaved)"
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    
    print(f"Saved NPU buffer to: {save_path}")
    print(f"Buffer dtype: {npu_buffer.dtype}")
    print(f"Buffer shape: {npu_buffer.shape}")
    print(f"Buffer size (MB): {npu_buffer.nbytes / 1024 / 1024:.2f} MB")
    
    # 確認用出力（先頭の数要素）
    print("\n--- First Row Preview (Hex interpretation) ---")
    print("Row 0 Indices (uint16):", npu_buffer[0:max_nnz])
    # bf16部分は整数として表示されるので人間には読みにくいですが、値が入っていればOK
    print("Row 0 Values  (bf16 as uint16):", npu_buffer[max_nnz:max_nnz*2]) 

def print_npu_matrix(matrix_npy_path,cols):
    npu_data = np.load(matrix_npy_path)
    
    print(f"Loaded data shape: {npu_data.shape}, dtype: {npu_data.dtype}")
    # 1行目のIndexとValueを分離して表示
    row0_indices = npu_data[0:cols]
    row0_values = npu_data[cols:cols*2]
    print("\n--- First Row of NPU Matrix ---")
    print("Indices (uint16):", row0_indices)
    print("Values (bf16 as uint16):", row0_values)



if __name__ == "__main__":
    # 例: HB/can_24
    #process_matrix_for_npu("HB", "can_24")
    process_matrix_for_npu("ML_Graph", "mnist_test_norm_10NN", alignment=32) 
    print_npu_matrix("npu_data/mnist_test_norm_10NN/mnist_test_norm_10NN_xdna_uint16.npy", 10)
    #print_npu_matrix("npu_data/can_24/can_24_xdna_int16.npy", 9)
    # 定番の Williams/pdb1HYS なども試せます
    # process_matrix_for_npu("Williams", "pdb1HYS")