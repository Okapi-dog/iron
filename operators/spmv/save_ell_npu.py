import numpy as np
import scipy.sparse
import scipy.io
import json
import os
import glob
import ssgetpy

# ==========================================
# 1. 設定エリア (Settings)
# ==========================================
USE_RANDOM = True   # ★ここをTrueにしてテストしてください

OUTPUT_DIR = "npu_data"

# ELL形式のアライメント設定
COL_ALIGNMENT = 32     # ELLの幅(width)をこの倍数に合わせる
ROW_ALIGNMENT = 128   # 行数(rows)をこの倍数に合わせる

# [Mode A] Random Settings
RAND_M = 5000
RAND_K = 1024
RAND_ELL_WIDTH = 64

# [Mode B] Download Settings
SS_GROUP = "ML_Graph"
SS_NAME  = "mnist_test_norm_10NN"
np.random.seed(42)  # 再現性のためのシード設定

# ==========================================
# 2. コア変換ロジック (Core Logic)
# ==========================================

def float32_to_bf16_bits_as_uint16(arr_float32):
    """
    Float32配列を、ビットシフトしてBFloat16相当のビット列にし、
    それをuint16のコンテナに入れて返す。
    """
    u32_view = arr_float32.view(np.uint32)
    bf16_bits_u32 = u32_view >> 16
    return bf16_bits_u32.astype(np.uint16)

def pack_for_xdna_ell(ell_data, ell_indices):
    """
    Output: [Row0_Index..., Row0_Val..., Row1_Index..., Row1_Val...]
    """
    rows, width = ell_data.shape
    
    indices_uint16 = ell_indices.astype(np.uint16)
    values_bf16_as_uint16 = float32_to_bf16_bits_as_uint16(ell_data.astype(np.float32))
    
    combined = np.empty((rows, 2, width), dtype=np.uint16)
    combined[:, 0, :] = indices_uint16        # Indexを先に
    combined[:, 1, :] = values_bf16_as_uint16 # Valueを後に
    
    packed_buffer = combined.reshape(-1)
    return packed_buffer

def calc_alignment_shape(n_rows_orig, max_nnz, row_align, col_align):
    """パディング後の行数とELL幅を計算"""
    if n_rows_orig % row_align == 0:
        n_rows_padded = n_rows_orig
    else:
        n_rows_padded = ((n_rows_orig // row_align) + 1) * row_align

    if max_nnz % col_align == 0:
        aligned_width = int(max_nnz)
    else:
        aligned_width = int(((max_nnz // col_align) + 1) * col_align)
        
    return int(n_rows_padded), int(aligned_width)

# ==========================================
# 3. データソース処理 (Data Source)
# ==========================================

def get_random_ell_matrix(output_root_dir):
    name = f"random_M{RAND_M}_K{RAND_K}"
    print(f"--- [Random Mode] Generating {name}, ELL_Target={RAND_ELL_WIDTH} ---")
    
    max_nnz = RAND_ELL_WIDTH
    n_rows_padded, aligned_width = calc_alignment_shape(RAND_M, max_nnz, ROW_ALIGNMENT, COL_ALIGNMENT)
    
    print(f"Rows: {RAND_M} -> Padded: {n_rows_padded}, Width: {max_nnz} -> Aligned: {aligned_width}")

    # 配列確保
    ell_data = np.zeros((n_rows_padded, aligned_width), dtype=np.float32)
    ell_indices = np.zeros((n_rows_padded, aligned_width), dtype=np.int32)
    
    # データ生成
    # ランダム値を生成し、各行でランダムな列インデックスを選びます
    temp_data = np.random.rand(RAND_M, aligned_width).astype(np.float32)
    for r in range(RAND_M):
        cols = np.random.choice(RAND_K, aligned_width, replace=False)
        cols.sort()
        ell_indices[r, :] = cols
        ell_data[r, :] = temp_data[r, :]

    # 統計情報
    actual_nnz = RAND_M * aligned_width
    csr_stats = {"nnz": actual_nnz, "rows": RAND_M, "cols": RAND_K}

    # 保存ディレクトリ
    save_dir = os.path.join(output_root_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # ★重要修正: Reference用の .mtx ファイルをここで保存する
    # ---------------------------------------------------------
    print(" -> Saving .mtx for Reference check...")
    
    # ★修正: NPUデータと完全に一致させるため、パディング部分(Index=0, Value=0)も含めてMTXにする
    # ELL形式では全行が aligned_width を持つため、構造がシンプルです
    
    # 座標データの作成 (n_rows_padded を使用)
    row_indices_vec = np.repeat(np.arange(n_rows_padded), aligned_width)
    col_indices_vec = ell_indices.flatten() # スライスせず全データ
    data_vec = ell_data.flatten()           # スライスせず全データ
    
    # CSR行列作成 (shapeもパディング後に合わせる)
    sparse_matrix = scipy.sparse.csr_matrix(
        (data_vec, (row_indices_vec, col_indices_vec)), 
        shape=(n_rows_padded, RAND_K)
    )
    
    # .mtx 保存
    mtx_path = os.path.join(save_dir, f"{name}.mtx")
    scipy.io.mmwrite(mtx_path, sparse_matrix)
    print(f"    Saved: {mtx_path}")
    # ---------------------------------------------------------
    
    return ell_data, ell_indices, csr_stats, n_rows_padded, aligned_width, max_nnz, name, save_dir

def get_downloaded_ell_matrix(output_dir):
    print(f"--- [Download Mode] Processing {SS_GROUP}/{SS_NAME} ---")
    
    # ダウンロード
    results = ssgetpy.search(group=SS_GROUP, name=SS_NAME, limit=1)
    if not results: raise AssertionError("Matrix not found.")
    item = results[0]
    item.download(format='MM', destpath=output_dir, extract=True)
    
    # .mtx特定
    mtx_pattern = os.path.join(output_dir, "**", f"{item.name}.mtx")
    files = glob.glob(mtx_pattern, recursive=True)
    mtx_file_path = files[0]
    save_dir = os.path.dirname(mtx_file_path)

    # 読み込み
    print("Converting to ELL...")
    sparse_mtx = scipy.io.mmread(mtx_file_path)
    csr = sparse_mtx.tocsr()
    n_rows_orig = csr.shape[0]
    
    # パディング計算
    row_nnz = np.diff(csr.indptr)
    max_nnz = int(row_nnz.max()) if n_rows_orig > 0 else 0
    
    n_rows_padded, aligned_width = calc_alignment_shape(n_rows_orig, max_nnz, ROW_ALIGNMENT, COL_ALIGNMENT)
    
    print(f"Rows: {n_rows_orig} -> Padded: {n_rows_padded} (Alignment: {ROW_ALIGNMENT})")
    print(f"max_nnz: {max_nnz} -> Aligned Width: {aligned_width}")
    
    # 配列確保 & データ埋め込み
    ell_data = np.zeros((n_rows_padded, aligned_width), dtype=np.float32)
    ell_indices = np.zeros((n_rows_padded, aligned_width), dtype=np.int32)
    
    for i in range(n_rows_orig):
        start = csr.indptr[i]
        end = csr.indptr[i+1]
        n = end - start
        if n > 0:
            copy_len = min(n, aligned_width)
            ell_data[i, :copy_len] = csr.data[start:start+copy_len]
            ell_indices[i, :copy_len] = csr.indices[start:start+copy_len]

    csr_stats = {"nnz": csr.nnz, "rows": n_rows_orig, "cols": csr.shape[1]}
    
    return ell_data, ell_indices, csr_stats, n_rows_padded, aligned_width, max_nnz, SS_NAME, save_dir

# ==========================================
# 4. メイン実行 (Main Execution)
# ==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_type = "Random" if USE_RANDOM else "SuiteSparse"

    # 1. データ取得
    if USE_RANDOM:
        ell_data, ell_indices, csr_stats, n_rows_padded, aligned_width, max_nnz, name, save_dir = get_random_ell_matrix(OUTPUT_DIR)
    else:
        ell_data, ell_indices, csr_stats, n_rows_padded, aligned_width, max_nnz, name, save_dir = get_downloaded_ell_matrix(OUTPUT_DIR)

    # 2. パッキング (NPUフォーマット)
    print("Packing for XDNA NPU (ELL format: bf16/uint16 interleaved)...")
    npu_buffer = pack_for_xdna_ell(ell_data, ell_indices)
    
    # 3. 保存
    npy_filename = f"{name}_xdna_ell.npy"
    save_path = os.path.join(save_dir, npy_filename)
    np.save(save_path, npu_buffer)

    # 4. メタデータ計算 (統計情報)
    buffer_bytes = npu_buffer.size * 2
    buffer_kb = buffer_bytes / 1024.0
    
    dense_bytes = n_rows_padded * csr_stats["cols"] * 2
    dense_kb = dense_bytes / 1024.0
    
    total_elements = csr_stats["rows"] * csr_stats["cols"]
    nonzero_percent = (csr_stats["nnz"] / total_elements * 100.0) if total_elements > 0 else 0.0

    # 5. JSON構築
    meta_info = {
        "dataset_info": {
            "name": name,
            "source_type": source_type,
            "format_version": "ELL_NPU"
        },
        "logical_shape": {
            "rows": int(csr_stats["rows"]),
            "cols": int(csr_stats["cols"]),
            "actual_nnz": int(csr_stats["nnz"])
        },
        "physical_layout": {
            "aligned_rows": int(n_rows_padded),
            "aligned_ell_width": int(aligned_width),
            "original_ell_width_max_nnz": int(max_nnz),
            "alignment_constraints": {
                "rows_must_be_multiple_of": int(ROW_ALIGNMENT),
                "ell_width_must_be_multiple_of": int(COL_ALIGNMENT)
            }
        },
        "buffer_stats": {
            "file_name": npy_filename,
            "buffer_size_kb": float(f"{buffer_kb:.2f}"),
            "dense_matrix_size_kb": float(f"{dense_kb:.2f}"),
            "nonzero_percent": float(f"{nonzero_percent:.4f}"),
            "format_description": "Interleaved BF16(val)/UINT16(idx). Structure: [Row0_Idx|Row0_Val, Row1_Idx|Row1_Val, ...]"
        }
    }
    
    meta_path = os.path.join(save_dir, f"{name}_ell_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    
    print("\n[Success]")
    print(f"  Saved to: {save_dir}")
    print(f"  Buffer: {buffer_kb:.2f} KB")
    print(f"  Metadata: {meta_path}")
    print(f"  Density: {nonzero_percent:.2f}%")

    # 確認表示
    #print_npu_matrix(save_path, aligned_width)

def print_npu_matrix(matrix_npy_path, cols):
    npu_data = np.load(matrix_npy_path)
    print(f"\nLoaded data shape: {npu_data.shape}, dtype: {npu_data.dtype}")
    
    # 1行目のIndexとValueを分離
    if npu_data.size >= cols * 2:
        row0_indices = npu_data[0:cols]
        row0_values_uint16 = npu_data[cols:cols*2]
        
        # --- BF16(uint16) -> Float32 変換ロジック ---
        # 1. uint16 を uint32 にキャスト
        # 2. 16ビット左シフト (float32の上位16ビットに配置)
        # 3. float32 としてメモリ解釈 (view)
        row0_values_float = (row0_values_uint16.astype(np.uint32) << 16).view(np.float32)

        print("--- First Row of NPU Matrix ---")
        print("Indices (uint16):", row0_indices)
        print("Values (decoded bf16):", row0_values_float)

if __name__ == "__main__":
    main()