import numpy as np
import scipy.sparse
import scipy.io
import json
import os
import glob
import ssgetpy

# ==========================================
# 1. 設定エリア
# ==========================================
USE_RANDOM = True

OUTPUT_DIR = "npu_data"
BLOCK_ALIGNMENT = 128    # ブロック数のアライメント単位 (例: 128ブロック単位で切り上げ)
ELL_WIDTH_ALIGNMENT = 1  # ELL幅のアライメント単位 (例: 1ならアライメントなし、4なら4の倍数)

# [Mode A] Random
RAND_M = 5000
RAND_K = 10000
RAND_ELL_WIDTH = 64

# [Mode B] Download
SS_GROUP = "ML_Graph"
SS_NAME  = "mnist_test_norm_10NN"
np.random.seed(42)  # 再現性のためのシード設定

# ==========================================
# 2. コア変換ロジック
# ==========================================

def float32_to_bf16_bits_as_uint16(arr_float32):
    u32_view = arr_float32.view(np.uint32)
    bf16_bits_u32 = u32_view >> 16
    return bf16_bits_u32.astype(np.uint16)

def pack_for_xdna_sell32(ell_data, ell_indices):
    rows, width = ell_data.shape
    BLOCK_SIZE = 32
    
    assert rows % BLOCK_SIZE == 0, f"Rows {rows} must be multiple of 32"
    num_blocks = rows // BLOCK_SIZE

    data_blocked = ell_data.reshape(num_blocks, BLOCK_SIZE, width)
    indices_blocked = ell_indices.astype(np.uint16).reshape(num_blocks, BLOCK_SIZE, width)

    data_transposed = data_blocked.transpose(0, 2, 1)
    indices_transposed = indices_blocked.transpose(0, 2, 1)

    vals_bf16 = float32_to_bf16_bits_as_uint16(data_transposed)

    combined = np.stack((indices_transposed, vals_bf16), axis=2)
    return combined.reshape(-1)

def calc_padded_rows(target_rows, block_alignment):
    SELL_ROW_SIZE = 32
    min_blocks = (target_rows + SELL_ROW_SIZE - 1) // SELL_ROW_SIZE
    
    if min_blocks == 0 and target_rows > 0:
        min_blocks = 1

    if min_blocks % block_alignment == 0:
        final_blocks = min_blocks
        if final_blocks == 0: final_blocks = block_alignment
    else:
        final_blocks = ((min_blocks // block_alignment) + 1) * block_alignment
        
    return final_blocks * SELL_ROW_SIZE, final_blocks

# ==========================================
# 3. データソース処理 (NNZを返すように変更)
# ==========================================

def get_random_ell_matrix(output_root_dir):
    name_prefix = f"random_M{RAND_M}_K{RAND_K}"
    print(f"--- [Random Mode] Generating {name_prefix}, ELL={RAND_ELL_WIDTH} ---")
    
    padded_rows, total_blocks = calc_padded_rows(RAND_M, BLOCK_ALIGNMENT)
    
    ell_data = np.random.rand(padded_rows, RAND_ELL_WIDTH).astype(np.float32)
    ell_indices = np.zeros((padded_rows, RAND_ELL_WIDTH), dtype=np.int32)
    
    for r in range(padded_rows):
        cols = np.random.choice(RAND_K, RAND_ELL_WIDTH, replace=False)
        cols.sort()
        ell_indices[r, :] = cols

    # Randomの場合、NNZは (論理行数 * ELL幅) とする
    actual_nnz = RAND_M * RAND_ELL_WIDTH 

    # --- mtx保存処理 (省略可だが維持) ---
    print("  -> Converting to sparse format for .mtx output ...")
    row_indices = np.repeat(np.arange(padded_rows), RAND_ELL_WIDTH)
    col_indices_flat = ell_indices.flatten()
    data_flat = ell_data.flatten()
    sparse_matrix = scipy.sparse.csr_matrix(
        (data_flat, (row_indices, col_indices_flat)), 
        shape=(padded_rows, RAND_K)
    )
    save_dir = os.path.join(output_root_dir, name_prefix)
    os.makedirs(save_dir, exist_ok=True)
    scipy.io.mmwrite(os.path.join(save_dir, f"{name_prefix}.mtx"), sparse_matrix)

    return ell_data, ell_indices, RAND_M, RAND_K, actual_nnz, name_prefix

def get_downloaded_ell_matrix(output_dir):
    print(f"--- [Download Mode] Fetching {SS_GROUP}/{SS_NAME} ---")
    
    os.makedirs(output_dir, exist_ok=True)
    results = ssgetpy.search(group=SS_GROUP, name=SS_NAME, limit=1)
    if not results: raise ValueError("Matrix not found.")
    
    item = results[0]
    item.download(format='MM', destpath=output_dir, extract=True)
    mtx_pattern = os.path.join(output_dir, "**", f"{item.name}.mtx")
    files = glob.glob(mtx_pattern, recursive=True)
    mtx_path = files[0]
    
    sparse_mtx = scipy.io.mmread(mtx_path)
    csr = sparse_mtx.tocsr()
    M, K = csr.shape
    actual_nnz = csr.nnz  # 実際の非ゼロ数
    
    padded_rows, total_blocks = calc_padded_rows(M, BLOCK_ALIGNMENT)
    
    row_nnz = np.diff(csr.indptr)
    max_nnz = row_nnz.max() if M > 0 else 0
    
    ell_width_aligned = int(max_nnz)
    if ell_width_aligned % ELL_WIDTH_ALIGNMENT != 0:
        ell_width_aligned = ((ell_width_aligned // ELL_WIDTH_ALIGNMENT) + 1) * ELL_WIDTH_ALIGNMENT
        
    ell_data = np.zeros((padded_rows, ell_width_aligned), dtype=np.float32)
    ell_indices = np.zeros((padded_rows, ell_width_aligned), dtype=np.int32)
    
    for i in range(M):
        start = csr.indptr[i]
        end = csr.indptr[i+1]
        n = end - start
        if n > 0:
            copy_len = min(n, ell_width_aligned)
            ell_data[i, :copy_len] = csr.data[start:start+copy_len]
            ell_indices[i, :copy_len] = csr.indices[start:start+copy_len]
            
    return ell_data, ell_indices, M, K, actual_nnz, SS_NAME

# ==========================================
# 4. メイン実行
# ==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_type = "Random" if USE_RANDOM else "SuiteSparse"

    # データ取得
    if USE_RANDOM:
        ell_data, ell_indices, orig_rows, orig_cols, actual_nnz, name_prefix = get_random_ell_matrix(OUTPUT_DIR)
    else:
        ell_data, ell_indices, orig_rows, orig_cols, actual_nnz, name_prefix = get_downloaded_ell_matrix(OUTPUT_DIR)

    # パッキング
    print("--- Packing data to XDNA SELL-32 format ---")
    packed_buffer = pack_for_xdna_sell32(ell_data, ell_indices)
    
    padded_rows, ell_width = ell_data.shape
    total_blocks = padded_rows // 32

    # 保存
    save_dir = os.path.join(OUTPUT_DIR, name_prefix)
    os.makedirs(save_dir, exist_ok=True)

    npy_filename = f"{name_prefix}_xdna_sell32.npy"
    npy_path = os.path.join(save_dir, npy_filename)
    np.save(npy_path, packed_buffer)
    
    # ----------------------------------------------------
    # ★ メタデータ計算 (サイズ、密度など)
    # ----------------------------------------------------
    # 1. バッファサイズ (KB)
    buffer_bytes = packed_buffer.size * 2  # uint16=2bytes
    buffer_kb = buffer_bytes / 1024.0

    # 2. 密行列だった場合のサイズ (KB) (BF16=2bytes計算)
    dense_bytes = orig_rows * orig_cols * 2
    dense_kb = dense_bytes / 1024.0

    # 3. 非ゼロ率 (Sparsity)
    total_elements = orig_rows * orig_cols
    nonzero_percent = (actual_nnz / total_elements * 100.0) if total_elements > 0 else 0.0

    # 4. JSON構築
    meta_info = {
        "dataset_info": {
            "name": name_prefix,
            "source_type": source_type,
            "format_version": "SELL-32_NPU"
        },
        "logical_shape": {
            "rows": int(orig_rows),
            "cols": int(orig_cols),
            "actual_nnz": int(actual_nnz)
        },
        "physical_layout": {
            "aligned_rows": int(padded_rows),
            "aligned_ell_width": int(ell_width),
            "total_blocks": int(total_blocks),
            "block_size": 32,
            # アライメント制約の説明をわかりやすく変更
            "alignment_constraints": {
                "block_count_must_be_multiple_of": int(BLOCK_ALIGNMENT),
                "ell_width_must_be_multiple_of": int(ELL_WIDTH_ALIGNMENT)
            }
        },
        "buffer_stats": {
            "file_name": npy_filename,
            "buffer_size_kb": float(f"{buffer_kb:.2f}"),
            "dense_matrix_size_kb": float(f"{dense_kb:.2f}"),
            "nonzero_percent": float(f"{nonzero_percent:.4f}"),
            "format_description": "Interleaved BF16(val)/UINT16(idx). Structure: [Block] -> [Column] -> [Row(32): Index|Value]"
        }
    }

    meta_filename = f"{name_prefix}_sell32_meta.json"
    meta_path = os.path.join(save_dir, meta_filename)
    
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)

    print("\n[Success]")
    print(f"  Saved to: {save_dir}")
    print(f"  Buffer: {buffer_kb:.2f} KB (Dense equivalent: {dense_kb:.2f} KB)")
    print(f"  Density: {nonzero_percent:.2f}%")

if __name__ == "__main__":
    main()