import numpy as np
import scipy.sparse
import scipy.io
import json
import os
import glob
import ssgetpy

# ==========================================
#　Unified Settings
# ==========================================
USE_RANDOM = True           # True: Random Mode, False: Download Mode
AUTO_PADDING = False        # True: 自動でPaddingして処理, False: アライメント不一致でエラー

OUTPUT_DIR = "npu_data"

# --- Common Random Settings ---
RAND_M = 368640             # 行数 (padding部分さえ気にすれば、いくらでも大きくできる)
RAND_K = 2048               # 列数(L1cacheの容量的に大体25600行ぐらいが最大)
RAND_NNZ_PER_ROW = 32       # Random生成時の1行あたりの非ゼロ要素数 (ELL Width相当)

# --- Download Settings ---
SS_GROUP = "ML_Graph"
SS_NAME  = "mnist_test_norm_10NN"

# --- SELL-32 Specific Alignment ---
SELL_BLOCK_ALIGNMENT = 32 # ブロック数のアライメント (例: 32ブロック単位)
SELL_ROW_SIZE = 32        # SELLの1ブロックあたりの行数 (固定)
SELL_WIDTH_ALIGNMENT = 1  # SELLではELL幅のアライメントは通常1

# --- ELL Specific Alignment ---
ELL_ROW_ALIGNMENT = 128   # 行数のアライメント
ELL_WIDTH_ALIGNMENT = 32  # ELL幅のアライメント

# Seed setting
np.random.seed(42)


def float32_to_bf16_bits_as_uint16(arr_float32):
    """Float32 -> BFloat16 bits (stored in uint16)"""
    u32_view = arr_float32.view(np.uint32)
    bf16_bits_u32 = u32_view >> 16
    return bf16_bits_u32.astype(np.uint16)

def pack_for_xdna_sell32(ell_data, ell_indices):
    """
    SELL-32 Format Packing
    Layout: [Block] -> [Column] -> [Row(32): [Index0, Val0, Index1, Val1...] ではなく [Index0...Index31, Val0...Val31]]
    """
    rows, width = ell_data.shape
    BLOCK_SIZE = 32
    
    assert rows % BLOCK_SIZE == 0, f"SELL32: Rows {rows} must be multiple of 32"
    num_blocks = rows // BLOCK_SIZE

    data_blocked = ell_data.reshape(num_blocks, BLOCK_SIZE, width)
    indices_blocked = ell_indices.astype(np.uint16).reshape(num_blocks, BLOCK_SIZE, width)

    # Transpose to: (Blocks, Width, RowsInBlock)
    data_transposed = data_blocked.transpose(0, 2, 1)
    indices_transposed = indices_blocked.transpose(0, 2, 1)

    vals_bf16 = float32_to_bf16_bits_as_uint16(data_transposed)

    # Interleave: [Index, Value] along last axis
    combined = np.stack((indices_transposed, vals_bf16), axis=2) # shape: (Blocks, Width, 2, 32) (conceptual)
    
    return combined.reshape(-1)

def pack_for_xdna_ell(ell_data, ell_indices):
    """
    ELL Format Packing
    Layout: [Row0_Idx, Row0_Val, Row1_Idx, Row1_Val...]
    """
    rows, width = ell_data.shape
    
    indices_uint16 = ell_indices.astype(np.uint16)
    values_bf16_as_uint16 = float32_to_bf16_bits_as_uint16(ell_data.astype(np.float32))
    
    # (Rows, 2, Width) -> Rowごとに [Index..., Value...]
    # 元コード: combined[:, 0, :] = indices; combined[:, 1, :] = values
    # flatten -> Row0_Idx(all cols), Row0_Val(all cols), Row1...
    combined = np.empty((rows, 2, width), dtype=np.uint16)
    combined[:, 0, :] = indices_uint16
    combined[:, 1, :] = values_bf16_as_uint16
    
    return combined.reshape(-1)

# ==========================================
# 3. データソース生成とMTX保存
# ==========================================

def generate_source_csr(output_dir):
    """
    共通のデータソースを生成し、CSR行列として返す。
    また、生の(パディングなし)MTXファイルを保存する。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if USE_RANDOM:
        name_prefix = f"random_M{RAND_M}_K{RAND_K}_ELL{RAND_NNZ_PER_ROW}"
        print(f"--- [Source] Generating Random CSR {name_prefix} ---")
        
        # 論理的なランダムデータ生成 (パディングなし)
        # 各行に固定数のNNZを持つように生成（元コードのロジックを継承）
        
        # データ生成用の一時バッファ
        temp_data = np.random.rand(RAND_M, RAND_NNZ_PER_ROW).astype(np.float32)
        temp_indices = np.zeros((RAND_M, RAND_NNZ_PER_ROW), dtype=np.int32)
        
        for r in range(RAND_M):
            cols = np.random.choice(RAND_K, RAND_NNZ_PER_ROW, replace=False)
            cols.sort()
            temp_indices[r, :] = cols
            
        # CSRに変換
        row_indices = np.repeat(np.arange(RAND_M), RAND_NNZ_PER_ROW)
        col_indices_flat = temp_indices.flatten()
        data_flat = temp_data.flatten()
        
        csr = scipy.sparse.csr_matrix(
            (data_flat, (row_indices, col_indices_flat)), 
            shape=(RAND_M, RAND_K)
        )
        base_name = name_prefix

    else:
        print(f"--- [Source] Downloading {SS_GROUP}/{SS_NAME} ---")
        results = ssgetpy.search(group=SS_GROUP, name=SS_NAME, limit=1)
        if not results: raise ValueError("Matrix not found.")
        
        item = results[0]
        item.download(format='MM', destpath=output_dir, extract=True)
        mtx_pattern = os.path.join(output_dir, "**", f"{item.name}.mtx")
        files = glob.glob(mtx_pattern, recursive=True)
        mtx_path = files[0]
        
        sparse_mtx = scipy.io.mmread(mtx_path)
        csr = sparse_mtx.tocsr()
        base_name = SS_NAME

    # MTX保存 (共通・パディングなし)
    save_dir = os.path.join(output_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)
    mtx_filename = f"{base_name}.mtx"
    mtx_path = os.path.join(save_dir, mtx_filename)
    
    print(f"--- [Source] Saving Reference MTX (Unpadded) to {mtx_path} ---")
    scipy.io.mmwrite(mtx_path, csr)
    
    return csr, base_name, save_dir

# ==========================================
# 4. パディングとアライメント処理
# ==========================================

def get_padded_numpy_arrays(csr, target_rows, target_width, format_name):
    """
    CSR行列を指定されたサイズにパディングし、DenseなNumpy配列(Index, Data)にして返す。
    AUTO_PADDING=False でサイズが合わない場合はエラーを出す。
    """
    M, K = csr.shape
    row_nnz = np.diff(csr.indptr)
    max_nnz = row_nnz.max() if M > 0 else 0
    
    # チェック
    if not AUTO_PADDING:
        if M != target_rows:
            raise ValueError(f"[{format_name}] Row alignment mismatch! Actual: {M}, Required: {target_rows}. Set AUTO_PADDING=True to fix.")
        if max_nnz > target_width:
            raise ValueError(f"[{format_name}] Width alignment mismatch! Actual Max NNZ: {max_nnz}, Target Width: {target_width}. Set AUTO_PADDING=True to fix.")
        # widthが target_width より小さい場合は、通常埋めるだけなので許容されることが多いが、
        # ここでは厳密に配列を作るため、target_widthに合わせる。
        
    print(f"   -> Padding [{format_name}]: ({M}, {max_nnz} [max]) -> ({target_rows}, {target_width})")

    # 配列確保 (0埋め)
    ell_data = np.zeros((target_rows, target_width), dtype=np.float32)
    ell_indices = np.zeros((target_rows, target_width), dtype=np.int32)
    
    # データ充填
    for i in range(M):
        start = csr.indptr[i]
        end = csr.indptr[i+1]
        n = end - start
        if n > 0:
            # target_widthを超えている場合はカット(基本的にはありえないが安全策)、足りない場合は0埋め(初期化済)
            copy_len = min(n, target_width)
            ell_data[i, :copy_len] = csr.data[start:start+copy_len]
            ell_indices[i, :copy_len] = csr.indices[start:start+copy_len]
            
    return ell_data, ell_indices

# ==========================================
# 5. 各フォーマットの実行フロー
# ==========================================

def run_sell32_flow(csr, base_name, save_dir):
    print(f"\n=== Processing SELL-32 Format for {base_name} ===")
    
    M, K = csr.shape
    row_nnz = np.diff(csr.indptr)
    max_nnz = int(row_nnz.max()) if M > 0 else 0
    
    # 1. アライメント計算
    # 行数計算: 32行(SELL_ROW_SIZE)のブロックが、SELL_BLOCK_ALIGNMENTの倍数になるように
    min_blocks = (M + SELL_ROW_SIZE - 1) // SELL_ROW_SIZE
    if min_blocks == 0 and M > 0: min_blocks = 1
    
    if min_blocks % SELL_BLOCK_ALIGNMENT == 0:
        final_blocks = min_blocks
        if final_blocks == 0: final_blocks = SELL_BLOCK_ALIGNMENT
    else:
        final_blocks = ((min_blocks // SELL_BLOCK_ALIGNMENT) + 1) * SELL_BLOCK_ALIGNMENT
    
    target_rows = final_blocks * SELL_ROW_SIZE
    
    # 幅計算
    target_width = max_nnz
    if target_width % SELL_WIDTH_ALIGNMENT != 0:
        target_width = ((target_width // SELL_WIDTH_ALIGNMENT) + 1) * SELL_WIDTH_ALIGNMENT
        
    # 2. パディング済みデータ取得
    ell_data, ell_indices = get_padded_numpy_arrays(csr, target_rows, target_width, "SELL-32")
    
    # 3. SELL形式へパッキング
    packed_buffer = pack_for_xdna_sell32(ell_data, ell_indices)
    
    # 4. 保存
    npy_filename = f"{base_name}_xdna_sell32.npy"
    npy_path = os.path.join(save_dir, npy_filename)
    np.save(npy_path, packed_buffer)
    
    # 5. メタデータ作成 (元のJSON構造を維持)
    buffer_bytes = packed_buffer.size * 2
    buffer_kb = buffer_bytes / 1024.0
    dense_kb = (M * K * 2) / 1024.0
    total_elements = M * K
    nonzero_percent = (csr.nnz / total_elements * 100.0) if total_elements > 0 else 0.0
    
    meta_info = {
        "dataset_info": {
            "name": base_name,
            "source_type": "Random" if USE_RANDOM else "SuiteSparse",
            "format_version": "SELL-32_NPU"
        },
        "logical_shape": {
            "rows": int(M),
            "cols": int(K),
            "actual_nnz": int(csr.nnz)
        },
        "physical_layout": {
            "aligned_rows": int(target_rows),
            "aligned_ell_width": int(target_width),
            "total_blocks": int(final_blocks),
            "block_size": 32,
            "alignment_constraints": {
                "block_count_must_be_multiple_of": int(SELL_BLOCK_ALIGNMENT),
                "ell_width_must_be_multiple_of": int(SELL_WIDTH_ALIGNMENT)
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
    
    meta_path = os.path.join(save_dir, f"{base_name}_sell32_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    print(f"   [SELL-32] Saved: {npy_filename}, Meta: {os.path.basename(meta_path)}")


def run_ell_flow(csr, base_name, save_dir):
    print(f"\n=== Processing ELL Format for {base_name} ===")
    
    M, K = csr.shape
    row_nnz = np.diff(csr.indptr)
    max_nnz = int(row_nnz.max()) if M > 0 else 0
    
    # 1. アライメント計算
    if M % ELL_ROW_ALIGNMENT == 0:
        target_rows = M
    else:
        target_rows = ((M // ELL_ROW_ALIGNMENT) + 1) * ELL_ROW_ALIGNMENT
        
    if max_nnz % ELL_WIDTH_ALIGNMENT == 0:
        target_width = max_nnz
    else:
        target_width = ((max_nnz // ELL_WIDTH_ALIGNMENT) + 1) * ELL_WIDTH_ALIGNMENT
        
    # 2. パディング済みデータ取得
    ell_data, ell_indices = get_padded_numpy_arrays(csr, target_rows, target_width, "ELL")
    
    # 3. ELL形式へパッキング
    packed_buffer = pack_for_xdna_ell(ell_data, ell_indices)
    
    # 4. 保存
    npy_filename = f"{base_name}_xdna_ell.npy"
    npy_path = os.path.join(save_dir, npy_filename)
    np.save(npy_path, packed_buffer)
    
    # 5. メタデータ作成 (元のJSON構造を維持)
    buffer_bytes = packed_buffer.size * 2
    buffer_kb = buffer_bytes / 1024.0
    dense_kb = (M * K * 2) / 1024.0
    total_elements = M * K
    nonzero_percent = (csr.nnz / total_elements * 100.0) if total_elements > 0 else 0.0

    meta_info = {
        "dataset_info": {
            "name": base_name,
            "source_type": "Random" if USE_RANDOM else "SuiteSparse",
            "format_version": "ELL_NPU"
        },
        "logical_shape": {
            "rows": int(M),
            "cols": int(K),
            "actual_nnz": int(csr.nnz)
        },
        "physical_layout": {
            "aligned_rows": int(target_rows),
            "aligned_ell_width": int(target_width),
            "original_ell_width_max_nnz": int(max_nnz),
            "alignment_constraints": {
                "rows_must_be_multiple_of": int(ELL_ROW_ALIGNMENT),
                "ell_width_must_be_multiple_of": int(ELL_WIDTH_ALIGNMENT)
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
    
    meta_path = os.path.join(save_dir, f"{base_name}_ell_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    print(f"   [ELL] Saved: {npy_filename}, Meta: {os.path.basename(meta_path)}")


# ==========================================
# 6. メイン実行 (Main Execution)
# ==========================================

def main():
    # 1. データソース取得 (共通処理・MTX保存)
    csr_matrix, base_name, save_dir = generate_source_csr(OUTPUT_DIR)
    
    # 2. SELL-32形式の生成
    run_sell32_flow(csr_matrix, base_name, save_dir)
    
    # 3. ELL形式の生成
    run_ell_flow(csr_matrix, base_name, save_dir)
    
    print("\n[All Done]")
    print(f"Results saved in: {save_dir}")

if __name__ == "__main__":
    main()