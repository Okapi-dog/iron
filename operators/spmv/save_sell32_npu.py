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

# --- 2. SELL-32形式にパッキングする関数 ---
def pack_for_xdna_sell32(ell_data, ell_indices):
    """
    SELL-32形式に変換する。
    """
    rows, width = ell_data.shape
    BLOCK_SIZE = 32
    
    assert rows % BLOCK_SIZE == 0, f"Rows ({rows}) must be a multiple of {BLOCK_SIZE} for SELL-32"
    
    num_blocks = rows // BLOCK_SIZE

    # 1. データをブロック単位に変形
    # Shape: (NumBlocks, 32, Width)
    data_blocked = ell_data.reshape(num_blocks, BLOCK_SIZE, width)
    indices_blocked = ell_indices.astype(np.uint16).reshape(num_blocks, BLOCK_SIZE, width)

    # 2. 転置 (Transpose) して、[Block][Width][RowInBlock] の順にする
    # Shape: (NumBlocks, Width, 32)
    data_transposed = data_blocked.transpose(0, 2, 1)
    indices_transposed = indices_blocked.transpose(0, 2, 1)

    # 3. 値の変換 (Float32 -> BF16 as Uint16)
    vals_bf16 = float32_to_bf16_bits_as_uint16(data_transposed)

    # 4. Col(Indices) と Val をインターリーブ
    # Stackを使って新しい次元を作る: Shape (NumBlocks, Width, 2, 32)
    combined = np.stack((indices_transposed, vals_bf16), axis=2)

    # 5. 一次元に平坦化
    # 順序: Block0 -> W0 -> (Inds(32), Vals(32)) -> W1...
    packed_buffer = combined.reshape(-1)
    
    return packed_buffer

# --- 3. メイン処理 (直接ブロックアライメント指定版) ---
def process_matrix_for_sell32(
    group, 
    name, 
    output_dir="npu_data", 
    col_alignment=8,
    block_alignment=64  # ここで指定したブロック数の倍数になるようパディングする
):
    """
    SELL-32を作成する。
    最終的なブロック総数が block_alignment の倍数になるようにパディングを行う。
    """
    SELL_ROW_SIZE = 32  # SELL-32なので高さは32固定
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Processing {group}/{name} for SELL-32 ---")
    print(f"Target Block Alignment: Must be a multiple of {block_alignment} blocks")

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
    print("Converting to CSR -> Standard ELL first...")
    sparse_mtx = scipy.io.mmread(mtx_file_path)
    csr = sparse_mtx.tocsr()
    
    n_rows_orig = csr.shape[0]
    
    # --- [変更点] ブロック数のパディング計算 ---
    
    # 1. データ格納に最低限必要なブロック数 (32行単位)
    min_blocks_needed = (n_rows_orig + SELL_ROW_SIZE - 1) // SELL_ROW_SIZE
    
    # 2. block_alignment (64など) の倍数に切り上げ
    if min_blocks_needed % block_alignment == 0:
        final_num_blocks = min_blocks_needed
        # 特例: データが空でない限り、0ブロックにはしない (最低 block_alignment 分は確保)
        if final_num_blocks == 0 and n_rows_orig > 0:
            final_num_blocks = block_alignment
    else:
        final_num_blocks = ((min_blocks_needed // block_alignment) + 1) * block_alignment
        
    # 3. 最終的な行数を計算 (ブロック数 * 32)
    n_rows_padded = final_num_blocks * SELL_ROW_SIZE
    
    print(f"Original Rows: {n_rows_orig}")
    print(f"Min Blocks needed: {min_blocks_needed}")
    print(f"Final Blocks: {final_num_blocks} (Aligned to {block_alignment})")
    print(f"Final Padded Rows: {n_rows_padded}")

    # --- 列幅(ELL Width)のパディング計算 ---
    row_nnz = np.diff(csr.indptr)
    max_nnz = row_nnz.max() if n_rows_orig > 0 else 0
    
    if max_nnz % col_alignment == 0:
        aligned_width = int(max_nnz)
    else:
        aligned_width = int(((max_nnz // col_alignment) + 1) * col_alignment)
        
    print(f"Max NNZ: {max_nnz} -> Aligned SELL-32 Width: {aligned_width}")
    
    # ELL配列確保
    ell_data = np.zeros((n_rows_padded, aligned_width), dtype=np.float32)
    # Indexのパディング値は 0 (無効値)
    ell_indices = np.zeros((n_rows_padded, aligned_width), dtype=np.int32)
    
    # データ埋め込み (CSR -> ELL)
    for i in range(n_rows_orig):
        start = csr.indptr[i]
        end = csr.indptr[i+1]
        n = end - start
        if n > 0:
            copy_len = min(n, aligned_width)
            ell_data[i, :copy_len] = csr.data[start:start+copy_len]
            ell_indices[i, :copy_len] = csr.indices[start:start+copy_len]

    # --- SELL-32 パッキング ---
    print("Packing for XDNA SELL-32...")
    npu_buffer = pack_for_xdna_sell32(ell_data, ell_indices)
    
    # 保存
    save_path = os.path.join(save_dir, f"{name}_xdna_sell32.npy")
    np.save(save_path, npu_buffer)

    # メタデータ保存
    meta_path = os.path.join(save_dir, f"{name}_sell32_meta.json")
    meta_info = {
        "name": name,
        "format": "SELL-32",
        "rows": int(n_rows_padded),
        "original_rows": int(n_rows_orig),
        "cols": int(csr.shape[1]),
        "ell_width": int(aligned_width),
        "block_size_rows": SELL_ROW_SIZE,
        "total_blocks": int(final_num_blocks),
        "block_alignment": block_alignment,
        "buffer_size_bytes": int(npu_buffer.size * 2),
        "layout": "Block(32) -> Width -> Interleaved(Indices[32], Values[32])",
        "dtype": "uint16 (indices and bf16 values)"
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    
    print(f"Saved SELL-32 buffer to: {save_path}")
    print(f"Saved Metadata to: {meta_path}")

# --- デバッグ表示用関数 ---
def print_sell32_head(matrix_npy_path, width):
    if not os.path.exists(matrix_npy_path):
        print(f"File not found: {matrix_npy_path}")
        return
        
    npu_data = np.load(matrix_npy_path)
    print(f"\n--- Loaded SELL-32 Data Check: {matrix_npy_path} ---")
    print(f"Total Elements (uint16): {npu_data.size}")
    
    # ブロック0のデータを表示
    chunk0 = npu_data[0:64]
    inds_0 = chunk0[0:32]
    vals_0 = chunk0[32:64]
    
    print(f"Block 0, Width 0 sample:")
    print(f"  Indices: {inds_0[:8]} ...")
    print(f"  Values : {vals_0[:8]} ...")

if __name__ == "__main__":
    # 使用例: ブロックアライメントを64に指定
    process_matrix_for_sell32(
        "ML_Graph", 
        "mnist_test_norm_10NN", 
        col_alignment=1,
        block_alignment=128
    )
    
    # 確認用
    base_dir = "npu_data/mnist_test_norm_10NN" 
    npy_file = os.path.join(base_dir, "mnist_test_norm_10NN_xdna_sell32.npy")
    json_file = os.path.join(base_dir, "mnist_test_norm_10NN_sell32_meta.json")
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            meta = json.load(f)
            print_sell32_head(npy_file, width=meta['ell_width'])