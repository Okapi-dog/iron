import save_sparse_matrix

# 設定を変えて実行
save_path = save_sparse_matrix.save(
    use_random=True,
    rand_m=1024,
    rand_k=512,
    rand_nnz=32,
    output_dir="./experiment_data"
)

print(f"データ生成完了: {save_path}")