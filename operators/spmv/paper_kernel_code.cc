// 疎行列ベクトル積 (SpMV) の単一コアにおけるカーネル疑似コード
// y = A * xを計算する
// A: 疎行列 (ELLPACK形式)  A_col: 列インデックス配列, A_val: 非ゼロ要素配列
// x: 入力ベクトル, y: 出力ベクトル
// ell_width: 1行あたりの最大非ゼロ要素数 (ELLPACKの列数)
// Rows: 単一コアが処理する行数
for (uint32_t i = 0; i < Rows; i++) {
    //部分和を保存するアキュムレータをゼロで初期化
    vector_acc = vector_zeros<float32, 32>();
    
    // 疎行列 A をELL形式で表した際の列インデックス(col)および非ゼロ要素(val)へのポインタ
    int16_t  *ptr_A_col = &A_col[i * ell_width];
    bfloat16 *ptr_A_val = &A_val[i * ell_width];

    // 非ゼロ要素数(ell_width)にわたるループ
    for (uint32_t j = 0; j < ell_width; j += 32) {
        // 【ベクトルロード】疎行列 A のデータをSIMDレジスタへ一括転送
        vector_A_col = vector_load<32>(ptr_A_col);
        vector_A_val = vector_load<32>(ptr_A_val);

        // 【スカラロード】不連続アクセスによるボトルネック
        // A の列インデックスに基づき、入力ベクトル x から要素を個別に読み出す
        aie::vector<bfloat16, 32> vector_x;
        for (int k = 0; k < 32; k++) {
            vector_x[k] = x[vector_A_col[k]]; 
        }
        
        // 積和演算(ベクトルUnitで一括計算)
        vector_acc = vector_mac(vector_acc, vector_A_val, vector_x);

        ptr_A_col += 32;
        ptr_A_val += 32;
    }

    // 32要素の加算結果を統合し、出力ベクトル y の第 i 要素に格納
    float row_sum = reduce_add(vector_acc);
    y[i] = (bfloat16)row_sum;
}

