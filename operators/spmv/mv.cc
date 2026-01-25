// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
extern "C" void event0();
extern "C" void event1();

#define REL_WRITE 0
#define REL_READ 1

#include "aie_kernel_utils.h"

#include <aie_api/aie.hpp>

using data_t = bfloat16;

void sparse_matvec_scalar(uint32_t m, 
                          uint32_t k, 
                          uint32_t ell_width, 
                          uint32_t row_offset, 
                          const bfloat16 *__restrict a, 
                          const bfloat16 *__restrict b, 
                          bfloat16 *__restrict c)
{
    event0();
    // 出力ポインタ移動
    c += row_offset * m;

    // 行ごとのストライド（要素数換算）: index領域 + value領域
    const uint32_t row_stride = 2 * ell_width;

    // 現在の行の先頭ポインタを初期化
    // Aは [idx...][val...], [idx...][val...] と並んでいる
    const bfloat16 *ptr_base = a;

    for (uint32_t row = 0; row < m; row++) {
        // インデックス部と値部のポインタセット
        const int16_t *ptr_idx = reinterpret_cast<const int16_t*>(ptr_base);
        const bfloat16 *ptr_val = ptr_base + ell_width;

        // アキュムレータを複数用意して依存関係を断ち切る (ILP向上)
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        uint32_t j = 0;

        // --- メインループ: 4要素ずつ処理 (Unrolling) ---
        // これにより、b[idx] のロード待ちの間に次の計算準備ができる
        for (; j + 3 < ell_width; j += 4) {
            // 値のロード (ポインタインクリメントはコンパイラが最適化しやすい)
            bfloat16 v0 = *ptr_val++;
            bfloat16 v1 = *ptr_val++;
            bfloat16 v2 = *ptr_val++;
            bfloat16 v3 = *ptr_val++;

            // インデックスのロード
            int16_t i0 = *ptr_idx++;
            int16_t i1 = *ptr_idx++;
            int16_t i2 = *ptr_idx++;
            int16_t i3 = *ptr_idx++;

            // 積和演算 (アキュムレータを分散)
            acc0 += (float)v0 * (float)b[i0];
            acc1 += (float)v1 * (float)b[i1];
            acc2 += (float)v2 * (float)b[i2];
            acc3 += (float)v3 * (float)b[i3];
        }

        // --- 残余ループ: 余った要素を処理 ---
        for (; j < ell_width; j++) {
            acc0 += (float)(*ptr_val++) * (float)b[*ptr_idx++];
        }

        // 行の処理完了、結果を格納
        c[row] = static_cast<bfloat16>(acc0 + acc1 + acc2 + acc3);

        // 次の行へベースポインタを進める
        ptr_base += row_stride;
    }
    event1();
}
void sparse_matvec_scalar_restrict(uint32_t m, 
                          uint32_t k, 
                          uint32_t ell_width, 
                          uint32_t row_offset, 
                          const bfloat16 *__restrict a, 
                          const bfloat16 *__restrict b, 
                          bfloat16 *__restrict c)
{
    event0();
    // 出力ポインタ移動
    c += row_offset * m;

    // 行ごとのストライド（要素数換算）: index領域 + value領域
    const uint32_t row_stride = 2 * ell_width;

    // 現在の行の先頭ポインタを初期化
    // Aは [idx...][val...], [idx...][val...] と並んでいる
    const bfloat16 *ptr_base = a;

    for (uint32_t row = 0; row < m; row++) {
        // インデックス部と値部のポインタセット
        const int16_t *__restrict ptr_idx = reinterpret_cast<const int16_t*>(ptr_base);
        const bfloat16 *__restrict ptr_val = ptr_base + ell_width;

        // アキュムレータを複数用意して依存関係を断ち切る (ILP向上)
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        uint32_t j = 0;

        // --- メインループ: 4要素ずつ処理 (Unrolling) ---
        // これにより、b[idx] のロード待ちの間に次の計算準備ができる
        for (; j + 3 < ell_width; j += 4) {
            // 値のロード (ポインタインクリメントはコンパイラが最適化しやすい)
            bfloat16 v0 = *ptr_val++;
            bfloat16 v1 = *ptr_val++;
            bfloat16 v2 = *ptr_val++;
            bfloat16 v3 = *ptr_val++;

            // インデックスのロード
            int16_t i0 = *ptr_idx++;
            int16_t i1 = *ptr_idx++;
            int16_t i2 = *ptr_idx++;
            int16_t i3 = *ptr_idx++;

            // 積和演算 (アキュムレータを分散)
            acc0 += (float)v0 * (float)b[i0];
            acc1 += (float)v1 * (float)b[i1];
            acc2 += (float)v2 * (float)b[i2];
            acc3 += (float)v3 * (float)b[i3];
        }

        // --- 残余ループ: 余った要素を処理 ---
        for (; j < ell_width; j++) {
            acc0 += (float)(*ptr_val++) * (float)b[*ptr_idx++];
        }

        // 行の処理完了、結果を格納
        c[row] = static_cast<bfloat16>(acc0 + acc1 + acc2 + acc3);

        // 次の行へベースポインタを進める
        ptr_base += row_stride;
    }
    event1();
}

void sparse_matvec_scalar_bf16acc(uint32_t m, 
                          uint32_t k, 
                          uint32_t ell_width, 
                          uint32_t row_offset, 
                          const bfloat16 *__restrict a, 
                          const bfloat16 *__restrict b, 
                          bfloat16 *__restrict c)
{
    event0();
    // 出力ポインタ移動
    c += row_offset * m;

    // 行ごとのストライド（要素数換算）: index領域 + value領域
    const uint32_t row_stride = 2 * ell_width;

    // 現在の行の先頭ポインタを初期化
    // Aは [idx...][val...], [idx...][val...] と並んでいる
    const bfloat16 *ptr_base = a;

    for (uint32_t row = 0; row < m; row++) {
        // インデックス部と値部のポインタセット
        const int16_t *ptr_idx = reinterpret_cast<const int16_t*>(ptr_base);
        const bfloat16 *ptr_val = ptr_base + ell_width;

        // アキュムレータを複数用意して依存関係を断ち切る (ILP向上)
        bfloat16 acc0 = 0.0f;
        bfloat16 acc1 = 0.0f;
        bfloat16 acc2 = 0.0f;
        bfloat16 acc3 = 0.0f;

        uint32_t j = 0;

        // --- メインループ: 4要素ずつ処理 (Unrolling) ---
        // これにより、b[idx] のロード待ちの間に次の計算準備ができる
        for (; j + 3 < ell_width; j += 4) {
            // 値のロード (ポインタインクリメントはコンパイラが最適化しやすい)
            bfloat16 v0 = *ptr_val++;
            bfloat16 v1 = *ptr_val++;
            bfloat16 v2 = *ptr_val++;
            bfloat16 v3 = *ptr_val++;

            // インデックスのロード
            int16_t i0 = *ptr_idx++;
            int16_t i1 = *ptr_idx++;
            int16_t i2 = *ptr_idx++;
            int16_t i3 = *ptr_idx++;

            // 積和演算 (アキュムレータを分散)
            acc0 += v0 * b[i0];
            acc1 += v1 * b[i1];
            acc2 += v2 * b[i2];
            acc3 += v3 * b[i3];
        }

        // --- 残余ループ: 余った要素を処理 ---
        for (; j < ell_width; j++) {
            acc0 += *ptr_val++ * b[*ptr_idx++];
        }

        // 行の処理完了、結果を格納
        c[row] = acc0 + acc1 + acc2 + acc3;

        // 次の行へベースポインタを進める
        ptr_base += row_stride;
    }
    event1();
}

// パイプライン効率を上げるためのヒント
// restrict: メモリ依存関係がないことを明示し、2つのロードユニット稼働を助ける
template <uint32_t r> //floatは4/8/16/32, int16,bfloat16は8/16/32/64まで対応
void sparse_matvec_vectorized(uint32_t m, 
                                uint32_t k,
                                uint32_t ell_width, 
                                uint32_t row_offset, 
                                const bfloat16 *__restrict a, 
                                const bfloat16 *__restrict b, 
                                bfloat16 *__restrict c)
{
    event0();
    c += row_offset * m;
    const uint32_t row_stride = 2 * ell_width;
    const bfloat16 *ptr_base = a;

    for (uint32_t row = 0; row < m; row++) {
        const int16_t *__restrict ptr_idx = reinterpret_cast<const int16_t*>(ptr_base);
        const bfloat16 *__restrict ptr_val = ptr_base + ell_width;

        // 仕様書にある "fp32 accumulator" を使用
        // これにより bfloat16 の積 -> float32 で加算 がハードウェアで行われる
        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();

        // コンパイラにパイプライン処理を強力に促す
        // (AIEコンパイラは通常これを自動で行いますが、明示も有効)
        uint32_t j = 0;
        
        // --- Vector Loop (r elems) ---
        // AIE-ML v2の "16-bit x r lanes" に対応
        for (; j + r <= ell_width; j += r) {
            // [Load Unit 1 & 2 Opportunity]
            // idx と val は連続領域なのでベクトルロード
            // ptr_idx と ptr_val のバンクが異なれば同時ロード可能
            aie::vector<int16_t, r> idx_vec = aie::load_unaligned_v<r>(ptr_idx);
            ptr_idx += r;
            
            aie::vector<bfloat16, r> val_vec = aie::load_unaligned_v<r>(ptr_val);
            ptr_val += r;

            // [Bottleneck: Software Gather]
            // ここが一番時間がかかる。
            // インデックスを使って b から値を拾う。
            // 仕様書の「スカラーからベクトル」機能をr回使うことになる。
            aie::vector<bfloat16, r> b_vec;
            
            
            // コンパイラによるLoop Unrolling + Pipeliningを期待
            #pragma unroll
            AIE_LOOP_MIN_ITERATION_COUNT(r)
            for (unsigned k = 0; k < r; ++k) {
                b_vec[k] = b[idx_vec[k]];
            }

            // [Vector Unit]
            // 仕様書の "Accumulate Unit" を使用
            // acc(FP32) += val(BF16) * b(BF16)
            acc = aie::mac(acc, val_vec, b_vec);
        }

        // --- Reduction ---
        // ベクトル(r個の部分和)を1つのスカラ値に畳み込む
        float total = aie::reduce_add(acc.template to_vector<float>());

        // --- Cleanup Loop (Scalar) ---
        #pragma unroll
        for (; j < ell_width; j++) {
            // ここも float で計算して精度維持
            total += (float)(*ptr_val++) * (float)b[*ptr_idx++];
        }

        // Store
        c[row] = static_cast<bfloat16>(total);
        ptr_base += row_stride;
    }
    event1();
}


template <uint32_t r> //floatは4/8/16/32, int16,bfloat16は8/16/32/64まで対応
void sparse_matvec_vectorized_aligned(uint32_t m, 
                                uint32_t k,
                                uint32_t ell_width, 
                                uint32_t row_offset, 
                                const bfloat16 *__restrict a, 
                                const bfloat16 *__restrict b, 
                                bfloat16 *__restrict c)
{
    event0();
    c += row_offset * m;
    const uint32_t row_stride = 2 * ell_width;
    const bfloat16 *ptr_base = a;
    #pragma unroll
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (uint32_t row = 0; row < m; row++) {
        const int16_t *__restrict ptr_idx = reinterpret_cast<const int16_t*>(ptr_base);
        const bfloat16 *__restrict ptr_val = ptr_base + ell_width;

        // 仕様書にある "fp32 accumulator" を使用
        // これにより bfloat16 の積 -> float32 で加算 がハードウェアで行われる
        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();

        // コンパイラにパイプライン処理を強力に促す
        // (AIEコンパイラは通常これを自動で行いますが、明示も有効)
        uint32_t j = 0;
        
        // --- Vector Loop (r elems) ---
        // AIE-ML v2の "16-bit x r lanes" に対応
        for (; j + r <= ell_width; j += r) {
            // [Load Unit 1 & 2 Opportunity]
            // idx と val は連続領域なのでベクトルロード
            // ptr_idx と ptr_val のバンクが異なれば同時ロード可能
            aie::vector<int16_t, r> idx_vec = aie::load_v<r>(ptr_idx);
            ptr_idx += r;
            
            aie::vector<bfloat16, r> val_vec = aie::load_v<r>(ptr_val);
            ptr_val += r;

            // [Bottleneck: Software Gather]
            // ここが一番時間がかかる。
            // インデックスを使って b から値を拾う。
            // 仕様書の「スカラーからベクトル」機能をr回使うことになる。
            aie::vector<bfloat16, r> b_vec;
            
            #pragma unroll
            AIE_LOOP_MIN_ITERATION_COUNT(r)
            for (unsigned k = 0; k < r; ++k) {
                b_vec[k] = b[idx_vec[k]];
            }

            // [Vector Unit]
            // 仕様書の "Accumulate Unit" を使用
            // acc(FP32) += val(BF16) * b(BF16)
            acc = aie::mac(acc, val_vec, b_vec);
        }

        // --- Reduction ---
        // ベクトル(r個の部分和)を1つのスカラ値に畳み込む
        float total = aie::reduce_add(acc.template to_vector<float>());

        // Store
        c[row] = static_cast<bfloat16>(total);
        ptr_base += row_stride;
    }
    event1();
}
#define MAX_K 1000
template <uint32_t r>
void sparse_matvec_vectorized_aligned_notb(
                                uint32_t m, 
                                uint32_t k,          // ループ境界として使うので k は残す
                                uint32_t ell_width, 
                                uint32_t row_offset, 
                                const bfloat16 *__restrict a, 
                                bfloat16 *__restrict c)
{
    // ★修正: 内部バッファを最大サイズで確保
    // staticをつけることでヒープ/スタックではなくデータメモリ領域に配置されます
    alignas(32) static bfloat16 b_internal[MAX_K];

    event0();

    // Trace用: 初回だけ初期化するなどの処理を入れると良いですが、
    // ランダムアクセス負荷の計測目的なら、未初期化(ゴミデータ)でもアクセス挙動は同じです。
    // 必要ならここで初期化してください。
    // if (row_offset == 0) { ... } 

    c += row_offset * m;
    const uint32_t row_stride = 2 * ell_width;
    const bfloat16 *ptr_base = a;
    
    #pragma unroll
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (uint32_t row = 0; row < m; row++) {
        const int16_t *__restrict ptr_idx = reinterpret_cast<const int16_t*>(ptr_base);
        const bfloat16 *__restrict ptr_val = ptr_base + ell_width;

        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
        uint32_t j = 0;
        
        for (; j + r <= ell_width; j += r) {
            aie::vector<int16_t, r> idx_vec = aie::load_v<r>(ptr_idx);
            ptr_idx += r;
            
            aie::vector<bfloat16, r> val_vec = aie::load_v<r>(ptr_val);
            ptr_val += r;

            aie::vector<bfloat16, r> b_vec;
            
            #pragma unroll
            AIE_LOOP_MIN_ITERATION_COUNT(r)
            for (unsigned k_idx = 0; k_idx < r; ++k_idx) {
                // ★修正: 引数 b の代わりに 内部バッファ b_internal を使用
                // ランダムアクセス(Gather)の負荷はこれで再現されます
                int idx = idx_vec[k_idx] % MAX_K;
                b_vec[k_idx] = b_internal[idx];
            }

            acc = aie::mac(acc, val_vec, b_vec);
        }

        float total = aie::reduce_add(acc.template to_vector<float>());
        c[row] = static_cast<bfloat16>(total);
        ptr_base += row_stride;
    }
    event1();
}

void sparse_matvec_scalar_easy(uint32_t m, 
                          uint32_t k, 
                          uint32_t ell_width, 
                          uint32_t row_offset, 
                          const bfloat16 *__restrict a, 
                          const bfloat16 *__restrict b, 
                          bfloat16 *__restrict c)
{
    event0();
    // 出力ポインタ移動
    c += row_offset * m;

    // 行ごとのストライド（要素数換算）: index領域 + value領域
    const uint32_t row_stride = 2 * ell_width;

    // 現在の行の先頭ポインタを初期化
    // Aは [idx...][val...], [idx...][val...] と並んでいる
    const bfloat16 *ptr_base = a;

    for (uint32_t row = 0; row < m; row++) {
        // インデックス部と値部のポインタセット
        const int16_t *ptr_idx = reinterpret_cast<const int16_t*>(ptr_base);
        const bfloat16 *ptr_val = ptr_base + ell_width;

        float acc = 0.0f;
        AIE_LOOP_MIN_ITERATION_COUNT(4)
        for (uint32_t j = 0; j < ell_width; j++) {
            acc += (float)(*ptr_val++) * (float)b[*ptr_idx++];
        }
        // 行の処理完了、結果を格納
        c[row] = static_cast<bfloat16>(acc);
        // 次の行へベースポインタを進める
        ptr_base += row_stride;
    }
    event1();
}


void matvec_scalar(uint32_t m, uint32_t k, uint32_t row_offset, 
                   const bfloat16 *__restrict a, 
                   const bfloat16 *__restrict b, 
                   bfloat16 *__restrict c)
{
    // 出力先のポインタを初期位置へ
    c += row_offset * m;

    for (uint32_t row = 0; row < m; row++) {
        // 依存関係を断ち切るためにアキュムレータを複数用意する (ILP向上)
        // VLIWのパイプラインを埋めるために4並列程度に展開
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        const bfloat16 *ptr_a = a + (row * k);
        const bfloat16 *ptr_b = b;
        
        uint32_t i = 0;
        
        // メインループ: 4要素ずつ処理 (Unrolling)
        for (; i + 3 < k; i += 4) {
            // bfloat16をfloatにキャストして演算
            acc0 += (float)(*ptr_a++) * (float)(*ptr_b++);
            acc1 += (float)(*ptr_a++) * (float)(*ptr_b++);
            acc2 += (float)(*ptr_a++) * (float)(*ptr_b++);
            acc3 += (float)(*ptr_a++) * (float)(*ptr_b++);
        }

        // 残りの要素を処理 (Cleanup loop)
        for (; i < k; i++) {
            acc0 += (float)(*ptr_a++) * (float)(*ptr_b++);
        }

        // 部分和を合計
        float total = (acc0 + acc1) + (acc2 + acc3);
        
        // 結果を格納
        c[row] = static_cast<bfloat16>(total);
    }
}


template <uint32_t r>
void matvec_vectorized(uint32_t m,
                       uint32_t k,
                       uint32_t row_offset,
                       const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b,
                       bfloat16 *__restrict c)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    c += row_offset * m;
    bfloat16 *c_end = c + m;
    const bfloat16 *b_end = b + k;
    for (; c < c_end; c++) {
        aie::accum acc = aie::zeros<accfloat, r>();
        // The following two pragmas enable pipelining the zero-overhead loop, but they do assume that k is at least
        // two. This assumption should hold for any useful use of this function; if k were one, this would be a simple
        // scalar multiplication of a vector.
        #pragma clang loop pipeline(disable)
        AIE_LOOP_MIN_ITERATION_COUNT(8)
        
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        *c = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
    event1();
}

extern "C" {

void matvec_scalar_bf16_bf16(uint32_t m,
                             uint32_t k,
                             uint32_t row_offset,
                             bfloat16 *a_in,
                             bfloat16 *b_in,
                             bfloat16 *c_out)
{
    matvec_scalar(m, k, row_offset, a_in, b_in, c_out);
}

void matvec_vectorized_bf16_bf16(uint32_t m,
                                 uint32_t k,
                                 uint32_t row_offset,
                                 bfloat16 *a_in,
                                 bfloat16 *b_in,
                                 bfloat16 *c_out)
{
    matvec_vectorized<64>(m, k, row_offset, a_in, b_in, c_out);
}

void sparse_matvec_scalar_bf16_bf16(uint32_t m, 
                                     uint32_t k, 
                                     uint32_t ell_width, 
                                     uint32_t row_offset, 
                                     const bfloat16 *a_in, 
                                     const bfloat16 *b_in, 
                                     bfloat16 *c_out)
{
    sparse_matvec_scalar_restrict(m, k, ell_width, row_offset, a_in, b_in, c_out);

}

void sparse_matvec_vectorized_bf16_bf16(uint32_t m, 
                                        uint32_t k, 
                                        uint32_t ell_width, 
                                        uint32_t row_offset, 
                                        const bfloat16 *a_in, 
                                        const bfloat16 *b_in, 
                                        bfloat16 *c_out)
{
    sparse_matvec_vectorized_aligned<32>(m, k, ell_width, row_offset, a_in, b_in, c_out);
    //sparse_matvec_vectorized_aligned_notb<32>(m, k, ell_width, row_offset, a_in, c_out);
} 

} // extern "C"