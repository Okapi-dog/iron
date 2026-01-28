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

template <uint32_t r = 32> 
void sell32_spmv_kernel(
    uint32_t num_blocks,
    uint32_t ell_width,
    const bfloat16 *__restrict data_ptr, 
    const bfloat16 *__restrict vec_x,   
    bfloat16 *__restrict vec_y          
)
{
    // パイプライン最適化のためのヒント: 
    // restrictポインタであることを明示し、ポインタ間のエイリアス（干渉）がないことを伝えます
    
    event0();

    const uint16_t *__restrict ptr_idx = reinterpret_cast<const uint16_t*>(data_ptr);
    const bfloat16 *__restrict ptr_val = data_ptr + 32; 

    const int block_stride = 64; 

    for (uint32_t b = 0; b < num_blocks; b++) {

        aie::accum<accfloat, 16> acc0 = aie::zeros<accfloat, 16>();
        aie::accum<accfloat, 16> acc1 = aie::zeros<accfloat, 16>();

        const uint16_t *__restrict p_idx_curr = ptr_idx;
        const bfloat16 *__restrict p_val_curr = ptr_val;
        
        // 最低反復回数の保証（パイプライン充填のため）
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_UNROLL(4)
        AIE_LOOP_MIN_ITERATION_COUNT(8)
        for (uint32_t k = 0; k < ell_width; k++) {
            
            // 1. ロード (Load Units)
            // 前半・後半を一気にロードします。
            // AIEは2つのロードユニットを持つため、並列ロードが期待できます。
            aie::vector<uint16_t, 16> idx0 = aie::load_v<16>(p_idx_curr);
            aie::vector<uint16_t, 16> idx1 = aie::load_v<16>(p_idx_curr + 16);
            
            aie::vector<bfloat16, 16> val0 = aie::load_v<16>(p_val_curr);
            aie::vector<bfloat16, 16> val1 = aie::load_v<16>(p_val_curr + 16);

            p_idx_curr += block_stride;
            p_val_curr += block_stride;

            // 2. Gather
            
            aie::vector<bfloat16, 16> x_gathered0;
            aie::vector<bfloat16, 16> x_gathered1;

            AIE_LOOP_UNROLL_FULL
            for (int i = 0; i < 16; i++) {
                x_gathered0[i] = vec_x[idx0[i]];
                x_gathered1[i] = vec_x[idx1[i]];
            }

            // 3. MAC (Vector Unit)
            acc0 = aie::mac(acc0, val0, x_gathered0);
            acc1 = aie::mac(acc1, val1, x_gathered1);
        }

        aie::store_v(vec_y, acc0.template to_vector<bfloat16>());
        aie::store_v(vec_y + 16, acc1.template to_vector<bfloat16>());
        
        ptr_idx = p_idx_curr;
        ptr_val = p_val_curr;
        
        vec_y += 32;
    }
    
    event1();
}


extern "C" { 


void sparse_matvec_vectorized_bf16_bf16(uint32_t m, 
                                        uint32_t k, 
                                        uint32_t ell_width, 
                                        uint32_t row_offset, 
                                        const bfloat16 *a_in, 
                                        const bfloat16 *b_in,
                                        bfloat16 *c_out
                                    )
{
    sparse_matvec_vectorized_aligned<32>(m, k, ell_width, row_offset, a_in, b_in, c_out);
} 


void sell32_spmv_vectorized_bf16_bf16(
    uint32_t num_blocks,
    uint32_t ell_width,
    const bfloat16 *data_ptr,
    const bfloat16 *vec_x,
    bfloat16 *vec_y
)
{
    sell32_spmv_kernel<32>(num_blocks, ell_width, data_ptr, vec_x, vec_y);
}

} // extern "C"