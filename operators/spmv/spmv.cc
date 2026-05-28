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
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(10)
    for (uint32_t row = 0; row < m; row++) {
        const uint16_t *__restrict ptr_idx = reinterpret_cast<const uint16_t*>(ptr_base);
        const bfloat16 *__restrict ptr_val = ptr_base + ell_width;

        // 仕様書にある "fp32 accumulator" を使用
        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();

        // コンパイラにパイプライン処理を強力に促す
        // (AIEコンパイラは通常これを自動で行いますが、明示も有効)
        uint32_t j = 0;
        
        // --- Vector Loop (r elems) ---
        // AIE-ML v2の "16-bit x r lanes" に対応
        for (; j + r <= ell_width; j += r) {
            aie::vector<uint16_t, r> idx_vec = aie::load_v<r>(ptr_idx);
            aie::vector<bfloat16, r> val_vec = aie::load_v<r>(ptr_val);
            ptr_idx += r;
            ptr_val += r;

            aie::vector<bfloat16, r> b_vec;
            
            AIE_LOOP_UNROLL_FULL
            for (unsigned k = 0; k < r; ++k) {
                b_vec[k] = b[idx_vec[k]];
            }

            acc = aie::mac(acc, val_vec, b_vec);
        }

        // --- Reduction ---
        float total = aie::reduce_add(acc.template to_vector<float>());

        // Store
        c[row] = static_cast<bfloat16>(total);
        
        ptr_base += row_stride;
    }
    event1();
}

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

        // 最低反復回数の保証（パイプライン充填のため）
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(4)
        for (uint32_t k = 0; k < ell_width; k++) {
            
            // 1. ロード (Load Units)
            // 前半・後半を一気にロードします。
            // AIEは2つのロードユニットを持つため、並列ロードが期待できます。
            aie::vector<uint16_t, 16> idx0 = aie::load_v<16>(ptr_idx);
            aie::vector<uint16_t, 16> idx1 = aie::load_v<16>(ptr_idx + 16);
            
            aie::vector<bfloat16, 16> val0 = aie::load_v<16>(ptr_val);
            aie::vector<bfloat16, 16> val1 = aie::load_v<16>(ptr_val + 16);
            ptr_idx += block_stride;
            ptr_val += block_stride;

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
        
        vec_y += 32;
    }
    
    event1();
}

void sell32_spmv_kernel_32wide(
    uint32_t num_blocks,
    uint32_t ell_width,
    const bfloat16 *__restrict data_ptr, 
    const bfloat16 *__restrict vec_x,   
    bfloat16 *__restrict vec_y          
)
{
    // [重要] 丸めモードを設定（acc -> vector 変換時のエラー防止と精度確保）
    // 参考コードにあるように、bf16への変換にはこれが必要です
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    event0();

    // ポインタのセットアップ
    const uint16_t *__restrict ptr_idx = reinterpret_cast<const uint16_t*>(data_ptr);
    const bfloat16 *__restrict ptr_val = data_ptr + 32; 

    const int block_stride = 64; 

    for (uint32_t b = 0; b < num_blocks; b++) {

        aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();

        // パイプライン化指示
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(16)
        for (uint32_t k = 0; k < ell_width; k++) {
            
            aie::vector<uint16_t, 32> idx = aie::load_v<32>(ptr_idx);
            aie::vector<bfloat16, 32> val = aie::load_v<32>(ptr_val);

            ptr_idx += block_stride;
            ptr_val += block_stride;
            aie::vector<bfloat16, 32> x_gathered;

            AIE_LOOP_UNROLL_FULL
            for (int i = 0; i < 32; i++) {
                x_gathered[i] = vec_x[idx[i]];
            }

            acc = aie::mac(acc, val, x_gathered);
        }

        aie::vector<bfloat16, 32> res = acc.template to_vector<bfloat16>();
        aie::store_v(vec_y, res);
        
        vec_y += 32;
    }
    
    event1();
}


void sell32_spmv_kernel_32_block(
    uint32_t ell_width,
    uint32_t reset,
    const bfloat16 *__restrict data_ptr, 
    const bfloat16 *__restrict vec_x,   
    bfloat16 *__restrict vec_y          
)
{
    // [重要] 丸めモードを設定（acc -> vector 変換時のエラー防止と精度確保）
    // 参考コードにあるように、bf16への変換にはこれが必要です
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    event0();

    // ポインタのセットアップ
    const uint16_t *__restrict ptr_idx = reinterpret_cast<const uint16_t*>(data_ptr);
    const bfloat16 *__restrict ptr_val = data_ptr + 32; 

    const int block_stride = 64; 

    aie::accum<accfloat, 32> acc;

    // 初期化またはロード
    if (reset == 0) {
        acc = aie::zeros<accfloat, 32>();
    } else {
        acc = aie::load_v<32>(vec_y);
    }
    // パイプライン化指示
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (uint32_t k = 0; k < ell_width; k++) {
        
        aie::vector<uint16_t, 32> idx = aie::load_v<32>(ptr_idx);
        aie::vector<bfloat16, 32> val = aie::load_v<32>(ptr_val);

        ptr_idx += block_stride;
        ptr_val += block_stride;
        aie::vector<bfloat16, 32> x_gathered;

        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < 32; i++) {
            x_gathered[i] = vec_x[idx[i]];
        }

        acc = aie::mac(acc, val, x_gathered);
    }

    aie::vector<bfloat16, 32> res = acc.template to_vector<bfloat16>();
    aie::store_v(vec_y, res);
    
    
    event1();
}
template <uint32_t r>
void sell32_mac_only_benchmark(
    uint32_t loop_count,
    const bfloat16 *__restrict data_ptr, 
    bfloat16 *__restrict vec_y          
)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // ポインタのセットアップ
    const bfloat16 *__restrict ptr_idx_as_bf16 = reinterpret_cast<const bfloat16*>(data_ptr);
    const bfloat16 *__restrict ptr_val = data_ptr + r; 

    // レジスタへのロード（ループ外）
    aie::vector<bfloat16, r> idx_reg = aie::load_v<r>(ptr_idx_as_bf16);
    aie::vector<bfloat16, r> val_reg = aie::load_v<r>(ptr_val);

    // [重要] アキュムレータを4つ用意して独立させる
    // これにより、acc0の計算が終わるのを待たずにacc1の計算を開始できる
    aie::accum<accfloat, r> acc0 = aie::zeros<accfloat, r>();
    aie::accum<accfloat, r> acc1 = aie::zeros<accfloat, r>();
    aie::accum<accfloat, r> acc2 = aie::zeros<accfloat, r>();
    aie::accum<accfloat, r> acc3 = aie::zeros<accfloat, r>();

    event0(); 

    AIE_PREPARE_FOR_PIPELINING
    // 4回分を1セットで回すので、ループ回数は1/4になる（またはk+=4にする）
    // loop_countは4の倍数であることを推奨
    for (uint32_t k = 0; k < loop_count; k += 4) {
        // 依存関係のない4つの命令を並べる
        acc0 = aie::mac(acc0, val_reg, idx_reg);
        acc1 = aie::mac(acc1, val_reg, idx_reg);
        acc2 = aie::mac(acc2, val_reg, idx_reg);
        acc3 = aie::mac(acc3, val_reg, idx_reg);
    }

    event1();

    // [重要] 最後に4つの結果を合算する
    // これをしないと、コンパイラが「acc1~3は不要」と判断して消してしまう可能性がある
    acc0 = aie::add(acc0, acc1);
    acc0 = aie::add(acc0, acc2);
    acc0 = aie::add(acc0, acc3);

    aie::vector<bfloat16, r> res = acc0.template to_vector<bfloat16>();
    aie::store_v(vec_y, res);
}


extern "C" { 


void sparse_matvec_vectorized_bf16_bf16(uint32_t m, 
                                        uint32_t k, 
                                        uint32_t ell_width, 
                                        uint32_t row_offset, 
                                        const bfloat16 *__restrict a_in, 
                                        const bfloat16 *__restrict b_in,
                                        bfloat16 *__restrict c_out
                                    )
{
    sparse_matvec_vectorized_aligned<32>(m, k, ell_width, row_offset, a_in, b_in, c_out);
} 


void sell32_spmv_vectorized_bf16_bf16(
    uint32_t num_blocks,
    uint32_t ell_width,
    const bfloat16 *__restrict data_ptr,
    const bfloat16 *__restrict vec_x,
    bfloat16 *__restrict vec_y
)
{
    sell32_spmv_kernel_32wide(num_blocks, ell_width, data_ptr, vec_x, vec_y);
}
void sell32_block_spmv_vectorized_bf16_bf16(
    uint32_t ell_width,
    uint32_t reset,
    const bfloat16 *__restrict data_ptr,
    const bfloat16 *__restrict vec_x,
    bfloat16 *__restrict vec_y
)
{
    sell32_spmv_kernel_32_block(ell_width, reset, data_ptr, vec_x, vec_y);
    //sell32_mac_only_benchmark<256>(ell_width, data_ptr, vec_y);
}

} // extern "C"