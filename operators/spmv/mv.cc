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

} // extern "C"