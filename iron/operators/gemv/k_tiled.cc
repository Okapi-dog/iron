// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP
#include "../../../aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// Reset two scalar output states before processing the K tiles of an 8-row
// output block.  Each core owns exactly two of those rows.
extern "C" void dense_gemv_k_tiled_init_2(float *__restrict state) {
  state[0] = 0.0f;
  state[1] = 0.0f;
}

// Fixed-size entries give LLVM a compile-time trip count, retaining the
// vector-loop scheduling of the stock GEMV kernel.  The 1376 entry is the
// largest exact 32-lane tile that divides Llama's K=11008.
template <unsigned KTile>
static __attribute__((always_inline)) inline void dense_gemv_k_tiled_accumulate_impl(
    const bfloat16 *__restrict a, const bfloat16 *__restrict x,
    float *__restrict state) {
  for (unsigned row = 0; row < 2; ++row) {
    const bfloat16 *a_row = a + row * KTile;
    aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (unsigned k = 0; k < KTile; k += 32) {
      const auto av = aie::load_v<32>(a_row + k);
      const auto xv = aie::load_v<32>(x + k);
      acc = aie::mac(acc, av, xv);
    }
    state[row] += aie::reduce_add(acc.template to_vector<float>());
  }
}

extern "C" void dense_gemv_k_tiled_accumulate_2x1376_bf16(
    const bfloat16 *__restrict a, const bfloat16 *__restrict x,
    float *__restrict state) {
  dense_gemv_k_tiled_accumulate_impl<1376>(a, x, state);
}

extern "C" void dense_gemv_k_tiled_accumulate_2x4096_bf16(
    const bfloat16 *__restrict a, const bfloat16 *__restrict x,
    float *__restrict state) {
  dense_gemv_k_tiled_accumulate_impl<4096>(a, x, state);
}

extern "C" void dense_gemv_k_tiled_finalize_2(
    const float *__restrict state, bfloat16 *__restrict y) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  y[0] = static_cast<bfloat16>(state[0]);
  y[1] = static_cast<bfloat16>(state[1]);
}
