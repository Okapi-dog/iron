// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#define NOCPP
#include "../../../aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C" void event0();
extern "C" void event1();

// This is the Phase-0 ELL kernel, retained verbatim in structure and ABI so a
// Phase-1 result compares the runtime/compiler rather than a new microkernel.
template <uint32_t lanes>
static void sparse_matvec_vectorized_aligned(
    uint32_t rows, uint32_t /*K*/, uint32_t ell_width, uint32_t row_offset,
    const bfloat16 *__restrict packed, const bfloat16 *__restrict x,
    bfloat16 *__restrict y) {
  event0();
  y += row_offset * rows;
  const uint32_t row_stride = 2 * ell_width;
  const bfloat16 *row_base = packed;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(10)
  for (uint32_t row = 0; row < rows; ++row) {
    const uint16_t *__restrict indices = reinterpret_cast<const uint16_t *>(row_base);
    const bfloat16 *__restrict values = row_base + ell_width;
    aie::accum<accfloat, lanes> acc = aie::zeros<accfloat, lanes>();
    uint32_t slot = 0;
    for (; slot + lanes <= ell_width; slot += lanes) {
      const auto idx = aie::load_v<lanes>(indices);
      const auto val = aie::load_v<lanes>(values);
      indices += lanes;
      values += lanes;
      aie::vector<bfloat16, lanes> gathered;
      AIE_LOOP_UNROLL_FULL
      for (uint32_t lane = 0; lane < lanes; ++lane)
        gathered[lane] = x[idx[lane]];
      acc = aie::mac(acc, val, gathered);
    }
    y[row] = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    row_base += row_stride;
  }
  event1();
}

extern "C" void sparse_matvec_vectorized_bf16_bf16(
    uint32_t rows, uint32_t K, uint32_t ell_width, uint32_t row_offset,
    const bfloat16 *__restrict packed, const bfloat16 *__restrict x,
    bfloat16 *__restrict y) {
  sparse_matvec_vectorized_aligned<32>(rows, K, ell_width, row_offset, packed, x, y);
}

extern "C" void sell32_spmv_bf16(
    uint32_t ell_width,
    const bfloat16 *__restrict packed,
    const bfloat16 *__restrict x,
    bfloat16 *__restrict y) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
  for (uint32_t slot = 0; slot < ell_width; ++slot) {
    const bfloat16 *slot_base = packed + slot * 64;
    const auto idx = aie::load_v<32>(reinterpret_cast<const uint16_t *>(slot_base));
    const auto val = aie::load_v<32>(slot_base + 32);
    aie::vector<bfloat16, 32> gathered;
    for (uint32_t lane = 0; lane < 32; ++lane) gathered[lane] = x[idx[lane]];
    acc = aie::mac(acc, val, gathered);
  }
  aie::store_v(y, acc.template to_vector<bfloat16>());
}

extern "C" void sell32_block_spmv_vectorized_bf16_bf16(
    uint32_t block_width, uint32_t reset,
    const bfloat16 *__restrict packed, const bfloat16 *__restrict x,
    bfloat16 *__restrict y) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  event0();
  aie::accum<accfloat, 32> acc;
  if (reset == 0)
    acc = aie::zeros<accfloat, 32>();
  else
    acc = aie::load_v<32>(y);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (uint32_t slot = 0; slot < block_width; ++slot) {
    const bfloat16 *slot_base = packed + slot * 64;
    const auto idx = aie::load_v<32>(reinterpret_cast<const uint16_t *>(slot_base));
    const auto val = aie::load_v<32>(slot_base + 32);
    aie::vector<bfloat16, 32> gathered;
    AIE_LOOP_UNROLL_FULL
    for (uint32_t lane = 0; lane < 32; ++lane) gathered[lane] = x[idx[lane]];
    acc = aie::mac(acc, val, gathered);
  }
  const auto result = acc.template to_vector<bfloat16>();
  aie::store_v(y, result);
  event1();
}
