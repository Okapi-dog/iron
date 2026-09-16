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

// Accumulate one 256-slot horizontal Slice-ELL block for one output row.
// ``always_inline`` is material here: p=2 must keep the caller's eight
// accumulators live across both A objects rather than materialize a partial
// vector in L1 at the function boundary.
static __attribute__((always_inline)) inline void slice_ell_accumulate_row(
    const bfloat16 *__restrict packed, const bfloat16 *__restrict x,
    aie::accum<accfloat, 32> &acc) {
  const auto *indices = reinterpret_cast<const uint16_t *>(packed);
  const auto *values = packed + 256;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (uint32_t slot = 0; slot < 256; slot += 32) {
    const auto idx = aie::load_v<32>(indices + slot);
    const auto val = aie::load_v<32>(values + slot);
    aie::vector<bfloat16, 32> gathered;
    AIE_LOOP_UNROLL_FULL
    for (uint32_t lane = 0; lane < 32; ++lane)
      gathered[lane] = x[idx[lane]];
    acc = aie::mac(acc, val, gathered);
  }
}

// Process exactly eight rows from each of one or two Slice-ELL blocks.  The
// specialisations deliberately expose eight independent 32-lane FP32
// accumulators to the register allocator; Phase 3 measures this A-plan before
// introducing a lower-register-pressure scalar-reduction alternative.
static __attribute__((always_inline)) inline void slice_ell_accumulate_block(
    const bfloat16 *__restrict packed, const bfloat16 *__restrict x,
    aie::accum<accfloat, 32> &acc0, aie::accum<accfloat, 32> &acc1,
    aie::accum<accfloat, 32> &acc2, aie::accum<accfloat, 32> &acc3,
    aie::accum<accfloat, 32> &acc4, aie::accum<accfloat, 32> &acc5,
    aie::accum<accfloat, 32> &acc6, aie::accum<accfloat, 32> &acc7) {
  slice_ell_accumulate_row(packed + 0 * 512, x, acc0);
  slice_ell_accumulate_row(packed + 1 * 512, x, acc1);
  slice_ell_accumulate_row(packed + 2 * 512, x, acc2);
  slice_ell_accumulate_row(packed + 3 * 512, x, acc3);
  slice_ell_accumulate_row(packed + 4 * 512, x, acc4);
  slice_ell_accumulate_row(packed + 5 * 512, x, acc5);
  slice_ell_accumulate_row(packed + 6 * 512, x, acc6);
  slice_ell_accumulate_row(packed + 7 * 512, x, acc7);
}

template <uint32_t blocks>
static __attribute__((always_inline)) inline void slice_ell_horizontal_impl(
    const bfloat16 *__restrict a0, const bfloat16 *__restrict a1,
    const bfloat16 *__restrict x, bfloat16 *__restrict y) {
  aie::accum<accfloat, 32> acc0 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> acc1 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> acc2 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> acc3 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> acc4 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> acc5 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> acc6 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> acc7 = aie::zeros<accfloat, 32>();
  slice_ell_accumulate_block(a0, x, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
  if constexpr (blocks == 2)
    slice_ell_accumulate_block(a1, x, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
  y[0] = static_cast<bfloat16>(aie::reduce_add(acc0.template to_vector<float>()));
  y[1] = static_cast<bfloat16>(aie::reduce_add(acc1.template to_vector<float>()));
  y[2] = static_cast<bfloat16>(aie::reduce_add(acc2.template to_vector<float>()));
  y[3] = static_cast<bfloat16>(aie::reduce_add(acc3.template to_vector<float>()));
  y[4] = static_cast<bfloat16>(aie::reduce_add(acc4.template to_vector<float>()));
  y[5] = static_cast<bfloat16>(aie::reduce_add(acc5.template to_vector<float>()));
  y[6] = static_cast<bfloat16>(aie::reduce_add(acc6.template to_vector<float>()));
  y[7] = static_cast<bfloat16>(aie::reduce_add(acc7.template to_vector<float>()));
}

extern "C" void slice_ell_horizontal_p1_bf16(
    const bfloat16 *__restrict a0, const bfloat16 *__restrict x,
    bfloat16 *__restrict y) {
  event0();
  slice_ell_horizontal_impl<1>(a0, nullptr, x, y);
  event1();
}

extern "C" void slice_ell_horizontal_p2_bf16(
    const bfloat16 *__restrict a0, const bfloat16 *__restrict a1,
    const bfloat16 *__restrict x, bfloat16 *__restrict y) {
  event0();
  slice_ell_horizontal_impl<2>(a0, a1, x, y);
  event1();
}
