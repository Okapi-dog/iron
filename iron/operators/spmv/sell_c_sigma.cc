// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#define NOCPP
#include "../../../aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// Step 2: keep the producer workers intentionally computationally trivial.
extern "C" void sell_route_copy2(const bfloat16 *input, bfloat16 *output) {
  for (unsigned i = 0; i < 2; ++i)
    output[i] = input[i];
}

extern "C" void sell_route_copy4(const bfloat16 *input, bfloat16 *output) {
  for (unsigned i = 0; i < 4; ++i)
    output[i] = input[i];
}

extern "C" void sell_zero_output(bfloat16 *output, int32_t rows) {
  for (int32_t row = 0; row < rows; ++row)
    output[row] = static_cast<bfloat16>(0);
}

// The MemTile join is [2 real | 3 real + dummy | 3 real + dummy].
// row_indices are local to one window, not absolute matrix row numbers.
extern "C" void sell_reorder_scatter8(
    const bfloat16 *joined, const int16_t *row_indices,
    bfloat16 *canonical_window, int32_t slice_in_window) {
  const unsigned joined_slot[8] = {0, 1, 2, 3, 4, 6, 7, 8};
  const unsigned base = static_cast<unsigned>(slice_in_window) * 8;
  for (unsigned row = 0; row < 8; ++row) {
    const unsigned destination = static_cast<uint16_t>(row_indices[base + row]);
    if (destination != 0xffff)
      canonical_window[destination] = joined[joined_slot[row]];
  }
}

// Step 3: the existing Slice-ELL scalar-state method, specialized to the
// (2,3,3) producer geometry.  The x vector starts after two int16 words.
extern "C" void sell_state_init(float *state) {
  for (unsigned row = 0; row < 4; ++row)
    state[row] = 0.0f;
}

template <unsigned Rows>
static void sell_accumulate_rows(
    const bfloat16 *packed, const int16_t *config_words, float *state) {
  const auto *x = reinterpret_cast<const bfloat16 *>(config_words + 2);
  for (unsigned row = 0; row < Rows; ++row) {
    const bfloat16 *row_packed = packed + row * 512;
    const auto *indices = reinterpret_cast<const uint16_t *>(row_packed);
    const auto *values = row_packed + 256;
    aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (unsigned slot = 0; slot < 256; slot += 32) {
      const auto idx = aie::load_v<32>(indices + slot);
      const auto val = aie::load_v<32>(values + slot);
      aie::vector<bfloat16, 32> gathered;
      AIE_LOOP_UNROLL_FULL
      for (unsigned lane = 0; lane < 32; ++lane)
        gathered[lane] = x[idx[lane]];
      acc = aie::mac(acc, val, gathered);
    }
    state[row] += aie::reduce_add(acc.template to_vector<float>());
  }
}

extern "C" void sell_accumulate2(
    const bfloat16 *packed, const int16_t *config_words, float *state) {
  sell_accumulate_rows<2>(packed, config_words, state);
}

extern "C" void sell_accumulate3(
    const bfloat16 *packed, const int16_t *config_words, float *state) {
  sell_accumulate_rows<3>(packed, config_words, state);
}

extern "C" void sell_finalize2(const float *state, bfloat16 *output) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  output[0] = static_cast<bfloat16>(state[0]);
  output[1] = static_cast<bfloat16>(state[1]);
}

extern "C" void sell_finalize3(const float *state, bfloat16 *output) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned row = 0; row < 3; ++row)
    output[row] = static_cast<bfloat16>(state[row]);
  output[3] = static_cast<bfloat16>(0);
}
