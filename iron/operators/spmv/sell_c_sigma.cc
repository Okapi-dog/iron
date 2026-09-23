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

// Step 3: the existing Slice-ELL scalar-state method.  The row count of each
// producer is selected by its kernel symbol.  x starts after two int16 words.
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

extern "C" void sell_accumulate1(
    const bfloat16 *packed, const int16_t *config_words, float *state) {
  sell_accumulate_rows<1>(packed, config_words, state);
}

extern "C" void sell_accumulate2(
    const bfloat16 *packed, const int16_t *config_words, float *state) {
  sell_accumulate_rows<2>(packed, config_words, state);
}

extern "C" void sell_accumulate3(
    const bfloat16 *packed, const int16_t *config_words, float *state) {
  sell_accumulate_rows<3>(packed, config_words, state);
}

extern "C" void sell_accumulate4(
    const bfloat16 *packed, const int16_t *config_words, float *state) {
  sell_accumulate_rows<4>(packed, config_words, state);
}

extern "C" void sell_finalize1(const float *state, bfloat16 *output) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  output[0] = static_cast<bfloat16>(state[0]);
  output[1] = static_cast<bfloat16>(0);
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

extern "C" void sell_finalize4(const float *state, bfloat16 *output) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned row = 0; row < 4; ++row)
    output[row] = static_cast<bfloat16>(state[row]);
}

// Skip a core's alignment padding while scattering physical rows to their
// original positions.  The local map uses 0xffff for final padded rows.
extern "C" void sell_reorder_scatter(
    const bfloat16 *joined, const int16_t *row_indices,
    bfloat16 *canonical_window, int32_t slice_in_window,
    int32_t rows0, int32_t rows1, int32_t rows2,
    int32_t slots0, int32_t slots1) {
  const unsigned height = static_cast<unsigned>(rows0 + rows1 + rows2);
  const unsigned base = static_cast<unsigned>(slice_in_window) * height;
  for (unsigned row = 0; row < height; ++row) {
    unsigned source;
    if (row < static_cast<unsigned>(rows0))
      source = row;
    else if (row < static_cast<unsigned>(rows0 + rows1))
      source = static_cast<unsigned>(slots0) + row - rows0;
    else
      source = static_cast<unsigned>(slots0 + slots1) + row - rows0 - rows1;
    const unsigned destination = static_cast<uint16_t>(row_indices[base + row]);
    if (destination != 0xffff)
      canonical_window[destination] = joined[source];
  }
}

// Step 4: four compute cores write disjoint two-row regions of two shared
// neighboring L1 buffers.  Core row 4 reads both buffers after lock handoff.
extern "C" void sell_finalize2_shared(
    const float *state, bfloat16 *half_window,
    int32_t slice_in_window, int32_t pair_in_half) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned base = static_cast<unsigned>(slice_in_window) * 4
                      + static_cast<unsigned>(pair_in_half) * 2;
  half_window[base] = static_cast<bfloat16>(state[0]);
  half_window[base + 1] = static_cast<bfloat16>(state[1]);
}

// The control object is [config | window-local row map].  Each half stores
// four physical rows per slice, so no full-window MemTile FIFO is required.
extern "C" void sell_reorder_shared(
    const bfloat16 *first_half, const bfloat16 *second_half,
    const int16_t *control, bfloat16 *canonical_window,
    int32_t config_words, int32_t rows_per_window) {
  for (int32_t row = 0; row < rows_per_window; ++row)
    canonical_window[row] = static_cast<bfloat16>(0);
  for (int32_t row = 0; row < rows_per_window; ++row) {
    const unsigned local = static_cast<unsigned>(row) % 8;
    const unsigned slice = static_cast<unsigned>(row) / 8;
    const unsigned source = slice * 4 + local % 4;
    const bfloat16 value = local < 4 ? first_half[source] : second_half[source];
    const unsigned destination = static_cast<uint16_t>(control[config_words + row]);
    if (destination != 0xffff)
      canonical_window[destination] = value;
  }
}
