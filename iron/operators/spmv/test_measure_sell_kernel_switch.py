# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the Step-4.5 benchmark protocol."""

from types import SimpleNamespace

import pytest
import torch

from iron.operators.spmv.measure_sell_kernel_switch import (
    TIMED_ITERS, WARMUP_ITERS, make_dummy_samples, measure_mode, select_modes,
)


class FakeOutput:
    def to_torch(self):
        return torch.ones(2, dtype=torch.bfloat16)


def test_default_mode_does_not_use_dummy():
    assert select_modes(False, False) == ("consecutive",)
    assert select_modes(True, False) == ("consecutive", "switched")
    assert select_modes(True, True) == ("switched", "consecutive")
    with pytest.raises(ValueError, match="requires --with-dummy"):
        select_modes(False, True)


def test_interleaving_excludes_dummy_time():
    calls = []
    count = WARMUP_ITERS + TIMED_ITERS
    target_args = [(sample, None, FakeOutput()) for sample in range(count)]
    dummy_args = [(sample,) for sample in range(count)]
    expected = [torch.ones(2, dtype=torch.bfloat16) for _ in range(count)]

    def target(sample, _unused, _output):
        calls.append(("target", sample))
        return SimpleNamespace(npu_time=1000)

    def dummy(sample):
        calls.append(("dummy", sample))
        return SimpleNamespace(npu_time=9000000)

    switched = measure_mode("switched", target, dummy, target_args, dummy_args, expected)
    assert calls == [item for sample in range(count)
                     for item in (("dummy", sample), ("target", sample))]
    assert switched["mean_npu_us"] == 1.0
    assert switched["dummy_npu_us_excluded"] == [9000.0] * TIMED_ITERS

    calls.clear()
    consecutive = measure_mode("consecutive", target, dummy, target_args, dummy_args, expected)
    assert calls == [("target", sample) for sample in range(count)]
    assert consecutive["mean_npu_us"] == 1.0
    assert consecutive["dummy_npu_us_excluded"] == []


def test_dummy_inputs_vary_and_have_32_core_shape():
    samples = make_dummy_samples(WARMUP_ITERS + TIMED_ITERS, seed=73)
    assert len(samples) == 7
    for matrix, tiled_x, expected in samples:
        assert matrix.shape == (64 * 4096,)
        assert tiled_x.shape == (8 * 4096,)
        assert expected.shape == (64,)
    assert not torch.equal(samples[0][0], samples[1][0])
    assert not torch.equal(samples[0][1], samples[1][1])
