#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync-exclusive Phase-1 SpMV benchmark, comparable in scope to Phase 0.

This script measures the new runtime's ``result.npu_time``.  It is the host
wall-clock around XRT kernel launch and ``wait()``, and excludes host tensor
creation and host/device BO synchronization.  That is the same scope as the
old ``run_runlist()`` timer, rather than a hardware cycle counter.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
import torch

from iron.common.test_utils import verify_buffer
from iron.operators.spmv.op import SpMVELL, SpMVSELL32Block
from iron.operators.spmv.reference import (
    make_uniform_ell,
    make_uniform_sell32,
    reference_ell,
    reference_sell32_block,
)


CASES = (
    (4096, 4096, 512, 8),
    (4096, 11008, 1376, 2),
    (28672, 8192, 1024, 4),
)


def _device_args(operator, packed, vector):
    tensor = aie_utils.DEFAULT_TENSOR_CLASS
    specs = operator.get_arg_spec()
    return [
        tensor.from_torch(packed),
        tensor.from_torch(vector),
        tensor(specs[2].shape, dtype=specs[2].dtype),
    ]


def _run(operator, packed, vector, expected, samples=5, idle_s=4):
    operator.compile()
    call = operator.get_callable()
    args = _device_args(operator, packed, vector)
    # Keep the Phase-0 convention: two unmeasured calls before every sample.
    times = []
    for i in range(samples):
        call(*args)
        call(*args)
        result = call(*args)
        times.append(result.npu_time / 1e3)
        if i + 1 < samples:
            time.sleep(idle_s)
    errors = verify_buffer(args[2].to_torch(), "output", expected, rel_tol=0.06, abs_tol=1e-4)
    if errors:
        raise RuntimeError(f"reference mismatch: first errors {errors[:10]}")
    payload_bytes = packed.numel() * 2 + vector.numel() * 2 + expected.numel() * 2
    return {
        "samples_us": times,
        "mean_us": statistics.mean(times),
        "min_us": min(times),
        "max_us": max(times),
        "std_us": statistics.pstdev(times),
        "effective_bandwidth_gbps": payload_bytes / (statistics.mean(times) * 1e-6) / 1e9,
        "payload_bytes": payload_bytes,
    }


def _legacy_packed(root, M, K, width, layout):
    """Load a Phase-0 packed payload, preserving its uint16 bit pattern."""
    name = f"random_M{M}_K{K}_ELL{width}"
    path = root / name / f"{name}_xdna_{layout}.npy"
    words = np.load(path)
    expected_words = M * width * 2
    if words.dtype != np.uint16 or words.size != expected_words:
        raise ValueError(
            f"{path}: expected {expected_words} uint16 words, got {words.shape} {words.dtype}"
        )
    # Index words and BF16 value words share a uint16 storage representation.
    # A copy makes the tensor writable for the XRT host buffer implementation.
    return torch.from_numpy(words.copy()).view(torch.int16).view(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--idle-s", type=float, default=4)
    parser.add_argument(
        "--legacy-data-root",
        type=Path,
        help="optional Phase-0 npu_data root; uses its exact ELL/SELL-32 payloads",
    )
    args = parser.parse_args()
    results = []
    for case_idx, (M, K, width, rows_per_core) in enumerate(CASES):
        vector = torch.rand(K, generator=torch.Generator().manual_seed(42)).to(torch.bfloat16)
        if args.legacy_data_root:
            ell = _legacy_packed(args.legacy_data_root, M, K, width, "ell")
            sell = _legacy_packed(args.legacy_data_root, M, K, width, "sell32")
        else:
            ell = make_uniform_ell(M, K, width, seed=1000 + case_idx)
            sell = make_uniform_sell32(M, K, width, seed=1000 + case_idx)
        plans = (
            ("ELL", SpMVELL(M, K, width, rows=4, cols=8, rows_per_core=rows_per_core), ell, reference_ell(ell, vector, M, width)),
            ("SELL-32 block", SpMVSELL32Block(M, K, width, rows=4, cols=8), sell, reference_sell32_block(sell, vector, M, width)),
        )
        for kernel, operator, packed, expected in plans:
            measured = _run(operator, packed, vector, expected, samples=args.samples, idle_s=args.idle_s)
            row = {"M": M, "K": K, "ell_width": width, "rows": 4, "cols": 8, "kernel": kernel, **measured}
            print(json.dumps(row, sort_keys=True))
            results.append(row)
    output = Path("npu_data/phase1_mlir_v1.4.3")
    output.mkdir(parents=True, exist_ok=True)
    name = "full_core_legacy_input.json" if args.legacy_data_root else "full_core_device_only.json"
    (output / name).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
