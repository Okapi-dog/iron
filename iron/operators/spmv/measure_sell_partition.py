#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare contiguous window cuts and measure them on NPU2.

Each policy keeps the original row order between windows and sorts only within
each window. Window object lengths remain fixed. Output DMA places the valid
prefixes at consecutive canonical row offsets; later windows overwrite the
preceding window's padding tail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aie.utils as aie_utils
from aie.iron.device import NPU2

from iron.common.test_utils import run_test
from iron.operators.spmv.evaluation import MatrixInput, load_or_generate_csr
from iron.operators.spmv.measure_llama_slice_ell import REPRESENTATIVE_WEIGHTS
from iron.operators.spmv.sell_c_sigma_op import SpMVSELLDedicated
from iron.operators.spmv.sell_c_sigma_runtime import (
    make_dedicated_inputs, make_vertical16_inputs,
)
from iron.operators.spmv.sell_partition import pad_csr_windows
from iron.operators.spmv.slice_ell import cpu_spmv_csr


POLICIES = ("equal_rows", "equal_nnz", "balanced_blocks")


def benchmark(matrix, packed, valid_ranges, profile, height: int, width: int,
              windows: int, geometry: str) -> dict:
    """Run correctness-checked NPU benchmark and report payload bandwidth."""
    if geometry == "vertical16":
        A, control, blocks = make_vertical16_inputs(packed, matrix.vector)
        rows_per_core = (16, 16, 16)
    else:
        rows_per_core = (2, 3, 3) if height == 8 else (height // 3,) * 3
        A, control, blocks = make_dedicated_inputs(packed, matrix.vector, rows_per_core)

    expected = cpu_spmv_csr(
        matrix.indptr, matrix.indices, matrix.values, matrix.vector
    )
    rows_per_window = int(profile["padded_rows_per_window"])
    valid_counts = tuple(last - first for first, last in valid_ranges)
    compact_dma = any(count < rows_per_window for count in valid_counts[:-1])

    operator = SpMVSELLDedicated(
        packed.padded_rows, packed.K, blocks, windows,
        rows_per_core=rows_per_core, block_width=width,
        vertical16=geometry == "vertical16",
        valid_rows_per_window=valid_counts if compact_dma else None,
    )
    # Keep the runtime explicitly selected for standalone CLI execution; pytest
    # normally supplies this through the repository's NPU setup hooks.
    aie_utils.set_current_device(NPU2())
    errors, latency, effective_gbps, samples = run_test(
        operator, {"packed": A, "control": control}, {"output": expected},
        rel_tol=0.08, abs_tol=0.025, warmup_iters=2, timed_iters=5,
        return_timings=True,
    )
    if errors:
        raise AssertionError(
            f"canonical output mismatch ({sum(map(len, errors.values()))} elements)"
        )
    a_bytes = A.numel() * A.element_size()
    control_bytes = control.numel() * control.element_size()
    output_bytes = packed.padded_rows * expected.element_size()
    return {
        "latency_us": latency, "samples_us": samples,
        "a_only_gbps": a_bytes / (latency * 1e3),
        "effective_gbps": effective_gbps,
        "a_bytes": a_bytes, "control_bytes": control_bytes,
        "output_bytes": output_bytes, "canonical_output_bytes": expected.numel() * expected.element_size(),
        "blocks_per_window": profile["blocks_per_window"],
        "block_imbalance_max_over_mean": profile["block_imbalance_max_over_mean"],
        "padded_rows_per_window": rows_per_window,
        "valid_rows_per_window": profile["valid_rows_per_window"],
        "compact_dma": compact_dma,
        "boundaries_in_slices": profile["slice_boundaries"],
    }


def run_case(matrix, name: str, height: int, width: int, windows: int,
             policy: str, geometry: str) -> dict:
    """Pack, correctness check, and measure one matrix/policy/geometry."""
    packed, valid_ranges, profile = pad_csr_windows(
        matrix.indptr, matrix.indices, matrix.values,
        height, width, windows, policy, matrix.profile.K,
    )
    actual_blocks = [
        int(packed.column_blocks_per_slice(i).sum()) for i in range(windows)
    ]
    if actual_blocks != profile["blocks_per_window"]:
        raise AssertionError(
            f"predicted block counts {profile['blocks_per_window']} != packed {actual_blocks}"
        )
    record = benchmark(matrix, packed, valid_ranges, profile,
                       height, width, windows, geometry)
    return {
        "weight": name, "M": matrix.profile.M, "K": matrix.profile.K,
        "nnz": matrix.profile.nnz,
        "density": matrix.profile.nnz / (matrix.profile.M * matrix.profile.K),
        "block_height": height, "block_width": width,
        "columns": windows, "cores": 4 * windows,
        "policy": policy, "geometry": geometry,
        "packed_a_storage_bytes": profile["packed_a_bytes"],
        "row_map_bytes": profile["row_map_bytes"],
        "control_bytes_estimate": profile["control_bytes"],
        "storage_bytes_a_plus_row_map": profile["storage_bytes"],
        "padding_overhead_rows": profile["padded_rows"] - matrix.profile.M,
        **record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--weight", action="append", choices=REPRESENTATIVE_WEIGHTS)
    parser.add_argument("--height", type=int, choices=(8, 48), default=8)
    parser.add_argument("--width", type=int, choices=(128, 256), default=None)
    parser.add_argument("--geometry", choices=("horizontal", "vertical16"), default=None)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    width = args.width or (128 if args.height == 48 else 256)
    geometry = args.geometry or ("vertical16" if args.height == 48 else "horizontal")
    if geometry == "vertical16" and (args.height, width) != (48, 128):
        parser.error("vertical16 is defined only for B_h=48, B_w=128")
    aie_utils.set_current_device(NPU2())

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    for name in args.weight or REPRESENTATIVE_WEIGHTS:
        matrix = load_or_generate_csr(MatrixInput(
            "safetensors", model_dir=str(args.model_dir.resolve()),
            tensor_name=name, x_seed=3000 + REPRESENTATIVE_WEIGHTS.index(name),
        ))
        for policy in args.policy or POLICIES:
            record = run_case(matrix, name, args.height, width, 8, policy, geometry)
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            with args.output_jsonl.open("a") as output:
                output.write(line + "\n")


if __name__ == "__main__":
    main()
