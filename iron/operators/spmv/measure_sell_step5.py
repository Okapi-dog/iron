#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare dense, Slice-ELL, and SELL on the same real pruned weight and x.

Each JSONL record identifies the matrix, packed format, execution design, and
canonical-output latency.  The default timing is two warmups and five timed
consecutive runs; a dummy kernel is deliberately not inserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
import torch
from aie.iron.device import NPU2

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.gemv.k_tiled_op import DenseGEMVKTile
from iron.operators.spmv.evaluation import (
    DesignSpec, FormatSpec, MatrixInput, estimate_storage, load_or_generate_csr,
    pack_for_design,
)
from iron.operators.spmv.measure_llama_dense_gemv_k_tiled import pack_dense_k_tiled
from iron.operators.spmv.measure_llama_slice_ell import (
    REPRESENTATIVE_WEIGHTS, make_runtime_config,
)
from iron.operators.spmv.op import SpMVSliceELLDynamicScalarMultiCol
from iron.operators.spmv.sell_c_sigma_runtime import prepare_sell_design
from iron.operators.spmv.slice_ell import cpu_spmv_csr


DESIGNS = (
    "dense_k_tiled", "slice_ell", "sell_dedicated_reorder",
    "sell_time_multiplex_reorder",
)
BLOCK_HEIGHT = 8
BLOCK_WIDTH = 256
COLUMNS = 8
WARMUP_ITERS = 2
TIMED_ITERS = 5


def sha256_array(array: np.ndarray) -> str:
    """Fingerprint one contiguous host-side payload without saving a binary."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_case(matrix, design_name: str, windows: int, block_height: int = BLOCK_HEIGHT):
    """Build one operator and payload, preserving matrix/vector identity."""
    M, K = matrix.profile.M, matrix.profile.K
    x = matrix.vector
    design = DesignSpec(design_name)
    if block_height != BLOCK_HEIGHT and design_name != "sell_dedicated_reorder":
        raise ValueError("variable B_h is implemented only for dedicated SELL")
    if design_name == "sell_dedicated_reorder" and block_height not in (6, 8, 9, 18, 36, 72):
        raise ValueError("dedicated SELL supports B_h=6,8,9,18,36,72")

    if design_name == "dense_k_tiled":
        fmt = FormatSpec("dense", columns=COLUMNS)
        dense = pack_for_design(matrix, fmt, design)
        k_tile = 1376 if K == 11008 else 4096
        packed_a, tiled_x = pack_dense_k_tiled(dense, x, COLUMNS, k_tile)
        operator = DenseGEMVKTile(M, K, COLUMNS, k_tile, context=AIEContext())
        inputs = {"matrix": packed_a, "vector_tiles": tiled_x}
        payload = packed_a.view(torch.uint16).numpy()
        extra = {"k_tile": k_tile, "row_indices_bytes": 0,
                 "control_bytes": 0, "packed_a_bytes": packed_a.numel() * 2}
    elif design_name == "slice_ell":
        fmt = FormatSpec("slice_ell", BLOCK_HEIGHT, BLOCK_WIDTH, COLUMNS)
        packed = pack_for_design(matrix, fmt, design)
        config, counts = make_runtime_config(x, packed, COLUMNS)
        operator = SpMVSliceELLDynamicScalarMultiCol(
            M, K, counts, block_height=BLOCK_HEIGHT,
        )
        inputs = {"packed": packed.packed_a_as_bf16, "config": config}
        payload = packed.packed_a
        extra = {"row_indices_bytes": 0, "control_bytes": config.numel() * 2,
                 "packed_a_bytes": packed.packed_a.nbytes,
                 "blocks_per_column": list(counts),
                 "blocks_per_slice_sha256": sha256_array(packed.blocks_per_slice),
                 "row_indices_sha256": None}
    else:
        fmt = FormatSpec(
            "sell_c_sigma", block_height, BLOCK_WIDTH, COLUMNS,
            window_count=windows,
        )
        packed = pack_for_design(matrix, fmt, design)
        rows_per_core = (2, 3, 3) if block_height == 8 else (block_height // 3,) * 3
        operator, inputs = prepare_sell_design(
            packed, x, design_name, rows_per_core=rows_per_core,
        )
        payload = packed.packed_a
        extra = {"row_indices_bytes": packed.row_indices.nbytes,
                 "control_bytes": inputs["control"].numel() * 2,
                 "packed_a_bytes": packed.packed_a.nbytes,
                 "blocks_per_column": [
                     int(packed.column_blocks_per_slice(col).sum())
                     for col in range(COLUMNS)
                 ],
                 "blocks_per_slice_sha256": sha256_array(packed.blocks_per_slice),
                 "row_indices_sha256": sha256_array(packed.row_indices)}

    storage = estimate_storage(matrix.profile, fmt, design)
    if design_name != "dense_k_tiled":
        if storage["packed_a_bytes"] != extra["packed_a_bytes"]:
            raise AssertionError("estimated and packed A sizes disagree")
    return operator, inputs, fmt, storage, {
        **extra, "packed_a_sha256": sha256_array(payload),
    }


def measure_case(matrix, design_name: str, windows: int, expected: torch.Tensor,
                 block_height: int = BLOCK_HEIGHT) -> dict:
    """Verify canonical output and time only a valid compatible design."""
    operator, inputs, fmt, storage, extra = make_case(matrix, design_name, windows, block_height)
    errors, latency_us, bandwidth_gbps, timed_samples_us = run_test(
        operator, inputs, {"output": expected}, rel_tol=0.08, abs_tol=0.025,
        warmup_iters=WARMUP_ITERS, timed_iters=TIMED_ITERS,
        return_timings=True,
    )
    if errors:
        raise AssertionError(
            f"{matrix.profile.spec.tensor_name} {design_name} "
            f"windows={windows}: {sum(map(len, errors.values()))} output errors"
        )
    profile = matrix.profile
    vector_bits = matrix.vector.contiguous().view(torch.uint16).numpy()
    # Fixed-size MemTile FIFO payloads, excluding compiler bookkeeping and
    # routing buffers. This is a documented lower bound, not an L2 allocation.
    if design_name == "sell_dedicated_reorder":
        memtile_fifo_bytes_per_column = (
            2 * block_height * BLOCK_WIDTH * 4
            + extra["control_bytes"] // COLUMNS // (windows // COLUMNS)
            + 2 * sum(r + r % 2 for r in ((2, 3, 3) if block_height == 8
                                          else (block_height // 3,) * 3)) * 2
        )
    elif design_name == "sell_time_multiplex_reorder":
        memtile_fifo_bytes_per_column = 2 * BLOCK_HEIGHT * BLOCK_WIDTH * 4
    elif design_name == "slice_ell":
        memtile_fifo_bytes_per_column = (
            2 * BLOCK_HEIGHT * BLOCK_WIDTH * 4 + 2 * BLOCK_HEIGHT * 2
        )
    else:
        memtile_fifo_bytes_per_column = None
    return {
        "matrix_id": profile.spec.matrix_id,
        "matrix_input": vars(profile.spec),
        "source_sha256": profile.source_sha256,
        "csr_sha256": hashlib.sha256(
            matrix.indptr.tobytes() + matrix.indices.tobytes()
            + matrix.values.tobytes()
        ).hexdigest(),
        "x_sha256": sha256_array(vector_bits),
        "format_id": fmt.format_id,
        "format_spec": vars(fmt),
        "design_id": design_name,
        "M": profile.M, "K": profile.K,
        "nnz": profile.nnz, "density": profile.density,
        "windows": windows if fmt.name == "sell_c_sigma" else 0,
        "padded_slots": storage["padded_slots"],
        "packed_a_bytes": extra["packed_a_bytes"],
        "row_indices_bytes": extra["row_indices_bytes"],
        "control_bytes": extra["control_bytes"],
        "total_blocks": storage.get("total_blocks"),
        "max_blocks_per_slice": storage.get("max_blocks_per_slice"),
        "column_blocks": storage.get("column_blocks"),
        "column_imbalance": storage.get("column_imbalance"),
        "npu_latency_us": latency_us,
        "timed_samples_us": timed_samples_us,
        "effective_bandwidth_gbps": bandwidth_gbps,
        "memtile_fifo_payload_lower_bound_bytes_per_column": memtile_fifo_bytes_per_column,
        "build_cache_hit": None,
        "build_cache_note": "Operator API does not report cache hit/miss reliably.",
        "canonical_output_verified": True,
        "cpu_error_count": 0,
        "warmup_iters": WARMUP_ITERS, "timed_iters": TIMED_ITERS,
        "dummy_between_runs": False,
        "spmv_only_latency_us": latency_us if design_name in ("dense_k_tiled", "slice_ell") else None,
        "spmv_only_note": None if design_name in ("dense_k_tiled", "slice_ell") else
            "Reorder is fused into this NPU design; no isolated SpMV-only timer.",
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--weight", action="append", choices=REPRESENTATIVE_WEIGHTS)
    parser.add_argument("--design", action="append", choices=DESIGNS)
    parser.add_argument("--windows", nargs="+", type=int, choices=(8, 16), default=(8,))
    parser.add_argument("--block-height", type=int, choices=(6, 8, 9, 18, 36, 72), default=8,
                        help="variable height is currently supported only by SELL dedicated")
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()

    aie_utils.set_current_device(NPU2())
    names = args.weight or REPRESENTATIVE_WEIGHTS
    designs = args.design or DESIGNS
    for name in names:
        spec = MatrixInput(
            "safetensors", model_dir=str(args.model_dir.resolve()),
            tensor_name=name, x_seed=3000 + REPRESENTATIVE_WEIGHTS.index(name),
        )
        matrix = load_or_generate_csr(spec)
        reference = cpu_spmv_csr(
            matrix.indptr, matrix.indices, matrix.values, matrix.vector,
        )
        print(f"loaded {name}: {matrix.profile.M}x{matrix.profile.K} "
              f"nnz={matrix.profile.nnz}", flush=True)
        for design_name in designs:
            cases = args.windows if design_name.startswith("sell_") else (0,)
            for windows in cases:
                record = measure_case(matrix, design_name, windows, reference, args.block_height)
                line = json.dumps(record, sort_keys=True)
                print(line, flush=True)
                if args.output_jsonl:
                    with args.output_jsonl.open("a") as handle:
                        handle.write(line + "\n")


if __name__ == "__main__":
    main()
