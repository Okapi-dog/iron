#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack and measure one SpMV matrix/design case on the NPU.

Real and synthetic experiment drivers both call ``measure_case``. This module
does not choose matrices, loop over paper conditions, or write results.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.gemv.k_tiled_op import DenseGEMVKTile
from iron.operators.spmv.matrix_preparation import (
    DesignSpec, FormatSpec, estimate_storage, pack_for_design,
)
from iron.operators.spmv.measure_llama_dense_gemv_k_tiled import pack_dense_k_tiled
from iron.operators.spmv.measure_llama_slice_ell import make_runtime_config
from iron.operators.spmv.op import SpMVSELL32Block, SpMVSliceELLDynamicScalarMultiCol
from iron.operators.spmv.sell_c_sigma_runtime import prepare_sell_design


DESIGNS = (
    "dense_k_tiled", "slice_ell", "sell_dedicated_reorder",
    "sell_time_multiplex_reorder",
)
AVAILABLE_DESIGNS = ("ell", *DESIGNS)
BLOCK_HEIGHT = 8
BLOCK_WIDTH = 256
COLUMNS = 8
WARMUP_ITERS = 2
TIMED_ITERS = 5
ELL_ROWS_PER_CORE = 32
ELL_CORES_PER_COLUMN = 4


def sha256_array(array: np.ndarray) -> str:
    """Fingerprint one contiguous host-side payload without saving a binary."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def ell_npu_rows(rows: int, columns: int = COLUMNS) -> int:
    """Round ELL output rows to a full 32-row block on every NPU core."""
    rows_per_dispatch = ELL_ROWS_PER_CORE * ELL_CORES_PER_COLUMN * columns
    return ((rows + rows_per_dispatch - 1) // rows_per_dispatch) * rows_per_dispatch


def estimate_case_storage(profile, fmt: FormatSpec, design: DesignSpec) -> dict:
    """Include the ELL kernel's physical row padding in its NPU payload bound."""
    storage = estimate_storage(profile, fmt, design)
    if design.name != "ell":
        return storage

    padded_rows = ell_npu_rows(profile.M, fmt.columns)
    slots = padded_rows * storage["ell_width"]
    packed_bytes = 4 * slots  # BF16 value + uint16 index per slot.
    dense_bytes = 2 * profile.M * profile.K  # Original, unpadded matrix.
    return {
        **storage,
        "padded_rows": padded_rows,
        "padded_slots": slots,
        "packed_a_bytes": packed_bytes,
        "total_storage_bytes": packed_bytes,
        "a_over_dense": packed_bytes / dense_bytes,
        "dense_over_a": dense_bytes / packed_bytes if packed_bytes else None,
        "storage_over_dense": packed_bytes / dense_bytes,
        "dense_over_storage": dense_bytes / packed_bytes if packed_bytes else None,
    }


def make_case(matrix, design_name: str, windows: int, block_height: int = BLOCK_HEIGHT):
    """Build one operator and payload, preserving matrix/vector identity."""
    M, K = matrix.profile.M, matrix.profile.K
    x = matrix.vector
    design = DesignSpec(design_name)
    if block_height != BLOCK_HEIGHT and design_name == "slice_ell" and block_height != 6:
        raise ValueError("row-order Slice-ELL additionally supports B_h=6")
    if block_height != BLOCK_HEIGHT and design_name == "sell_time_multiplex_reorder":
        raise ValueError("variable B_h is not implemented for multiplexed SELL")
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
                 "control_bytes": 0, "packed_a_bytes": packed_a.numel() * 2,
                 "x_transfer_bytes": tiled_x.numel() * 2}
    elif design_name == "ell":
        fmt = FormatSpec("ell", columns=COLUMNS)
        row_major = pack_for_design(matrix, fmt, design)
        width = row_major.numel() // (2 * M)
        npu_rows = ell_npu_rows(M)
        row_words = row_major.view(torch.uint16).numpy().reshape(M, 2, width)
        if npu_rows != M:
            # These all-zero rows live only in the ELL payload, not in CSR.
            padded = np.zeros((npu_rows, 2, width), dtype=np.uint16)
            padded[:M] = row_words
            row_words = padded
        # Transpose each 32-row group to the old vertical-vector kernel's ABI.
        words = np.ascontiguousarray(
            row_words.reshape(npu_rows // 32, 32, 2, width).transpose(0, 3, 2, 1)
        )
        operator = SpMVSELL32Block(npu_rows, K, width)
        inputs = {"packed": torch.from_numpy(words.reshape(-1)).view(torch.bfloat16),
                  "vector": x}
        payload = words
        extra = {"ell_width": width, "row_indices_bytes": 0,
                 "control_bytes": 0, "packed_a_bytes": words.nbytes,
                 "ell_npu_rows": npu_rows, "ell_padding_rows": npu_rows - M}
    elif design_name == "slice_ell":
        fmt = FormatSpec("slice_ell", block_height, BLOCK_WIDTH, COLUMNS)
        packed = pack_for_design(matrix, fmt, design)
        core_rows = 3 if block_height == 6 else 4
        config, counts = make_runtime_config(
            x, packed, COLUMNS, block_height=block_height, core_rows=core_rows,
        )
        operator = SpMVSliceELLDynamicScalarMultiCol(
            packed.padded_rows, K, counts, block_height=block_height,
            core_rows=core_rows,
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

    storage = estimate_case_storage(matrix.profile, fmt, design)
    if design_name != "dense_k_tiled":
        if storage["packed_a_bytes"] != extra["packed_a_bytes"]:
            raise AssertionError("estimated and packed A sizes disagree")
    return operator, inputs, fmt, storage, {
        **extra, "packed_a_sha256": sha256_array(payload),
    }


def measure_case(matrix, design_name: str, windows: int, expected: torch.Tensor,
                 block_height: int = BLOCK_HEIGHT,
                 warmup_iters: int = WARMUP_ITERS,
                 timed_iters: int = TIMED_ITERS,
                 idle_s: float = 0.0) -> dict:
    """Verify canonical output and time only a valid compatible design."""
    operator, inputs, fmt, storage, extra = make_case(matrix, design_name, windows, block_height)
    expected_output = expected
    if design_name == "ell" and extra["ell_padding_rows"]:
        expected_output = torch.zeros(extra["ell_npu_rows"], dtype=expected.dtype)
        expected_output[:matrix.profile.M] = expected
    errors, latency_us, bandwidth_gbps, timed_samples_us = run_test(
        operator, inputs, {"output": expected_output}, rel_tol=0.08, abs_tol=0.025,
        warmup_iters=warmup_iters, timed_iters=timed_iters,
        return_timings=True, idle_s=idle_s,
    )
    error_count = sum(map(len, errors.values()))
    allowed_errors = max(1, matrix.profile.M // 1000) if design_name == "ell" else 0
    if error_count > allowed_errors:
        raise AssertionError(
            f"{matrix.profile.spec.tensor_name} {design_name} "
            f"windows={windows}: {error_count} output errors"
        )
    profile = matrix.profile
    vector_bits = matrix.vector.contiguous().view(torch.uint16).numpy()
    x_copies = windows if design_name.startswith("sell_") else (
        COLUMNS if design_name in ("dense_k_tiled", "slice_ell") else 1
    )
    # Config/control embeds x for every column or window. Count it for
    # transfer, but not for stored matrix-format capacity.
    format_metadata_bytes = max(0, extra["control_bytes"] - 2 * profile.K * x_copies)
    if design_name == "dense_k_tiled":
        format_metadata_bytes = 0
    storage_bytes = extra["packed_a_bytes"] + format_metadata_bytes
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
            2 * block_height * BLOCK_WIDTH * 4 + 2 * block_height * 2
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
        "design_id": "ell_vertical_block" if design_name == "ell" else design_name,
        "M": profile.M, "K": profile.K,
        "nnz": profile.nnz, "density": profile.density,
        "windows": windows if fmt.name == "sell_c_sigma" else 0,
        "padded_slots": storage["padded_slots"],
        "packed_a_bytes": extra["packed_a_bytes"],
        "row_indices_bytes": extra["row_indices_bytes"],
        "control_bytes": extra["control_bytes"],
        "format_metadata_bytes": format_metadata_bytes,
        "storage_bytes": storage_bytes,
        "storage_over_dense": storage_bytes / (2 * profile.M * profile.K),
        "dense_bf16_bytes": 2 * profile.M * profile.K,
        "x_transfer_bytes": extra.get("x_transfer_bytes", 2 * profile.K * x_copies),
        "y_transfer_bytes": 2 * extra.get("ell_npu_rows", profile.M),
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
        "canonical_output_verified": error_count == 0,
        "canonical_output_tolerance_exceptions": error_count,
        "cpu_error_count": error_count,
        "status": "ok_with_tolerance_exceptions" if error_count else "ok",
        "warmup_iters": warmup_iters, "timed_iters": timed_iters,
        "idle_seconds_before_timed": idle_s,
        "dummy_between_runs": False,
        "spmv_only_latency_us": latency_us if design_name in ("dense_k_tiled", "ell", "slice_ell") else None,
        "spmv_only_note": None if design_name in ("dense_k_tiled", "ell", "slice_ell") else
            "Reorder is fused into this NPU design; no isolated SpMV-only timer.",
        **extra,
    }
