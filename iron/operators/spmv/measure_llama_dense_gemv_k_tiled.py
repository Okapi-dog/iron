#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Measure the 4x8 K-tiled dense GEMV baseline on Llama Phase-5 weights.

The dense matrix itself is unchanged, including all pruned zeros.  This script
only repacks transfer order into ``[column][8 rows][K tile]`` objects so that a
core never needs the complete input vector in its 64 KiB L1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import aie.utils as aie_utils
import torch
from aie.iron.device import NPU2
from safetensors.torch import safe_open

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.gemv.k_tiled_op import DenseGEMVKTile


REPRESENTATIVE_WEIGHTS = (
    "model.layers.3.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.25.mlp.down_proj.weight",
)

# 4096 keeps a two-row core input FIFO + x tile around 40 KiB with ping-pong.
# K=11008 uses 1376 (= 43*32) instead, eliminating tail padding.
DEFAULT_K_TILE = 4096
BLOCK_HEIGHT = 8


def index_safetensors(model_dir: Path) -> dict[str, Path]:
    """Map tensor names to checkpoint shards without materializing their data."""
    result: dict[str, Path] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            result.update({name: path for name in handle.keys()})
    if not result:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return result


def pack_dense_k_tiled(matrix: torch.Tensor, vector: torch.Tensor, cols: int, k_tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack dense A and repeated x tiles in the exact ObjectFIFO consume order."""
    M, K = matrix.shape
    blocks_per_col = M // (BLOCK_HEIGHT * cols)
    a_parts: list[torch.Tensor] = []
    x_parts: list[torch.Tensor] = []
    for col in range(cols):
        col_row0 = col * blocks_per_col * BLOCK_HEIGHT
        for block in range(blocks_per_col):
            rows = matrix[col_row0 + block * BLOCK_HEIGHT : col_row0 + (block + 1) * BLOCK_HEIGHT]
            for k0 in range(0, K, k_tile):
                a_tile = torch.zeros((BLOCK_HEIGHT, k_tile), dtype=torch.bfloat16)
                x_tile = torch.zeros(k_tile, dtype=torch.bfloat16)
                width = min(k_tile, K - k0)
                a_tile[:, :width] = rows[:, k0 : k0 + width]
                x_tile[:width] = vector[k0 : k0 + width]
                a_parts.append(a_tile.reshape(-1))
                x_parts.append(x_tile)
    return torch.cat(a_parts), torch.cat(x_parts)


def measure_weight(paths: dict[str, Path], name: str, cols: int, seed: int, verify: bool) -> None:
    """Run one real pruned Llama matrix at the selected 4*cols core geometry."""
    with safe_open(paths[name], framework="pt", device="cpu") as handle:
        matrix = handle.get_tensor(name).to(torch.bfloat16).contiguous()
    M, K = matrix.shape
    k_tile = 1376 if K == 11008 else DEFAULT_K_TILE
    if M % (BLOCK_HEIGHT * cols):
        raise ValueError(f"{name}: shape {M}x{K} is not compatible with cols={cols}, k_tile={k_tile}")
    vector = torch.rand(K, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
    packed_a, tiled_x = pack_dense_k_tiled(matrix, vector, cols, k_tile)
    context = AIEContext()
    operator = DenseGEMVKTile(M=M, K=K, cols=cols, k_tile=k_tile, context=context)
    expected = matrix @ vector if verify else None
    errors, latency_us, bandwidth_gbps = run_test(
        operator, {"matrix": packed_a, "vector_tiles": tiled_x}, {"output": expected},
        rel_tol=0.05, abs_tol=2e-2, warmup_iters=2, timed_iters=5,
    )
    if errors:
        raise AssertionError(f"{name} cols={cols}: numerical verification failed: {errors}")
    actual_input_bytes = (packed_a.numel() + tiled_x.numel()) * 2
    print(
        f"name={name} M={M} K={K} cols={cols} cores={4 * cols} k_tile={k_tile} "
        f"input_bytes={actual_input_bytes} latency_us={latency_us:.4f} "
        f"effective_bandwidth_gbps={bandwidth_gbps:.4f}", flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--columns", choices=("1", "8"), default=None)
    parser.add_argument("--weight", action="append", choices=REPRESENTATIVE_WEIGHTS)
    parser.add_argument("--verify", action="store_true", help="check a selected case against CPU BF16 matvec")
    args = parser.parse_args()
    aie_utils.set_current_device(NPU2())
    paths = index_safetensors(args.model_dir)
    names = tuple(args.weight) if args.weight else REPRESENTATIVE_WEIGHTS
    columns = (int(args.columns),) if args.columns else (8, 1)
    for weight_index, name in enumerate(names):
        for cols in columns:
            measure_weight(paths, name, cols, seed=5000 + weight_index, verify=args.verify)


if __name__ == "__main__":
    main()
