# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the existing dense IRON GEMV on Phase-5 Llama weight matrices.

This is a dense BF16 baseline: the original pruned matrices are transferred
with their zeros intact.  The stock GEMV places one worker in each AIE column,
so ``--columns 4`` uses four cores and ``--columns 8`` uses eight cores.  It
is intentionally reported separately from Slice-ELL's four-rows-per-column
layout.
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import aie.utils as aie_utils
import torch
from aie.iron.device import NPU2
from safetensors.torch import safe_open

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.gemv.op import GEMV
from iron.operators.spmv.measure_llama_slice_ell import REPRESENTATIVE_WEIGHTS


def index_safetensors(model_dir: Path) -> dict[str, Path]:
    """Map tensor names to checkpoint shards without materializing their data."""
    result: dict[str, Path] = {}
    for path_string in sorted(glob(str(model_dir / "*.safetensors"))):
        path = Path(path_string)
        with safe_open(path, framework="pt", device="cpu") as handle:
            result.update({name: path for name in handle.keys()})
    if not result:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return result


def measure_weight(paths: dict[str, Path], name: str, columns: int, seed: int) -> None:
    """Transfer one dense BF16 matrix, including zeros, and measure its GEMV."""
    with safe_open(paths[name], framework="pt", device="cpu") as handle:
        matrix = handle.get_tensor(name)
    if matrix.ndim != 2:
        raise ValueError(f"{name} is not a matrix")
    M, K = matrix.shape
    if M % columns or K % 64:
        raise ValueError(f"{name}: GEMV requires M divisible by columns and K divisible by 64")
    vector = torch.rand(K, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
    context = AIEContext()
    operator = GEMV(
        M=M,
        K=K,
        num_aie_columns=columns,
        tile_size_input=1,
        tile_size_output=M // columns,
        # K=11008 needs 22 KiB each for A and x.  The normal A ping-pong
        # allocation would exceed one core's 64 KiB L1, so use a single A tile.
        a_fifo_depth=1 if K > 8192 else 2,
        context=context,
    )
    _, latency_us, bandwidth_gbps = run_test(
        operator,
        {"matrix": matrix.to(torch.bfloat16).flatten(), "vector": vector},
        {"output": None}, warmup_iters=2, timed_iters=5,
    )
    print(
        f"name={name} M={M} K={K} columns={columns} cores={columns} a_fifo_depth={operator.a_fifo_depth} "
        f"latency_us={latency_us:.4f} effective_bandwidth_gbps={bandwidth_gbps:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--columns", type=int, choices=(4, 8), default=None)
    parser.add_argument("--weight", action="append", choices=REPRESENTATIVE_WEIGHTS)
    args = parser.parse_args()
    aie_utils.set_current_device(NPU2())
    paths = index_safetensors(args.model_dir)
    names = tuple(args.weight) if args.weight else REPRESENTATIVE_WEIGHTS
    columns = (args.columns,) if args.columns else (8, 4)
    for weight_index, name in enumerate(names):
        if name not in paths:
            raise KeyError(f"missing representative tensor: {name}")
        for column_count in columns:
            measure_weight(paths, name, column_count, seed=4000 + weight_index)


if __name__ == "__main__":
    main()
