# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-5 NPU measurement on representative Llama-2-7B pruned weights.

The matrix payload is read directly from safetensors, converted to the
row-order-preserving ``B_h=8, B_w=256`` Slice-ELL format, then measured on
four cores and all 32 cores.  No host CPU reference is run for these large
performance cases; Phase-4 tests cover the packed-format numerical contract.

When safetensors is only installed in the ELSA environment, expose it to the
IRON environment, for example::

  NPU_RUNTIME=xrt PYTHONPATH=/home/hitoshi/elsa/elsa_venv/lib/python3.12/site-packages:$PYTHONPATH \\
    python operators/spmv/measure_llama_slice_ell.py MODEL_DIR
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import torch
import aie.utils as aie_utils
from aie.iron.device import NPU2
from safetensors.torch import safe_open

from iron.common.test_utils import run_test
from iron.operators.spmv.op import SpMVSliceELLDynamicScalarMultiCol
from iron.operators.spmv.slice_ell import SliceELLConfig, csr_to_slice_ell


# Deliberately include three common projection shapes and one low-ELL-benefit
# counterexample.  Their storage-only statistics are recorded in the tuning
# note before this script is used for NPU latency measurements.
REPRESENTATIVE_WEIGHTS = (
    "model.layers.3.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.25.mlp.down_proj.weight",
)

BLOCK_HEIGHT = 8
BLOCK_WIDTH = 256
CORE_ROWS = 4


def index_safetensors(model_dir: Path) -> dict[str, Path]:
    """Map every tensor name to its owning checkpoint shard without loading it."""
    tensor_paths: dict[str, Path] = {}
    for path_string in sorted(glob(str(model_dir / "*.safetensors"))):
        path = Path(path_string)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensor_paths:
                    raise ValueError(f"duplicate tensor name across shards: {name}")
                tensor_paths[name] = path
    if not tensor_paths:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return tensor_paths


def load_csr(tensor_path: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one 2D safetensor and return its CSR arrays for the Slice-ELL packer."""
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        matrix = handle.get_tensor(name)
    if matrix.ndim != 2:
        raise ValueError(f"{name} is not a matrix: {tuple(matrix.shape)}")
    rows, cols = matrix.shape
    if cols > np.iinfo(np.uint16).max:
        raise ValueError(f"{name} has K={cols}, beyond the uint16 index contract")
    csr = matrix.to_sparse_csr()
    return (
        csr.crow_indices().cpu().numpy().astype(np.int64, copy=False),
        csr.col_indices().cpu().numpy().astype(np.uint16, copy=False),
        csr.values().cpu().float().numpy(),
    )


def make_runtime_config(
    vector: torch.Tensor, packed, cols: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Build per-column ``[C_h, reserved, x, p..., pad]`` config objects."""
    M, K = packed.M, packed.K
    slices_per_column = M // (BLOCK_HEIGHT * cols)
    p_by_column = packed.blocks_per_slice.reshape(cols, slices_per_column)
    blocks_per_column = tuple(int(p.sum()) for p in p_by_column)
    if any(count == 0 for count in blocks_per_column):
        raise ValueError("every active column must own at least one A block")

    config_words = 2 + K + slices_per_column
    config_words += config_words % 2  # ObjectFIFO config DMA needs 4-byte alignment.
    config = torch.zeros(cols * config_words, dtype=torch.int16)
    x_words = vector.view(torch.uint16).view(torch.int16)
    for column in range(cols):
        base = column * config_words
        config[base] = BLOCK_HEIGHT // CORE_ROWS
        config[base + 2 : base + 2 + K] = x_words
        config[base + 2 + K : base + 2 + K + slices_per_column] = torch.from_numpy(
            p_by_column[column].astype(np.int16, copy=False)
        )
    return config, blocks_per_column


def measure_weight(tensor_paths: dict[str, Path], name: str, cols: int, seed: int) -> None:
    """Pack and measure one representative weight for one core-column count."""
    indptr, indices, values = load_csr(tensor_paths[name], name)
    M = indptr.size - 1
    K = int(indices.max()) + 1 if indices.size else 0
    # Shape K must come from the tensor, not its maximum occupied column.
    with safe_open(tensor_paths[name], framework="pt", device="cpu") as handle:
        K = int(handle.get_slice(name).get_shape()[1])
    if M % (BLOCK_HEIGHT * cols):
        raise ValueError(f"{name}: M={M} is not divisible by B_h*columns={BLOCK_HEIGHT * cols}")
    packed = csr_to_slice_ell(
        indptr, indices, values, K=K,
        config=SliceELLConfig(
            core_rows=CORE_ROWS,
            block_height=BLOCK_HEIGHT,
            block_width=BLOCK_WIDTH,
            shim_columns=cols,
        ),
    )
    vector = torch.rand(K, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
    config, blocks_per_column = make_runtime_config(vector, packed, cols)
    operator = SpMVSliceELLDynamicScalarMultiCol(
        M=M, K=K, blocks_per_column=blocks_per_column, block_height=BLOCK_HEIGHT
    )
    _, latency_us, bandwidth_gbps = run_test(
        operator,
        {"packed": packed.packed_a_as_bf16, "config": config},
        {"output": None}, warmup_iters=2, timed_iters=5,
    )
    physical_width = int(packed.blocks_per_slice.max()) * BLOCK_WIDTH
    nnz = int(indptr[-1])
    print(
        f"name={name} M={M} K={K} nnz={nnz} density={nnz / (M * K):.6f} "
        f"cols={cols} cores={CORE_ROWS * cols} B_h={BLOCK_HEIGHT} B_w={BLOCK_WIDTH} "
        f"physical_width_max={physical_width} blocks_per_column={blocks_per_column} "
        f"latency_us={latency_us:.4f} effective_bandwidth_gbps={bandwidth_gbps:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--columns", type=int, choices=(1, 8), default=None)
    parser.add_argument("--weight", action="append", choices=REPRESENTATIVE_WEIGHTS)
    args = parser.parse_args()
    aie_utils.set_current_device(NPU2())
    tensor_paths = index_safetensors(args.model_dir)
    names = tuple(args.weight) if args.weight else REPRESENTATIVE_WEIGHTS
    columns = (args.columns,) if args.columns else (8, 1)
    for weight_index, name in enumerate(names):
        if name not in tensor_paths:
            raise KeyError(f"missing representative tensor: {name}")
        for cols in columns:
            measure_weight(tensor_paths, name, cols, seed=3000 + weight_index)


if __name__ == "__main__":
    main()
