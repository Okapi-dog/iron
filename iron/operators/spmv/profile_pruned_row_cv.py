#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile row-NNZ variation of every pruned Llama projection weight.

This is a CPU-only, read-only checkpoint scan. It does not load the full model,
pack sparse data, or run an NPU kernel. One tensor is inspected at a time.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open


PROJECTION_NAME = re.compile(
    r"^model\.layers\.(\d+)\.(?:self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))\.weight$"
)


def profile_tensor(name: str, weight: torch.Tensor, shard: Path) -> dict:
    """Return shape, density, and population CV of NNZ counts across rows."""
    match = PROJECTION_NAME.fullmatch(name)
    if match is None or weight.ndim != 2:
        raise ValueError(f"not a 2-D projection weight: {name}")
    counts = weight.count_nonzero(dim=1).cpu().numpy().astype(np.int64)
    M, K = (int(dimension) for dimension in weight.shape)
    mean = float(counts.mean())
    quantiles = np.quantile(counts, [0.25, 0.5, 0.75])
    return {
        "weight": name,
        "layer": int(match.group(1)),
        "projection": match.group(2) or match.group(3),
        "shard": shard.name,
        "M": M,
        "K": K,
        "nnz": int(counts.sum()),
        "density": float(counts.sum() / (M * K)),
        "mean_row_nnz": mean,
        "row_nnz_cv": float(counts.std(ddof=0) / mean) if mean else 0.0,
        "row_nnz_min": int(counts.min()),
        "row_nnz_q1": float(quantiles[0]),
        "row_nnz_median": float(quantiles[1]),
        "row_nnz_q3": float(quantiles[2]),
        "row_nnz_max": int(counts.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    args = parser.parse_args()
    shards = sorted(args.model_dir.glob("*.safetensors"))
    if not shards:
        parser.error("model_dir contains no .safetensors shards")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    rows = []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                if PROJECTION_NAME.fullmatch(name):
                    rows.append(profile_tensor(name, handle.get_tensor(name), shard))
        print(f"scanned {shard.name}: {len(rows)} projection weights", flush=True)
    rows.sort(key=lambda row: (row["layer"], row["projection"]))
    if len(rows) != 224:
        raise RuntimeError(f"expected 32 layers x 7 projections = 224, found {len(rows)}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"saved {len(rows)} rows to {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
