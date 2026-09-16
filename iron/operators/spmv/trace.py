#!/usr/bin/env python3
"""Capture one static SpMV NPU trace and write a Perfetto JSON file.

This uses a one-step full-ELF ``OperatorSequence``.  The ordinary single-op
xclbin callable has no host-visible trace-buffer argument, while the full-ELF
sequence allocates, dispatches, and synchronizes that buffer for us.
"""

import argparse
import os
from pathlib import Path

import torch

from iron.common.context import AIEContext
from iron.common.sequence import OperatorSequence
from iron.common.test_utils import verify_buffer
from iron.common.tracing_utils import dump_traces
from iron.operators.spmv.op import SpMVELL, SpMVSELL32Block
from iron.operators.spmv.reference import (
    make_uniform_ell,
    make_uniform_sell32,
    reference_ell,
    reference_sell32_block,
)


def _stage(run, name, data):
    buf = run.get_buffer(name)
    buf.torch_view()[: data.numel()] = data.reshape(-1)
    buf.to("npu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("ell", "sell32-block"), default="ell")
    parser.add_argument("--trace-size", type=int, default=65536)
    parser.add_argument("--trace-tiles", type=int, default=1)
    parser.add_argument(
        "--cols",
        type=int,
        default=1,
        help="AIE columns in the trace build (default: 1; full 8-column routing is congested)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spmv_traces"))
    args = parser.parse_args()
    if args.trace_size <= 0 or args.trace_tiles <= 0 or args.cols <= 0:
        raise ValueError("trace-size, trace-tiles, and cols must be positive")

    # Keep generation and compilation deterministic, and make the choice visible
    # to the shared trace helper during MLIR generation.
    os.environ["IRON_TRACE_SIZE"] = str(args.trace_size)
    os.environ["IRON_TRACE_NTILES"] = str(args.trace_tiles)
    os.environ["IRON_TRACE_DIR"] = str(args.output_dir)

    M, K, width = 1024, 2048, 256
    vector = torch.rand(K, generator=torch.Generator().manual_seed(42)).to(torch.bfloat16)
    context = AIEContext()
    if args.kernel == "ell":
        packed = make_uniform_ell(M, K, width, seed=17)
        expected = reference_ell(packed, vector, M, width)
        op = SpMVELL(M, K, width, rows=4, cols=args.cols, rows_per_core=2,
                     trace_size=args.trace_size, context=context)
    else:
        packed = make_uniform_sell32(M, K, width, seed=31)
        expected = reference_sell32_block(packed, vector, M, width)
        op = SpMVSELL32Block(M, K, width, rows=4, cols=args.cols,
                              trace_size=args.trace_size, context=context)

    sequence = OperatorSequence(
        name=f"spmv_{args.kernel}_trace",
        runlist=[(op, "packed", "vector", "output")],
        input_args=["packed", "vector"],
        output_args=["output"],
        dispatch="fused",
        trace_size=args.trace_size,
        context=context,
    )
    sequence.compile()
    run = sequence.get_callable()
    _stage(run, "packed", packed)
    _stage(run, "vector", vector)
    run()

    output = run.get_buffer("output").torch_view()[:M].clone()
    errors = verify_buffer(output, "output", expected, rel_tol=0.06, abs_tol=1e-4)
    if errors:
        raise RuntimeError(f"reference mismatch: {errors[:10]}")

    paths = dump_traces(run, f"spmv_{args.kernel}", out_dir=args.output_dir)
    if not paths:
        raise RuntimeError("trace capture produced no Perfetto JSON")
    print("Perfetto JSON:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
