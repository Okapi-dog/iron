#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run paper SpMV measurements on real or seeded synthetic matrices.

Both sources use the same matrix preparation and one-case NPU measurement.
``synthetic --preflight`` uses seed 999; ``--profile-only`` needs no NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
import torch
from aie.iron.device import NPU2

from iron.operators.spmv.matrix_preparation import (
    DesignSpec, FormatSpec, MatrixInput, estimate_storage,
    load_or_generate_csr, synthetic_profile,
)
from iron.operators.spmv.matrix_measure import (
    AVAILABLE_DESIGNS, BLOCK_WIDTH, COLUMNS, TIMED_ITERS, WARMUP_ITERS,
    measure_case, sha256_array,
)
from iron.operators.spmv.paper_config import (
    PAPER_DESIGNS, PAPER_WEIGHTS, paper_conditions, timing_protocol_id,
)
from iron.operators.spmv.paper_provenance import paper_provenance
from iron.operators.spmv.slice_ell import cpu_spmv_csr


PAPER_OUTPUT = Path(__file__).resolve().parent / "paper_evaluation"


def storage_estimate(profile, design: str) -> dict:
    """CPU-only packed-A/row-map bound, available even if NPU build fails."""
    format_name = {
        "dense_k_tiled": "dense", "ell": "ell", "slice_ell": "slice_ell",
        "sell_dedicated_reorder": "sell_c_sigma",
    }[design]
    fmt = FormatSpec(format_name, block_height=6, block_width=256,
                     columns=8, window_count=8)
    storage = estimate_storage(profile, fmt, DesignSpec(design))
    return {
        "estimated_packed_a_bytes": storage["packed_a_bytes"],
        "estimated_a_plus_row_map_bytes": storage["total_storage_bytes"],
        "estimated_storage_over_dense": storage["storage_over_dense"],
        "estimated_ell_width": storage.get("ell_width"),
    }


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    """Resume a paper run without silently replacing failed or valid records."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if "matrix_id" in record and "requested_design" in record:
                keys.add((record["matrix_id"], record["requested_design"],
                          record.get("timing_protocol_id", "legacy")))
    return keys


def emit(record: dict, output: Path | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        with output.open("a") as handle:
            handle.write(line + "\n")


def profile_record(spec: MatrixInput, condition: str, profile) -> dict:
    """Record achieved, not merely requested, row statistics."""
    counts = profile.row_nnz
    actual_cv = float(counts.std() / counts.mean()) if counts.mean() else 0.0
    assert spec.M is not None and spec.K is not None and spec.row_cv is not None
    target_nnz = round(spec.M * spec.K * (spec.density or 0))
    if profile.nnz != target_nnz:
        raise AssertionError("density rounding changed total NNZ")
    if (spec.row_pattern == "cv_calibrated"
            and abs(actual_cv - spec.row_cv) > max(0.002, 0.05 * spec.row_cv)):
        raise AssertionError("achieved row CV is outside the paper tolerance")
    return {
        "matrix_id": spec.matrix_id,
        "source": "calibrated_synthetic_cv" if spec.row_pattern == "cv_calibrated"
                  else "seeded_synthetic_cv",
        "condition": condition, "M": spec.M, "K": spec.K,
        "seed": spec.seed, "x_seed": spec.x_seed,
        "target_density": spec.density, "target_sparsity": 1 - spec.density,
        "target_row_cv": spec.row_cv, "row_pattern": spec.row_pattern,
        "actual_density": profile.density, "actual_row_cv": actual_cv,
        "actual_mean_nnz": float(counts.mean()),
        "row_nnz_sd": float(counts.std()),
        "row_nnz_p25": float(np.percentile(counts, 25)),
        "row_nnz_p50": float(np.percentile(counts, 50)),
        "row_nnz_p75": float(np.percentile(counts, 75)),
        "row_nnz_p99": float(np.percentile(counts, 99)),
        "row_nnz_max": int(counts.max()), "nnz": profile.nnz,
        "row_nnz_sha256": profile.row_nnz_sha256,
        "dense_bf16_bytes": 2 * spec.M * spec.K,
        "block_height": 6, "block_width": 256,
        "columns": 8, "window_count": 8,
    }


def measure_synthetic(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="11 conditions, seed 999")
    mode.add_argument("--custom", action="store_true", help="Ad-hoc density/CV values")
    parser.add_argument("--profile-only", action="store_true", help="Check row counts/storage, no CSR or NPU")
    parser.add_argument("--condition", action="append", help="Select named paper condition for a staged run")
    parser.add_argument("--paper-seed", type=int, action="append",
                        help="Select one or more of the fixed paper seeds 1000..1009")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--density", type=float, action="append", help="Legacy/custom nonzero fraction")
    parser.add_argument("--cv", type=float, action="append", help="Legacy/custom target row CV")
    parser.add_argument("--seed", type=int, default=42, help="Legacy/custom seed")
    parser.add_argument("--x-seed", type=int, default=3000)
    parser.add_argument("--design", action="append", choices=PAPER_DESIGNS)
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--timed-iters", type=int)
    parser.add_argument("--idle-seconds", type=float)
    parser.add_argument("--inter-case-seconds", type=float,
                        help="Optional host pause between different matrix/design cases")
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args(argv)
    paper_mode = not args.custom
    if paper_mode and (args.M, args.K) != (4096, 4096):
        parser.error("paper and preflight modes fix M=K=4096")
    if paper_mode and (args.density or args.cv):
        parser.error("paper conditions are fixed; omit --density and --cv")
    if paper_mode and args.x_seed != 3000:
        parser.error("paper conditions fix x_seed=3000; use --custom otherwise")
    if args.condition and not paper_mode:
        parser.error("--condition is only available in the fixed paper sweep")
    if args.paper_seed and (args.custom or args.preflight):
        parser.error("--paper-seed is only for the full paper sweep")
    if args.paper_seed and any(seed not in range(1000, 1010) for seed in args.paper_seed):
        parser.error("--paper-seed must be in 1000..1009")
    if args.M <= 0 or args.K <= 0:
        parser.error("M and K must be positive")
    warmups = args.warmup_iters if args.warmup_iters is not None else (2 if paper_mode else 10)
    timed = args.timed_iters if args.timed_iters is not None else (1 if args.preflight else 5)
    idle = args.idle_seconds if args.idle_seconds is not None else 0.0
    inter_case = (args.inter_case_seconds if args.inter_case_seconds is not None
                  else 4.0 if paper_mode and not args.preflight else 0.0)
    if warmups < 0 or timed <= 0 or idle < 0 or inter_case < 0:
        parser.error("invalid warmup/timed/idle setting")
    protocol_id = timing_protocol_id(warmups, timed, idle, inter_case)
    if paper_mode:
        conditions = paper_conditions()
        if args.condition:
            known = {name for name, _, _ in conditions}
            unknown = set(args.condition) - known
            if unknown:
                parser.error(f"unknown paper condition(s): {sorted(unknown)}")
            conditions = [case for case in conditions if case[0] in args.condition]
        seeds = (999,) if args.preflight else (tuple(dict.fromkeys(args.paper_seed))
                                                if args.paper_seed else range(1000, 1010))
    else:
        conditions = [(f"custom_{d:g}_{cv:g}", d, cv)
                      for cv in (args.cv or (0.10, 0.25))
                      for d in (args.density or (0.50, 0.40, 0.30, 0.20, 0.10, 0.05))]
        seeds = (args.seed,)
    designs = args.design or PAPER_DESIGNS
    output = args.output_jsonl
    if output is None and paper_mode:
        filename = ("synthetic_profiles.jsonl" if args.profile_only else
                    "synthetic_preflight.jsonl" if args.preflight else
                    f"synthetic_paper_{protocol_id}.jsonl")
        output = PAPER_OUTPUT / filename
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(output) if output and paper_mode and not args.profile_only else set()

    if not args.profile_only:
        aie_utils.set_current_device(NPU2())
    order = len(done)
    for condition, density, cv in conditions:
        for seed in seeds:
            spec = MatrixInput(
                source="synthetic", M=args.M, K=args.K, density=density,
                row_cv=cv, row_pattern="cv_calibrated" if paper_mode else "cv",
                seed=seed, x_seed=args.x_seed,
            )
            if paper_mode and not args.profile_only and all(
                (spec.matrix_id, design, protocol_id) in done for design in designs
            ):
                continue
            profile = synthetic_profile(spec)
            common = profile_record(spec, condition, profile)
            if args.profile_only:
                emit({**common, "status": "profile_only",
                      "storage_estimates": {d: storage_estimate(profile, d)
                                            for d in designs}}, output)
                continue

            matrix = load_or_generate_csr(spec)
            expected = cpu_spmv_csr(matrix.indptr, matrix.indices,
                                    matrix.values, matrix.vector)
            common.update({
                "csr_sha256": hashlib.sha256(
                    matrix.indptr.tobytes() + matrix.indices.tobytes()
                    + matrix.values.tobytes()).hexdigest(),
                "x_sha256": sha256_array(matrix.vector.view(torch.uint16).numpy()),
                "x_bytes": 2 * args.K, "y_bytes": 2 * args.M,
                "warmup_iters": warmups, "timed_iters": timed,
                "idle_seconds_before_timed": idle,
                "paper_protocol": bool(paper_mode and not args.preflight
                                       and warmups == 2 and timed == 5
                                       and idle == 0.0 and inter_case == 4.0),
                "timing_protocol_id": protocol_id,
                "environment": paper_provenance(),
            })
            # Rotating the order reduces a systematic thermal bias by design.
            rotated = designs[seed % len(designs):] + designs[:seed % len(designs)]
            for design in rotated:
                if (spec.matrix_id, design, protocol_id) in done:
                    continue
                if order and inter_case:
                    time.sleep(inter_case)
                order += 1
                base = {**common, **storage_estimate(profile, design),
                        "requested_design": design, "run_order": order,
                        "inter_case_seconds": inter_case,
                        "measured_at_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    result = measure_case(matrix, design, 8, expected,
                                          block_height=6, warmup_iters=warmups,
                                          timed_iters=timed, idle_s=idle)
                    record = {**result, **base}
                except Exception as exc:
                    record = {**base, "status": "failed",
                              "error_type": type(exc).__name__,
                              "error": str(exc)[:1000]}
                emit(record, output)


def measure_real(argv: list[str]) -> None:
    """Load named checkpoint tensors one at a time and run the shared cases."""
    parser = argparse.ArgumentParser(description="Measure real pruned SpMV weights")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--weight", action="append", help="Full safetensors tensor name")
    parser.add_argument("--design", action="append", choices=AVAILABLE_DESIGNS)
    parser.add_argument("--windows", nargs="+", type=int, choices=(8, 16), default=(8,))
    parser.add_argument("--block-height", type=int, choices=(6, 8, 9, 18, 36, 72), default=6)
    parser.add_argument("--x-seed", type=int, default=3000)
    parser.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS)
    parser.add_argument("--timed-iters", type=int, default=TIMED_ITERS)
    parser.add_argument("--idle-seconds", type=float, default=0.0)
    parser.add_argument("--inter-case-seconds", type=float, default=4.0)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args(argv)
    if (args.warmup_iters < 0 or args.timed_iters <= 0
            or args.idle_seconds < 0 or args.inter_case_seconds < 0):
        parser.error("invalid warmup/timed/idle setting")
    designs = args.design or PAPER_DESIGNS
    paper_geometry = (args.block_height == 6 and tuple(args.windows) == (8,)
                      and args.x_seed == 3000
                      and all(design in PAPER_DESIGNS for design in designs)
                      and all(name in PAPER_WEIGHTS for name in (args.weight or PAPER_WEIGHTS)))
    protocol_id = timing_protocol_id(args.warmup_iters, args.timed_iters,
                                     args.idle_seconds, args.inter_case_seconds)
    paper_protocol = (paper_geometry and args.warmup_iters == 2
                      and args.timed_iters == 5 and args.idle_seconds == 0.0
                      and args.inter_case_seconds == 4.0)
    output = args.output_jsonl
    if output is None:
        prefix = ("real_weights" if paper_protocol else
                  f"real_exploratory_h{args.block_height}_w{'-'.join(map(str, args.windows))}")
        output = PAPER_OUTPUT / f"{prefix}_{protocol_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if "matrix_id" in record and "requested_design" in record:
                    completed.add((record["matrix_id"], record["requested_design"],
                                   record.get("windows", 0),
                                   record.get("timing_protocol_id", "legacy")))

    aie_utils.set_current_device(NPU2())
    run_order = len(completed)
    for name in args.weight or PAPER_WEIGHTS:
        spec = MatrixInput("safetensors", model_dir=str(args.model_dir.resolve()),
                           tensor_name=name, x_seed=args.x_seed)
        matrix = load_or_generate_csr(spec)
        expected = cpu_spmv_csr(matrix.indptr, matrix.indices,
                                matrix.values, matrix.vector)
        counts = matrix.profile.row_nnz
        actual_cv = float(counts.std() / counts.mean()) if counts.mean() else 0.0
        print(f"loaded {name}: {matrix.profile.M}x{matrix.profile.K} "
              f"nnz={matrix.profile.nnz}", flush=True)
        for design in designs:
            for windows in args.windows if design.startswith("sell_") else (0,):
                key = (spec.matrix_id, design, windows, protocol_id)
                if key in completed:
                    continue
                if run_order > len(completed) and args.inter_case_seconds:
                    time.sleep(args.inter_case_seconds)
                run_order += 1
                record = {
                    "paper_protocol": paper_protocol,
                    "timing_protocol_id": protocol_id,
                    "requested_design": design,
                    "matrix_id": spec.matrix_id, "matrix_input": vars(spec),
                    "tensor_name": name, "actual_row_cv": actual_cv,
                    "actual_density": matrix.profile.density,
                    "M": matrix.profile.M, "K": matrix.profile.K,
                    "nnz": matrix.profile.nnz, "windows": windows,
                    "row_nnz_max": int(counts.max()),
                    "row_nnz_p99": float(np.percentile(counts, 99)),
                    "row_nnz_sha256": matrix.profile.row_nnz_sha256,
                    "csr_sha256": hashlib.sha256(
                        matrix.indptr.tobytes() + matrix.indices.tobytes()
                        + matrix.values.tobytes()).hexdigest(),
                    "x_sha256": sha256_array(matrix.vector.view(torch.uint16).numpy()),
                    "dense_bf16_bytes": 2 * matrix.profile.M * matrix.profile.K,
                    "x_bytes": 2 * matrix.profile.K,
                    "y_bytes": 2 * matrix.profile.M,
                    "run_order": run_order,
                    "inter_case_seconds": args.inter_case_seconds,
                    "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                    "environment": paper_provenance(),
                }
                format_name = {"dense_k_tiled": "dense", "ell": "ell",
                               "slice_ell": "slice_ell",
                               "sell_dedicated_reorder": "sell_c_sigma"}.get(design)
                if format_name is not None:
                    estimate = estimate_storage(
                        matrix.profile,
                        FormatSpec(format_name, block_height=args.block_height,
                                   block_width=BLOCK_WIDTH, columns=COLUMNS,
                                   window_count=windows or 8),
                        DesignSpec(design),
                    )
                    record["estimated_packed_a_bytes"] = estimate["packed_a_bytes"]
                    record["estimated_storage_over_dense"] = estimate["storage_over_dense"]
                try:
                    result = measure_case(
                        matrix, design, windows, expected,
                        block_height=args.block_height,
                        warmup_iters=args.warmup_iters,
                        timed_iters=args.timed_iters,
                        idle_s=args.idle_seconds,
                    )
                    record = {**result, **record}
                except Exception as exc:
                    record.update(status="failed", error_type=type(exc).__name__,
                                  error=str(exc)[:1000], warmup_iters=args.warmup_iters,
                                  timed_iters=args.timed_iters,
                                  idle_seconds_before_timed=args.idle_seconds)
                emit(record, output)


def main(argv: list[str] | None = None) -> None:
    """Choose a paper data source; the NPU measurement path is shared."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print("Usage: python -m iron.operators.spmv.measure_paper "
              "{real MODEL_DIR|synthetic} [options]\n"
              "Use 'real --help' or 'synthetic --help' for source-specific options.")
        return
    source, *source_args = arguments
    if source == "real":
        measure_real(source_args)
    elif source == "synthetic":
        measure_synthetic(source_args)
    else:
        raise SystemExit(f"unknown source {source!r}; choose real or synthetic")


if __name__ == "__main__":
    main()
