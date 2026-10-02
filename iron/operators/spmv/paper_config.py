# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed, CPU-readable choices for the paper's main SpMV comparison."""

PAPER_WEIGHTS = (
    "model.layers.19.mlp.down_proj.weight",
    "model.layers.5.self_attn.o_proj.weight",
    "model.layers.11.mlp.up_proj.weight",
    "model.layers.18.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
)
PAPER_DESIGNS = ("dense_k_tiled", "ell", "slice_ell", "sell_dedicated_reorder")
PAPER_DENSITIES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
PAPER_CVS = (0.05, 0.13, 0.44, 0.88, 1.06, 1.89)
PAPER_WARMUP_ITERS = 5
PAPER_TIMED_ITERS = 5
PAPER_IDLE_SECONDS = 0.0
PAPER_INTER_CASE_SECONDS = 0.0


def paper_conditions() -> list[tuple[str, float, float]]:
    """Return 11 points, measuring the density/CV intersection only once."""
    density_sweep = [(f"density_{int(d * 100):02d}", d, 0.44)
                     for d in PAPER_DENSITIES]
    cv_sweep = [(f"cv_{cv:.2f}", 0.10, cv)
                for cv in PAPER_CVS if cv != 0.44]
    return density_sweep + cv_sweep


def timing_protocol_id(warmups: int, timed: int, idle: float,
                       inter_case: float) -> str:
    """Keep resumed runs and figure inputs from mixing timing schedules."""
    return (f"w{warmups}_t{timed}_idle{idle:g}s_"
            f"between{inter_case:g}s")


PAPER_TIMING_PROTOCOL_ID = timing_protocol_id(
    PAPER_WARMUP_ITERS, PAPER_TIMED_ITERS,
    PAPER_IDLE_SECONDS, PAPER_INTER_CASE_SECONDS,
)
