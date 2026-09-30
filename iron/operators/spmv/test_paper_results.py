# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for paper condition selection and paired aggregation."""

from iron.operators.spmv.paper_config import paper_conditions
import pytest

from iron.operators.spmv.plot_paper_results import (
    apply_corrections, check_completeness, check_timing_health, paired_rows,
)


def test_paper_conditions_have_one_shared_center():
    cases = paper_conditions()
    assert len(cases) == 11
    assert len({(density, cv) for _, density, cv in cases}) == 11
    assert (0.10, 0.44) in {(density, cv) for _, density, cv in cases}


def test_aggregation_pairs_matrix_means_and_preserves_failures():
    common = {"paper_protocol": True, "matrix_id": "one", "condition": "density_10",
              "seed": 1000, "M": 4096, "K": 4096, "actual_density": 0.10,
              "actual_row_cv": 0.44, "warmup_iters": 2, "timed_iters": 5,
              "idle_seconds_before_timed": 0.0, "inter_case_seconds": 4.0}
    records = [
        {**common, "requested_design": "dense_k_tiled", "status": "ok",
         "npu_latency_us": 200.0, "timed_samples_us": [180, 200, 200, 200, 220],
         "storage_over_dense": 1.0},
        {**common, "requested_design": "slice_ell", "status": "ok",
         "npu_latency_us": 100.0, "timed_samples_us": [90, 100, 100, 100, 110],
         "storage_over_dense": 0.3},
        {**common, "requested_design": "ell", "status": "failed",
         "estimated_storage_over_dense": 1.2, "error": "L1 overflow"},
    ]
    rows = {row["design"]: row for row in paired_rows(records)}
    assert rows["slice_ell"]["speedup_over_dense"] == 2.0
    assert rows["ell"]["speedup_over_dense"] is None
    assert rows["ell"]["storage_over_dense"] == 1.2
    assert rows["ell"]["status"] == "failed"


def test_publication_figure_rejects_periodic_timing_spike():
    with pytest.raises(ValueError, match="timing sets"):
        check_timing_health([{"matrix_id": "m", "requested_design": "ell",
                              "timed_samples_us": [300, 305, 295, 310, 2900]}])
    assert check_timing_health([{"matrix_id": "m", "requested_design": "ell",
                                 "timed_samples_us": [300, 305, 295, 310, 2900]}],
                               allow_isolated_spike=True) == 1


def test_publication_figure_rejects_incomplete_run():
    with pytest.raises(ValueError, match="incomplete paper input"):
        check_completeness([{"matrix_id": "m", "requested_design": "ell",
                             "paper_protocol": True}], 1)


def test_correction_overlay_keeps_original_and_checks_same_input():
    original = {"matrix_id": "m", "requested_design": "sell_dedicated_reorder",
                "csr_sha256": "a", "x_sha256": "b", "row_nnz_sha256": "c",
                "timing_protocol_id": "w2_t5_idle0s_between4s", "format_id": "f",
                "timed_samples_us": [900, 310, 300, 305, 295], "npu_latency_us": 422.0,
                "status": "ok"}
    correction = {**original, "timed_samples_us": [300, 310, 300, 305, 295],
                  "npu_latency_us": 302.0}
    combined, audit = apply_corrections([original], [correction])
    assert combined == [correction]
    assert original["timed_samples_us"][0] == 900
    assert audit[0]["original_mean_us"] == 422.0
    with pytest.raises(ValueError, match="correction mismatch"):
        apply_corrections([original], [{**correction, "csr_sha256": "other"}])
