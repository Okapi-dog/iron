# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the unified paper driver without running the NPU."""

import json

import torch
from safetensors.torch import save_file

from iron.operators.spmv import measure_paper
from iron.operators.spmv.matrix_preparation import MatrixInput
from iron.operators.spmv.paper_config import PAPER_TIMING_PROTOCOL_ID


def test_synthetic_profile_mode_uses_one_paper_condition(tmp_path):
    output = tmp_path / "profile.jsonl"
    measure_paper.main([
        "synthetic", "--profile-only", "--condition", "density_10",
        "--paper-seed", "1000", "--design", "ell",
        "--output-jsonl", str(output),
    ])
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["condition"] == "density_10"
    assert records[0]["seed"] == 1000
    assert records[0]["status"] == "profile_only"
    assert list(records[0]["storage_estimates"]) == ["ell"]


def test_real_driver_loads_one_weight_and_calls_shared_measurement(tmp_path, monkeypatch):
    weight = torch.zeros((4, 8), dtype=torch.bfloat16)
    weight[0, 1] = 1
    weight[2, 6] = 2
    save_file({"example.weight": weight}, tmp_path / "model.safetensors")
    output = tmp_path / "result.jsonl"
    calls = []

    monkeypatch.setattr(measure_paper.aie_utils, "set_current_device", lambda _: None)

    def fake_measure(matrix, design, windows, expected, **kwargs):
        calls.append((matrix.profile.M, matrix.profile.K, design, windows))
        return {"status": "ok", "npu_latency_us": 12.0}

    monkeypatch.setattr(measure_paper, "measure_case", fake_measure)
    measure_paper.main([
        "real", str(tmp_path), "--weight", "example.weight",
        "--design", "dense_k_tiled", "--inter-case-seconds", "0",
        "--output-jsonl", str(output),
    ])
    record = json.loads(output.read_text())
    assert calls == [(4, 8, "dense_k_tiled", 0)]
    assert record["tensor_name"] == "example.weight"
    assert record["nnz"] == 2
    assert record["requested_design"] == "dense_k_tiled"
    assert record["paper_protocol"] is False


def test_real_paper_defaults_use_five_warmups_without_case_sleep(tmp_path, monkeypatch):
    weight = torch.zeros((4, 8), dtype=torch.bfloat16)
    weight[0, 1] = 1
    save_file({"example.weight": weight}, tmp_path / "model.safetensors")
    output = tmp_path / "result.jsonl"
    calls = []
    monkeypatch.setattr(measure_paper, "PAPER_WEIGHTS", ("example.weight",))
    monkeypatch.setattr(measure_paper.aie_utils, "set_current_device", lambda _: None)

    def fake_measure(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "ok", "npu_latency_us": 12.0,
                "timed_samples_us": [12.0] * 5}

    monkeypatch.setattr(measure_paper, "measure_case", fake_measure)
    measure_paper.main([
        "real", str(tmp_path), "--design", "dense_k_tiled",
        "--output-jsonl", str(output),
    ])
    record = json.loads(output.read_text())
    assert record["paper_protocol"] is True
    assert record["timing_protocol_id"] == PAPER_TIMING_PROTOCOL_ID
    assert record["inter_case_seconds"] == 0.0
    assert calls[0]["warmup_iters"] == 5
    assert calls[0]["timed_iters"] == 5
    assert calls[0]["idle_s"] == 0.0


def test_real_driver_retries_a_failed_case_without_overwriting_it(tmp_path, monkeypatch):
    weight = torch.zeros((4, 8), dtype=torch.bfloat16)
    weight[0, 1] = 1
    save_file({"example.weight": weight}, tmp_path / "model.safetensors")
    output = tmp_path / "result.jsonl"
    matrix_id = MatrixInput(
        "safetensors", model_dir=str(tmp_path.resolve()),
        tensor_name="example.weight", x_seed=3000,
    ).matrix_id
    failed = {
        "matrix_id": matrix_id, "requested_design": "ell", "windows": 0,
        "timing_protocol_id": PAPER_TIMING_PROTOCOL_ID, "run_order": 7,
        "status": "failed", "error": "old shape limit",
    }
    output.write_text(json.dumps(failed) + "\n")
    monkeypatch.setattr(measure_paper.aie_utils, "set_current_device", lambda _: None)
    monkeypatch.setattr(measure_paper, "measure_case", lambda *args, **kwargs: {
        "status": "ok", "npu_latency_us": 12.0,
    })

    measure_paper.main([
        "real", str(tmp_path), "--weight", "example.weight",
        "--design", "ell", "--inter-case-seconds", "0",
        "--output-jsonl", str(output),
    ])
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0] == failed
    assert len(records) == 2
    assert records[1]["status"] == "ok"
    assert records[1]["run_order"] == 8
    assert records[1]["estimated_ell_npu_rows"] == 1024
