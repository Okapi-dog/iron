# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the unified paper driver without running the NPU."""

import json

import torch
from safetensors.torch import save_file

from iron.operators.spmv import measure_paper


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
