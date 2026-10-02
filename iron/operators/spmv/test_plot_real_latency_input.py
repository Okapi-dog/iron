# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the publication latency/input-data figure."""

import json
import tempfile
import unittest
from pathlib import Path

from iron.operators.spmv.paper_config import PAPER_DESIGNS, PAPER_WEIGHTS
from iron.operators.spmv.plot_real_latency_input import (
    Measurement, metric_values, read_measurements,
)


class RealLatencyInputTests(unittest.TestCase):
    def test_row_map_inside_control_is_counted_once(self):
        item = Measurement("weight", "sell_dedicated_reorder", 4096, 4096, 0.88,
                           200.0, 10_000, 2_000, 400)
        self.assertEqual(item.input_bytes, 12_000)

    def test_reader_matches_names_instead_of_jsonl_position(self):
        records = []
        for weight in PAPER_WEIGHTS:
            for index, design in enumerate(PAPER_DESIGNS):
                records.append({
                    "tensor_name": weight, "requested_design": design,
                    "status": "ok", "timing_protocol_id": "w5_t5_idle0s_between0s",
                    "timed_samples_us": [100.0] * 5, "npu_latency_us": 100.0 + index,
                    "M": 4096, "K": 4096, "actual_row_cv": 0.4,
                    "packed_a_bytes": 1_000_000 + 100_000 * index,
                    "control_bytes": 2_000 if index == 3 else 0,
                    "row_indices_bytes": 400 if index == 3 else 0,
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in reversed(records)))
            items = read_measurements(path)
        latency, input_data = metric_values(items, "absolute")
        self.assertEqual(latency[0], [100.0, 101.0, 102.0, 103.0])
        self.assertEqual(input_data[0], [1.0, 1.1, 1.2, 1.302])
        speedup, reduction = metric_values(items, "normalized")
        self.assertEqual(speedup[0][0], 1.0)
        self.assertEqual(reduction[0][0], 1.0)


if __name__ == "__main__":
    unittest.main()
