# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the paper's effective-bandwidth figures."""

import json
import tempfile
import unittest
from pathlib import Path

from iron.operators.spmv.paper_config import PAPER_TIMING_PROTOCOL_ID
from iron.operators.spmv.plot_paper_bandwidth import (
    BandwidthCase, npu_io_bytes, read_cases, sweep_statistics,
)


class PaperBandwidthTests(unittest.TestCase):
    def test_io_payload_does_not_double_count_embedded_x_or_row_map(self):
        record = {"packed_a_bytes": 10_000, "control_bytes": 2_000,
                  "x_transfer_bytes": 1_000, "y_transfer_bytes": 400}
        for design, expected in (("dense_k_tiled", 11_400), ("ell", 11_400),
                                 ("slice_ell", 12_400),
                                 ("sell_dedicated_reorder", 12_400)):
            self.assertEqual(npu_io_bytes({**record, "requested_design": design}), expected)

    def test_reader_rejects_bandwidth_inconsistent_with_buffer_sizes(self):
        record = {
            "tensor_name": "model.layers.5.self_attn.o_proj.weight",
            "requested_design": "sell_dedicated_reorder",
            "timing_protocol_id": PAPER_TIMING_PROTOCOL_ID,
            "status": "ok", "timed_samples_us": [200.0] * 5,
            "npu_latency_us": 200.0, "M": 4096, "K": 4096,
            "actual_row_cv": 0.13, "packed_a_bytes": 1_000_000,
            "control_bytes": 20_000, "x_transfer_bytes": 10_000,
            "y_transfer_bytes": 2_000, "effective_bandwidth_gbps": 5.11032,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.jsonl"
            path.write_text(json.dumps(record) + "\n")
            cases = read_cases(path, "real")
            self.assertEqual(cases[0].io_bytes, 1_022_064)
            record["effective_bandwidth_gbps"] = 5.06
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "disagree"):
                read_cases(path, "real")

    def test_sweep_mean_uses_ten_case_bandwidths(self):
        cases = [BandwidthCase("synthetic", f"density_10/{seed}", "ell",
                               4096, 4096, 0.44, 200.0, 1_000_000,
                               float(seed - 1000))
                 for seed in range(1000, 1010)]
        mean, standard_deviation = sweep_statistics(cases, "density_10", "ell")
        self.assertEqual(mean, 4.5)
        self.assertAlmostEqual(standard_deviation, 3.027650354)


if __name__ == "__main__":
    unittest.main()
