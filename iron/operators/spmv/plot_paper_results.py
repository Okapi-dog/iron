#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize paper JSONL and draw paired speedup/capacity figures.

One plotted synthetic point is one matrix seed, not one of its five dispatches.
Run with an environment containing matplotlib (the ELSA venv on ws007 does).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iron.operators.spmv.paper_config import (
    PAPER_CVS, PAPER_DENSITIES, PAPER_DESIGNS, PAPER_TIMING_PROTOCOL_ID,
    PAPER_WEIGHTS,
)


LABELS = {
    "dense_k_tiled": "Dense", "ell": "ELL",
    "slice_ell": "Blocked Slice-ELL",
    "sell_dedicated_reorder": "Blocked SELL-C-σ",
}
COLORS = {
    "dense_k_tiled": "#505050", "ell": "#d08721",
    "slice_ell": "#2374ab", "sell_dedicated_reorder": "#a64b80",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_timing_health(records: list[dict], allow_isolated_spike: bool = False) -> int:
    """Reject periodic spikes; optionally disclose one isolated timing anomaly."""
    protocols = {record.get("timing_protocol_id") for record in records
                 if record.get("timed_samples_us")}
    if len(protocols) > 1:
        raise ValueError(f"mixed timing protocols in one figure input: {protocols}")
    unhealthy = []
    for record in records:
        samples = record.get("timed_samples_us", [])
        if len(samples) == 5 and max(samples) > 2 * float(np.median(samples)):
            unhealthy.append((record.get("matrix_id"), record.get("requested_design")))
    if unhealthy and not (allow_isolated_spike and len(unhealthy) == 1):
        raise ValueError(
            f"{len(unhealthy)} timing sets have >2x spikes; "
            "investigate/re-measure before drawing publication figures. "
            f"First: {unhealthy[0]}"
        )
    return len(unhealthy)


def apply_corrections(records: list[dict], corrections: list[dict]) -> tuple[list[dict], list[dict]]:
    """Overlay independently saved remeasurements without changing the original JSONL."""
    updated = list(records)
    positions = {(r.get("matrix_id"), r.get("requested_design")): i
                 for i, r in enumerate(records)}
    if len(positions) != len(records):
        raise ValueError("duplicate matrix/design keys in original measurement")
    seen = set()
    audit = []
    for correction in corrections:
        key = (correction.get("matrix_id"), correction.get("requested_design"))
        if key in seen or key not in positions:
            raise ValueError(f"duplicate or unknown correction key: {key}")
        seen.add(key)
        original = records[positions[key]]
        for field in ("csr_sha256", "x_sha256", "row_nnz_sha256",
                      "timing_protocol_id", "format_id"):
            if correction.get(field) != original.get(field):
                raise ValueError(f"correction mismatch in {field}: {key}")
        old_samples = original.get("timed_samples_us", [])
        new_samples = correction.get("timed_samples_us", [])
        if (len(old_samples) != 5 or max(old_samples) <= 2 * float(np.median(old_samples))
                or len(new_samples) != 5 or max(new_samples) > 2 * float(np.median(new_samples))
                or correction.get("status") not in ("ok", "ok_with_tolerance_exceptions")):
            raise ValueError(f"correction is not a healthy rerun of a >2x spike: {key}")
        updated[positions[key]] = correction
        audit.append({"matrix_id": key[0], "design": key[1],
                      "condition": original.get("condition"), "seed": original.get("seed"),
                      "original_samples_us": old_samples, "rerun_samples_us": new_samples,
                      "original_mean_us": original.get("npu_latency_us"),
                      "rerun_mean_us": correction.get("npu_latency_us"),
                      "original_measured_at_utc": original.get("measured_at_utc"),
                      "rerun_measured_at_utc": correction.get("measured_at_utc")})
    return updated, audit


def check_completeness(records: list[dict], expected_matrices: int) -> None:
    """Prevent partial long-running JSONL files from becoming final figures."""
    if not records:
        return
    keys = {(record.get("matrix_id"), record.get("requested_design"))
            for record in records if record.get("paper_protocol")}
    matrices = {matrix_id for matrix_id, _ in keys}
    expected_cases = expected_matrices * len(PAPER_DESIGNS)
    if len(matrices) != expected_matrices or len(keys) != expected_cases:
        raise ValueError(
            f"incomplete paper input: {len(matrices)}/{expected_matrices} matrices, "
            f"{len(keys)}/{expected_cases} matrix/design cases"
        )


def paired_rows(records: list[dict]) -> list[dict]:
    """Pair all designs by the exact CSR recipe, not by nominal CV/density."""
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        if (record.get("paper_protocol")
                and record.get("timing_protocol_id") == PAPER_TIMING_PROTOCOL_ID
                and "requested_design" in record):
            groups[record["matrix_id"]][record["requested_design"]] = record
    rows = []
    for group in groups.values():
        dense = group.get("dense_k_tiled", {})
        dense_time = dense.get("npu_latency_us") if dense.get("status") in (
            "ok", "ok_with_tolerance_exceptions",
        ) and len(dense.get("timed_samples_us", [])) == 5 else None
        for design in PAPER_DESIGNS:
            record = group.get(design)
            if record is None:
                continue
            valid = (record.get("status") in ("ok", "ok_with_tolerance_exceptions")
                     and len(record.get("timed_samples_us", [])) == 5)
            latency = record.get("npu_latency_us") if valid else None
            samples = record.get("timed_samples_us", [])
            spike = bool(samples and max(samples) > 2 * float(np.median(samples)))
            # The CPU storage model remains visible when an NPU build fails.
            capacity = (record.get("storage_over_dense") if valid else
                        record.get("estimated_storage_over_dense") if design == "ell" else None)
            rows.append({
                "matrix_id": record["matrix_id"],
                "condition": record.get("condition", "real"),
                "seed": record.get("seed", ""),
                "tensor_name": record.get("tensor_name", ""),
                "M": record.get("M"), "K": record.get("K"),
                "ell_npu_rows": record.get("ell_npu_rows", ""),
                "ell_padding_rows": record.get("ell_padding_rows", ""),
                "target_density": record.get("target_density", ""),
                "actual_density": record.get("actual_density", record.get("density", "")),
                "target_row_cv": record.get("target_row_cv", ""),
                "actual_row_cv": record.get("actual_row_cv", ""),
                "design": design, "status": record.get("status", "missing"),
                "timed_samples_us": json.dumps(record.get("timed_samples_us", [])),
                "timing_spike_over_2x": spike,
                "latency_us": latency, "dense_latency_us": dense_time,
                "speedup_over_dense": dense_time / latency if dense_time and latency else None,
                "storage_over_dense": capacity,
                "capacity_source": "NPU packed" if valid else
                    "CPU exact ELL model" if capacity is not None else "unavailable",
                "packed_a_bytes": record.get("packed_a_bytes", ""),
                "format_metadata_bytes": record.get("format_metadata_bytes", ""),
                "x_transfer_bytes": record.get("x_transfer_bytes", ""),
                "y_transfer_bytes": record.get("y_transfer_bytes", ""),
                "error_type": record.get("error_type", ""),
                "error": record.get("error", ""),
            })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], title: str, path: Path) -> None:
    """Expose valid seed counts and failures next to each plotted condition."""
    if not rows:
        return
    lines = [f"# {title}", "", "One point = one matrix's five-call mean. "
             "N/A is never inferred from format capacity.", "",
             "| Condition | Design | Valid / attempted | Median speedup | Median storage/Dense | Achieved CV range |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    for condition in conditions:
        for design in PAPER_DESIGNS:
            subset = [row for row in rows if row["condition"] == condition
                      and row["design"] == design]
            if not subset:
                continue
            good = [row for row in subset if row["latency_us"] is not None]
            ratios = [row["speedup_over_dense"] for row in good
                      if row["speedup_over_dense"] is not None]
            capacities = [row["storage_over_dense"] for row in subset
                          if row["storage_over_dense"] is not None]
            cvs = [float(row["actual_row_cv"]) for row in subset
                   if row["actual_row_cv"] != ""]
            speed = f"{np.median(ratios):.3f}×" if ratios else "N/A"
            capacity = f"{np.median(capacities):.3f}" if capacities else "N/A"
            cv_range = f"{min(cvs):.3f}–{max(cvs):.3f}" if cvs else "N/A"
            lines.append(f"| {condition} | {LABELS[design]} | {len(good)}/{len(subset)} "
                         f"| {speed} | {capacity} | {cv_range} |")
    failures = [row for row in rows if row["status"] not in
                ("ok", "ok_with_tolerance_exceptions")]
    spikes = [row for row in rows if row["timing_spike_over_2x"]]
    if spikes:
        lines += ["", "## Timing anomalies retained in five-call means", "",
                  "| Condition / weight | Seed | Design | Five samples (µs) |",
                  "| --- | ---: | --- | --- |"]
        for row in spikes:
            lines.append(f"| {row['condition']} / {row['tensor_name']} | {row['seed']} "
                         f"| {LABELS[row['design']]} | `{row['timed_samples_us']}` |")
    if failures:
        lines += ["", "## Failed combinations", "",
                  "| Condition / weight | Seed | Design | Error |",
                  "| --- | ---: | --- | --- |"]
        for row in failures:
            detail = (str(row["error_type"]) + ": " + str(row["error"])).replace("|", "/")
            lines.append(f"| {row['condition']} / {row['tensor_name']} | {row['seed']} "
                         f"| {LABELS[row['design']]} | {detail[:160]} |")
    if title == "Five actual weights":
        lines += ["", "## SELL-C-σ compared directly with Slice-ELL", "",
                  "| Weight | Slice latency / SELL latency | Slice storage / SELL storage |",
                  "| --- | ---: | ---: |"]
        for name in PAPER_WEIGHTS:
            subset = {row["design"]: row for row in rows
                      if row["tensor_name"] == name}
            slice_row = subset.get("slice_ell", {})
            sell_row = subset.get("sell_dedicated_reorder", {})
            if slice_row.get("latency_us") and sell_row.get("latency_us"):
                latency_ratio = slice_row["latency_us"] / sell_row["latency_us"]
                capacity_ratio = (slice_row["storage_over_dense"]
                                  / sell_row["storage_over_dense"])
                lines.append(f"| {name} | {latency_ratio:.3f}× | {capacity_ratio:.3f}× |")
    path.write_text("\n".join(lines) + "\n")


def write_correction_audit(audit: list[dict], path: Path) -> None:
    lines = ["# Timing correction audit", "", "The original raw JSONL is unchanged.", ""]
    if audit:
        lines += ["Each pre-identified >2×-median timing set below was measured once "
                  "again with the same matrix, design and timing protocol. Only the "
                  "derived CSV and figures use the rerun.", ""]
    else:
        lines += ["No timing correction applied.", ""]
    for entry in audit:
        lines += [f"- Condition / seed / design: `{entry['condition']}` / "
                  f"`{entry['seed']}` / `{entry['design']}`",
                  f"- Matrix ID: `{entry['matrix_id']}`",
                  f"- Original (`{entry['original_measured_at_utc']}`): "
                  f"`{entry['original_samples_us']}` µs; mean "
                  f"{entry['original_mean_us']:.3f} µs",
                  f"- One rerun (`{entry['rerun_measured_at_utc']}`): "
                  f"`{entry['rerun_samples_us']}` µs; mean "
                  f"{entry['rerun_mean_us']:.3f} µs", ""]
    path.write_text("\n".join(lines))


def save_figure(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")


def real_figure(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(PAPER_WEIGHTS)
    fig, (speed_ax, capacity_ax) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    x = np.arange(len(names))
    width = 0.20
    for index, design in enumerate(PAPER_DESIGNS):
        subset = {row["tensor_name"]: row for row in rows if row["design"] == design}
        speed = [subset.get(name, {}).get("speedup_over_dense") or np.nan for name in names]
        capacity = [subset.get(name, {}).get("storage_over_dense") or np.nan for name in names]
        offset = (index - 1.5) * width
        speed_ax.bar(x + offset, speed, width, label=LABELS[design], color=COLORS[design])
        capacity_ax.bar(x + offset, capacity, width, color=COLORS[design])
        for position, name in enumerate(names):
            row = subset.get(name, {})
            if row.get("latency_us") is None:
                speed_ax.text(x[position] + offset, 0.12, "N/A", rotation=90,
                              ha="center", va="bottom", fontsize=8)
            if row.get("capacity_source") == "CPU exact ELL model":
                capacity_ax.text(x[position] + offset, capacity[position] + 0.025,
                                 "est.", ha="center", va="bottom", fontsize=8)
    speed_ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    capacity_ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    speed_ax.set_ylabel("Dense latency / method latency")
    capacity_ax.set_ylabel("Stored format bytes / Dense BF16 A")
    labels = []
    for name in names:
        record = next((row for row in rows if row["tensor_name"] == name), None)
        short = name.replace("model.layers.", "L").replace(".mlp.", " ")
        short = short.replace(".self_attn.", " ").replace(".weight", "")
        if record:
            labels.append(f"{short}\n{record['M']}×{record['K']}; CV={record['actual_row_cv']:.2f}")
        else:
            labels.append(short)
    capacity_ax.set_xticks(x, labels, rotation=15, ha="right")
    speed_ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.suptitle("Five actual Llama-2-7B pruned weights (CV quantiles)")
    fig.tight_layout()
    save_figure(fig, output / "real_five_speedup_capacity")
    plt.close(fig)


def sweep_figure(rows: list[dict], conditions: list[str], labels: list[str],
                 title: str, stem: str, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.1, 1.1]})
    x = np.arange(len(conditions))
    designs = PAPER_DESIGNS[1:]
    offsets = np.linspace(-0.25, 0.25, len(designs))
    for design, offset in zip(designs, offsets):
        color = COLORS[design]
        for j, condition in enumerate(conditions):
            points = [row["speedup_over_dense"] for row in rows
                      if row["condition"] == condition and row["design"] == design
                      and row["speedup_over_dense"] is not None]
            if points:
                axes[0].boxplot([points], positions=[j + offset], widths=0.16,
                                patch_artist=True, showfliers=False,
                                boxprops={"facecolor": color, "alpha": 0.55},
                                medianprops={"color": "black"})
                jitter = np.linspace(-0.045, 0.045, len(points))
                axes[0].scatter(j + offset + jitter, points, color=color, s=10, alpha=0.8)
        for panel, field in ((axes[1], "latency_us"), (axes[2], "storage_over_dense")):
            medians = [np.median([row[field] for row in rows
                                  if row["condition"] == condition
                                  and row["design"] == design and row[field] is not None])
                       if any(row["condition"] == condition and row["design"] == design
                              and row[field] is not None for row in rows) else np.nan
                       for condition in conditions]
            panel.plot(x + offset, medians, marker="o", markersize=3,
                       label=LABELS[design], color=color)
    dense_times = [np.median([row["latency_us"] for row in rows
                              if row["condition"] == condition
                              and row["design"] == "dense_k_tiled"
                              and row["latency_us"] is not None])
                   if any(row["condition"] == condition and row["design"] == "dense_k_tiled"
                          and row["latency_us"] is not None for row in rows) else np.nan
                   for condition in conditions]
    axes[0].axhline(1, color=COLORS["dense_k_tiled"], linestyle="--",
                    label="Dense baseline")
    axes[1].plot(x, dense_times, marker="o", color=COLORS["dense_k_tiled"],
                 label="Dense")
    axes[2].axhline(1, color=COLORS["dense_k_tiled"], linestyle="--")
    axes[0].set_ylabel("Paired speedup over Dense\n(one point / matrix seed)")
    axes[1].set_ylabel("Median latency (µs)")
    axes[2].set_ylabel("Median storage / Dense")
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel(title)
    handles = [Line2D([0], [0], color=COLORS["dense_k_tiled"], linestyle="--",
                      label="Dense baseline")]
    handles += [Patch(facecolor=COLORS[design], label=LABELS[design])
                for design in designs]
    axes[0].legend(handles=handles, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    save_figure(fig, output / stem)
    plt.close(fig)


def scatter_figure(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"ell": "o", "slice_ell": "s", "sell_dedicated_reorder": "^"}
    scatter = None
    for design in PAPER_DESIGNS[1:]:
        points = [(row["storage_over_dense"], row["speedup_over_dense"],
                   row["actual_density"])
                  for row in rows if row["design"] == design
                  and row["storage_over_dense"] is not None
                  and row["speedup_over_dense"] is not None]
        if points:
            values = np.array(points)
            scatter = ax.scatter(values[:, 0], values[:, 1], c=values[:, 2],
                                 cmap="viridis", vmin=0.05, vmax=0.50,
                                 marker=markers[design], label=LABELS[design],
                                 alpha=0.75, s=24)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Stored format bytes / Dense BF16 A")
    ax.set_ylabel("Paired latency speedup over Dense")
    ax.legend()
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="Nonzero fraction")
    fig.tight_layout()
    save_figure(fig, output / "synthetic_capacity_vs_speedup")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=Path(__file__).resolve().parent /
                        "paper_evaluation3/synthetic_paper_w5_t5_idle0s_between0s.jsonl")
    parser.add_argument("--real", type=Path, default=Path(__file__).resolve().parent /
                        "paper_evaluation3/real_weights_w5_t5_idle0s_between0s.jsonl")
    parser.add_argument("--synthetic-corrections", type=Path,
                        default=Path(__file__).resolve().parent /
                        "paper_evaluation3/synthetic_corrections.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent /
                        "paper_evaluation3/figures")
    parser.add_argument("--allow-partial", action="store_true",
                        help="For diagnostic previews only; not publication figures")
    parser.add_argument("--allow-isolated-spike", action="store_true",
                        help="Retain and disclose exactly one >2x sample set; no trimming")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_records = read_jsonl(args.synthetic)
    real_records = read_jsonl(args.real)
    synthetic_records, audit = apply_corrections(
        synthetic_records, read_jsonl(args.synthetic_corrections))
    write_correction_audit(audit, args.output_dir / "timing_correction_audit.md")
    synthetic_spikes = check_timing_health(synthetic_records, args.allow_isolated_spike)
    real_spikes = check_timing_health(real_records, args.allow_isolated_spike)
    if synthetic_spikes or real_spikes:
        print(f"WARNING: retaining {synthetic_spikes} synthetic and {real_spikes} real "
              "timing spike sets in five-call means; see summary.md")
    if not args.allow_partial:
        check_completeness(synthetic_records, 110)
        check_completeness(real_records, 5)
    synthetic = paired_rows(synthetic_records)
    real = paired_rows(real_records)
    write_csv(synthetic, args.output_dir / "synthetic_paired.csv")
    write_csv(real, args.output_dir / "real_paired.csv")
    write_summary(synthetic, "Synthetic sweeps", args.output_dir / "synthetic_summary.md")
    write_summary(real, "Five actual weights", args.output_dir / "real_summary.md")
    if synthetic:
        sweep_figure(synthetic, [f"density_{int(d * 100):02d}" for d in PAPER_DENSITIES],
                     [f"{d:.0%}" for d in PAPER_DENSITIES], "Nonzero fraction",
                     "synthetic_density_sweep", args.output_dir)
        cv_conditions = ["density_10" if cv == 0.44 else f"cv_{cv:.2f}"
                         for cv in PAPER_CVS]
        sweep_figure(synthetic, cv_conditions, [str(cv) for cv in PAPER_CVS],
                     "Achieved row-NNZ CV (target)", "synthetic_cv_sweep", args.output_dir)
        scatter_figure(synthetic, args.output_dir)
    if real:
        real_figure(real, args.output_dir)
    print(f"synthetic records={len(synthetic)}, real records={len(real)}; "
          f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
