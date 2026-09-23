# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Window partitioning by rows, NNZ, or estimated packed block work."""

from __future__ import annotations

import numpy as np


def compact_window_output(window_major_output, valid_ranges):
    """Remove per-window tail padding and restore contiguous original row order.

    This host helper specifies the output contract for tests/profiling only;
    a deployment that consumes the next layer on NPU needs an on-device compact
    path or must preserve the segmented-vector descriptor through that layer.
    """
    import torch

    if not isinstance(window_major_output, torch.Tensor):
        raise TypeError("window_major_output must be a torch.Tensor")
    if window_major_output.numel() % len(valid_ranges):
        raise ValueError("output length must be divisible by the number of windows")
    rows_per_window = window_major_output.numel() // len(valid_ranges)
    pieces = []
    for window, (start, end) in enumerate(valid_ranges):
        count = int(end) - int(start)
        if count < 0 or count > rows_per_window:
            raise ValueError("valid row range exceeds its fixed output window")
        base = window * rows_per_window
        pieces.append(window_major_output[base:base + count])
    return torch.cat(pieces) if pieces else window_major_output[:0]


def slice_nnz(row_nnz: np.ndarray, block_height: int) -> np.ndarray:
    """Sum row NNZ for each contiguous row group of one B_h slice."""
    counts = np.asarray(row_nnz, dtype=np.int64)
    return np.add.reduceat(
        counts, np.arange(0, counts.size, block_height, dtype=np.int64)
    )


def exact_window_blocks(row_nnz: np.ndarray, first_slice: int, last_slice: int,
                        block_height: int, block_width: int) -> int:
    """Estimate packed blocks after descending-NNZ sort within one window.

    Each slice needs ceil(max_row_nnz / B_w) A blocks.  Sorting the rows
    within the window gives the exact block count for this row-length model.
    """
    first_row = first_slice * block_height
    last_row = min(last_slice * block_height, len(row_nnz))
    if last_row <= first_row:
        return 0
    sorted_rows = np.sort(np.asarray(row_nnz[first_row:last_row], dtype=np.int64))[::-1]
    slice_maxima = sorted_rows[::block_height]
    return int(((slice_maxima + block_width - 1) // block_width).sum())


def _greedy_cuts(metric: np.ndarray, windows: int) -> list[int]:
    """Choose nonempty contiguous groups near equal cumulative metric."""
    total_slices = int(metric.size)
    if windows > total_slices:
        raise ValueError("every window needs at least one row slice")
    cumulative = np.cumsum(metric, dtype=np.int64)
    cuts = [0]
    previous = 0
    for window in range(1, windows):
        target = cumulative[-1] * window / windows
        candidate = int(np.searchsorted(cumulative, target, side="left")) + 1
        candidate = max(previous + 1, min(candidate, total_slices - (windows - window)))
        cuts.append(candidate)
        previous = candidate
    cuts.append(total_slices)
    return cuts


def _balance_block_cuts(row_nnz: np.ndarray, cuts: list[int], windows: int,
                        block_height: int, block_width: int) -> list[int]:
    """Locally refine contiguous cuts to reduce the largest predicted A load."""
    total_slices = len(slice_nnz(row_nnz, block_height))
    radius = max(1, int(np.ceil(total_slices / (2 * windows))))

    def score(candidate_cuts: list[int]) -> tuple[int, int]:
        loads = [exact_window_blocks(row_nnz, a, b, block_height, block_width)
                 for a, b in zip(candidate_cuts[:-1], candidate_cuts[1:])]
        # Minimize the slowest column, then the total squared imbalance.
        return max(loads), sum((load * windows - sum(loads)) ** 2 for load in loads)

    result = list(cuts)
    for _ in range(4):
        changed = False
        for boundary in range(1, windows):
            low = max(result[boundary - 1] + 1, result[boundary] - radius)
            high = min(result[boundary + 1] - 1, result[boundary] + radius)
            best_cut, best_score = result[boundary], score(result)
            for candidate in range(low, high + 1):
                trial = list(result)
                trial[boundary] = candidate
                trial_score = score(trial)
                if trial_score < best_score:
                    best_cut, best_score = candidate, trial_score
            if best_cut != result[boundary]:
                result[boundary] = best_cut
                changed = True
        if not changed:
            break
    return result


def partition_row_slices(row_nnz: np.ndarray, block_height: int, block_width: int,
                         windows: int, policy: str) -> list[int]:
    """Return contiguous slice-index cuts, including 0 and the final endpoint.

    ``equal_nnz`` balances the input nonzero count. ``balanced_blocks`` first
    targets each original slice's padded-block estimate, then locally refines
    boundaries using the exact post-sort packed-block cost per window.
    """
    counts = np.asarray(row_nnz, dtype=np.int64)
    per_slice_nnz = slice_nnz(counts, block_height)
    total_slices = int(per_slice_nnz.size)
    if windows <= 0 or windows > total_slices:
        raise ValueError("windows must be between one and the number of row slices")
    if policy == "equal_rows":
        # Match the existing NPU packer: every physical window reserves
        # ceil(total_slices / windows) slices and only the final one is short.
        # If that would leave an entirely empty window, distribute slices
        # nearly evenly so every active NPU column still receives A work.
        reserved = (total_slices + windows - 1) // windows
        cuts = [min(window * reserved, total_slices) for window in range(windows)]
        cuts.append(total_slices)
        if len(set(cuts)) == windows + 1:
            return cuts
        sizes = np.full(windows, total_slices // windows, dtype=np.int64)
        sizes[: total_slices % windows] += 1
        return [0, *sizes.cumsum().tolist()]
    if policy == "equal_nnz":
        return _greedy_cuts(per_slice_nnz, windows)
    if policy == "balanced_blocks":
        per_slice_blocks = (np.maximum.reduceat(
            counts, np.arange(0, counts.size, block_height, dtype=np.int64)
        ) + block_width - 1) // block_width
        initial = _greedy_cuts(per_slice_blocks, windows)
        return _balance_block_cuts(counts, initial, windows, block_height, block_width)
    raise ValueError(f"unknown partition policy: {policy}")


def window_profile(row_nnz: np.ndarray, block_height: int, block_width: int,
                   windows: int, policy: str, K: int) -> dict:
    """Predict fixed-length NPU payloads and per-column packed work."""
    counts = np.asarray(row_nnz, dtype=np.int64)
    cuts = partition_row_slices(counts, block_height, block_width, windows, policy)
    window_slices = np.diff(cuts)
    slices_per_window = int(window_slices.max())
    rows_per_window = slices_per_window * block_height
    blocks = [exact_window_blocks(counts, int(a), int(b), block_height, block_width)
              for a, b in zip(cuts[:-1], cuts[1:])]
    total_blocks = sum(blocks)
    padded_rows = windows * rows_per_window
    a_bytes = total_blocks * block_height * block_width * 4
    row_map_bytes = padded_rows * 2
    config_words = 2 + K + slices_per_window
    config_words += config_words % 2
    control_bytes = windows * (config_words + rows_per_window) * 2
    max_blocks = max(blocks)
    return {
        "policy": policy,
        "slice_boundaries": cuts,
        "slices_per_window": window_slices.tolist(),
        "padded_slices_per_window": slices_per_window,
        "valid_rows_per_window": (window_slices * block_height).tolist(),
        "padded_rows_per_window": rows_per_window,
        "padded_rows": padded_rows,
        "blocks_per_window": blocks,
        "total_blocks": total_blocks,
        "block_imbalance_max_over_mean": max_blocks / (total_blocks / windows),
        "packed_a_bytes": a_bytes,
        "row_map_bytes": row_map_bytes,
        "control_bytes": control_bytes,
        "output_bytes": padded_rows * 2,
        "storage_bytes": a_bytes + row_map_bytes,
    }


def pad_csr_windows(indptr: np.ndarray, indices: np.ndarray, values: np.ndarray,
                    block_height: int, block_width: int, windows: int, policy: str,
                    K: int):
    """Insert empty rows between variable contiguous windows for fixed-size FIFOs.

    Returns the SELL pack, padded per-window source ranges, and profile.  The
    CSR payload itself remains in canonical row order; no row data is copied.
    """
    from iron.operators.spmv.slice_ell import SliceELLConfig, csr_to_slice_ell

    row_nnz = np.diff(np.asarray(indptr, dtype=np.int64))
    profile = window_profile(row_nnz, block_height, block_width, windows, policy, K)
    padded_rows = int(profile["padded_rows_per_window"])
    new_lengths = np.zeros(windows * padded_rows, dtype=np.int64)
    valid_ranges = []
    for window, (first, last) in enumerate(zip(
            profile["slice_boundaries"][:-1], profile["slice_boundaries"][1:])):
        row_start = int(first) * block_height
        row_end = min(int(last) * block_height, row_nnz.size)
        valid_ranges.append((row_start, row_end))
        count = row_end - row_start
        base = window * padded_rows
        new_lengths[base:base + count] = row_nnz[row_start:row_end]
    virtual_indptr = np.empty(new_lengths.size + 1, dtype=np.int64)
    virtual_indptr[0] = 0
    np.cumsum(new_lengths, out=virtual_indptr[1:])
    core_rows = 3 if block_height % 3 == 0 else 4
    packed = csr_to_slice_ell(
        virtual_indptr, indices, values, K=K,
        config=SliceELLConfig(
            core_rows=core_rows, block_height=block_height, block_width=block_width,
            shim_columns=windows, window_count=windows,
        ),
    )
    return packed, valid_ranges, profile
