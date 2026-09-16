# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


def make_uniform_ell(M: int, K: int, ell_width: int, seed: int = 0):
    """Make row-major ELL words: all uint16 indices, then all BF16 values per row."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, K, size=(M, ell_width), dtype=np.uint16)
    values = torch.from_numpy(rng.uniform(-1, 1, size=(M, ell_width)).astype(np.float32)).to(torch.bfloat16)
    words = np.empty((M, 2, ell_width), dtype=np.uint16)
    words[:, 0, :] = indices
    words[:, 1, :] = values.view(torch.uint16).numpy()
    return torch.from_numpy(words.reshape(-1)).view(torch.int16).view(torch.bfloat16)


def reference_ell(packed: torch.Tensor, vector: torch.Tensor, M: int, ell_width: int):
    words = packed.contiguous().view(torch.uint16).view(M, 2, ell_width)
    indices = words[:, 0, :].to(torch.long)
    values = words[:, 1, :].view(torch.int16).view(torch.bfloat16).float()
    return (values * vector.float()[indices]).sum(dim=1).to(torch.bfloat16)
