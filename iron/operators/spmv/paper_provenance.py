# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only software provenance attached to each paper measurement."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import cache
from importlib import metadata
from pathlib import Path


def _git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        timeout=3, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@cache
def paper_provenance() -> dict:
    """Fingerprint code/toolchain; unavailable package versions stay explicit."""
    root = Path(__file__).resolve().parents[3]
    versions = {}
    for package in ("torch", "numpy", "mlir-aie", "safetensors"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    git_status = _git_value(root, "status", "--porcelain")
    return {
        "host": platform.node(), "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "git_branch": _git_value(root, "branch", "--show-current"),
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_status) if git_status is not None else None,
        "package_versions": versions,
        "NPU_RUNTIME": os.environ.get("NPU_RUNTIME"),
        "XILINX_XRT": os.environ.get("XILINX_XRT"),
    }
