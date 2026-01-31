# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path
import time
from operators.common.aie_device_manager import pyxrt

from operators.common import (
    AIEOperatorBase,
    AIEOperatorConstraintError,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)
from operators.common.utils import torch_to_numpy


class AIESPMV(AIEOperatorBase):
    """AIE-accelerated General Matrix-Vector/Vector-Matrix Multiplication layer"""

    def __init__(
        self,
        M,
        K,
        ell_width,
        tile_size=1,
        num_core_rows=1,
        num_core_cols=1,
        design_name="sell32_block", # "ell" or "sell32" or "sell32_block"
        is_mv=True,
        use_static_weight=False,
        context=None,
        trace_ddr_id=None,
        trace_size=8192,
    ):

        self.M = M  # matrix rows  (if is_mv=False, matrix columns)
        self.K = K  # matrix columns, vector rows  (if is_mv=False, matrix rows, vector columns)
        self.ell_width = ell_width
        self.tile_size = tile_size
        self.num_core_rows = num_core_rows
        self.num_core_cols = num_core_cols
        self.design_name = design_name
        self.is_mv = is_mv
        self.trace_ddr_id = trace_ddr_id
        self.trace_size = trace_size
        if use_static_weight:
            self.weight = torch.zeros(
                (M, K) if is_mv else (K, M), dtype=torch.bfloat16
            ).T  # weights are stored col-major/transposed
        else:
            self.weight = None

        # For compatibility with my_matvec parameters
        self.m = self.tile_size

        # Artifacts created by set_up_artifacts()
        self.xclbin_artifact = None
        self.insts_artifact = None

        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="spmv_"):
        trace_suffix = f"_traceddr{self.trace_ddr_id}" if self.trace_ddr_id is not None else ""
        operator_dir = Path(__file__).parent
        file_name_base = (
            f"{prefix}{self.M}x{self.K}_ellwidth{self.ell_width}_tile{self.tile_size}_core{self.num_core_rows}x{self.num_core_cols}{trace_suffix}"
        )

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / f"design_{self.design_name}.py",
            callback_fn="my_matvec",
            callback_args=[
                self.context.device_manager.device_type,
                self.M,
                self.K,
                self.ell_width,
                self.tile_size,
                self.num_core_rows,
                self.num_core_cols,
                self.trace_ddr_id,
                self.trace_size,
            ],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    f"mv.o",
                    depends=[
                        SourceArtifact.new(
                            operator_dir / "mv.cc"
                        )
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_base}.bin", depends=[mlir_artifact]
        )

        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        # If this operator is only used as a sub-operator in another operator that sets it up, we should skip the setup here as those artifacts and buffers may not be needed.
        # Compilation Artifacts
        # ---
        xclbin_artifact, insts_artifact = self.get_artifacts()

        self.xclbin_artifact = xclbin_artifact
        self.insts_artifact = insts_artifact

        artifacts = [xclbin_artifact, insts_artifact]
        self.add_artifacts(artifacts)

    def set_up_runtime(self):
        # If this operator is only used as a sub-operator in another operator that sets it up, we should skip the setup here as those artifacts and buffers may not be needed.
        # Runtime Setup
        # ---
        static_weights = None
        if self.weight is not None:
            raise AssertionError("Static weight is not supported in SpMV.")
            # Kernel expects row-major weights, so might need to transpose;
            # also might need to transpose if is_mv
            if self.is_mv:
                static_weights = self.weight.T
            else:
                # Double transpose cancels out
                static_weights = self.weight
            if isinstance(static_weights, torch.Tensor):
                static_weights = torch_to_numpy(static_weights)
        self.add_kernel(
            "spmv",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
            self.insts_artifact,
        )
        self.add_buffer("sparse_matrix", self.M * self.ell_width * 2, static_data=static_weights)
        self.add_buffer("vector", self.K)
        self.add_buffer("output", self.M)
        runlist_args = ["spmv", "sparse_matrix", "vector", "output"]
        if self.trace_ddr_id is not None:
            # ワークアラウンド: 2倍確保(bf16換算)
            TRACE_BUFFER_SIZE = self.trace_size * 4
            self.add_buffer("trace", TRACE_BUFFER_SIZE)
            runlist_args.append("trace")
        self.add_to_runlist(*runlist_args)

    def forward(self, vector, matrix=None):
        raise NotImplementedError("SpMV operator does not support dense matrix input.")
        """Forward pass through GEMV operation

        Args:
            matrix: Input matrix of shape (..., M, K)
            vector: Input vector of shape (..., K) for MV or (..., M) for VM
            is_mv: True for matrix-vector multiplication, False for vector-matrix

        Returns:
            Output vector of shape (..., M) for MV or (..., K) for VM
        """
        t_start = time.perf_counter()
        # Flatten batch dimensions if needed
        if matrix is not None:
            matrix = matrix.reshape(*matrix.shape[-2:])
        vector = vector.reshape(*vector.shape[-1:])

        # For vector-matrix, we'll transpose the matrix internally
        if matrix is not None and not self.is_mv:
            # Transpose the matrix for vector-matrix multiplication
            # (if using static weights, the matrix is already transposed once at setup if needed)
            matrix = matrix.transpose(-2, -1)

        if matrix is not None:
            matrix_rows = matrix.shape[-2]
            matrix_cols = matrix.shape[-1]
        else:
            matrix_rows = self.M
            matrix_cols = self.K

        vector_size = vector.shape[-1]

        applicable = (
            matrix_cols == vector_size
            and matrix_rows == self.M
            and matrix_cols == self.K
            and (matrix is None or matrix.dtype == torch.bfloat16)
            and vector.dtype == torch.bfloat16
        )
        if not applicable:
            raise AIEOperatorConstraintError(
                "AIEElementwiseAdd: incompatible tensor shape(s)"
            )
        t_prep_end = time.perf_counter()
        self.run_runlist()
        if matrix is not None:
            # If matrix is none, we are using static weights that have already been written to the buffer
            self.write_buffer("sparse_matrix", matrix)
        self.write_buffer("vector", vector)
        t_writebuffer_end = time.perf_counter()

        # 実行に必要なバッファオブジェクトを集める
        bos = set(
            self.buffer_bos[buffer_arg]
            for _, *buffer_args in self.runlist
            for buffer_arg in buffer_args
        )
        insts_bos = set(
            self.xrt_kernels[kernel_name][2] for (kernel_name, *_) in self.runlist
        )

        for bo in bos | insts_bos:
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        t_sync_end = time.perf_counter()

        # カーネル実行 (AIEでの計算時間)
        self.xrt_runlist.execute()
        self.xrt_runlist.wait()
        t_exec_end = time.perf_counter()
        for bo in bos:
            bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        # 結果の取り出し
        result = self.read_buffer_as_torch("output", (self.M,))
        t_end = time.perf_counter()
        if self.trace_ddr_id is not None:
            try:
                trace_data = self.read_buffer("trace", (self.trace_size,), dtype=np.uint32)
                
                filename = "trace_gemv.txt"
                with open(filename, "w") as f:
                    for val in trace_data.flatten():
                        f.write(f"{val:08x}\n")
                trace_suffix = f"_tr{self.trace_ddr_id}" if self.trace_ddr_id is not None else ""
                file_name_base = (
                    f"gemv_{self.num_aie_columns}c_{self.M}x{self.K}_{self.tile_size}t{trace_suffix}"
                )
                print(f"[AIEGEMV] Trace saved to {filename}.this program is {file_name_base}")
                
            except Exception as e:
                print(f"[AIEGEMV] Trace save failed: {e}")

        # --- 時間の表示 ---
        # M, K, is_mv, num_aie_columns, Total(ms), prep(ms), Py=>MEM(ms),CPUMEM=>NPUMEM(ms), Kernel(ms), NPUMEM=>Py(ms)
        
        print(f"{self.M}, {self.K}, {self.is_mv}, {self.num_aie_columns}, "
              f"{(t_end - t_start)*1000:.3f}, "
              f"{(t_prep_end - t_start)*1000:.3f}, "
              f"{(t_writebuffer_end - t_prep_end)*1000:.3f}, "
              f"{(t_sync_end - t_writebuffer_end)*1000:.3f}, "
              f"{(t_exec_end - t_sync_end)*1000:.3f}, "
              f"{(t_end - t_exec_end)*1000:.3f}")

        return result
