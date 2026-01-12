module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_3, %tile_0_4, %tile_1_3, %tile_1_2, %tile_0_5, %tile_1_5, %tile_1_4, %tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_2(%tile_0_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_1_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo @fifo_4(%tile_0_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @fifo_5(%mem_tile_2_1 dimensionsToStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_2] -> [@fifo_3]([] [])
    aie.objectfifo.link [@fifo_4] -> [@fifo_5]([] [])
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<64x64xbf16> 
    %buffer_0_2_0 = aie.buffer(%tile_0_2) : memref<4096xbf16> 
    %buffer_0_3 = aie.buffer(%tile_0_3) : memref<64x64xbf16> 
    %buffer_0_3_1 = aie.buffer(%tile_0_3) : memref<4096xbf16> 
    %buffer_0_4 = aie.buffer(%tile_0_4) : memref<64x64xbf16> 
    %buffer_0_4_2 = aie.buffer(%tile_0_4) : memref<4096xbf16> 
    %buffer_0_5 = aie.buffer(%tile_0_5) : memref<64x64xbf16> 
    %buffer_0_5_3 = aie.buffer(%tile_0_5) : memref<4096xbf16> 
    %buffer_1_2 = aie.buffer(%tile_1_2) : memref<64x64xbf16> 
    %buffer_1_2_4 = aie.buffer(%tile_1_2) : memref<4096xbf16> 
    %buffer_1_3 = aie.buffer(%tile_1_3) : memref<64x64xbf16> 
    %buffer_1_3_5 = aie.buffer(%tile_1_3) : memref<4096xbf16> 
    %buffer_1_4 = aie.buffer(%tile_1_4) : memref<64x64xbf16> 
    %buffer_1_4_6 = aie.buffer(%tile_1_4) : memref<4096xbf16> 
    %buffer_1_5 = aie.buffer(%tile_1_5) : memref<64x64xbf16> 
    %buffer_1_5_7 = aie.buffer(%tile_1_5) : memref<4096xbf16> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 1.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_64_64_vector(%buffer_0_2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %0 = aie.objectfifo.acquire @fifo_2(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %buffer_0_2, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_2(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 1.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_64_64_vector(%buffer_0_3) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %0 = aie.objectfifo.acquire @fifo_4(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @fill_zeros_bf16_64_64_vector(%1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %buffer_0_3, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_4(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 1.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_64_64_vector(%buffer_0_4) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        func.call @fill_zeros_bf16_64_64_vector(%buffer_0_4) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %buffer_0_4, %buffer_0_4) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 1.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_64_64_vector(%buffer_0_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        func.call @fill_zeros_bf16_64_64_vector(%buffer_0_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %buffer_0_5, %buffer_0_5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 1.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_64_64_vector(%buffer_1_2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        func.call @fill_zeros_bf16_64_64_vector(%buffer_1_2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %buffer_1_2, %buffer_1_2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 1.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_64_64_vector(%buffer_1_3) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        func.call @fill_zeros_bf16_64_64_vector(%buffer_1_3) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %buffer_1_3, %buffer_1_3) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 1.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_64_64_vector(%buffer_1_4) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        func.call @fill_zeros_bf16_64_64_vector(%buffer_1_4) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %buffer_1_4, %buffer_1_4) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 1.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_64_64_vector(%buffer_1_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        func.call @fill_zeros_bf16_64_64_vector(%buffer_1_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
        func.call @matmul_bf16_bf16(%3, %buffer_1_5, %buffer_1_5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
        aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_2_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_3, Trace : 0>
      aie.packet_dest<%shim_noc_tile_2_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(3) {
      aie.packet_source<%tile_0_4, Trace : 0>
      aie.packet_dest<%shim_noc_tile_2_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(4) {
      aie.packet_source<%tile_0_5, Trace : 0>
      aie.packet_dest<%shim_noc_tile_2_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(5) {
      aie.packet_source<%mem_tile_0_1, Trace : 0>
      aie.packet_dest<%shim_noc_tile_2_0, DMA : 1>
    } {keep_pkt_header = true}
    aiex.runtime_sequence(%arg0: memref<4096xbf16>, %arg1: memref<262144xbf16>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 559107915 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 622466850 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 74273 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228132 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 2 : i32, row = 0 : i32, value = 15 : ui32}
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 3 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 3 : i32, value = 2 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 3 : i32, value = 559107915 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 3 : i32, value = 622466850 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 3 : i32, value = 74273 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 3 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 3 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 14 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 2 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228100 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 2 : i32, row = 0 : i32, value = 14 : ui32}
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 4 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 4 : i32, value = 3 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 4 : i32, value = 559107915 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 4 : i32, value = 622466850 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 4 : i32, value = 74273 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 4 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 4 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 13 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 3 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228068 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 2 : i32, row = 0 : i32, value = 13 : ui32}
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 5 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 5 : i32, value = 4 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 5 : i32, value = 559107915 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 5 : i32, value = 622466850 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 5 : i32, value = 74273 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 5 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 5 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 4 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228036 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 2 : i32, row = 0 : i32, value = 12 : ui32}
      aiex.npu.write32 {address = 606416 : ui32, column = 0 : i32, row = 1 : i32, value = 2627534848 : ui32}
      aiex.npu.write32 {address = 606420 : ui32, column = 0 : i32, row = 1 : i32, value = 12293 : ui32}
      aiex.npu.write32 {address = 606432 : ui32, column = 0 : i32, row = 1 : i32, value = 1549292624 : ui32}
      aiex.npu.write32 {address = 606436 : ui32, column = 0 : i32, row = 1 : i32, value = 1818780768 : ui32}
      aiex.npu.write32 {address = 724736 : ui32, column = 0 : i32, row = 1 : i32, value = 16780832 : ui32}
      aiex.npu.write32 {address = 724740 : ui32, column = 0 : i32, row = 1 : i32, value = 572588802 : ui32}
      aiex.npu.write32 {address = 606208 : ui32, column = 0 : i32, row = 1 : i32, value = 7424 : ui32}
      aiex.npu.writebd {bd_id = 11 : i32, buffer_length = 65536 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 2 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 5 : i32, packet_type = 3 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 67228004 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 2 : i32, row = 0 : i32, value = 11 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 2 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 2 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 2 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][8, 1, 64, 64][0, 0, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<4096xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 1, 64, 64][512, 64, 4096, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 64][8, 1, 64, 64][512, 64, 4096, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<262144xbf16>
      aiex.npu.dma_wait {symbol = @fifo_3}
      aiex.npu.dma_wait {symbol = @fifo_5}
      aie.end
    }
  }
}
