duration_ms=390
tile_size=2
K=2048
event_count=306
clock_MHz=1800
num_core=4
DMA_starvation_ms=78


clock_cycles=duration_ms *1000 #trace上ではusが1clock

duration_real_s = clock_cycles / (clock_MHz * 1e6)
data_GB = (K*2*event_count*tile_size) / 1024**3
bandwidth_GBs = data_GB / duration_real_s
print(f"Estimated Bandwidth(1 core): {bandwidth_GBs} GB/s")
print(f"Estimated Bandwidth({num_core} core): {bandwidth_GBs * num_core} GB/s")
print(f"DMA Starvation Ratio: {DMA_starvation_ms / duration_ms * 100:.2f} %")

