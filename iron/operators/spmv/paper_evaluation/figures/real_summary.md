# Five actual weights

One point = one matrix's five-call mean. N/A is never inferred from format capacity.

| Condition | Design | Valid / attempted | Median speedup | Median storage/Dense | Achieved CV range |
| --- | --- | ---: | ---: | ---: | ---: |
| real | Dense | 5/5 | 1.000× | 1.000 | 0.050–1.891 |
| real | ELL | 4/5 | 0.935× | 1.172 | 0.050–1.891 |
| real | Blocked Slice-ELL | 5/5 | 2.588× | 0.342 | 0.050–1.891 |
| real | Blocked SELL-C-σ | 5/5 | 3.049× | 0.265 | 0.050–1.891 |

## Failed combinations

| Condition / weight | Seed | Design | Error |
| --- | ---: | --- | --- |
| real / model.layers.11.mlp.up_proj.weight |  | ELL | ValueError: vertical ELL kernel requires M divisible by 32*4*columns |

## SELL-C-σ compared directly with Slice-ELL

| Weight | Slice latency / SELL latency | Slice storage / SELL storage |
| --- | ---: | ---: |
| model.layers.19.mlp.down_proj.weight | 1.013× | 1.006× |
| model.layers.5.self_attn.o_proj.weight | 1.083× | 1.028× |
| model.layers.11.mlp.up_proj.weight | 1.365× | 1.459× |
| model.layers.18.self_attn.k_proj.weight | 1.131× | 1.237× |
| model.layers.0.self_attn.v_proj.weight | 1.396× | 1.909× |
