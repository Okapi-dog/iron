# Five actual weights

One point = one matrix's five-call mean. N/A is never inferred from format capacity.

| Condition | Design | Valid / attempted | Median speedup | Median storage/Dense | Achieved CV range |
| --- | --- | ---: | ---: | ---: | ---: |
| real | Dense | 5/5 | 1.000× | 1.000 | 0.050–1.891 |
| real | ELL | 5/5 | 0.957× | 1.172 | 0.050–1.891 |
| real | Blocked Slice-ELL | 5/5 | 2.695× | 0.342 | 0.050–1.891 |
| real | Blocked SELL-C-σ | 5/5 | 3.406× | 0.265 | 0.050–1.891 |

## SELL-C-σ compared directly with Slice-ELL

| Weight | Slice latency / SELL latency | Slice storage / SELL storage |
| --- | ---: | ---: |
| model.layers.19.mlp.down_proj.weight | 1.004× | 1.006× |
| model.layers.5.self_attn.o_proj.weight | 1.020× | 1.028× |
| model.layers.11.mlp.up_proj.weight | 1.420× | 1.459× |
| model.layers.18.self_attn.k_proj.weight | 1.180× | 1.237× |
| model.layers.0.self_attn.v_proj.weight | 1.426× | 1.909× |
