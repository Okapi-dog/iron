# SpMV: Phase 1（最新 IRON への移植）

このdirectoryは、旧 `devel` の `operators/spmv` を参照して、最新 AMD IRON の
operator interfaceへ段階的に移植する作業場所である。旧 checkout や旧 venv を更新しない。

## 固定した環境

| 項目 | 値 |
| --- | --- |
| IRON upstream | `2191fe0741e6b6c2b5efe245a33fba5afa6b717a` (`devel`, 2026-09-14) |
| 作業branch | `spmv/mlir-v1.4.3` |
| mlir-aie | `1.4.3.dev85+gdf48abc` |
| llvm-aie | `22.0.0.2026090701+3e93bf7b` |
| XRT | `/opt/xilinx/xrt/setup.sh`（Phase 0と同じws007環境） |
| venv | `/home/hitoshi/ironenv-mlir-v1.4.3` |
| 実機 | AMD Ryzen AI 9 HX 370 / NPU2 (Strix) |

環境を有効化する。

```bash
source /opt/xilinx/xrt/setup.sh
source /home/hitoshi/ironenv-mlir-v1.4.3/bin/activate
cd /home/hitoshi/IRON-mlir-v1.4.3
```

## 最初の通過条件

最新upstreamの既存 operator も先に実機で確認済みである。

```bash
pytest -q 'iron/operators/axpy/test.py::test_axpy[input_length_2048-num_aie_columns_8-tile_size_256-scalar_factor_3.0]' \
  --iterations 1 --csv-output /tmp/phase1_axpy.csv
# 1 passed
```

## Static ELL baseline

`SpMVELL` はSlice-ELLやdynamic ObjectFIFOを含まないPhase 1用baselineである。行は元の順のまま、
columnごとに連続範囲を担当し、各column内で4 core-rowへsplitする。出力も同じ順でjoinして連続した
`y`へdrainする。

- `packed` は各行について **`[ell_width 個の uint16 index][ell_width 個の BF16 value]`** の順である。
  index/valueをslotごとに交互配置しない。この平面配置は旧 `sparse_matvec_vectorized_bf16_bf16`
  kernelと互換である。
- `x` は各columnへ一度だけ転送し、column内の4 coreが再利用する。
- `M=1024, K=2048, ell_width=256, rows=4, cols=8, rows_per_core=2` を固定最小ケースとする。
  全32 core、12.5%密度相当である。

再現コマンド:

```bash
pytest -q -s iron/operators/spmv/test.py --iterations 1 \
  --csv-output /tmp/phase1_spmv_ell.csv
```

2026-09-16のNPU2実測ではCPU reference一致（`1 passed`）、NPU execution latencyは
`170.2 us`、payloadベースeffective bandwidthは`6.196 GB/s`だった。これは一回のsanity runであり、
Phase 0の5-sample性能baselineとは比較しない。

### SELL-32 のL1制約

SELL-32でwidth 256、`m=1`のcore入力objectは`32 × 256 × (index 2 B + value 2 B)`、すなわち
32 KiBである。最新compilerはdepth=2のA FIFO（64 KiB）にxとstackを加える配置を明確に拒否した。
この最小baselineではcore側A FIFOをdepth=1にする。これは機能移植の設定であり、overlapを含む性能設定は
L1容量を満たすwidth/depthの組を別途選ぶ。

再現コマンド:

```bash
pytest -q -s iron/operators/spmv/test.py::test_static_sell32_1024x2048 \
  --iterations 1 --csv-output /tmp/phase1_spmv_sell32.csv
```

2026-09-16の実機結果はCPU reference一致（`1 passed`）、NPU execution latency `217.2 us`、
effective bandwidth `4.855 GB/s`である。これも機能移植のsanity runである。

### SELL-32 block

`SpMVSELL32Block` はSELL-32の物理配置を変えず、横16 slotを一つのA objectとして送る。
従ってcoreごとのA objectは`32 × 16 × 4 B = 2 KiB`で、depth=2でもL1に余裕がある。各coreは
32行のy objectを一度acquireし、16 slotずつBF16へ丸めながら更新してから一度だけreleaseする。

```bash
pytest -q -s iron/operators/spmv/test.py::test_static_sell32_block_1024x2048 \
  --iterations 1 --csv-output /tmp/phase1_spmv_sell32_block.csv
```

2026-09-16の単独sanity runはCPU reference一致、NPU execution latency `211.6 us`、effective
bandwidth `4.985 GB/s`だった。三形式を同一sessionで再実行した結果もすべてPASSで、ELL `166.1 us`、
SELL-32 `182.8 us`、SELL-32 block `197.4 us`だった。これらは一回ずつのsanity値であり、Phase 0の
5-sample baselineとは比較しない。

生成した`input_with_addresses.mlir`も確認した。SELL-32 blockはcore A BDが`memref<1024xbf16>`
（2 KiB）、SELL-32は`memref<16384xbf16>`（32 KiB, depth=1）、ELLは`memref<1024xbf16>`である。
column内join後のy drain offsetはELLで`0,2,4,6`、SELL-32/blockで`0,32,64,96`となり、いずれも
4 core-rowの出力を元のrow順で連続している。今回の最小caseのTAP outer iterationは1であり、d3の
65上限に達しない。大きい行列でのTAP chunking/BD数評価は次の性能評価で行う。

## 次の順序

1. `design_sell32.py` 相当を最新 `MLIROperator` / `Runtime`へ移し、固定幅SELL-32のjoin/splitと
   contiguous y drainを検証する。
2. `design_sell32_block.py` 相当の横block pathを移し、y accumulateとx broadcastを検証する。
3. 各段階で生成MLIRと`input_with_addresses.mlir`を調べ、TAPのd3、BD数、y drain offsetを旧baselineと
   比較する。Slice-ELLとdynamic ObjectFIFOはこの段階では導入しない。
