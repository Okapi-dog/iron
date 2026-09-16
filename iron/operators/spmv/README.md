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

## Phase 0 / Phase 1: 全32 coreの移行比較

Phase 0の旧実装（`devel`）と同じ3 shape、全32 core（4 core-row × 8 core-column）、
width=`K/8`（12.5%密度相当）で測った。通常SELL-32はPhase 0の比較対象ではなかったため、
ここでも **ELL と SELL-32 blockだけ**を表に載せる。

Phase 1の再現コマンドは次である。結果JSONはgitignore対象の
`npu_data/phase1_mlir_v1.4.3/full_core_device_only.json`に置く。

```bash
python iron/operators/spmv/measure.py
```

各sampleは2回warm-up後の1回を採用し、5 sampleのmean/min/max/std. dev.を出す。Phase 1は
新版runtimeの`result.npu_time`、Phase 0は旧runtimeの`run_runlist()`戻り値であり、どちらも
host側のtensor生成・BO同期を含まないdevice-only値である。sample間は4秒idleにした。全6条件で
CPU reference一致した。

| `M × K` | width | kernel | Phase 0 mean | Phase 1 mean | Phase 1 min | Phase 1 max | Phase 1 std. dev. | Phase 1 effective BW | 結果 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `4096 × 4096` | 512 | ELL | 265.13 us | 590.19 us | 572.17 us | 618.37 us | 16.69 us | 14.241 GB/s | PASS |
| `4096 × 4096` | 512 | SELL-32 block | 261.96 us | 571.58 us | 561.40 us | 580.53 us | 7.29 us | 14.705 GB/s | PASS |
| `4096 × 11008` | 1376 | ELL | 517.97 us | 1341.38 us | 1333.46 us | 1361.27 us | 10.11 us | 16.829 GB/s | PASS |
| `4096 × 11008` | 1376 | SELL-32 block | 520.63 us | 1336.94 us | 1329.56 us | 1348.00 us | 7.74 us | 16.885 GB/s | PASS |
| `28672 × 8192` | 1024 | ELL | 2207.14 us | 6533.19 us | 6521.80 us | 6546.40 us | 8.74 us | 17.987 GB/s | PASS |
| `28672 × 8192` | 1024 | SELL-32 block | 2182.89 us | 6515.70 us | 6508.55 us | 6524.51 us | 5.57 us | 18.036 GB/s | PASS |

この表は「移植が正しく動くこと」と同一規模のdevice-only測定を示すものであり、**toolchainだけの
速度比較ではない**。Phase 1は最新Runtimeに合わせてkernelとdata pathを最小から書き直しており、
旧kernelのloop pragma・allocation・artifact構成をまだ性能同等に移植していない。従って上表の
遅さをmlir-aie更新による性能劣化とは結論付けない。Phase 2以降で同じkernel最適化とTAP/BD構成を
揃えてから、初めて性能回帰として評価する。

## 次の順序

Phase 1の静的ELL / SELL-32 / SELL-32 block移植は完了した。次はSlice-ELLのpacker/referenceを
先に作り、静的slice path、最後にdynamic ObjectFIFOを小さいmatrixで段階的に検証する。
