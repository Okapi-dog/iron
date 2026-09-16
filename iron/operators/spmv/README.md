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

旧kernel相当のpipeline/unroll構造を復元した2026-09-16のNPU2実測ではCPU reference一致（`1 passed`）、
NPU execution latencyは`135.6 us`、payloadベースeffective bandwidthは`7.775 GB/s`だった。これは一回のsanity runであり、
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

旧kernel相当のpipeline/unroll構造を復元した2026-09-16の単独sanity runはCPU reference一致、
NPU execution latency `109.9 us`、effective bandwidth `9.598 GB/s`だった。これは一回のsanity値であり、
Phase 0の5-sample baselineとは比較しない。

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

各sampleは2回warm-up後の1回を採用し、5 sampleのmean/min/max/std. dev.を出す。Phase 1の
`result.npu_time`とPhase 0の`run_runlist()`戻り値は、いずれもXRTのkernel launchから`wait()`までを
host側の時計で測る値であり、host側のtensor生成・BO同期を含まない。これはhardware cycle counterでは
ないが、旧新で同じ範囲である。sample間は4秒idleにした。全6条件でCPU reference一致した。

| `M × K` | width | kernel | Phase 0 mean | Phase 1 mean | Phase 1 min | Phase 1 max | Phase 1 std. dev. | Phase 1 effective BW | 結果 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `4096 × 4096` | 512 | ELL | 265.13 us | 288.31 us | 278.79 us | 305.60 us | 9.94 us | 29.152 GB/s | PASS |
| `4096 × 4096` | 512 | SELL-32 block | 261.96 us | 280.22 us | 274.86 us | 288.65 us | 5.45 us | 29.994 GB/s | PASS |
| `4096 × 11008` | 1376 | ELL | 517.97 us | 525.77 us | 504.28 us | 539.59 us | 14.19 us | 42.936 GB/s | PASS |
| `4096 × 11008` | 1376 | SELL-32 block | 520.63 us | 513.30 us | 491.78 us | 532.04 us | 16.26 us | 43.979 GB/s | PASS |
| `28672 × 8192` | 1024 | ELL | 2207.14 us | 2205.59 us | 2193.71 us | 2223.78 us | 10.56 us | 53.280 GB/s | PASS |
| `28672 × 8192` | 1024 | SELL-32 block | 2182.89 us | 2216.48 us | 2171.57 us | 2276.03 us | 33.68 us | 53.018 GB/s | PASS |

この表は、旧kernelの性能上重要な構造を復元した後の値である。すなわち32 lane gatherの強制unroll、
`AIE_PREPARE_FOR_PIPELINING`、最低反復数指定、およびA/x DMAを同じtask groupへ投入する順序を
旧版と揃えた。したがって3 shapeすべてでPhase 0と概ね同等であり、最新toolchainへの移植は性能上も
Phase 1の通過条件を満たした。

この前に得た590--6533 usの値は、最小kernelを書き直した際に上記のunroll/pipeline指示とDMAの
同時投入を落とした非等価版の値であり、性能比較から除外する。新版の`result.npu_time`がhost同期を
含んだことによる数倍差ではない。

## 次の順序

Phase 1の静的ELL / SELL-32 / SELL-32 block移植は完了した。次はSlice-ELLのpacker/referenceを
先に作り、静的slice path、最後にdynamic ObjectFIFOを小さいmatrixで段階的に検証する。
