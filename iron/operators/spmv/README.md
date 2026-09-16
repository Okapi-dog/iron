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

通常の再現コマンドは次である。結果JSONはgitignore対象の
`npu_data/phase1_mlir_v1.4.3/full_core_device_only.json`に置く。

```bash
python iron/operators/spmv/measure.py
```

Phase 0と保存済みの行列ビット列まで揃える比較には、旧checkoutの `npu_data` を明示する。

```bash
python iron/operators/spmv/measure.py \
  --legacy-data-root /home/hitoshi/IRON/operators/spmv/npu_data
```

`--legacy-data-root` は旧 `*_xdna_ell.npy` / `*_xdna_sell32.npy` の `uint16` 表現を
BF16 storage bit patternのまま読込む。従ってindexの局所性を含めて同一であり、単に同じ
`M,K,width` を乱数生成する比較ではない。各sampleは2回warm-up後の1回を採用し、5 sampleの
mean/min/max/std. dev.を出す。Phase 1の`result.npu_time`とPhase 0の`run_runlist()`戻り値は、
いずれもXRTのkernel launchから`wait()`までをhost側の時計で測る値で、host側のtensor生成・BO同期を
含まない。hardware cycle counterではない。全6条件でCPU reference一致した。

| `M × K` | width | kernel | Phase 0 mean | Phase 1 mean | 差分 | Phase 1 min | Phase 1 max | Phase 1 std. dev. | Phase 1 effective BW | 結果 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `4096 × 4096` | 512 | ELL | 265.13 us | 276.91 us | +4.4% | 262.43 us | 294.25 us | 10.92 us | 30.353 GB/s | PASS |
| `4096 × 4096` | 512 | SELL-32 block | 261.96 us | 275.99 us | +5.4% | 263.77 us | 282.58 us | 6.40 us | 30.454 GB/s | PASS |
| `4096 × 11008` | 1376 | ELL | 517.97 us | 518.49 us | +0.1% | 498.82 us | 530.35 us | 10.66 us | 43.539 GB/s | PASS |
| `4096 × 11008` | 1376 | SELL-32 block | 520.63 us | 524.59 us | +0.8% | 511.59 us | 536.59 us | 7.97 us | 43.032 GB/s | PASS |
| `28672 × 8192` | 1024 | ELL | 2207.14 us | 2198.66 us | -0.4% | 2188.87 us | 2224.77 us | 13.37 us | 53.448 GB/s | PASS |
| `28672 × 8192` | 1024 | SELL-32 block | 2182.89 us | 2182.95 us | +0.0% | 2165.17 us | 2190.88 us | 9.34 us | 53.833 GB/s | PASS |

この表は、旧kernelの性能上重要な構造を復元し、同一NPYで測った値である。具体的には32 lane gatherの
強制unroll、`AIE_PREPARE_FOR_PIPELINING`、最低反復数指定、旧ELLの関数ABI、さらにDMA taskの投入順を
揃えた。SELL-32 blockについては、d3の64回制限を守る旧方式（A TAPを64 iteration以下へ分割し、
4 chunkずつtask group化）も復元している。

中・大の4条件は旧値との差が -0.4%〜+0.8% であり、帯域律速のSpMV本体に新版IRON/mlir-aieの
性能低下は観測されない。一方、`4096 × 4096` は +4〜5%（絶対 +11〜14 us）が残る。ただしこれは
XRT launch/waitをhost時計で測る値であり、このshapeではsample std. dev. が6〜11 us、別run間の揺れも
同程度に現れる。旧1.1.3 venvは現在ws007に残っていないため、同一時刻に旧xclbinと新xclbinを交互実行する
paired testはまだできない。従ってこの小shapeの差を新版toolchainの回帰とは断定しないが、厳密にゼロ差を
要求する場合はPhase 0の固定venvを再構築し、paired measurementを追加する。

この前に得た590--6533 usの値は、最小kernelを書き直した際に上記のunroll/pipeline指示とDMAの
同時投入を落とした非等価版の値であり、性能比較から除外する。新版の`result.npu_time`がhost同期を
含んだことによる数倍差ではない。

## 次の順序

Phase 1の静的ELL / SELL-32 / SELL-32 block移植は機能面・帯域律速の性能面で完了した。次は
Slice-ELLのpacker/referenceを先に作り、静的slice path、最後にdynamic ObjectFIFOを小さいmatrixで
段階的に検証する。小shapeのhost launch/wait差を厳密に詰める場合だけは、その前に上記paired testを行う。
