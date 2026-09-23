# SELL-C-σ 実装報告書

各Stepの完了時点で確認できた事実を記録する。折り畳まれた過去Stepの記述は、当時の状態を示す。

<details>
<summary>Step 0 — 共通評価導線とstorage予測（展開して表示）</summary>

## Step 0 報告（NPUなし）

## 状態と再現条件

- 実装branch: `spmv/sell-c-sigma`。基点は `spmv/slice-ell` の `3c603e9c64c5167ac39bb92034cd23aa6662bcdb`。
- このStepでは **行列の容量と列別block負荷のみ** を評価した。SELL-C-σのpacker、kernel、reorder、およびNPU実測はまだない。global sortはoffline容量下限であり、NPUの候補として扱わない。
- 初期パラメータ: `B_h=8`, `B_w=256`, 8 column、BF16 value 2 B + uint16 A index 2 B。比較対象はdense BF16、従来ELL、row-order-preserving Slice-ELL、window-local sort 8/16、global sort。
- 計測にはws007上の `/home/hitoshi/ironenv-mlir-v1.4.3/bin/python` を使用した。元の実モデルは `/home/hitoshi/elsa/pruned_model/Llama-2-7b-hf_pruned0.9_admm_lr5e-05_20260301_2016`。モデル名だけでなく各tensorの内容SHA-256もJSONL recordへ出力される。

## 用意した共通入口

- `evaluation.py`: `MatrixInput` が合成行列の生成条件、またはsafetensors tensorの読み込み元を指定する。`load_or_generate_csr()` はformatと独立な `CSRMatrix`（`indptr`、`indices`、`values`、共通BF16入力ベクトル `x`）を返す。合成行列だけを直接生成するときは `generate_synthetic_csr()` を使う。
- 使用経路は、実体が必要なら `MatrixInput → load_or_generate_csr() → CSRMatrix → pack_existing_format()`。容量評価だけなら `MatrixInput → synthetic_profile() / safetensors_profile() → estimate_storage()` とし、大行列の全CSRを作らない。`canonical` は行・出力の元順序を表す語として使い、CSR型の名前には使わない。
- `FormatSpec` と `pack_existing_format()` で、同一CSRから既存dense/ELL/Slice-ELLへ変換できる。SELL-C-σを選んだ場合はStep 1まで明示的に未実装エラーとし、Slice-ELLを偽って返さない。
- `DesignSpec` はformatと実行方式の組を検査し、`existing` / `planned` / `unproven` を記録する。NPU dispatch自体は後続Stepで追加する。
- `evaluate_step0.py` は行列生成・format評価をNPU実行から分離し、matrix recipe ID、実tensor SHA-256（合成行列ではseed/生成条件と行NNZから作る再現用hash）、format ID、全設定、storage、負荷をJSONLに記録できる。短い表の表示も可能。
- 8/16 windowは元の行順で連続し、境界を `B_h` 単位で揃える。等行数と等NNZの両方を比較できる。16 windowには連続割当と容量モデル上のbalanced割当を設けた。balanced側のruntime TAP/BD/同期は**未検証**。

## 実モデル4行列のstorage評価

値はMiB。`A` はpacked行列本体のみ。`dense比` は **packed A + row map** をBF16 dense行列サイズで割った値。`max blocks` は8 column中の最大block数であり、処理時間ではない。SELLは8 window・等行数境界・連続割当。16 windowは等行数境界・balanced割当。

| weight | M×K | dense A | 従来ELL A | Slice-ELL A | SELL 8 A / dense比 / max blocks | SELL 16 A / dense比 / max blocks | global A (参考) | reorder L1下限 8→16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| layer 3 `o_proj` | 4096×4096 | 32.000 | 42.500 | 8.750 | 8.234 / 0.258 / 137 | 8.336 / 0.261 / 138 | 8.148 | 2.00→1.00 KiB |
| layer 0 `gate_proj` | 11008×4096 | 86.000 | 155.875 | 47.000 | 22.727 / 0.265 / 375 | 23.078 / 0.269 / 372 | 22.391 | 5.38→2.69 KiB |
| layer 0 `down_proj` | 4096×11008 | 86.000 | 101.000 | 27.969 | 19.531 / 0.227 / 322 | 19.867 / 0.231 / 320 | 19.117 | 2.00→1.00 KiB |
| layer 25 `down_proj` | 4096×11008 | 86.000 | 20.000 | 20.000 | 19.867 / 0.231 / 319 | 19.891 / 0.231 / 319 | 19.820 | 2.00→1.00 KiB |

4行列とも実際の密度は約10%。ただし行NNZ分布が違うので効果は均一ではない。例えばlayer 0 `gate_proj` はSlice-ELLの47.0 MiBからSELL 8の22.7 MiBまで減るが、ほぼ均一なlayer 25 `down_proj` では20.0→19.9 MiB程度である。8→16 windowにすると、ここでは主にreorderのL1下限が半減し、A paddingは少し増える。global sortのA容量は参考下限だが、例えば `M=28672` なら出力とrow mapだけでも112 KiBとなり、単一64 KiB L1には収まらない。

`reorder L1下限` はwindow内のcanonical BF16出力とrow mapの `σ×(2+index_bytes)` のみ。実際の採否にはFIFO object、stack、code/data配置、アラインメントも含めて生成物を検査する必要がある。ここに表示された4行列の下限は64 KiB未満だが、これはL1 fitの証明ではない。

## 合成行列の分布・密度sweep

`M×K=28672×8192`、seed 42、dense BF16 448 MiB。uniformは各行ほぼ同じNNZ、skewedはlognormal由来の不均一NNZで、いずれも指定密度に総NNZを正確に合わせた。以下はpacked A MiB。SELLは等行数境界・連続割当。

| 密度 | uniform Slice-ELL / SELL 8 | skewed Slice-ELL | skewed SELL 8 | skewed SELL 16 | skewed SELL 8 最大列blocks |
|---:|---:|---:|---:|---:|---:|
| 5% | 56.0 / 56.0 | 180.0 | 61.6 | 62.5 | 1024 |
| 10% | 112.0 / 112.0 | 332.1 | 104.9 | 105.8 | 1735 |
| 12.5% | 112.0 / 112.0 | 400.8 | 126.9 | 127.8 | 2101 |
| 20% | 196.0 / 196.0 | 573.1 | 193.5 | 194.3 | 3200 |

均一分布ではソートの利益はなく、row map分だけ総storageがわずかに増える。不均一分布では利益が大きいが、これをNPU速度向上と同一視してはいけない。特にreorder・3 compute core/column化・DMA・同期の追加コストがある。

## 実行方法と検証

リポジトリrootをカレントディレクトリとする。Python環境にtorch、numpy、safetensorsが必要。リポジトリのpytest設定はNPU runtime初期化を行うため、CPU-onlyの本テストにもws007ではXRTのsetupが必要だった。

```bash
source /opt/xilinx/xrt/setup.sh
export NPU_RUNTIME=xrt
export PYTHONPATH="$PWD:/home/hitoshi/elsa/elsa_venv/lib/python3.12/site-packages:$PYTHONPATH"
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m pytest iron/operators/spmv/test_evaluation.py -q

/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m iron.operators.spmv.evaluate_step0 \
  --model-dir /home/hitoshi/elsa/pruned_model/Llama-2-7b-hf_pruned0.9_admm_lr5e-05_20260301_2016 \
  --format all --output jsonl

/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m iron.operators.spmv.evaluate_step0 \
  --synthetic 28672 8192 0.125 skewed 42 --format all --output markdown
```

テスト結果は **35 passed**（リポジトリのiteration設定により7件×5回）。同一CSR/xからの既存format変換、再現性、window境界、storage/負荷の整合性、global-sort参考値、safetensorsの内容hash変化を確認した。safetensors→PyTorch CSRでPyTorchのbeta警告が1件あるがテストは成功した。

## Step 1へ渡す判断

初期対象は計画通り **8 window / 等行数 / 連続割当** が妥当。4行列でreorder L1下限は2–5.38 KiB、global sortよりAはやや大きいが、column内完結ができる。16 windowはL1の余裕を増やす選択肢として残す。等NNZ境界やbalanced割当はmodel上の候補であり、最初のNPU実装に直結させない。最も重要な未確定点は、実SELL packerのcorrectnessとreorderを含むend-to-end NPU latencyである。

</details>

## Step 1 — windowed packerとCPU reference

### 実装した契約

- `SliceELLConfig.window_count=0` は従来の行順保存形式。`1` はglobal-sortのoffline参照、`8/16` はwindow-local stable sort。`window_slice_boundaries` は `B_h` 行単位の境界を指定する。省略時は各windowにsliceを均等配分する。
- sortは各window内の行NNZ降順、同じNNZの行は元の順序を維持する。`PackedSliceELL.row_indices[physical_row]` はそのwindow内の元の行番号。末尾のpadding行はdtypeの最大値をsentinelとする。window内最大行数が65535以下ならuint16、それを超えればuint32。
- `packed_a` のblock/core-row/local-row順と `blocks_per_slice` の固定長control contractは従来と同じ。`cpu_spmv_slice_ell()` はphysical `y'` を返し、`cpu_unpermute_windows()` が元の行順の `y` に戻す。ソートなしでは両者は同じ値。
- `dense_to_slice_ell()` も同じpackerを通る。共通入口 `pack_for_design()` はformatとdesignの整合性を検査する。方式Aと方式Bは**同一CSR・同一FormatSpecなら同一packed A/row mapを共有**できるが、実行方式の配線・L1配置は別問題。
- 16 window等の連続割当はpackできる。不均等境界で列別slice数が異なる場合、固定長NPU ABIの `slices_per_column` は明示的に失敗する。Step 0容量モデルの `balanced` window→column割当は、まだpayload/runtime契約がないためpackerでは黙って実装せず拒否する。
- optional cacheはsorted時に `row_indices.bin` も保存し、manifestへwindow境界・dtype・hashを記録する。通常のテストと実行はメモリ上のpacked objectを使う。

### CPU検証結果

ws007の既存Python環境で、従来の `test_slice_ell.py`、Step 0の `test_evaluation.py`、新規 `test_sell_c_sigma.py` を実行し、**130 passed**（pytestのiteration設定で26件×5回）。無ソートのA payloadは旧 `spmv/slice-ell` branchから取得したSHA-256ともbitwise一致した。stable tie-break、zero row/zero-block slice、末尾partial slice、global sort、任意境界、複数column、8/16 window、uint32 row mapへの拡張、実packed byte数とStep 0モデルの一致を確認した。

Llama-2-7B pruning modelの代表4 weightについて、8/16 window・等行数境界・8 column・`B_h=8, B_w=256`で実際にpackし、`cpu_unpermute_windows(cpu_spmv_slice_ell(...))` を同じCSRのCPU参照と比較した。**全8条件で一致**。下表は実packed A容量で、Step 0予測値とも一致する。これはNPU実測ではない。

| weight | M×K | 8 window A MiB | 16 window A MiB | row map KiB | 最大絶対誤差 |
|---|---:|---:|---:|---:|---:|
| layer 3 `o_proj` | 4096×4096 | 8.234 | 8.336 | 8.0 | 0 |
| layer 0 `gate_proj` | 11008×4096 | 22.727 | 23.078 | 21.5 | 0 |
| layer 0 `down_proj` | 4096×11008 | 19.531 | 19.867 | 8.0 | 0 |
| layer 25 `down_proj` | 4096×11008 | 19.867 | 19.891 | 8.0 | 0.000061 |

末尾にcolumn配置用のpadding sliceがある小行列では、Step 0の旧モデルはrow map容量を過小評価していた。Step 1の実packerとの照合で発見し、モデルを **padding行のsentinelを含む実配置** に修正した。上記4行列には追加padding sliceがなく、Step 0の掲載値は変わらない。

### 再現コマンド

リポジトリrootで実行する。`verify_step1.py` は行列を一度だけ読み込み、同一CSR/xとCPU参照を各formatへ再利用する。JSONLにはmatrix/format hash、実packed A bytes、CPU正誤などが残る。

```bash
source /opt/xilinx/xrt/setup.sh
export NPU_RUNTIME=xrt
export PYTHONPATH="$PWD:/home/hitoshi/elsa/elsa_venv/lib/python3.12/site-packages:$PYTHONPATH"
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m pytest \
  iron/operators/spmv/test_slice_ell.py \
  iron/operators/spmv/test_evaluation.py \
  iron/operators/spmv/test_sell_c_sigma.py -q

/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m iron.operators.spmv.verify_step1 \
  --model-dir /home/hitoshi/elsa/pruned_model/Llama-2-7b-hf_pruned0.9_admm_lr5e-05_20260301_2016 \
  --format sell_c_sigma --windows 8 16 --boundary equal_rows

/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m iron.operators.spmv.verify_step1 \
  --synthetic 128 128 0.1 skewed 42 --columns 2 --block-height 8 \
  --block-width 32 --windows 2 4 --boundary equal_rows equal_nnz
```

### 次Stepへ残ること

このStepではNPU compile/実行、固定長ObjectFIFOのlowering、row mapのFIFO転送、MemTile joinからreorder coreへの配線、L1実配置、16 windowのTAP/BD切替を検証していない。`window_count=1` は容量・CPU参照専用であり、8 columnから単一coreへのfan-inを実装した意味ではない。これらは下記Step 2/3で検証した。Step 2のmicro-testは既存の一時ファイルの移植ではなく、このbranchで新規に書いた。

<details>
<summary>Step 2 — 新規マイクロテストと固定長データ経路（展開して表示）</summary>

## 実装と結果

`sell_c_sigma_design.py` の `sell_reorder_route()`、`sell_c_sigma.cc` のcopy/scatter kernel、`sell_c_sigma_op.py` のoperator、および `test_sell_route.py` を新規作成した。3 producerは**単純コピー**でありSpMVではない。各列で2/4/4 BF16の3 producer出力をMemTileで10要素にjoinし、専用coreがdummy 2要素を除く8行をwindow内row mapでscatterする。完成した1 windowのみをcanonical出力としてdrainする。producer→reorderの途中にDRAM書戻しはない。

ws007のNPU2、`mlir-aie v1.4.3`環境で `M=1024, 8 column/8 window` のidentity/reverse/random、`M=28672, 8 column/8 window` と `M=28672, 8 column/16 window` のrandomを実行し、BF16出力がCPU期待値と**bitwise一致**した（5 passed）。16 windowでは1列のreorder workerとL1 FIFOを2 windowで再利用した。

生成された `input_with_addresses.mlir` の `M=28672/16 window` 代表列: reorder L1はcanonical `1792×BF16=3584 B`、map `1792×int16=3584 B`、join FIFO `2×10×BF16=40 B`を割当。reorder tileには4 buffer・6 lock、MemTileには4 buffer・12 lock。Shimは入力MM2S 2本（physical/map）と出力S2MM 1本。reorder tileのDMAは入力S2MM 2 channel・出力MM2S 1 channel、計4 BD。これらは代表列の生成MLIR上の数であり、全デバイス合計ではない。MemTileのbank配置はphysicalの2 objectがbank 0/1、joinedの2 objectがbank 2/3。worker stack/code等は上記payload容量に含めていない。

</details>

## Step 3 — 実SpMVと専用reorder core

### データ契約と実装

`sell_spmv_dedicated()` は各列のrow 2/3/4に3計算worker、row 5にreorder workerを置く。`B_h=8, B_w=256`、AはStep 1の `PackedSliceELL` をそのまま使い、行順を変えずに2/3/3行へsplitする。3行側の出力はDMAの4-byte整列のため4要素とし、4番目をdummy zeroとする。MemTileで2/4/4を10要素にjoinし、reorder側で実8行だけをscatterする。scalar-stateは各compute coreの4×FP32 L1 bufferに置き、各sliceで初期化、`p` 個のA blockを既存Slice-ELLと同様の32-lane BF16 MACで処理してからBF16出力にする。

Shim入力は **Aとcontrolの2本**。`make_dedicated_inputs()` はwindowごとに `[2 header words | BF16 x | p per slice | alignment pad | uint16 row map]` を作る。MemTileでconfigを3 compute coreへ、row mapをreorder coreへsplitする。出力は1本。3入力Shimが必要なA/config/map別送を避けた。control・A・出力のObjectFIFO object長は各window/columnで固定し、`p` のみruntime値である。canonical windowはreorder L1上でまずzero初期化し、末尾padding行のsentinelをscatterせずzero出力する。完成windowを連続drainし、full outputをMemTileに保持しない。

初期ABIの制限は `K<=65535`、8列なら `window_count=8/16`、等slice数の連続window、row mapはuint16、1列なら `window_count=1/2`、各列のA block数>0。任意の不均等window境界、uint32 row map、完全zero列の特別経路は未実装で、明示的に拒否する。`M`が8×window数で割り切れない場合はpackerの `padded_rows` をNPU出力長にして末尾zeroを含める。

### 実機検証

ws007 NPU2で新規 `test_sell_spmv.py` の **6条件すべて成功**。1列はidentity/reverseと全zero slice、8列はrandom・8/16 window、さらに `M=1021` の末尾不完全sliceを確認した。行NNZにはzero rowと `p=0/1/2` のsliceが混在する。全条件でCPU CSR参照に対し指定BF16許容誤差内でcanonical outputが一致し、deadlockなし。`M=1024,K=512`、8列の短時間測定では8 window約117 µs、16 window約111 µsだった。これは小さいテスト行列の参考値で、実モデルの速度を予測するものではない。

`M=4096,K=4096`、seed 73、packed A 4,489,216 Bでも、**同一packed Aとx**を使い、24計算core＋8 reorder coreのidentity map（physical出力）・real map（canonical出力）・従来32計算core（physical出力）を比較した。2回の実機測定は順に `209.34/183.88/187.74 µs` と `201.86/181.70/190.59 µs`。24-core canonicalはこの例で32-core physicalと概ね同程度だが、前者にreorder、後者にreorderがなく、短時間計測の揺れもあるため、一般的な優劣は主張しない。同じAを使うidentity-map実験は**sort無効の別packingとの比較ではない**。行ソートによるA転送削減を含む公平な性能比較は後続Stepの評価とする。

`input_with_addresses.mlir` の `M=1024,K=512,8 window` 代表列では、MemTileにA ping-pong `2×4096×BF16=16 KiB`、control `658×int16=1316 B`、joined `2×10×BF16=40 B`、計5 buffer・16 lock。ShimはA/controlのMM2S 2本、outputのS2MM 1本。compute coreのA ping-pongは2行coreで4 KiB、3行coreで6 KiB。各compute coreにconfig `530×int16=1060 B`、出力FIFO 2 object、FP32 state 16 Bを置く。reorder coreはmap 256 B、canonical window 256 B、joined ping-pong 40 B、計4 buffer・6 lock・4 BD、入力S2MM 2 channel/出力MM2S 1 channel。これらは**payload/MLIR配置**であり、stack・ELF・空きbankまで含めた一般的なL1容量保証ではない。

### 再現方法

ws007のIRON checkout rootで、XRTを有効化して実行する。`--iterations 1` はpytestのNPU test反復を1回にする。

```bash
source /opt/xilinx/xrt/setup.sh
export NPU_RUNTIME=xrt
export PYTHONPATH="$PWD:$PYTHONPATH"
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m pytest \
  iron/operators/spmv/test_sell_route.py \
  iron/operators/spmv/test_sell_spmv.py \
  iron/operators/spmv/test_sell_c_sigma.py \
  iron/operators/spmv/test_slice_ell.py -q --iterations 1

/home/hitoshi/ironenv-mlir-v1.4.3/bin/python \
  -m iron.operators.spmv.measure_sell_dedicated --M 4096 --K 4096 --seed 73
```

上記回帰テストは **30 passed**。NPU compileの生成物は `build/<operator-name>.mlir.d/input_with_addresses.mlir` で確認できる。
