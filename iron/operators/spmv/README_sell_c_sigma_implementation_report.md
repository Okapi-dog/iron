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

## Step 3 追補 — 計算coreの行分割を設定化

現行の8行slice `(2,3,3)` はデフォルトのまま維持した。`SELLCoreLayout.rows_per_core` を3計算coreの唯一の行数設定とし、`B_h=sum(rows_per_core)`、AのMemTile split offset、各coreのA FIFO型、出力FIFOの転送スロット数、MemTile join offset、reorder scatterの実データ位置を導出する。出力スロット数は各coreについて `rows + rows % 2` BF16。3行なら4要素転送し、末尾1要素をdummyとする。**計算する行数まで4になるわけではない**。MemTileはjoined objectを2個持ち、reorder coreはdummyを読み飛ばす。現行8行ではjoinedは `2+4+4=10` BF16（20 B/object、ping-pong 40 B）で、実出力8行に対するdummyは2要素。

`(1,1,1)→B_h=3`、`(2,2,2)→6`、`(2,3,3)→8`、`(3,3,3)→9`、`(4,4,4)→12` を設定可能にした。6/12行はcore間の計算行数が同じで、出力の転送paddingもない。9行では各coreに1 dummyが必要。奇数 `B_h` のwindowはBF16出力とuint16 row mapの4-byte DMA長を保つため、packerが各windowのslice数を偶数に丸める。末尾の追加行はsentinel mapとzero出力で処理する。この丸めはwindowed/奇数 `B_h` の場合だけで、以前の8行形式は変更しない。

ws007実機でデフォルト8行、8行の別分割 `(3,2,3)`、均等な3/6/9/12行、9行の16 windowをCPU CSR参照と照合した。関連する `test_evaluation.py` も含めた最終回帰は **47 passed、1 skipped**。以下は同じseed 73の合成CSR `4096×4096`、8列・8 window、canonical出力の一度の短時間測定。形式ごとにpacked Aとpadding行数が違うため、**行分割だけの純粋な速度差ではない**。現行8行の計測が以前の約182–184 µsに対して178 µsで、大きな性能後退は見えなかった。

| 行数/core | `B_h` | padded M | packed A bytes | canonical latency |
|---|---:|---:|---:|---:|
| 2/3/3 | 8 | 4096 | 4,489,216 | 178.35 µs |
| 2/2/2 | 6 | 4128 | 4,491,264 | 172.46 µs |
| 3/3/3 | 9 | 4176 | 4,497,408 | 173.86 µs |
| 4/4/4 | 12 | 4128 | 4,521,984 | 180.23 µs |

同じ行列・seedで繰り返し測定し、packed A容量と転送量も併記してからデフォルト値を選ぶ。現時点では8行を変えず、均等分割は選択肢として提供する。

```bash
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python \
  -m iron.operators.spmv.measure_sell_dedicated \
  --M 4096 --K 4096 --seed 73 --rows-per-core 2 2 2
```

## Step 4 — 4計算coreを時間再利用する方式B

### 実装前の仕様確認と設計変更

元計画の「4 coreが計算した全windowのphysical `y'`をMemTileに集め、1 coreへ再入力」は、再利用coreに **A、control、`y'` の3本の静的入力**を要求する。AIE-ML v2 compute tileのDMAはS2MM入力2 channel、MM2S出力2 channelであり、現在のObjectFIFO loweringでもcore入力stream channelはFIFOごとに静的割当される。`TaskGroup.finish()`でruntime taskを終えてもこの配線・L1割当は切り替わらない。したがって、現行の高水準FIFOだけで元計画をそのまま構築するのは無理がある。低水準BDの再設定を先に導入する前に、同じ目的を2入力のまま達成できる隣接L1共有を選んだ。参照: [AMD AIE-ML v2 Memory Module](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/Memory-Module)、[Tile Architecture](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/AIE-ML-v2-Tile-Architecture)、mlir-aie v1.4.3の`aie.iron.Buffer`/`Lock`実装と`AIEObjectFifoStatefulTransform.cpp`。

各列row 2/3/4/5が8行sliceを2行ずつ担当する。row 2/3はrow 3のL1に置いた前半window bufferの互いに異なる位置へ、row 4/5はrow 4のL1に置いた後半bufferへ書く。row 4は自分の2行の計算後に、他の3 coreの完了を明示`Lock`で待ち、隣接するrow 3の前半と自身の後半を読み、combined control内のwindow-local row mapで正規順のoutput FIFOへscatterする。**同じrow 4 worker/ELFが計算と並び替えを順に行い、PDI/ELF再load・中間DRAM出力・追加のFIFO入力はない**。4 coreへのcontrol broadcastが次windowへの進行をゲートするため、共有bufferを並び替え中に上書きしない。`p=0`、zero row、末尾sentinelは既存のA/control契約のまま処理する。

`sell_c_sigma_design.py`に`SELLTimeMultiplex`用デザイン、`sell_c_sigma.cc`に2行の共有bufferへのfinalizeとwindow単位scatterを追加した。Step 1の**同じpack済みA/row map**を`make_window_inputs()`から方式A/Bへ渡せる。`prepare_sell_design()`は`DesignSpec("sell_dedicated_reorder")`または`DesignSpec("sell_time_multiplex_reorder")`を指定してoperator/inputを切り替える共通入口で、未対応名は拒否する。測定スクリプトも`--include-time-multiplex`で同じCSR・A・xのまま両方式を選べる。

### 実機検証と資源配置

ws007、NPU2、mlir-aie v1.4.3で新規`test_sell_time_multiplex.py`の **6条件すべて成功**: 1列identity/reverse/zero slice、8列8/16 windowのrandom、末尾padding行。出力はCPU CSR参照と指定BF16許容誤差内で一致し、deadlockしなかった。Step 0–3の既存経路を合わせた回帰は **48 passed、1 skipped**。`M=28672,K=8192`でも8/16 windowのcompile・実行・CPU一致を確認した。

`M=28672,K=8192,8 window`の`input_with_addresses.mlir`における代表列のrow 4には、control `12226×int16=24452 B`、A ping-pong `2×1024×BF16=4096 B`、canonical output `3584×BF16=7168 B`、後半共有buffer `1792×BF16=3584 B`、FP32 state 16 Bが配置された。前半buffer `1792×BF16=3584 B`は隣のrow 3に配置。row 4の最大明示buffer末端アドレスは52736 Bで64 KiB内。これは生成MLIRの割当であり、任意形状のfit保証ではない。代表列のShimはA/controlにMM2S 0/1、outputにS2MM 0を使用。row 4はA/controlのS2MM 2 channel、outputのMM2S 1 channelで、3入力問題は生じていない。3つの明示lockも生成MLIRで確認した。

### 同一packed Aでの速度比較

seed 73の**合成CSR**（行NNZは`[4,300)`、13行ごとにzero）を一度packし、8列・8 window・BF16、warmup 2回＋timed 5回で計測。以下はcanonical出力の1回の短時間測定で、実モデルの行NNZ分布や密度とは異なる。方式Aは24計算core＋8専用並び替えcore、方式Bは32計算coreのうち各列1 coreをwindow後半で再利用する。両者は同じA、x、row map、CPU参照を用いた。

| M×K | packed A bytes | 方式A canonical | 方式B canonical | B/A |
|---|---:|---:|---:|---:|
| 4096×4096 | 4,489,216 | 183.63 µs | 187.54 µs | 1.02× |
| 11008×4096 | 11,960,320 | 311.14 µs | 334.98 µs | 1.08× |
| 4096×11008 | 4,489,216 | 187.55 µs | 188.65 µs | 1.01× |
| 28672×8192 | 31,014,912 | 666.36 µs | 737.79 µs | 1.11× |

`28672×8192,16 window`でも方式A 664.51 µs、方式B 734.92 µs。同じAを32計算coreだけでphysical `y'`として出す参考経路は、8 windowの`28672×8192`で649.43 µsだった。時分割方式Bは**実現可能だが、この測定では高速化しない**。4 core計算の利益だけでは、full-window共有bufferへの書込み、4 core分のcontrol配布、完了同期、window単位scatterの追加を相殺できない可能性がある。個々の原因の切り分けにはtraceや単独コスト測定が必要で、現時点で断定しない。2形状を同時に走らせた測定値は競合の恐れがあるため表には採用せず、単独再測定値を掲載した。

### 再現方法と次段階への判断

```bash
source /opt/xilinx/xrt/setup.sh
export NPU_RUNTIME=xrt
export PYTHONPATH="$PWD:$PYTHONPATH"
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m pytest \
  iron/operators/spmv/test_sell_route.py \
  iron/operators/spmv/test_sell_spmv.py \
  iron/operators/spmv/test_sell_time_multiplex.py \
  iron/operators/spmv/test_sell_c_sigma.py \
  iron/operators/spmv/test_evaluation.py -q --iterations 1

/home/hitoshi/ironenv-mlir-v1.4.3/bin/python \
  -m iron.operators.spmv.measure_sell_dedicated \
  --M 28672 --K 8192 --seed 73 --include-time-multiplex
```

実pruning weightでの比較は次のStep 5で行う。現時点では方式Aを性能上の第一候補、方式Bを成立済みの比較対象として保持する。方式Bの適用可能域は`B_h=8, B_w=256`、均等8/16 window、各columnにA blockあり、window内row mapがuint16である場合。Kやwindow行数をさらに増やす場合は、operatorの入力検査に加えて生成MLIRのL1配置を毎回確認する。

## Step 4.5 — 別32-core GEMVを挟むカーネル切替測定

### 先に確認した測定上の相違

旧`iron-devel-4/operators/spmv/measure.py`は、各ループで対象を2回続けて実行し、**その後**にダミーGEMVを1回実行していた。旧表の`Mean(us)`は対象の2回目であり、ダミー直後の1回目は別欄の`W1_Mean(us)`だった。さらに旧`GEMV(M=1024,K=1024,num_aie_columns=8,tile_size_input=2)`のdesignは1列に1 worker、計8 workerであり、32 coreダミーではない。従って旧`Mean`や前Stepの連続実行値を「別32-coreカーネルから切り替えたレイテンシ」と解釈してはいけない。

現行の`run_test()`はoperatorをcompile/loadしたあと、**warmup 2回＋timed 5回を同じcallableで連続実行**する。ws007のmlir-aie v1.4.3 XRT runtimeでは、`result.npu_time`は`kernel(...)` submit直前から`wait()`終了までのhost clock差である。BOの`to("npu")`と`DefaultNPURuntime.load()`はこの時間の外。名称は`npu_time`だが純粋なcore計算サイクルではなく、実行時のコンテキスト切替なども含み得る。[MLIR-AIE runtime cache説明](https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/iron_configuration.md)、[XRT native run/wait API](https://xilinx.github.io/XRT/master/html/xrt_native_apis.html)も参照。

### 計測プロトコル

新規`measure_sell_kernel_switch.py`は、同一SELL packed A・seed 73・8列/8 windowを用い、方式A（24計算core＋reorder）、方式B（32計算core時分割）、既存32-core Slice-ELL physicalを選択可能にした。**標準では対象だけを連続測定し、ダミーGEMVのcompile/load/実行は一切行わない**。`--with-dummy`を指定した場合だけ連続・交互の両条件を測る。ダミーは**別xclbin**の`DenseGEMVKTile(M=64,K=4096,cols=8,k_tile=4096)`で、8列×各4 worker＝32計算core。ダミーのAとxは毎回別seedのデータ。対象側も毎回別xをcontrolに埋め込むが、Aと7個の対象xは連続・交互条件で同じものを再利用する。各target/dummyの出力はCPU参照と照合した。

`--with-dummy`指定時には両operatorを測定前にcompile/loadし、次の順で実行する。標準実行は「連続」の行のみ。

```text
連続: [対象 × 2 warmup] → [対象 × 5 timed]
交互: [(ダミー → 対象) × 2 warmup] → [(ダミー → 対象) × 5 timed]
```

記録する対象時間は各`target(*args).npu_time`のみ。**ダミーの約2.5 msは表の対象時間に足していない**。`--reverse-order`では交互→連続の順で繰り返せる。コンパイル、初回load、ホストでのpacking、CPU参照生成、BO syncは計時外。これは「事前load済みの異なるxclbinコンテキストを交互にdispatchする」測定であり、初回load時間やLLM全体のend-to-end時間を示すものではない。

### ws007/NPU2 実測結果

seed 73のStep 4と同じ**合成CSR**。単位はµs、各欄は本計測5 sampleの`npu_time`平均。`差`は交互−連続であり、そのまま純粋なPDI転送時間と断定はしない。target入力は条件間で完全に同一、ダミーの実行時間は除外。1ケースずつ直列に測定した。

| 対象 | M×K | 連続 | 毎回32-core GEMV後 | 差 |
|---|---:|---:|---:|---:|
| SELL方式A canonical | 4096×4096 | 175.30 | 2580.80 | 2405.50 |
| SELL方式A canonical | 11008×4096 | 310.21 | 2712.19 | 2401.98 |
| SELL方式A canonical | 4096×11008 | 193.80 | 2582.04 | 2388.24 |
| SELL方式A canonical | 28672×8192 | 665.95 | 3065.32 | 2399.37 |
| SELL方式B canonical | 4096×4096 | 182.04 | 2071.81 | 1889.78 |
| SELL方式B canonical | 11008×4096 | 334.38 | 2224.16 | 1889.78 |
| SELL方式B canonical | 4096×11008 | 184.22 | 2075.59 | 1891.37 |
| SELL方式B canonical | 28672×8192 | 726.72 | 2617.06 | 1890.34 |
| 32-core Slice-ELL physical | 4096×4096 | 174.47 | 2584.03 | 2409.57 |
| 32-core Slice-ELL physical | 11008×4096 | 302.92 | 2715.23 | 2412.31 |
| 32-core Slice-ELL physical | 4096×11008 | 170.93 | 2582.44 | 2411.52 |
| 32-core Slice-ELL physical | 28672×8192 | 642.43 | 3057.27 | 2414.84 |

交互測定中、対象の直後に実行するダミー自身の`npu_time`は概ね2494–2519 µs（表から除外）。`4096×4096`の3方式について測定順を交互→連続に逆転しても、交互の平均は方式A 2599.71 µs、方式B 2084.06 µs、Slice-ELL 2597.31 µsで、上表と同じ傾向だった。連続値には通常のばらつきがあり、特に方式Aの`4096×11008`は5 sample中2件が約218 µs、残り3件が約174–180 µsで平均が上振れした。交互条件の差は方式A/従来Slice-ELLで約2.4 ms、方式Bで約1.89 msと形状にほぼ依存せず、計算時間の増加より切替に関連する固定費が支配的と推測する。ただし、どの内部PDI操作に何µs使ったかはこのベンチマークだけでは分解できない。

### 再現方法と評価上の扱い

```bash
source /opt/xilinx/xrt/setup.sh
export NPU_RUNTIME=xrt
export PYTHONPATH="$PWD:$PYTHONPATH"
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m pytest \
  iron/operators/spmv/test_measure_sell_kernel_switch.py -q --iterations 1

/home/hitoshi/ironenv-mlir-v1.4.3/bin/python \
  -m iron.operators.spmv.measure_sell_kernel_switch \
  --shape 4096 4096 --design dedicated
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python \
  -m iron.operators.spmv.measure_sell_kernel_switch \
  --shape 4096 4096 --design dedicated --with-dummy
# --design time_multiplex / slice_ell_physical も選択可能
# --with-dummy --reverse-order で交互→連続の順に変更
```

スクリプトは平均だけでなく5個の生sample、中央値、ホストcall全体の時間、除外したダミー時間をJSONで出す。今後のSELL/Dense比較では、**連続dispatchと交互dispatchを別指標**として同じ行列・同じ実行条件で並記する。今回のダミーは小さなdense GEMVであり、実際のLlama layer間で使われる具体的なoperator列を模擬したわけではない。また別xclbin/別hw_contextの交互dispatchであって、複数operatorを単一ELF・同一contextへまとめた推論実装にも同じ約2 msが必ず掛かると主張するものではない。Step 4.5の表は合成行列で、実pruning weightのStep 5評価を置き換えない。
