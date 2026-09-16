# SpMV 実験・再現手順

このディレクトリには、AMD Ryzen AI NPU 向けの疎行列ベクトル積（SpMV）の実装、入力データ生成、CPU 参照実装、テスト、性能計測コードが含まれています。

この文書は ws007 の `/home/hitoshi/IRON`、NPU2 環境を前提にしています。2026-09-16 時点で、ELL 実装の `1024 x 2048` ケースが NPU 上で正常終了し、CPU 参照値と一致することを確認しています。

## 重要: MLIR-AIE のバージョン

> **この実験で使用・動作確認した MLIR-AIE は `1.1.3` です。**
>
> `requirements.txt` でも `mlir_aie==1.1.3` に固定されています。異なるバージョンでは Python API、MLIR 生成、配置、コンパイル結果が変わる可能性があるため、再現実験では必ず同じバージョンを使用してください。

実際のインストール済みバージョンは次のコマンドで確認できます。

```bash
source /home/hitoshi/IRON/ironenv/bin/activate
python -m pip show mlir-aie
```

期待される出力:

```text
Name: mlir-aie
Version: 1.1.3
```

## 1. 現在確認済みの構成

```text
Git branch : devel
Git commit (Phase 0測定開始時) : fdcddda6c6de64f04831a423b8720f2ce1fd9561
Device     : NPU2
Format     : ELL
M          : 1024
K          : 2048
ELL width  : 256
Tile size  : 2
Core array : 4 x 8
```

実行経路は次のとおりです。

```text
test.py
  -> op.py (AIESPMV、成果物・バッファ・runlist の定義)
  -> design_ell.py (MLIR生成、NPU配置・データ転送)
  -> spmv.cc (AIEコア上の計算カーネル)
  -> reference.py (CPU参照値)
```

## 2. 環境の準備

ws007 にログインし、必ず XRT と IRON の仮想環境を読み込みます。

```bash
ssh ws007
cd /home/hitoshi/IRON/operators/spmv

source /opt/xilinx/xrt/setup.sh
source ../../ironenv/bin/activate

python --version
python -m pytest --version
```

`source /opt/xilinx/xrt/setup.sh` を忘れると、通常は `ModuleNotFoundError: No module named 'pyxrt'` になります。

再現時には Git の状態も記録してください。

```bash
cd /home/hitoshi/IRON
git rev-parse HEAD
git status --short --branch
cd operators/spmv
```

## 3. 入力データの準備

テストはカレントディレクトリから `npu_data/` を探します。したがって、テストは原則として `/home/hitoshi/IRON/operators/spmv` から実行してください。

確認済みケースのデータがない場合は、次のコマンドで生成できます。

```bash
python -c 'from save_sparse_matrix import save; print(save(output_dir="./npu_data", auto_padding=False, use_random=True, rand_m=1024, rand_k=2048, rand_nnz=256))'
```

生成される主なファイルは次のとおりです。

```text
npu_data/random_M1024_K2048_ELL256/
  random_M1024_K2048_ELL256.mtx
  random_M1024_K2048_ELL256_xdna_ell.npy
  random_M1024_K2048_ELL256_ell_meta.json
  random_M1024_K2048_ELL256_xdna_sell32.npy
  random_M1024_K2048_ELL256_sell32_meta.json
```

乱数 seed は `save_sparse_matrix.py` 内で `42` に固定されています。

注意: `python save_sparse_matrix.py --help` はヘルプを表示しません。このスクリプトは CLI 引数を解析せず、そのまま `main()` のデータ生成を開始します。再現実験では上記の `save(...)` 呼び出しを使用してください。

## 4. 単発の動作テスト

まず、意図したケースが pytest に収集されることを確認します。

```bash
python -m pytest --collect-only -q test.py \
  --iterations=1 \
  --csv-output=/tmp/iron_spmv_collect.csv
```

出力に次のようなケース名が含まれていれば、入力データが認識されています。

```text
random_M1024_K2048_ELL256_1024x2048_ellwidth256_tile2_core4x8
```

単発の NPU 動作テストは次のコマンドです。

```bash
python -m pytest -s -q test.py \
  --iterations=1 \
  --csv-output=/tmp/iron_spmv_test_results.csv
```

成功時は最後に `1 passed` と表示されます。レイテンシはマシン状態によって変動します。2026-09-16 の確認では約 113--123 us でした。

`test.py` 内の現在の主要設定は次の箇所です。

```python
design_name = "ell"
tile_size = 2

REGULAR_TEST_CONFIGS = [
    ("random_M1024_K2048_ELL256", tile_size, 4, 8),
]
```

## 5. README の一般コマンドとの違い

リポジトリ直下から次のように実行すると、SpMV 固有の `npu_data` が見つからず、パラメータが `NOTSET` になることがあります。

```bash
cd /home/hitoshi/IRON
python -m pytest operators/spmv/test.py
```

SpMV については、次のようにディレクトリを移動してから実行してください。

```bash
cd /home/hitoshi/IRON/operators/spmv
python -m pytest -s -q test.py --iterations=1
```

## 6. ビルド方法

SpMV には専用 Makefile/CMakeLists.txt はありません。`run_test()` が IRON の Python ビルドシステムを通して自動的にビルドします。

処理順は次のとおりです。

1. `design_<design_name>.py` の `my_matvec()` から MLIR を生成する。
2. Peano の `clang++` で `spmv.cc` を AIE2P 用の `spmv.o` にコンパイルする。
3. `aiecc.py` で MLIR と `spmv.o` から `.xclbin` と NPU instruction `.bin` を生成する。
4. XRT で xclbin を登録し、バッファを確保して runlist を実行する。

概略のカーネルコンパイルコマンドは次の形です。通常は手動実行する必要はありません。

```text
$PEANO/bin/clang++ \
  -O2 -std=c++20 \
  --target=aie2p-none-unknown-elf \
  -I$MLIR_AIE/include \
  -c operators/spmv/spmv.cc \
  -o build/spmv.o
```

生成物は、通常 `operators/spmv/build/` に置かれます。

```text
build/spmv_<M>x<K>_ellwidth<W>_tile<T>_core<R>x<C>.mlir
build/spmv_<M>x<K>_ellwidth<W>_tile<T>_core<R>x<C>.xclbin
build/spmv_<M>x<K>_ellwidth<W>_tile<T>_core<R>x<C>.bin
build/spmv.o
```

既存成果物が依存ソースより新しい場合は再利用されます。

## 7. キャッシュに影響されないクリーンビルド

`AIEContext` は実行時のカレントディレクトリに `build/` を作ります。既存 `operators/spmv/build/` を削除せずにクリーンビルドを確認する場合は、隔離した一時ディレクトリを使用できます。

```bash
mkdir -p /tmp/iron-spmv-clean-run
ln -sfn /home/hitoshi/IRON/operators/spmv/npu_data \
  /tmp/iron-spmv-clean-run/npu_data

cd /tmp/iron-spmv-clean-run
export PYTHONPATH=/home/hitoshi/IRON:${PYTHONPATH}

python -m pytest -s -q \
  /home/hitoshi/IRON/operators/spmv/test.py \
  --iterations=1 \
  --csv-output=/tmp/iron_spmv_clean_results.csv
```

この場合、コンパイル成果物は `/tmp/iron-spmv-clean-run/build/` に生成されます。

## 8. design ファイルを直接 MLIR にする

ws007 は NPU2 です。直接 design スクリプトを実行する場合は `--dev npu2` を明示してください。

ELL:

```bash
python design_ell.py --dev npu2 \
  -M 1024 -K 2048 -ell_width 256 -m 2 \
  --rows 4 --cols 8 \
  -o /tmp/spmv_ell.mlir
```

SELL-32:

```bash
python design_sell32.py --dev npu2 \
  -M 1024 -K 2048 -ell_width 256 -m 1 \
  --rows 4 --cols 8 \
  -o /tmp/spmv_sell32.mlir
```

SELL-32 block:

```bash
python design_sell32_block.py --dev npu2 \
  -M 1024 -K 2048 -ell_width 256 -m 1 \
  --rows 4 --cols 8 \
  -o /tmp/spmv_sell32_block.mlir
```

3種類とも NPU2 向け MLIR の生成までは確認済みです。NPU 上で CPU 参照値との一致まで確認済みなのは、現在は ELL だけです。

`--dev npu` のまま 4 x 8 配置を指定すると、ws007 では次のような配置エラーになります。

```text
Partial Placement Error: Tile Tile(4, 2) not available on device NPU1
```

## 9. design 切り替え時の注意

`op.py` が生成する成果物名には `design_name` が含まれていません。そのため、同じ `M/K/ELL幅/tile/core` のまま `ell`、`sell32`、`sell32_block` を切り替えると、別 design の古い `.mlir/.xclbin/.bin` が再利用される可能性があります。

design 比較時は、design ごとに別の作業ディレクトリと `build/` を使用してください。少なくとも、既存成果物をそのまま使った結果を別 design の結果として扱わないでください。

また、現在の推奨組み合わせは次のとおりです。

```text
design_name="ell"          -> ELL形式の *_xdna_ell.npy
design_name="sell32"       -> SELL-32形式の *_xdna_sell32.npy
design_name="sell32_block" -> SELL-32形式の *_xdna_sell32.npy
```

## 10. `measure.py` について

`measure.py` は単発テストではなく、大規模な性能スイープです。

- 多数の `M x K` を列挙する。
- SRAM 制約から tile size を計算する。
- import/pytest収集時に `save_sparse_matrix.save()` を呼び、入力データを生成する。
- 各ケースを5回測定する。
- warmup 2回と本計測を行う。
- 各ループ間に別の GEMV を実行して NPU 状態を変える。
- 各ループ後に4秒待機する。
- 結果を `spmv_results.csv` に追記する。

したがって、動作確認だけが目的なら `measure.py` は使用せず、`test.py --iterations=1` を使用してください。

`measure.py` を実行する前には、必ず `REGULAR_TEST_CONFIGS` の生成範囲、ディスク容量、実行時間、CSV 出力先を確認してください。単に `--collect-only` した場合でも、モジュール import 時のデータ生成が走るため注意が必要です。

## 11. ファイルの役割

| ファイル | 役割 |
|---|---|
| `test.py` | 単一または少数ケースの end-to-end NPU テスト |
| `measure.py` | 多数ケースの性能測定・CSV出力 |
| `op.py` | `AIESPMV`、ビルド成果物、バッファ、runlist の定義 |
| `design_ell.py` | ELL 用 NPU design / MLIR生成 |
| `design_sell32.py` | SELL-32 用 NPU design / MLIR生成 |
| `design_sell32_block.py` | SELL-32 block 用 NPU design / MLIR生成 |
| `spmv.cc` | AIE2P 計算カーネル |
| `reference.py` | CPU 参照値生成 |
| `save_sparse_matrix.py` | MTX、ELL/SELL-32 NPY、メタデータ生成 |
| `save_ell_npu.py` | ELL データ生成の個別スクリプト |
| `save_sell32_npu.py` | SELL-32 データ生成の個別スクリプト |
| `test_gen.py` | `save_sparse_matrix.save()` の使用例。NPUテストではない |
| `design_mem_*`, `design_trace*` | 実験用 design。現行 `op.py` の標準選択肢ではない |
| `format_fig_*`, `transfer_problem*` | 論文・説明用の図生成 |

## 12. 現時点の制限・測定値の読み方

- `AIESPMV.forward()` は先頭で `NotImplementedError` を送出します。現在動作確認済みなのは、`run_test()` がバッファへ直接書き込み、runlist を実行する経路です。
- `test.py` が表示する Throughput は `(2 * M * K) / time` です。実際の SpMV 演算量 `2 * nnz` ではなく、dense-equivalent の値です。
- `test.py` の bandwidth は実際の転送バイト数に基づく値と、疎行列を dense とみなした参考値の両方を出力します。比較時に混同しないでください。
- PyTorch の Sparse CSR に関する beta warning は、確認済みテストでは失敗原因ではありません。

## 13. 最小チェックリスト

実験ログには最低限、次を残してください。

```text
Git commit
git status の差分有無
device type (NPU2)
design_name
入力フォーマット (ELL / SELL-32)
M, K, ell_width
tile_size
num_core_rows, num_core_cols
テストコマンド
クリーンビルドかキャッシュ再利用か
pytest の PASS/FAIL
latency と bandwidth
CSV出力先
```

## 14. branch と再現環境の記録方針

この作業では Git tag は使わず、各 branch の `operators/spmv/README.md` を再現実験の
正とする。ws007 の現行 branch 名は **`devel`** である（`develop` ではない）。新しい
branch は、必ず前段の通過 commit から作る。

```text
devel
  Phase 0: 現行 toolchain の baseline を commit / push
  └─ spmv/mlir-v1.4.3
       Phase 1: 最新 upstream を clean に構築し、最新 operator 構造へ SpMV を移植。
                ELL, sell32, sell32_block を再現
       └─ spmv/slice-ell
            Phase 2--5: packer/reference, static Slice-ELL, dynamic p_g, 実機評価
```

`spmv/mlir-v1.4.3` では旧 checkout を in-place 更新しない。最新 AMD IRON / mlir-aie を
別 checkout と venv に clean に構築し、upstream の既存 NPU2 operator で環境成立を確認する。
その後、旧 `devel` の SpMV を参照元として、最新 upstream の他 operator の構造に合わせて
`operators/spmv` を組み直す。古い `op.py` や artifact/cache の構造をそのまま複製せず、
まず固定幅 ELL、次に sell32、sell32_block を順に再現する。`spmv/slice-ell` は、その
確認済み commit を親にして継続的に実装する branch とする。実験的な試行が必要な場合だけ、
`spmv/slice-ell` から一時 branch を切る。

### 各 branch で最初に記録する項目

環境を作った直後と、測定結果を出す直前に次を README の「現在確認済みの構成」へ実値で
追記する。branch 名から toolchain の正確な build 番号を推測してはならない。

```bash
cd /path/to/IRON
git branch --show-current
git rev-parse HEAD
git status --short --branch

source /opt/xilinx/xrt/setup.sh
source /path/to/venv/bin/activate
python --version
python -m pip show mlir-aie
python -m pip show iron
which aiecc.py
```

README には少なくとも次を残す。

```text
IRON checkout / commit
mlir-aie version（wheel の完全な Version）
llvm-aie / Peano version または commit
XRT version と setup.sh の path
Python / virtual environment path
device type と実行 host
design_name、input format、M/K/ELL width/h/B/R
build command、test command、測定 command
clean build か artifact cache 再利用か
CPU reference の PASS/FAIL と許容誤差
```

Phase 0 では既存の `devel` に、baseline の再現手順・既知の測定結果・この README を
commit/push する。`npu_data/`、`build/`、xclbin、trace、CSV などの生成物は commit
しない。Phase 1 と Phase 2 以降では、環境または format が変わるごとに同じ README の
該当節を更新し、実際に実行したコマンドを残す。

## 15. Phase 0 baseline 実測（2026-09-16、`devel`）

Phase 1 の移植前比較用として、ws007 の NPU2 で既存実装を隔離した clean build
directory から実行した。測定開始時のIRON commitは
`fdcddda6c6de64f04831a423b8720f2ce1fd9561`、branchは`devel`である。環境は
Python 3.12.3、`mlir-aie==1.1.3`、`/opt/xilinx/xrt/setup.sh`、
`/home/hitoshi/IRON/ironenv`である。`iron` は独立したpip packageではなく、この環境では
`mlir-aie`が提供する`aie.iron` APIを使用する。

```text
aiecc.py  : 19cce64c85c3f22bf4908819b1038a5c0ae42f52
llvm-aie  : clang 20.0.0, ae321ba3819b3575d63c3173993104fe532d692c
XRT       : 2.21.0
device    : AMD Ryzen AI 9 HX 370 / NPU Strix
NPU FW    : 1.1.2.64
```
実行時には必ず次のように、XRT が設定した Python path を残す。

```bash
source /opt/xilinx/xrt/setup.sh
source /home/hitoshi/IRON/ironenv/bin/activate
export PYTHONPATH=/home/hitoshi/IRON:${PYTHONPATH}
```

以下だけを正式な性能baselineとする。各shapeは固定seed random matrixで、全行が
`K/8` 個のnnzを持つため、**行列密度は厳密に12.5%**、`ell_width=K/8`である。同じ論理行列を
ELLとSELL-32 blockのそれぞれのlayoutへpackし、4 core-row × 8 core-column、すなわち
**全32 core**を使った。ELLはL1に収まる最大の安全な`m`を、SELL-32 blockは実装上固定の`m=1`を
使う。

各行はdesignごと・shapeごとに新しいempty build directoryからcompileした。そのため、
`op.py`の成果物名が`design_name`を含まない既知のcache collisionは起きない。

性能値は **`measure.py` と同一の方法**で5回採取した。各sampleでは入力をdeviceへ書いた後、
1回目を初期化、2回目をwarm-up、3回目の`run_runlist()`戻り値を測定値とする。この戻り値は
`xrt_runlist.execute()`から`wait()`まで、すなわちdevice executionのみであり、host側のBO同期
（host→NPU DMA）を含まない。sample間には1024要素GEMVを実行して4秒待機した。全ケースは
CPU referenceと`rel_tol=0.04`、`abs_tol=1e-4`で全要素一致した。

| `M × K` | ELL width | kernel | `m` | mean | min | max | std. dev. | effective BW | 結果 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `4096 × 4096` | 512 | ELL | 8 | 265.13 us | 254.74 us | 270.36 us | 5.49 us | 31.702 GB/s | PASS |
| `4096 × 4096` | 512 | SELL-32 block | 1 | 261.96 us | 256.28 us | 269.46 us | 4.43 us | 32.085 GB/s | PASS |
| `4096 × 11008` | 1376 | ELL | 2 | 517.97 us | 510.97 us | 523.97 us | 4.45 us | 43.583 GB/s | PASS |
| `4096 × 11008` | 1376 | SELL-32 block | 1 | 520.63 us | 500.48 us | 529.08 us | 10.38 us | 43.360 GB/s | PASS |
| `28672 × 8192` | 1024 | ELL | 4 | 2207.14 us | 2203.54 us | 2217.13 us | 5.09 us | 53.243 GB/s | PASS |
| `28672 × 8192` | 1024 | SELL-32 block | 1 | 2182.89 us | 2175.14 us | 2190.12 us | 6.19 us | 53.834 GB/s | PASS |

したがって、過去の`measure.py`の`random_M28672_K8192_ELL1024`（約2161--2179 us）と
今回のELL 2207.14 usは同じ定義であり、差は約1--2%である。一方、以前ここに記録した
4115.7 usは、外側で時刻を取ったため約117 MiBの入力BO同期を含むhost-inclusive値だった。
これはend-to-end診断には使えるが、`measure.py`の性能値や上表との比較には使わない。

この限定したuniform ELL inputでは、4096×4096はSELL-32 block、4096×11008はELL、
28672×8192はSELL-32 blockがわずかに速い。この差だけからformat一般の優劣を結論付けない。
実modelのunstructured pruning後の不均一な行長、またはSlice-ELLの結果とは別に扱う。

traceはユーザ判断によりこのPhase 0の要件から外す。DMA/FIFO traceは採取しない。

再実行用runnerと今回のJSON・logは、gitignore対象の
`npu_data/phase0_devel_2026-09-16/`へ`measure_*`として保存する。`rerun_*`は上記の
host-inclusive診断値であり、baselineには使わない。xclbinとMLIRは各一時clean build directoryに
のみ置き、いずれの生成物もcommitしない。commit前には
`git status --short --branch`が意図したREADME変更だけであることを確認する。
