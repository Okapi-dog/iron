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
Git commit : 7c210b788df7c000aeda4322d1beac93498749fb
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
