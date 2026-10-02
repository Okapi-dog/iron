# 論文用再測定（2026-10-01、ws007）

## 測定条件

- NPU: AMD Ryzen AI 9 HX 370 の NPU Strix、8 columns、Power Mode `Turbo`。
- 測定前後に `xrt-smi examine -r platform` で `Turbo` を確認。XDNA の clock query では MP-NPU Clock 1,267 MHz、H Clock 1,800 MHzを確認。ただし約200 µsの個々の実行中に連続サンプリングして「常時固定」を証明したわけではない。
- XRT 2.21.0、amdxdna `2.21.0_20260303`、NPU firmware `1.1.2.64`、Python 3.12.3、mlir-aie `1.4.3.dev85+gdf48abc`。
- 各行列・方式につき **5回連続warmup → 5回連続計測**。計測間・ケース間とも明示的なsleepなし。5個の `result.npu_time` の算術平均をそのケースのレイテンシとする。プロトコルIDは `w5_t5_idle0s_between0s`。
- 形式と構成: Dense K-tiled、縦方向ELL、Blocked Slice-ELL、専用reorder core付きBlocked SELL-C-σ。Slice/SELLは `B_h=6, B_w=256, 8 columns`、SELLは `window_count=8`。方式ごとのカーネルやコア配分も違うので、形式だけを変更した比較とは呼ばない。
- 実行コードはブランチ `spmv/sell-c-sigma-h6-comparison` の **未コミット変更を含む** ws007 checkout。JSONLの `environment.git_commit=21e2da0c...` はベースコミットであって、実行コード全体の版を単独では表さない。下のハッシュも保存しておく。旧 `paper_evaluation/` と `paper_evaluation2/` の生データは変更していない。

## 入力と実行コマンド

実重みは `/home/hitoshi/elsa/pruned_model/Llama-2-7b-hf_pruned0.9_admm_lr5e-05_20260301_2016` からCV分位点に対応する5行列を読み込む。合成行列は `paper_config.py` の11条件×seed `1000..1009`。同じCSRとBF16 `x` を4方式で共有する。XRTを初期化し、`/home/hitoshi/IRON` から次の順で実行した。

```bash
source /opt/xilinx/xrt/setup.sh
cd /home/hitoshi/IRON
/home/hitoshi/IRON/ironenv/bin/python3 -m iron.operators.spmv.measure_paper real /home/hitoshi/elsa/pruned_model/Llama-2-7b-hf_pruned0.9_admm_lr5e-05_20260301_2016
/home/hitoshi/IRON/ironenv/bin/python3 -m iron.operators.spmv.measure_paper synthetic
/home/hitoshi/IRON/ironenv/bin/python3 -m iron.operators.spmv.plot_paper_results
```

`measure_paper.py` と `plot_paper_results.py` の既定入出力は、この `paper_evaluation3/` に設定済み。JSONLは1行が1行列×1方式で、5回の生時間、入力・packed Aのhash、数値照合結果、環境を含む。`figures/` にペア済みCSV・SVG/PNG・要約を生成した。

## 完了状況と品質確認

| データ | 行列数 | 方式×行列のレコード | 結果 |
| --- | ---: | ---: | --- |
| 実重み | 5 | 20/20 | `ok` 19件、`ok_with_tolerance_exceptions` 1件 |
| 合成 | 110 | 440/440 | 全件 `ok` |

数値許容例は `model.layers.18.self_attn.k_proj.weight` の ELLで1要素。全460組が同一プロトコルで、各組に5個の時間がある。`max(sample)/median(sample) > 2` に該当する組は0。全組を通した最大比は約1.256。ただしこの事前規則に抵触しない通常の計測揺れまで無いという意味ではない。数値許容例は厳密一致と扱わず、JSONLの `cpu_error_count` を残す。

### 実重みのレイテンシ（µs、各5回平均）

| Weight | Dense | ELL | Slice-ELL | SELL-C-σ | Dense / SELL | Slice / SELL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L19 `down_proj` | 1826.85 | 769.63 | 432.63 | 430.76 | 4.241× | 1.004× |
| L5 `o_proj` | 717.38 | 749.63 | 214.85 | 210.60 | 3.406× | 1.020× |
| L11 `up_proj` | 1835.06 | 1403.58 | 681.04 | 479.69 | 3.826× | 1.420× |
| L18 `k_proj` | 717.79 | 803.01 | 276.19 | 234.12 | 3.066× | 1.180× |
| L0 `v_proj` | 717.61 | 1024.02 | 456.92 | 320.43 | 2.240× | 1.426× |

L19とL5のSlice対SELL差は小さく、1回の5-sample平均だけで統計的な優劣が確定したとはしない。容量比、全生時間、合成スイープの各seed値は `figures/real_paired.csv` と `figures/synthetic_paired.csv` を参照する。5実重みの速度比の平均をモデル全体の推論高速化と解釈しない。

## ファイルと整合性

| ファイル | レコード/行数 | SHA-256 |
| --- | ---: | --- |
| `real_weights_w5_t5_idle0s_between0s.jsonl` | 20 | `6eacb04ed016f6e84990732a8c16d95f9a6600bbe98d8d962ebd1e4e747ccef1` |
| `synthetic_paper_w5_t5_idle0s_between0s.jsonl` | 440 | `c70cee0f9452da6c521196c28aaf90d4c56d2c25b7d616ed5d6ce3052af983dc` |
| `figures/real_paired.csv` | 20 data rows | — |
| `figures/synthetic_paired.csv` | 440 data rows | — |

実行時の主要コードSHA-256: `paper_config.py` = `677bebcef2dd48fc1312797cd242cf9b32594556e171990f6d38082b2d7799ca`、`measure_paper.py` = `f1cf925ec666267d8e19013630ffb3ff3cf2a2c0bc65e96d55dfc6836de14d84`、`matrix_measure.py` = `60423f9e768d353fe72b2260e5a421fe9bfd94afda3d65bcbe38aaa5d0d22807`。論文提出前には変更をコミットして、版の参照を簡潔にするのが望ましい。
