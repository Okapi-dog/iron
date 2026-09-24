# Llama-2-7B pruning weight: 高さ6の形式比較

2026-09-24、ws007のNPU2で、同じpruning済みLlama-2-7B checkpointの4行列を比較した。**追加のpruningはしていない**。`B_h=6`, `B_w=256`, 8 column、SELL-C-σは8 window・等行数境界・各列 `(2,2,2)` の3計算core＋1並び替え専用core（方式A）。行順維持Slice-ELLも比較のため3計算core×2行へ対応させた。Denseは既存の32-core K-tiled GEMV、ELLの`B_h`は該当しない。

## 容量

分母は元行列のDense BF16 `2MK` byte。ELL/Slice-ELLは`BF16値+uint16列index`を計上し、ELLの幅は32の倍数へ切り上げる。SELL-C-σのみ`packed A + row map`を計上する。全形式で`x`、control、出力、FIFOのping-pong複製、xclbinは容量比に含めない。行列本体をDRAMに保存する際の比較であって、実機転送byte数の完全な比較ではない。

| Weight | Shape | 密度 | ELL幅 | ELL / Dense | Slice-ELL / Dense | SELL-C-σ / Dense | Dense / ELL | Dense / Slice-ELL | Dense / SELL-C-σ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L3 `o_proj` | 4096×4096 | 10.00% | 2720 | 132.81% | 26.81% | 25.66% | 0.75× | 3.73× | 3.90× |
| L0 `gate_proj` | 11008×4096 | 10.00% | 3712 | 181.25% | 49.23% | 26.31% | 0.55× | 2.03× | 3.80× |
| L0 `down_proj` | 4096×11008 | 10.00% | 6464 | 117.44% | 30.59% | 22.53% | 0.85× | 3.27× | 4.44× |
| L25 `down_proj` | 4096×11008 | 10.00% | 1280 | 23.26% | 23.27% | 23.11% | 4.30× | 4.30× | 4.33× |

高さ8の旧表と違い、高さ6では行順維持Slice-ELLにもpaddingや端数列の影響が出る。特にL0 `gate_proj`ではSELL-C-σの行並び替えで49.23%→26.31%へ下がる。一方、ほぼ均一なL25 `down_proj`ではrow mapを足しても差は0.16ポイントで、reorderを実行するだけの容量上の利益は小さい。

## NPU実測

同じweightから1つのCSRとBF16入力ベクトルを作り、各形式のcanonical行順出力をCPU CSR参照と照合してから採用した。8 column同時実行、別kernelダミーなし、warmup 2回＋timed 5回の平均µs。Dense、Slice-ELL、SELL-C-σを**同じ測定run内で**行列ごとに順番に測った。SELL-C-σの時間には並び替えcoreも含む。ELLは下記の実装制約により、容量のみであり、この時間表には載せない。

| Weight | Dense実測 | Slice-ELL実測 | SELL-C-σ実測 | Dense / ELL | Dense / Slice-ELL | Dense / SELL-C-σ | Slice-ELL / SELL-C-σ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L3 `o_proj` | 753.9 µs | 273.9 µs | 274.6 µs | — | 2.75× | 2.75× | 1.00× |
| L0 `gate_proj` | 1867.9 µs | 912.6 µs | 543.5 µs | — | 2.05× | 3.44× | 1.68× |
| L0 `down_proj` | 1878.0 µs | 619.3 µs | 482.8 µs | — | 3.03× | 3.89× | 1.28× |
| L25 `down_proj` | 1857.6 µs | 469.7 µs | 470.6 µs | — | 3.95× | 3.95× | 1.00× |

速度比は左側の実測時間を右側の実測時間で割った値。ここでいう通常の「SELL」は行順維持Slice-ELLを指す。`Dense / ELL`の`—`は、4行列で比較可能なELL実機測定値が揃っていないためであり、容量比から速度比を推定していない。

L3とL25のSlice-ELL/SELL差は5 sample中の変動より小さく、優劣は未確定。特にL3 SELLのsampleは約234–286 µsに散っている。L0 gate/downでの大差は容量の変化と整合するが、容量だけを原因と断定はしない。異なる設計では計算core数、reorder、制御転送が違う。速度比は今回の連続実行条件であり、別xclbin間の切り替え時間を含まない。

### ELLの実機比較が欠ける理由

既存`SpMVELL`は4 core row×8 column。L3 `o_proj`は1 core 2行で出力一致・計測（896.2 µs）したが、L0 `gate_proj`の幅3712では同じ設定のL1 allocationが失敗した。1 core 1行へ下げると、今度は出力ObjectFIFOがBF16 1要素＝2 Bとなり、NPU DMAの4 B転送アラインメント制約でコンパイル失敗した。L0 `down_proj`など残り2行列について、ELLの実機時間は測っていない。容量表のELL数値はCPU側の正確なpack容量だが、**4行列共通のELL実機ベースラインの数値ではない**。これを埋めるには、1行ずつAを読みつつ2行分の出力を4 Bでまとめる等、別のELL設計と検証が必要になる。本比較のためだけに未検証の新kernelを作って速度表へ混ぜない。

## 再現情報

- branch: `spmv/sell-c-sigma-h6-comparison`。モデル: `/home/hitoshi/elsa/pruned_model/Llama-2-7b-hf_pruned0.9_admm_lr5e-05_20260301_2016`。
- 実測: [`h6_comparison_results/sell-h6-three-running-formats-20260924.jsonl`](h6_comparison_results/sell-h6-three-running-formats-20260924.jsonl)。容量: [`h6_comparison_results/sell-h6-storage-20260924.jsonl`](h6_comparison_results/sell-h6-storage-20260924.jsonl)。個々のNNZ、hash、転送byte数、列block負荷、timed sampleはJSONLを参照。
- 既存`measure_sell_step5.py`に高さ6の行順維持Slice-ELL経路を追加した。`SpMVSliceELLDynamicScalarMultiCol`は`core_rows=3`、packerとruntime configも同じ設定を使う。既存の高さ8・4core-row経路はデフォルトのまま。
- ws007: `/home/hitoshi/ironenv-mlir-v1.4.3/bin/python`、XRT setup `/opt/xilinx/xrt/setup.sh`。`safetensors`は既存ELSA venvのsite-packagesを`PYTHONPATH`に足した。

```bash
source /opt/xilinx/xrt/setup.sh
export PYTHONPATH="$PWD:/home/hitoshi/elsa/elsa_venv/lib/python3.12/site-packages:$PYTHONPATH"
MODEL=/home/hitoshi/elsa/pruned_model/Llama-2-7b-hf_pruned0.9_admm_lr5e-05_20260301_2016
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m iron.operators.spmv.evaluate_step0 \
  --model-dir "$MODEL" --format dense --format ell --format slice_ell \
  --format sell_c_sigma --block-height 6 --block-width 256 --columns 8 \
  --windows 8 --boundary equal_rows --assignment contiguous --output jsonl
/home/hitoshi/ironenv-mlir-v1.4.3/bin/python -m iron.operators.spmv.measure_sell_step5 \
  "$MODEL" --design dense_k_tiled --design slice_ell \
  --design sell_dedicated_reorder --block-height 6 --windows 8
```

実機検証: 4行列×3実行方式の全12ケースがCPU出力と一致。追加の高さ6・末尾padding付きmicrotestは1列/8列×pytest 5 iterations＝10件合格。CPU側の既存容量評価50件と、従来高さ8・4 core-row実機テスト5件も合格。ELLの2種類の失敗は上記に記録し、速度表に架空値を入れていない。
