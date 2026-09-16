# 行順を保存する Slice-ELL の設計探索と DMA 制約

このノートは、unstructured pruning 後の SpMV を対象に、元の行順を
維持する Slice-ELL を設計・探索するためのものです。SELL-C-σ のような
行ソートや、出力ベクトルの permutation は行いません。そのため、
ObjectFIFO の join と Shim DMA による固定順・固定長の出力経路と両立します。

## branch と再現環境の記録方針

この作業では Git tag は使わず、各 branch の `operators/spmv/README.md` とこの設計ノートを
再現実験の正とする。ws007 の現行 baseline branch 名は **`devel`** である。
新しい branch は必ず前段の通過 commit から作る。

```text
devel
  Phase 0: 現行 toolchain の baseline を commit / push
  └─ spmv/mlir-v1.4.3
       Phase 1: 最新 IRON / mlir-aie への移植。ELL, sell32, sell32_block を再現
       └─ spmv/slice-ell
            Phase 2--5: packer/reference, static Slice-ELL, dynamic slice_blocks, 実機評価
```

`spmv/mlir-v1.4.3` には Slice-ELL を混ぜず、固定幅 ELL / SELL が最新環境で再現することを
通過条件にする。`spmv/slice-ell` はその通過 commit を親にし、Slice-ELL の実装と評価を
継続する branch とする。実験を分ける必要がある場合だけ、後者から一時 branch を切る。

環境を作った直後と測定直前に、README へ次の実値を残す。branch 名から正確な toolchain
build 番号を推測してはならない。

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

記録対象は、IRON checkout/commit、mlir-aie wheel の完全な version、llvm-aie/Peano の
version または commit、XRT と venv の path、device/host、design/format/形状/parameter、
build・test・測定 command、clean build の有無、CPU reference の結果と許容誤差である。
`npu_data/`、`build/`、xclbin、trace、CSV などの生成物は commit しない。

## 1. block と core の形を分けて書く

SpMV は数式では `y = A x` と書く。出力を `C` と呼ばず `y` とすることで、以下の
`C_h/C_w`（core の形）と衝突させない。

| 記号 | 意味 | 主なトレードオフ |
|---|---|---|
| `R` | 一つの MemTile 配下で使う物理 core-row 数（既定値は 4） | ハードウェア配置。NPU2 では `1 <= R <= 4` のみ |
| `B_h` | Slice-ELL block の高さ、すなわち一 slice の行数 | padding と一 block 当たりの仕事量 |
| `B_w` | 一つの固定幅 A block の横 slot 数。例: 256 | 丸め padding と acquire/kernel call 回数 |
| `C_h` | 一 core が担当する行数。`C_h = B_h / R` | core L1 の A/y object 高さ |
| `C_w` | 一 core が担当する横 slot 数。現設計では `C_w = B_w` | 横方向を core 間で分割しない |
| `\ell_s` | slice `s` 内の最大実 nnz/row | その slice が必要とする実幅 |
| `slice_blocks[s]` | slice `s` を横方向に処理する `B_h × B_w` block 数 | `x` と同じ config object に格納する可変値 |
| `W_s` | slice `s` の `B_w` 丸め後の ELL 幅 | `W_s = slice_blocks[s] × B_w` |

ここで `B` は **Block**、`C` は **Core** を表す。`B_h/B_w` と `C_h/C_w` は図でも
そのまま縦横の対応として使える。`C_w` は今は `B_w` と同じであり、独立した tuning
parameter ではない。

slice `s` について、必要な block 数と padded slot 数は次である。

```
slice_blocks[s] = ceil(ell_s / B_w)       # 横方向の B_h × B_w block 数
W_s             = slice_blocks[s] * B_w  # slice の丸め後 ELL 幅
slots(s)        = B_h * W_s               # value/index の padding 込み slot 数
```

`R` 個の core は全員が同じ `slice_blocks[s]` 個の A block を消費する。ただし各 core
が扱うのは一 block 内の `C_h × C_w` slot だけである。短い row の余りには value=0 を
入れる。各 core は最後に必ず `C_h` 行を出力するので、join 後の `y_s` object は常に
固定の `B_h` 行となる。

つまり、**入力 A block 数だけは slice ごとに変わる**一方で、**出力 `y_s` は固定長かつ
元の行順のまま**にできる。

### `x + slice_blocks[]` config と固定回数の `y_s` release

通常の ObjectFIFO object には、任意の利用者定義 `metadata.is_last` はありません。
API に現れる `metadata` は Shim DMA allocation と ObjectFIFO を関連付ける名前です。
したがって終端 flag を設計に使うなら、A payload 内に明示的に header/flag を格納する
必要があります。

最初の実装では、A object ごとの `is_last` と独立した control FIFOは使わない。
NPU2の各Shim tileにはhostからarrayへ送るMM2S DMA channelが2本しかなく、現行構成は既に
packed Aと入力vector `x`で2本を使用するためである。代わりに、column `c` が担当する
sliceの`slice_blocks[s]`列を`x`の末尾へ連結し、一つの固定長config objectとして四coreへ
broadcastする。

```text
config_c = [ BF16 x[0:K] | uint16 slice_blocks[s_begin[c]:s_end[c]] | alignment padding ]
```

各columnのconfig型を共通にするため、control部分は全column中の最大local slice数まで
paddingする。`x`のBF16 bit列とcontrolのuint16列を同じraw uint16 bufferにpackし、kernel側では
先頭`K`要素をBF16としてreinterpretする。config objectは各coreがjobの最初に一回だけacquireし、
全sliceが終わるまで保持する。これによりShimの入力はAとconfigの2本に収まる。

```text
config = acquire(x_and_slice_blocks) # jobごとに一回
for s in assigned_slices:
  p = load_u16(config, K + local_s)  # p = slice_blocks[s]
  y = acquire(C_h rows)
  clear(fp32_row_accumulator)
  for i = 0 .. p-1:                  # dynamic な A acquire/release 回数
    A = acquire(C_h x C_w slots)
    accumulate(A, fp32_row_accumulator)
    release(A)
  finalize_bf16(fp32_row_accumulator, y)
  release(y)                          # slice ごとに必ず一回
release(config)
```

ここで core が Shim DMA を新たに start するわけではない。Runtime 側は `y_s` の drain DMA
task をあらかじめ arm しておき、`release(y)` が data-ready を知らせる。DMA は release
済み object を drain する。このため y の物理 object 長は常に `C_h` 行で固定のまま、A の
block 数だけを slice ごとに可変にできる。

`--dynamic-objFifos` が必要なのは、slice ごとに A の acquire/release 回数が変わるため
です。一方 y はsliceごとに一回、configはjobごとに一回acquire/releaseするので回数は
静的です。この形は「最後の flag を読んだ時だけ y release」よりlockの対応関係が明瞭です。
古い静的/cyclostatic lowering を前提にしてはならないため、最初に小さい NPU2 design で
dynamic A loop が lower・実行できることを確認します。

`ell_s=0` の全ゼロsliceも仕様に含める。`slice_blocks[s]=0`ならAを一つもacquireせず、
zero clearしたaccumulatorをBF16化してyをreleaseする。LAST flag方式と違ってdummy A blockは
不要である。

4 core の join を維持する限り、同一 slice の四 core は同じ `slice_blocks[s]` 個の A block を
消費し、全 core が y を release する必要がある。短い row の余りは padding する。
core ごとに異なる `slice_blocks[s]` にして終了順で y を返す方式は、この固定順 join/drain の範囲外であり、
PacketFifo と row-id 付き出力、または別の scatter/reorder 経路が必要になる。

### vector laneは行方向ではなく横slot方向に使う

現行`SELL-32` kernelは32 vector laneを32行へ割り当てるため、1 coreが32行、4 core join後の
`B_h`が128行になる。この方式はreduceを避けられる一方、Slice-ELLでは128行中の最大nnzへ
全行をpaddingするため、sliceを導入する利点が小さくなる。

Slice-ELL kernelでは通常のELL kernelと同様に、vector lane数`V=32`を一行内の横slotへ
割り当てる。`V`と`C_h`は独立であり、`B_w`だけを`V`の倍数にする。
これを最初のSlice-ELL実装の方針とし、縦32行をlaneへ割り当てる方式は比較baselineに留める。

```text
for row = 0 .. C_h-1:
  block_acc = fp32_vector_zero(V)
  for j = 0 .. B_w-1 step V:
    idx = load_u16xV(A[row].index[j:j+V])
    val = load_bf16xV(A[row].value[j:j+V])
    xval = gather_x(idx)
    block_acc += val * xval
  row_acc[row] += reduce_add(block_acc)
```

`row_acc[C_h]`はsliceの全A blockを処理する間FP32で保持し、最後に一度だけBF16のyへ変換する。
これによりblockごとにBF16へ丸め直す誤差を避ける。`V=32`を使うため、`B_w` は32の倍数にする。
`C_h={2,4,8,16}`、`B_w={128,256}`は最初に profile する**例**であり、format やCLIの選択肢を
限定するものではない。実装に渡す基本 parameter は`R`、`B_h`、`B_w`であり、
`C_h = B_h/R`として導出する。従って32 laneを使うことは`B_h>=128`を意味しない。最初の実装における実用上の最小slice
heightは`B_h=8`であり、`B_h=128`ではない。`C_h=1, B_h=4`はBF16 yがcore当たり2 byteと
小さすぎるため主候補にはせず、DMA/object alignmentを確認する診断候補としてだけ残す。

横方向vector化では各行に`reduce_add`が必要であり、index gatherもscalar loadを含むため、
32行をlane化するSELL-32より常に速いとは仮定しない。padding削減、A object数、reduce/gather
costを含めて実機で比較する。

## 2. L1 容量の制約

BF16 value、uint16 index、BF16 出力、A の ping-pong buffer、FP32 の行 accumulator を
仮定する。column `c` の local slice 数を `N_ctl,c`、64 byte alignment を `align64` とすると、
一 core の保守的な L1 使用量はおおよそ次です。

```
config_bytes = align64(2*K + 2*N_ctl,c)
L1(C_h, B_w, K) ~= config_bytes
                    + 2 * C_h*B_w * (2 + 2) # A value/index, depth 2
                    + 2 * C_h * 2           # BF16 y, depth 2
                    + 4 * C_h               # FP32 row accumulator
                    + state/stack
                 = config_bytes + 8*C_h*B_w + 8*C_h + state/stack   [byte]
```

ここで `K` は入力 vector の長さです。`N_ctl,c` は小さいため、config の大半は x です。
y の depth は実装時の ObjectFIFO depth に合わせて式を更新する。現行 `measure.py` は
64 KiB から 2 KiB を引いた 62 KiB を探索上限に使っていますが、NPU2 の data memory
64 KiB と program memory 16 KiB は別領域です。したがって、この 62 KiB は program code
を差し引く式ではなく、state・stack・alignment・compiler が置く buffer の余裕を確保する
経験的な上限として扱う。コンパイル後の L1 layout を確認するまでは、48--56 KiB 程度を
実用的な上限とします。

`/home/hitoshi/elsa` の pruning 済み Llama-2-7B checkpoint には、主に次の
2-D weight shape があります。

```
(4096, 4096), (4096, 11008), (11008, 4096), (32000, 4096)
```

したがって最大入力 vector 長は `K=11008`、BF16 で 22,016 B です。`B_w=256` の
場合、62 KiB という探索上限なら `K=4096` で最大 `C_h=26`、`K=11008` で最大
`C_h=20` です。

たとえば `C_h=16, B_w=256, K=11008` で、column 当たりの control が8要素なら、
`config_bytes=22,080 B`、A ping-pongが32,768 B、yとFP32 accumulatorが128 Bで、
state/stackを除いて54,976 B（約53.7 KiB）です。`R=4` ならslice block全体は
`B_h=64` 行です。この候補は容量上は可能性がありますが、余裕は大きくありません。
必ずコンパイル後の L1 layout と実測で確認します。

この場合の A object は core 当たり 16 KiB、4 core 全体で 64 KiB です。
ping-pong では slice block 全体で 128 KiB となります。これは最初に測る価値のある
候補であり、最速と仮定してはいけません。

入力 vector 全体が入らない場合は、これはもはや通常の Slice-ELL ではありません。
vector を column tile に分割し、各 row の partial sum を最後の column tile まで
L1 に保持する、2-D tiled SpMV が必要です。A も `(row-block, column-tile)` 単位
に再 pack します。現在の Llama-2-7B では `K <= 11008` で全 vector が入るため、
この複雑な方式はまだ導入しません。

## 3. 実装前に固定する format contract

Slice-ELL の数式だけでは packer と kernel を別々に実装できない。最初の実装前に、
以下を manifest とテストで固定する。

- **A の線形順序**: `slice s -> horizontal block i -> core-row -> local row -> slot` の
  ように、DDR buffer と MemTile split の両方で同じ順序を定義する。value/index の
  AoS/SoA と alignment を含める。A は column ごとの一つの線形 stream とし、slice ごとの
  独立した Shim DMA task にはしない。
- **config の layout**: raw uint16 buffer を
  `[BF16 x の bit列 | local slice_blocks の uint16列 | 64-byte alignment padding]` とする。
  全columnで同じobject長になるようcontrol部分を最大local slice数までpaddingし、各columnの
  `slice_begin/slice_end`、有効control数、config byte数をmanifestに記録する。
- **padding の有効性**: padding slot は `value=0` だけでは不十分である。kernel が
  value を掛ける前に index gather するので、index も必ず有効な `0 <= index < K`
  （通常は 0）にする。
- **index 型の範囲**: index を uint16 とするなら `K <= 65535` を format の前提として
  明記する。実際の kernel/packer が index を BF16 として運んでいるなら、精度と表現可能な
  範囲を別途検証する。
- **row 端部と core-column への配置**: `M` が `B_h`、または全 Shim column への slice
  分配で割り切れない場合の zero-row padding、元の `M` 行だけを host に返す規則、各 Shim
  column が持つ連続 slice range を定める。y の drain offset はこの表から生成する。
- **数値仕様**: slice の先頭で y をゼロ初期化する場所、accumulator の型、BF16 へ丸める
  時点、CPU reference の許容誤差を定める。`slice_blocks[s]=0` ではゼロ y を確実に出力する。
- **artifact identity**: xclbin/MLIR/packed data の名前と manifest に、少なくとも
  `M,K,R,B_h,B_w,C_h,C_w,format version,index type,toolchain commit` を含める。異なる design の古い
  artifact を再利用しない。

`slice_blocks` の制御列は slice 数 `N_slice=ceil(M/B_h)` 個であり、A 本体の
`sum_s slice_blocks[s]` block に比べて小さい。offline packer は各 matrix の
`packed_A`, `slice_blocks`, `manifest.json` を一組として生成・検証する。x は実行時入力なので、
host runtime が各 invocation で x とそのcolumnの `slice_blocks` を一つの `packed_config`
へ詰める。静的なpacked weightへ特定のxを埋め込んではならない。

## 4. 現在の `design_sell32_block.py` がしていること

現行 design は、

```
R = 4, B_h = 128, B_w = 16, C_h = 32, C_w = 16
```

です。一つの MemTile object は四つの core 用 object に split されます。各 core
は K 要素すべての入力 vector を取得して全反復の間保持し、横方向の `B_w=16` block ごとに
一つの A object を消費し、固定 `C_h=32` 行の y object を生成します。

各 Shim column には連続した row range が割り当てられます。四つの y object は
MemTile で join され、その連続 row range を一つの y TAP で DDR に drain します。
したがって各 row は一つの core だけが計算しており、出力 reduction は不要です。

Slice-ELL に移行しても、この y の所有権を維持します。slice の A block
`slice_blocks[s]` 個を連続 pack して四 core に送り、固定長の `B_h` 出力を join/drain します。
core が終了した順に任意の y を出力する方式は使いません。Shim DMA の drain は
順次出力であり、任意位置の DDR scatter writer ではないためです。

### 現行 SELL design の位置付け

`measure.py` の現在の既定値は `design_name = "ell"` であり、何も変更せず実行すると
`design_ell.py` を測定する。SELL を測定するには明示的に design 名を切り替える必要がある。

| design | 方式 | 工程での用途 |
|---|---|---|
| `design_sell32.py` | core 当たり `m*32` 行と全 `ell_width` を一 object で処理し、kernel を一回呼ぶ標準的な静的 SELL-32 | **Phase 1 の固定幅 SELL baseline** |
| `design_sell32_block.py` | `32 x 16` の A object を横方向にストリーミングし、同じ y に逐次 accumulate する。`m=1` 固定 | **split/join と block stream の bridge design**。Phase 3 の出発点だが、32行をlane化するkernel自体はbaselineに留める |
| `design_sell32_c.py` | 一 Shim column / 一 core を使い、128 回分の y を MemTile の大きな蓄積 object に置く出力集約実験 | baseline にしない。generic `op.py` の callback 引数・kernel object とも互換でない |

`measure.py` の自動 tile-size 探索が渡す `tile_size` は `m` として design に渡る。
`design_sell32_block.py` は `m == 1` を assert するため、同じ測定設定をそのまま
`sell32_block` へ切り替えてはいけない。block design 用には `m=1` を固定した別の
test parameter / measurement entry point を用意する。

## 5. TAP / DMA の制約

### TAP の「次元数」と BD の「値域」は別の制約である

- `TensorAccessPattern` / Runtime の Shim DMA task API に指定できる size/stride は
  最大 4 次元です。
- ただし Shim DMA はそのうち先頭の `sizes[0]` を address-generation 次元としてでは
  なく、同じ BD task を繰り返す **d3 iteration/repeat** として使います。実際に
  address generation へ使えるのは右側の三次元 `sizes[1:4]` です。
- Compute tile と Shim tile の address generation は最大 3 次元、MemTile は最大 4
  次元です。この違いを「TAP が 4 次元まで書ける」ことと混同しないこと。

現行 design の TAP は論理的に

```
[iteration, horizontal ELL block, core-row, contiguous object]
```

の四次元です。先頭の `iteration` が Shim の d3 repeat、残り三つが通常の
address-generation 次元になります。

### NPU2 Shim DMA の値域

最新 toolchain の `verifyStridesWraps` と PR #3392 の実装に基づくと、NPU2 の
Shim/MemTile の non-contiguous BD では d0--d2 の wrap (size) field は 10 bit、
すなわち最大 **1023** です。Compute tile の wrap field は 8 bit です。
また Shim の d3 iteration count は最大 **65** であり、それを超える場合は現時点でも
分解されず reject されます。現在の `MAX_BD_SIZE3 = 64` はこの上限の内側に収める
保守的な既存回避策です。

なお、size と stride は element 幅で書く一方、hardware field への encoding では
element size による換算も入ります。したがって「BF16 の 1024 要素が必ず失敗する」と
単純には言えません。最終判定は compiler の generated MLIR と verifier に任せます。

### PR #3392 が解決した範囲

[PR #3392](https://github.com/Xilinx/mlir-aie/pull/3392) は merge 済みです。これは
**非連続** DMA transfer の d0--d2 wrap/stride が hardware field に収まらないとき、
順序を保ったまま BD を自動的に合法化する pass
`aie-decompose-large-dma-bd` を追加したものです。対象は
`aiex.npu.dma_memcpy_nd` と、IRON の `rt.fill/drain(tap=...)` が使う
`aie.dma_bd` task path の両方です。

- 可能なら大きすぎる次元を隣接次元へ factor し、一つの合法 BD にする。
- factor できない outer 次元は、offset を調整した複数の順序保存 BD / BD chain にする。
- したがって d0--d2 に置かれた大きな非連続 access は、多くの場合は手動 TAP 分割
  なしで通るようになりました。**ただし現行 `design_sell32_block.py` の
  `MAX_BD_SIZE3 = 64` は d3 iteration の制限に対する分割であり、この PR の対象外です。**

一方で、次は **解決していません**。

- d3 iteration count が 65 を超える場合。
- 空き次元がなく、順序を壊さずには分割できない prime な inner 次元が 1023 を超える場合。
- 4 次元を超える logical TAP。
- Slice-ELL kernel がconfigから読んだ `slice_blocks[s]` 回だけA ObjectFIFOを正しく
  acquire/releaseすること。
- core が終了順に任意 DDR address へ y を scatter すること。

従って PR #3392 により「TAP の大きい非連続次元のため compile が失敗する」問題は
大幅に緩和されましたが、ragged Slice-ELL の制御・固定順出力問題まで解消したわけでは
ありません。生成された MLIR/XCLBIN で BD chain 数と d3 count を確認します。

Shim tile あたり利用できる BD は 16 個です。Slice-ELL slice ごとに独立した host DMA
task を発行してはなりません。A は連続 packed stream として送り、完了後にだけ BD を
再利用します。連続 row-major 転送は compiler が linear DMA に正規化できます。実際の
難しさは `B_h*B_w` の単純な byte 上限ではなく、非連続 layout、object 順序、BD 数です。

仕様の根拠は [MLIR-AIE の data-movement guide](https://xilinx.github.io/mlir-aie/dev/programming_guide/section-2/section-2c/)
および [DMA task guide](https://xilinx.github.io/mlir-aie/dev/programming_guide/section-2/section-2d/DMATasks/) を参照します。


## 6. 行列ごとの tuning、pack plan、Pareto table

### parameter は固定の候補リストではない

`R=4` は「4 core を使う」という既定値であって、packer に埋め込む定数ではない。CLI/API は
`--core-rows R`（既定4）、`--block-height B_h`、`--block-width B_w` を任意の整数として受け、
次だけを validation する。

```text
1 <= R <= 4                 # 一 MemTile 下の使用 core-row 数
B_h > 0 and B_h % R == 0    # C_h = B_h/R が整数
B_w > 0 and B_w % 32 == 0   # 横32 lane kernel
L1(R, B_h/R, B_w, K) <= L1 budget
```

したがって `B_h=12, 20, 28` や `B_w=96, 160, 320` も合法なら評価対象である。探索時には
ユーザーが指定した上下限・刻み、または全ての合法な32倍数を列挙する。`{2,4,8,16}` や
`{128,256}` は探索を最初に短時間で回すための sample grid に過ぎない。

ここで重要なのは、**NPUなしの探索だけで実行時間の真の最適値は決められない**ことです。
row nnz 分布から正確に分かるのは padding、A object 数、L1 使用量、column 間の仕事量だけであり、
dynamic ObjectFIFO と gather/reduce の実時間はまだ分からない。従って自動 tuner・コストモデル・
layerごとの最終選択は、dynamic Slice-ELL が実機で動いてからの後続作業とする。

最初の実装では探索を行わず、次の一つに固定する。

```text
R = 4, B_h = 32, C_h = 8, B_w = 256, V = 32
```

これは最大の入力 vector `K=11008` に対しても、core 当たりのconfigが約22 KiB、A ping-pongが
16 KiBであり、state/stackの余裕を残せる。join後のyは32行（64 B）で、`B_h=128`よりpaddingの
悪化を抑えつつ、極端に小さいobjectにもならない。packer のCLIは将来の比較のため任意の
`R,B_h,B_w`を受けられるようにするが、Phase 2--4の機能確認で自動探索はしない。

各 weight matrix ごとに row の nnz を一度数えれば、各候補について
次を計算できます。

```
S_c(B_h,B_w)      = Σ{s in column c} B_h*B_w*slice_blocks[s]
Nblock_c(B_h,B_w) = Σ{s in column c} slice_blocks[s]
T_c               = max(4*S_c / BW_DMA,
                        S_c / throughput_slots + Nslice_c*C_h*t_reduce)
                    + Nblock_c*t_object + Nslice_c*t_slice
Tmodel            = max_c T_c
```

`4*S_c` はBF16 valueとuint16 indexを合わせたcolumn `c`の行列転送byte数です。
`t_reduce`は横32 laneのreductionを一行行うcost、`t_slice`はconfig lookup、accumulator
初期化、y finalize/releaseのslice固定costです。全columnを並列に使うため、総和ではなく
最も遅いcolumnを予測値にする。sliceは元の行順を保つ連続rangeのまま、予測workが均等に
なる境界を選ぶ。`BW_DMA`、`throughput_slots`、`t_object`、`t_reduce`、`t_slice`は推測で
決めず、少数のコンパイル済みkernelをtrace/profileして測定します。

まず、転送量・object 数・L1 使用量・最遅columnの仕事量の四目的で支配される候補を除外し、残る
Pareto frontier だけを実機で測ります。ここで **matrix ごとの Pareto table** とは、各行列について
この「他候補より全指標で悪くない」候補を並べた JSON/CSV です。これは実際に一つを選んだ記録ではなく、
測定すべき候補集合です。ある候補が別候補より転送量・object 数・L1 使用量・最遅column仕事量の
すべてで悪ければ、測定する必要がありません。

自動 tuner を導入する段階では `pareto.json`（支配されない候補群）と、packerに渡す一つの
`plan.json`（選択済みparameter）を分ける。この二つは現時点では生成しない。Phase 2 の packer は
上記固定parameterを既定にし、明示的に指定されたparameterでだけpackする。

### packed data と manifest の役割

一つの行列・一つの選択済み plan から packer は、概念上以下を生成する。

```text
packed_A      : slice -> horizontal block -> core-row -> local row -> slot 順の index/value 本体
slice_blocks  : 各 slice の uint16 p。p 個の A block を読む、というruntime control表
manifest.json : この packed data を解釈・検証・再現するための静的なsidecar
```

`packed_A` は大きな重みデータ、`slice_blocks[s]` は「slice `s` の横block数」であり、どちらも
実行時に x から計算する値ではない。`manifest.json` はデータ本体ではなく、`M,K,R,B_h,B_w,C_h,V`、
format version、index型、slice/column境界、各columnの A offset と control長、padding量、元行列と
packing parameter の hash を記録する。host は manifest を読んで正しい packed_A と slice_blocks を選び、
slice_blocks を x の後ろへ詰めた config object を作る。これにより異なる matrix 用の重み・xclbin・
plan を取り違えない。

今回の Llama-2-7B pruning model に対する storage-only simulation で、`B_h=8` の
結果は次でした。**これは全 2-D weight matrix の nnz 分布を合算した集計値であり、
単一行列の結果でも、全行列へ同一 parameter を適用する提案でもありません。**

| `B_w` | padded slots / 真の nnz | A block object 数 | 解釈 |
|---:|---:|---:|---|
| 32 | 1.365x | 4.85 M | padding は最小だが object/call が多すぎる |
| 128 | 1.440x | 1.28 M | 比較すべき候補 |
| 256 | 1.524x | 0.677 M | 最初の実装候補 |
| 512 | 1.723x | 0.383 M | call は減るが padding が大きい |

この表は候補の初期優先順位を付けるためだけに使います。実装では各 matrix / layer ごとに
row nnz を調べ、L1 容量、`B_h` による padding、`B_w` による object 数から別々の Pareto
候補を作り、最終的に **行列ごとに異なる `(C_h, B_w)` を選びます**（`B_h=R*C_h`）。
ただし y の row ownership と出力順は共通に保ちます。

`B_w=256` の全モデル集計では、`B_h=4` の storage は約 1.266 B slot、`B_h=8` は約
1.386 B slot、`B_h=16` は約 1.535 B slot でした。大きい `B_h` は L1 をよく使うから
選ぶのではなく、その行列の DMA/kernel 固定コストを実測して初めて正当化します。

## 7. 実装の工程

### Phase 0: 現行 baseline を凍結する

1. 現行 IRON、mlir-aie、llvm-aie、XRT の version、現行の commit、実行 command、
   既知の正しい ELL / SELL-32 の結果と trace を保存する。
2. 最小行列と実 model の代表 shape `(4096,4096)`、`(4096,11008)` で、CPU reference
   との一致、compile 時間、実行時間、DMA/FIFO stall を記録する。

これは toolchain 更新後の性能差と functional regression を識別する基準である。

#### 2026-09-16 の指定条件による再baseline

operator sourceの測定開始commitは`fdcddda6c6de64f04831a423b8720f2ce1fd9561`、環境はNPU2、
`mlir-aie==1.1.3`、XRT 2.21.0である。詳細な環境、command、全数値は
[`README.md`](README.md) の「Phase 0 baseline 実測」を正とする。

- `4096×4096`、`4096×11008`、`28672×8192`を、各行`K/8` nnzの固定seed random matrix、
  すなわち12.5%密度のuniform ELLとして生成した。
- 各shapeについてELLとSELL-32 blockを、4×8の全32 coreでCPU reference一致まで実行した。
  ELLの`m`は順に8、2、4、SELL-32 blockの`m`は実装上1である。
- designごと・shapeごとにempty build directoryを用意し、`op.py`のdesign名なしartifact cacheが
  結果を混ぜないようにした。性能値は`measure.py`と同じdevice-onlyの5 sample平均であり、
  結果JSON/logは`npu_data/phase0_devel_2026-09-16/measure_*`に保存した。host→NPU BO同期を
  含む`rerun_*`は診断用で、baselineの性能比較には使わない。
- traceは本Phase 0の要件から外し、採取しない。

### Phase 1: 最新 toolchain への移行を先に完了する

1. 現行の `/home/hitoshi/IRON` を in-place 更新しない。最新 AMD IRON / mlir-aie を別の
   checkout と virtual environment に置き、version を lock する。これは旧 checkout の
   version up ではなく、最新 upstream からの clean な構築である。
2. まず upstream に既にある NPU2 operator を一つ build/run し、XRT、compiler、device、
   host runtime がその環境で成立することを確認する。
3. 旧 `devel` の `operators/spmv` は参照元として残す。そこから新 checkout の
   `operators/spmv` へ、kernel、CPU reference、入力生成、test を必要最小限で移す。
   古い `op.py`、artifact 名、Runtime API、build/cache の構造をそのまま複製しない。
   最新 upstream の他 operator を手本にして、directory 構造と host/operator interface を
   組み直す。
4. まず Slice-ELL を入れず、既知の `1024 x 2048` 固定幅 ELL を新しい構造で NPU2 compile、
   xclbin 作成、host 実行、CPU reference 一致まで動かす。
5. 次に **`design_sell32.py`** 相当を固定幅 SELL baseline として移植し、固定幅 SELL の
   join/split と contiguous y drain が新しい runtime でも同じ結果になることを確認する。
6. 続けて `design_sell32_block.py` 相当の横方向 block data path を移植し、y accumulate、
   contiguous y drain、入力 vector `x` の broadcast が同じ結果になることを確認する。
7. 各段階で生成 MLIR と `input_with_addresses.mlir` を調べ、TAP の d3、BD 数、y drain
   offset を旧 baseline と比較する。

**通過条件:** 最新 upstream の operator 構造上で、静的 ELL、`sell32`、`sell32_block` の
三つが NPU2 実行・参照一致・trace 採取まで再現できること。dynamic ObjectFIFO と
Slice-ELL を同時に導入しない。

### Phase 2: 固定parameterの format と reference packer

1. 最初のparameterを `R=4, B_h=32, B_w=256` に固定する。packer は将来の比較用に
   `--core-rows 4 --block-height 32 --block-width 256` のような任意値を受けるが、自動探索・
   `pareto.json`・`plan.json` は作らない。
2. row-order-preserving packer を作り、`packed_A`, `slice_blocks`, `manifest.json` を出力する。
   `slice_blocks[s]=0`、末尾 row、padding index=0、複数 Shim column への連続 slice 割当を含める。
   runtime用 host reference は x と column-local `slice_blocks` から固定長 `packed_config` を生成する。
3. 同じ packed format を読む Python reference と property test を作り、元 sparse matrix reference と
   全行一致させる。pack/unpack、padding、slice/column境界もNPUなしで検証する。
4. manifest に固定parameter、容量・padding、format layoutを記録する。最終的な latency 最適化は
   dynamic design が実機で動いた後の別工程にする。

#### Phase 2 の実装入口

`slice_ell.py` はNPU/MLIRに依存しない format module である。CSR の
`(indptr, indices, values, shape)` を入力とし、既定の
`R=4, B_h=32, B_w=256, shim_columns=8` で pack する。個別の比較には
`--core-rows`、`--block-height`、`--block-width` を明示してよいが、Phase 2 の
baselineを変えてはならない。

```bash
python -m iron.operators.spmv.slice_ell \
  --csr-npz matrix.npz --output-dir npu_data/slice_ell --stem layer_name

# pruning 済み safetensors の一weightを直接packする場合
python -m iron.operators.spmv.slice_ell \
  --safetensors model-00001-of-00006.safetensors \
  --tensor model.layers.0.self_attn.q_proj.weight \
  --output-dir npu_data/slice_ell --stem layer0_q_proj

source /opt/xilinx/xrt/setup.sh
python -m pytest iron/operators/spmv/test_slice_ell.py -q
```

CLI入力のNPZには`indptr`、`indices`、`values`、および`shape=[M,K]`（または`K`）を入れる。
`--safetensors`では一つの2-D tensorをその場でCSR化する（`safetensors` packageが必要）。
出力は`*_packed_A.npy`、`*_slice_blocks.npy`、`*_manifest.json`である。testはNPUを使用せず、
元CSR referenceとの一致、row順、末尾zero-row padding、`slice_blocks=0`、config objectの
`[x | control | 64-byte padding]` layoutを検証する。

### Phase 3: 静的な Slice-ELL data path を確認する

1. `slice_blocks[s]` が全sliceで同じ小さなsynthetic matrixを使い、A split、config broadcast、
   y join、連続y drainを検証する。kernelは`V=32`を横slot方向に使い、block間はFP32の
   `row_acc[C_h]`へ蓄積し、slice末尾で一度だけBF16へ変換する。
2. この段階では dynamic ObjectFIFO を使わない。`C_h` 行/core、`B_h=R*C_h` 行/slice の y が
   元の row 順に一度だけ DDR へ書かれることを確認する。

### Phase 4: dynamic `slice_blocks[s]` を四 core micro-test で検証する

1. `slice_blocks=[0,1,4,2]` のように全ゼロ slice と異なる長さを混在させる小さな matrix を使う。
2. xとcontrol表を含む一つの固定長config objectを四coreへbroadcastし、そこからpを読む。
   `for i in range(p)` のdynamic A acquire/releaseと、sliceごと一回のy releaseを
   `--dynamic-objFifos`でlowerする。
3. column当たりのShim MM2SがAとconfigのちょうど2本であり、独立control DMAが生成されない
   ことを生成MLIRで確認する。
4. timeout/deadlock、出力順、全coreのacquire/release数、生成BD/lockを検査する。

**通過条件:** `slice_blocks[s]` の分布にかかわらず、y の長さ・順序が固定で、CPU reference と一致し、
deadlock しないこと。ここが Slice-ELL 固有の最大リスクである。

### Phase 5（後続）: 実 model layer と tuning

1. dynamic Slice-ELL が動いた後、代表 shape に対して複数の `(R,B_h,B_w)` を手動比較する。
2. 必要性が確認できた場合だけ、Pareto探索・コストモデル・autotuningを導入する。
3. 最終parameterとpacked dataを再現可能なcommandで保存し、fixed ELL、SELL-32、Slice-ELLを
   同じ精度・同じinput vectorで比較する。

この順序では、最初に toolchain 差分、次に format correctness、最後に dynamic control と
性能を分離するため、失敗した原因を切り分けられる。

この方針なら、行順と y の固定長転送を保ったまま、各 layer の分布に応じて A の
packing parameter を変えられます。
