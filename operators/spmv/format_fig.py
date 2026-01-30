import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 1. 設定
# ==========================================
ROWS = 32       # 行数
COLS = 32       # 列数
ELL_WIDTH = 8   # ELLの幅
BLOCK_SIZE = 16 # NPUのブロックサイズ

# カラー設定
C_EMPTY = '#f0f0f0' # 空（グレー）
C_NZ    = '#4a90e2' # 通常の非ゼロ（青）
C_TG    = '#ff9900' # ターゲット（2番目の要素・橙）
C_GRID  = 'white'   # グリッド線

# 線の設定
ARROW_COLOR = '#222222'
LINE_WIDTH_SOLID = 1.2    # 矢印の太さ
LINE_WIDTH_DOTTED = 1.0   # 遷移線の太さ
BLOCK_LINE_WIDTH = 2.5    # ブロック区切り線の太さ（太い実線）

# ==========================================
# 2. データ生成
# ==========================================
np.random.seed(99) # シード固定

dense_map = {} 
ell_map = {}   

for r in range(ROWS):
    # 各行に 2 ～ ELL_WIDTH 個の要素をランダム配置
    n_items = np.random.randint(2, ELL_WIDTH + 1)
    col_indices = np.random.choice(range(COLS), size=n_items, replace=False)
    col_indices.sort()
    
    for w, c in enumerate(col_indices):
        color = C_TG if w == 1 else C_NZ
        dense_map[(r, c)] = color
        ell_map[(r, w)] = color

# ==========================================
# 3. 描画用ヘルパー関数
# ==========================================
def draw_base_grid(ax, rows, cols, color_map, title, show_block_line=False):
    """基本グリッドを描画"""
    ax.set_aspect('equal')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis() 
    
    # 背景
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, facecolor=C_EMPTY))
    
    # セル描画
    for r in range(rows):
        for c in range(cols):
            if (r, c) in color_map:
                face = color_map[(r, c)]
                rect = patches.Rectangle((c, r), 1, 1, facecolor=face, edgecolor=C_GRID, lw=0.5)
                ax.add_patch(rect)
            else:
                rect = patches.Rectangle((c, r), 1, 1, fill=False, edgecolor='white', lw=0.5)
                ax.add_patch(rect)

    # 外枠
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, fill=False, edgecolor='black', lw=1.5))
    
    # ★変更点: NPUブロック境界線（太い実線）
    if show_block_line:
        for y in range(BLOCK_SIZE, rows, BLOCK_SIZE):
            # linestyle='-' (実線), linewidth=BLOCK_LINE_WIDTH (太め), color='black'
            ax.axhline(y, color='black', linestyle='-', linewidth=BLOCK_LINE_WIDTH)
            
            # ラベル (位置調整)
            ax.text(-0.8, y - BLOCK_SIZE/2, f"Block {int((y/BLOCK_SIZE)-1)}", 
                    va='center', ha='right', fontsize=9, rotation=90, fontweight='bold')
            ax.text(-0.8, y + BLOCK_SIZE/2, f"Block {int(y/BLOCK_SIZE)}", 
                    va='center', ha='right', fontsize=9, rotation=90, fontweight='bold')

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel("Width Index" if cols == ELL_WIDTH else "Logical Columns")
    ax.set_ylabel("Logical Rows")
    ax.set_xticks([])
    ax.set_yticks([])

def draw_long_arrow(ax, start_rc, end_rc):
    """連続領域を表す実線の矢印を描く"""
    r1, c1 = start_rc
    r2, c2 = end_rc
    
    x1, y1 = c1 + 0.5, r1 + 0.5
    x2, y2 = c2 + 0.5, r2 + 0.5
    
    # 始点から終点へ矢印
    ax.annotate("", 
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, 
                                linestyle="-", lw=LINE_WIDTH_SOLID))

def draw_dotted_connector(ax, start_rc, end_rc):
    """領域間をつなぐ点線（矢印なし）を描く"""
    r1, c1 = start_rc
    r2, c2 = end_rc
    
    x1, y1 = c1 + 0.5, r1 + 0.5
    x2, y2 = c2 + 0.5, r2 + 0.5
    
    # 単なる点線
    ax.plot([x1, x2], [y1, y2], color=ARROW_COLOR, linestyle=":", lw=LINE_WIDTH_DOTTED, alpha=0.7)


# ==========================================
# 4. パス描画ロジック
# ==========================================

def plot_ell_arrows(ax):
    """Standard ELL: 列ごとに上から下へ、終わったら次の列の上へ"""
    rows = ROWS
    width = ELL_WIDTH
    
    for w in range(width):
        # 1. その列の上から下への実線矢印
        draw_long_arrow(ax, (0, w), (rows - 1, w))
        
        # 2. その列の最後から、次の列の先頭への点線 (最後の列以外)
        if w < width - 1:
            draw_dotted_connector(ax, (rows - 1, w), (0, w + 1))

def plot_npu_arrows(ax):
    """NPU Format: ブロックごとに処理。ブロック内では列ごとにスキャン。"""
    rows = ROWS
    width = ELL_WIDTH
    blk_size = BLOCK_SIZE
    num_blocks = rows // blk_size
    
    for b in range(num_blocks):
        row_start = b * blk_size
        row_end   = (b + 1) * blk_size - 1
        
        # ブロック内の列ループ
        for w in range(width):
            # 1. ブロック内、特定列の実線矢印 (上から下へ)
            draw_long_arrow(ax, (row_start, w), (row_end, w))
            
            # 2. 列間の遷移 (ブロック内)
            if w < width - 1:
                # この列の終わりから、同じブロックの次の列の頭へ
                draw_dotted_connector(ax, (row_end, w), (row_start, w + 1))
        
        # 3. ブロック間の遷移
        # このブロックの「最後の列の最後」から、次のブロックの「最初の列の最初」へ
        if b < num_blocks - 1:
            next_b_start = (b + 1) * blk_size
            draw_dotted_connector(ax, (row_end, width - 1), (next_b_start, 0))


# ==========================================
# 5. プロット実行
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# (a) Dense Matrix (矢印なし)
draw_base_grid(axes[0], ROWS, COLS, dense_map, 
               title=f"(a) Dense Matrix ({ROWS}x{ROWS})\nTarget in Orange")

# (b) Standard ELL
draw_base_grid(axes[1], ROWS, ELL_WIDTH, ell_map,
               title="(b) Standard ELLPACK\nScan: Global Column-Major")
plot_ell_arrows(axes[1])

# (c) NPU Format
draw_base_grid(axes[2], ROWS, ELL_WIDTH, ell_map,
               title=f"(c) NPU Format ({BLOCK_SIZE}-row Blocked)\nScan: Block-Column-Major",
               show_block_line=True)
plot_npu_arrows(axes[2])

plt.tight_layout()
plt.show()
plt.savefig("spmv_ell_npu_format_comparison.png", dpi=300)