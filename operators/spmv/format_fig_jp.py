import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 0. 日本語フォント設定
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 
    'TakaoExGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP', 'sans-serif'
]

# ==========================================
# 1. 設定
# ==========================================
ROWS = 32       # 行数
COLS = 32       # 列数
ELL_WIDTH = 12  # ELLの幅

# (b)用の設定
TILE_SIZE_B = 16   # 行方向のみのタイルサイズ（16行ごと）

# (c)用の設定
BLOCK_SIZE_H = 16 # NPUの縦ブロックサイズ
BLOCK_SIZE_W = 4  # NPUの横ブロックサイズ

# カラー設定
C_EMPTY = '#f0f0f0' # 空
C_NZ    = '#4a90e2' # 非ゼロ
C_MARK  = '#ff9900' # 目印
C_GRID  = 'white'   # グリッド線

# 線の設定
ARROW_COLOR = '#222222'
LINE_WIDTH_SOLID = 1.2
LINE_WIDTH_DOTTED = 1.0
BLOCK_LINE_WIDTH = 3.5  # 太く強調

# ==========================================
# 2. データ生成
# ==========================================
np.random.seed(99) 

dense_map = {} 
ell_map = {}   

for r in range(ROWS):
    n_items = np.random.randint(2, ELL_WIDTH + 1)
    col_indices = np.random.choice(range(COLS), size=n_items, replace=False)
    col_indices.sort()
    
    for w, c in enumerate(col_indices):
        color = C_MARK if w == 1 else C_NZ
        dense_map[(r, c)] = color
        ell_map[(r, w)] = color

# ==========================================
# 3. 描画用ヘルパー関数
# ==========================================
def draw_base_grid(ax, rows, cols, color_map, title, h_block=None, w_block=None):
    """
    h_block: 縦方向(行)のブロック区切り間隔 (Noneなら描画しない)
    w_block: 横方向(列)のブロック区切り間隔 (Noneなら描画しない)
    """
    ax.set_aspect('equal')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis() 
    
    # 背景
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, facecolor=C_EMPTY))
    
    # セル
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
    
    # ブロック境界線 (太線)
    # 横線 (行ブロック)
    if h_block is not None:
        for y in range(h_block, rows, h_block):
            ax.axhline(y, color='black', linestyle='-', linewidth=BLOCK_LINE_WIDTH)
            
    # 縦線 (列ブロック)
    if w_block is not None:
        for x in range(w_block, cols, w_block):
            ax.axvline(x, color='black', linestyle='-', linewidth=BLOCK_LINE_WIDTH)

    ax.set_title(title, fontsize=11, pad=12)
    
    if cols == ELL_WIDTH:
        ax.set_xlabel("ELLPACK Columns", fontsize=10)
    else:
        ax.set_xlabel("Matrix Columns", fontsize=10)
    ax.set_ylabel("Matrix Rows", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

def draw_long_arrow(ax, start_rc, end_rc):
    r1, c1 = start_rc
    r2, c2 = end_rc
    x1, y1 = c1 + 0.5, r1 + 0.5
    x2, y2 = c2 + 0.5, r2 + 0.5
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=ARROW_COLOR, linestyle="-", lw=LINE_WIDTH_SOLID))

def draw_dotted_connector(ax, start_rc, end_rc):
    r1, c1 = start_rc
    r2, c2 = end_rc
    x1, y1 = c1 + 0.5, r1 + 0.5
    x2, y2 = c2 + 0.5, r2 + 0.5
    ax.plot([x1, x2], [y1, y2], color=ARROW_COLOR, linestyle=":", lw=LINE_WIDTH_DOTTED, alpha=0.7)


# ==========================================
# 4. パス描画ロジック
# ==========================================

def plot_row_major_ell_arrows(ax):
    """
    (b) 行優先 (1D Tiling)
    アクセス順序自体は行ごとのスキャンだが、
    視覚的に「ブロック」の概念があることを実線で示す
    """
    rows = ROWS
    width = ELL_WIDTH
    for r in range(rows):
        draw_long_arrow(ax, (r, 0), (r, width - 1))
        if r < rows - 1:
            draw_dotted_connector(ax, (r, width - 1), (r + 1, 0))

def plot_npu_tiled_arrows(ax):
    """
    (c) Tiled Block-ELL Format (2D Tiling)
    """
    rows = ROWS
    width = ELL_WIDTH
    blk_h = BLOCK_SIZE_H
    blk_w = BLOCK_SIZE_W
    
    num_blk_rows = rows // blk_h
    num_blk_cols = width // blk_w
    
    # 1. 行ブロックのループ
    for br in range(num_blk_rows):
        row_start = br * blk_h
        row_end   = (br + 1) * blk_h - 1
        
        # 2. 列ブロックのループ
        for bc in range(num_blk_cols):
            col_start = bc * blk_w
            col_end   = (bc + 1) * blk_w - 1
            
            # 3. ブロック内のスキャン
            for w in range(col_start, col_end + 1):
                draw_long_arrow(ax, (row_start, w), (row_end, w))
                
                if w < col_end:
                    draw_dotted_connector(ax, (row_end, w), (row_start, w + 1))
            
            # 列ブロック間の遷移
            if bc < num_blk_cols - 1:
                next_col_start = (bc + 1) * blk_w
                draw_dotted_connector(ax, (row_end, col_end), (row_start, next_col_start))
        
        # 行ブロック間の遷移
        if br < num_blk_rows - 1:
            next_row_start = (br + 1) * blk_h
            draw_dotted_connector(ax, (row_end, width - 1), (next_row_start, 0))


# ==========================================
# 5. プロット実行
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# (a) Dense Matrix
draw_base_grid(axes[0], ROWS, COLS, dense_map, 
               title=f"(a) 疎行列\nサイズ: {ROWS}x{ROWS}",
               h_block=None, w_block=None)

# (b) Row-Major ELL (1D Tiled)
# ★変更点: h_block=8を指定して横線のみ引く
title_b = f"(b) 行優先ELLPACK\n行タイルサイズ: {TILE_SIZE_B}"
draw_base_grid(axes[1], ROWS, ELL_WIDTH, ell_map,
               title=title_b,
               h_block=TILE_SIZE_B, w_block=None)
plot_row_major_ell_arrows(axes[1])

# (c) NPU Format (2D Tiled)
# ★変更点: h_block=16, w_block=4を指定して格子状に引く
title_c = f"(c) 列優先ブロックELLPACK\nタイルサイズ: {BLOCK_SIZE_H}x{BLOCK_SIZE_W}"
draw_base_grid(axes[2], ROWS, ELL_WIDTH, ell_map,
               title=title_c,
               h_block=BLOCK_SIZE_H, w_block=BLOCK_SIZE_W)
plot_npu_tiled_arrows(axes[2])

# 凡例
legend_elements = [
    patches.Patch(facecolor=C_EMPTY, edgecolor=C_GRID, label='ゼロ要素'),
    patches.Patch(facecolor=C_NZ,    edgecolor=C_GRID, label='非ゼロ要素'),
    patches.Patch(facecolor=C_MARK,  edgecolor=C_GRID, label='各行で左から2番目の非ゼロ要素 (目印)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1]) 
plt.show()
plt.savefig("spmv_formats_fig.png", dpi=300)