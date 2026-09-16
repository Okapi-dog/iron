import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 0. 日本語フォント設定
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo',
    'TakaoExGothic', 'IPAPGothic', 'VL PGothic',
    'Noto Sans CJK JP', 'sans-serif'
]

# ==========================================
# 1. 設定
# ==========================================
ROWS = 32
COLS = 32
ELL_WIDTH = 12

# (b)用の設定
TILE_SIZE_B = 16

# カラー設定
C_EMPTY = '#f0f0f0'
C_NZ = '#4a90e2'
C_MARK = '#ff9900'
C_GRID = 'white'

# 線の設定
ARROW_COLOR = '#222222'
LINE_WIDTH_SOLID = 1.2
LINE_WIDTH_DOTTED = 1.0
BLOCK_LINE_WIDTH = 3.5

# fontsize
FONT_SIZE = 18

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
    ax.set_aspect('equal')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()

    ax.add_patch(
        patches.Rectangle((0, 0), cols, rows, facecolor=C_EMPTY)
    )

    for r in range(rows):
        for c in range(cols):
            if (r, c) in color_map:
                face = color_map[(r, c)]
                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    facecolor=face,
                    edgecolor=C_GRID,
                    lw=0.5
                )
            else:
                rect = patches.Rectangle(
                    (c, r), 1, 1,
                    fill=False,
                    edgecolor='white',
                    lw=0.5
                )
            ax.add_patch(rect)

    ax.add_patch(
        patches.Rectangle(
            (0, 0), cols, rows,
            fill=False,
            edgecolor='black',
            lw=1.5
        )
    )

    if h_block is not None:
        for y in range(h_block, rows, h_block):
            ax.axhline(
                y,
                color='black',
                linestyle='-',
                linewidth=BLOCK_LINE_WIDTH
            )

    if w_block is not None:
        for x in range(w_block, cols, w_block):
            ax.axvline(
                x,
                color='black',
                linestyle='-',
                linewidth=BLOCK_LINE_WIDTH
            )

    ax.set_title(title, fontsize=FONT_SIZE, pad=12)

    if cols == ELL_WIDTH:
        ax.set_xlabel("ELLPACK Columns", fontsize=FONT_SIZE)
    else:
        ax.set_xlabel("Matrix Columns", fontsize=FONT_SIZE)

    ax.set_ylabel("Matrix Rows", fontsize=FONT_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_long_arrow(ax, start_rc, end_rc):
    r1, c1 = start_rc
    r2, c2 = end_rc

    x1, y1 = c1 + 0.5, r1 + 0.5
    x2, y2 = c2 + 0.5, r2 + 0.5

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color=ARROW_COLOR,
            linestyle="-",
            lw=LINE_WIDTH_SOLID
        )
    )


def draw_dotted_connector(ax, start_rc, end_rc):
    r1, c1 = start_rc
    r2, c2 = end_rc

    x1, y1 = c1 + 0.5, r1 + 0.5
    x2, y2 = c2 + 0.5, r2 + 0.5

    ax.plot(
        [x1, x2],
        [y1, y2],
        color=ARROW_COLOR,
        linestyle=":",
        lw=LINE_WIDTH_DOTTED,
        alpha=0.7
    )

# ==========================================
# 4. 行優先ELLPACKのアクセス順序
# ==========================================
def plot_row_major_ell_arrows(ax):
    rows = ROWS
    width = ELL_WIDTH

    for r in range(rows):
        draw_long_arrow(ax, (r, 0), (r, width - 1))

        if r < rows - 1:
            draw_dotted_connector(
                ax,
                (r, width - 1),
                (r + 1, 0)
            )

# ==========================================
# 5. プロット実行
# ==========================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 8),
    gridspec_kw={'width_ratios': [COLS, ELL_WIDTH]},
    constrained_layout=True
)

# (a) Sparse Matrix
draw_base_grid(
    axes[0],
    ROWS,
    COLS,
    dense_map,
    title=f"(a) Sparse Matrix\nMatrix Size: {ROWS} × {COLS}",
    h_block=None,
    w_block=None
)

# (b) Row-Major ELLPACK
title_b = f"(b) Row-Major ELLPACK\nRow Tile Size: {TILE_SIZE_B}"

draw_base_grid(
    axes[1],
    ROWS,
    ELL_WIDTH,
    ell_map,
    title=title_b,
    h_block=TILE_SIZE_B,
    w_block=None
)

plot_row_major_ell_arrows(axes[1])

# 凡例
legend_elements = [
    patches.Patch(
        facecolor=C_EMPTY,
        edgecolor=C_GRID,
        label='Zero elements'
    ),
    patches.Patch(
        facecolor=C_NZ,
        edgecolor=C_GRID,
        label='Nonzero elements'
    ),
    patches.Patch(
        facecolor=C_MARK,
        edgecolor=C_GRID,
        label='Second nonzero element in each row'
    )
]

fig.legend(
    handles=legend_elements,
    loc='outside lower center',
    ncol=3,
    frameon=False,
    fontsize=FONT_SIZE,
    borderaxespad=-0.5
)

plt.savefig("spmv_formats_fig_en.png", dpi=300, pad_inches=0.1)
plt.show()
