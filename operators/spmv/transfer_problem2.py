import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 0. フォント等の設定
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 
    'TakaoExGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP', 'sans-serif'
]

# ==========================================
# 1. パラメータ設定
# ==========================================
ROWS = 5       
COLS = 6       
WIDTH = 5      
MIN_WIDTH = 1  
ELEMENTS_PER_CORE = 4  

CORE_COLORS = [
    '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#E2CBF7', '#FFDFBA', '#D0F0C0'
]
C_EMPTY_BG = '#fcfcfc'
C_GRID_EMPTY = '#e0e0e0'
C_GRID_NZ = 'white'
C_HEADER_BG = '#eeeeee' # ヘッダー背景色

# フォントサイズ
FONT_LABEL = 30      # 「行」「列」
FONT_CELL = 40       # データ
FONT_INDEX = 24      # ヘッダー番号
FONT_TABLE = 14

# ==========================================
# 2. データ生成 (固定シード)
# ==========================================
np.random.seed(42) 

matrix_data = {} 
valid_element_counter = 0
core_stats = {} 

for r in range(ROWS):
    n_items = np.random.randint(MIN_WIDTH, WIDTH + 1)
    col_indices = np.random.choice(range(COLS), size=n_items, replace=False)
    col_indices.sort()
    for c in col_indices:
        core_id = (valid_element_counter // ELEMENTS_PER_CORE) + 1
        valid_element_counter += 1
        matrix_data[(r, c)] = core_id
        if core_id not in core_stats:
            core_stats[core_id] = set()
        core_stats[core_id].add(r)

# ==========================================
# 3. 表データの作成
# ==========================================
sorted_core_ids = sorted(core_stats.keys())
col_labels = [f"Core {cid}" for cid in sorted_core_ids]
row_labels = ["担当行の数", "出力行番号"]
row_counts = []
row_ranges = []

for cid in sorted_core_ids:
    rows = sorted(list(core_stats[cid]))
    row_counts.append(str(len(rows)))
    if len(rows) > 0:
        if rows[0] == rows[-1]:
             row_ranges.append(f"{rows[0]}")
        else:
             row_ranges.append(f"{rows[0]}~{rows[-1]}")
    else:
        row_ranges.append("-")
table_cells = [row_counts, row_ranges]

# ==========================================
# 4. 描画関数
# ==========================================
def draw_colored_sparse_matrix(ax, rows, cols, data_map):
    ax.set_aspect('equal')
    # 範囲設定: ヘッダー分(-0.8)とラベル分を考慮
    ax.set_xlim(-0.8, cols)
    ax.set_ylim(rows, -0.8) # Y軸反転
    
    # 1. データ領域の描画
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, facecolor=C_EMPTY_BG))
    
    for r in range(rows):
        for c in range(cols):
            # 空枠
            rect = patches.Rectangle((c, r), 1, 1, fill=False, edgecolor=C_GRID_EMPTY, lw=0.5)
            ax.add_patch(rect)
            
            # データあり
            if (r, c) in data_map:
                core_id = data_map[(r, c)]
                color_idx = (core_id - 1) % len(CORE_COLORS)
                face_color = CORE_COLORS[color_idx]
                
                rect_nz = patches.Rectangle((c, r), 1, 1, facecolor=face_color, edgecolor=C_GRID_NZ, lw=1.0)
                ax.add_patch(rect_nz)
                
                ax.text(c + 0.5, r + 0.5, str(core_id), 
                        ha='center', va='center', 
                        fontsize=FONT_CELL, fontweight='bold', color='black')

    # データ領域の外枠
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, fill=False, edgecolor='black', lw=2.0))

    # 2. ヘッダー領域の描画
    
    # 行ヘッダー (左側)
    for r in range(rows):
        rect = patches.Rectangle((-0.8, r), 0.8, 1, facecolor=C_HEADER_BG, edgecolor='white', lw=1)
        ax.add_patch(rect)
        ax.text(-0.4, r + 0.5, str(r), ha='center', va='center', fontsize=FONT_INDEX, color='#555555', fontweight='bold')

    # 列ヘッダー (上側)
    for c in range(cols):
        rect = patches.Rectangle((c, -0.8), 1, 0.8, facecolor=C_HEADER_BG, edgecolor='white', lw=1)
        ax.add_patch(rect)
        ax.text(c + 0.5, -0.4, str(c), ha='center', va='center', fontsize=FONT_INDEX, color='#555555', fontweight='bold')
        
    # 3. 軸ラベル (英語削除・位置調整)
    # 列ラベル（上）
    ax.text(cols / 2, -1.0, "列", ha='center', va='bottom', fontsize=FONT_LABEL, fontweight='bold')
    
    # 行ラベル（左）
    ax.text(-1.0, rows / 2, "行", ha='right', va='center', fontsize=FONT_LABEL, fontweight='bold')

    # 軸設定のクリア
    ax.axis('off')


# ==========================================
# 5. プロット実行
# ==========================================
fig = plt.figure(figsize=(10, 11))

# 上下の間隔を詰める (hspace=0.1)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.25], hspace=0.1)

ax_matrix = fig.add_subplot(gs[0, 0])
ax_table = fig.add_subplot(gs[1, 0])

# --- 行列描画 ---
draw_colored_sparse_matrix(ax_matrix, ROWS, COLS, matrix_data)

# --- 表描画 ---
ax_table.axis('off')
table = ax_table.table(
    cellText=table_cells,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(FONT_TABLE)
table.scale(1, 2.5) 

for j, cid in enumerate(sorted_core_ids):
    color_idx = (cid - 1) % len(CORE_COLORS)
    bg_color = CORE_COLORS[color_idx]
    cell = table[(0, j)]
    cell.set_facecolor(bg_color)
    cell.set_text_props(weight='bold')

for i in range(len(row_labels)):
    cell = table[(i + 1, -1)]
    cell.set_facecolor('#f0f0f0')
    cell.set_text_props(weight='bold')

# 余白調整: leftとtopを詰めて全体を大きく見せる
plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.05)
plt.show()
plt.savefig("sparse_core_spreadsheet_style_compact.png", dpi=300, bbox_inches='tight')