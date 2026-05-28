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
    '#E57373', # Core 1
    '#81C784', # Core 2
    '#64B5F6', # Core 3
    '#FFD54F'  # Core 4
]

C_EMPTY_BG = '#fcfcfc'
C_GRID_EMPTY = '#e0e0e0'
C_GRID_NZ = 'white'
C_HEADER_BG = '#eeeeee'

FONT_LABEL = 30      
FONT_INDEX = 24      
FONT_LEGEND = 30  

# ==========================================
# 2. データ生成 (固定シード)
# ==========================================
np.random.seed(42) 
matrix_data = {} 
valid_element_counter = 0

for r in range(ROWS):
    n_items = np.random.randint(MIN_WIDTH, WIDTH + 1)
    col_indices = np.random.choice(range(COLS), size=n_items, replace=False)
    col_indices.sort()
    for c in col_indices:
        core_id = (valid_element_counter // ELEMENTS_PER_CORE) + 1
        if core_id > 4: break
        valid_element_counter += 1
        matrix_data[(r, c)] = core_id

# ==========================================
# 3. 描画関数
# ==========================================
def draw_colored_sparse_matrix(ax, rows, cols, data_map):
    ax.set_aspect('equal')
    ax.set_xlim(-0.8, cols)
    ax.set_ylim(rows, -0.8) 
    
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, facecolor=C_EMPTY_BG))
    
    for r in range(rows):
        for c in range(cols):
            rect = patches.Rectangle((c, r), 1, 1, fill=False, edgecolor=C_GRID_EMPTY, lw=0.5)
            ax.add_patch(rect)
            
            if (r, c) in data_map:
                color_idx = (data_map[(r, c)] - 1) % len(CORE_COLORS)
                face_color = CORE_COLORS[color_idx]
                rect_nz = patches.Rectangle((c, r), 1, 1, facecolor=face_color, edgecolor=C_GRID_NZ, lw=1.0)
                ax.add_patch(rect_nz)

    ax.add_patch(patches.Rectangle((0, 0), cols, rows, fill=False, edgecolor='black', lw=2.0))

    for r in range(rows):
        rect = patches.Rectangle((-0.8, r), 0.8, 1, facecolor=C_HEADER_BG, edgecolor='white', lw=1)
        ax.add_patch(rect)
        ax.text(-0.4, r + 0.5, str(r), ha='center', va='center', fontsize=FONT_INDEX, color='#555555', fontweight='bold')

    for c in range(cols):
        rect = patches.Rectangle((c, -0.8), 1, 0.8, facecolor=C_HEADER_BG, edgecolor='white', lw=1)
        ax.add_patch(rect)
        ax.text(c + 0.5, -0.4, str(c), ha='center', va='center', fontsize=FONT_INDEX, color='#555555', fontweight='bold')
        
    ax.text(cols / 2, -1.0, "列", ha='center', va='bottom', fontsize=FONT_LABEL, fontweight='bold')
    ax.text(-1.0, rows / 2, "行", ha='right', va='center', fontsize=FONT_LABEL, fontweight='bold')
    ax.axis('off')

# ==========================================
# 4. プロット実行
# ==========================================
fig = plt.figure(figsize=(12, 10)) # 横長に並べるため横幅を少し広げました
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.1], hspace=0.1)

ax_matrix = fig.add_subplot(gs[0, 0])
ax_legend = fig.add_subplot(gs[1, 0])
ax_legend.axis('off')

draw_colored_sparse_matrix(ax_matrix, ROWS, COLS, matrix_data)

# --- 凡例（1行に横並び） ---
# 順序通りに作成
legend_elements = [
    patches.Patch(facecolor=CORE_COLORS[0], edgecolor='#555555', label='Core 1'),
    patches.Patch(facecolor=CORE_COLORS[1], edgecolor='#555555', label='Core 2'),
    patches.Patch(facecolor=CORE_COLORS[2], edgecolor='#555555', label='Core 3'),
    patches.Patch(facecolor=CORE_COLORS[3], edgecolor='#555555', label='Core 4'),
]

ax_legend.legend(
    handles=legend_elements, 
    loc='center', 
    ncol=4,               # 4列に設定（1行に並ぶ）
    fontsize=FONT_LEGEND, 
    frameon=False,
    handletextpad=0.3,    
    columnspacing=1.0     # 1行に収まるよう間隔を調整
)

plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.05)
plt.show()
plt.savefig("spmv_transfer_problem2_5.png", dpi=300, bbox_inches='tight')