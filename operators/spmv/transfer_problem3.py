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

# 色設定
C_NZ_BLUE = '#4a90e2'    # 非ゼロ要素の青
C_EMPTY_BG = '#fcfcfc'   # 空セルの背景
C_GRID_EMPTY = '#e0e0e0' # 空セルの枠線
C_GRID_NZ = 'white'      # 非ゼロセルの枠線
C_HEADER_BG = '#eeeeee'  # ヘッダー背景色

# フォントサイズ
FONT_LABEL = 30      # 「行」「列」
FONT_INDEX = 24      # ヘッダー番号
FONT_LEGEND = 30     # 凡例 (大きくしました)

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
        valid_element_counter += 1
        matrix_data[(r, c)] = core_id

# ==========================================
# 3. 描画関数
# ==========================================
def draw_simple_sparse_matrix(ax, rows, cols, data_map):
    ax.set_aspect('equal')
    ax.set_xlim(-0.8, cols)
    ax.set_ylim(rows, -0.8) # Y軸反転
    
    # 1. データ領域
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, facecolor=C_EMPTY_BG))
    
    for r in range(rows):
        for c in range(cols):
            rect = patches.Rectangle((c, r), 1, 1, fill=False, edgecolor=C_GRID_EMPTY, lw=0.5)
            ax.add_patch(rect)
            
            if (r, c) in data_map:
                rect_nz = patches.Rectangle((c, r), 1, 1, facecolor=C_NZ_BLUE, edgecolor=C_GRID_NZ, lw=1.0)
                ax.add_patch(rect_nz)

    # データ領域の外枠
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, fill=False, edgecolor='black', lw=2.0))

    # 2. ヘッダー領域
    
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
        
    # 3. 軸ラベル
    ax.text(cols / 2, -1.0, "列", ha='center', va='bottom', fontsize=FONT_LABEL, fontweight='bold')
    ax.text(-1.0, rows / 2, "行", ha='right', va='center', fontsize=FONT_LABEL, fontweight='bold')

    # 軸設定のクリア
    ax.axis('off')

# ==========================================
# 4. プロット実行
# ==========================================
fig, ax = plt.subplots(figsize=(10, 9.5)) # 縦を少し伸ばして余白確保

# --- 行列描画 ---
draw_simple_sparse_matrix(ax, ROWS, COLS, matrix_data)

# --- 凡例 ---
legend_elements = [
    patches.Patch(facecolor=C_NZ_BLUE, edgecolor='white', label='非ゼロ要素')
]

# 図の下部に配置 (フォントが大きいので位置をさらに下げる)
ax.legend(handles=legend_elements, 
          loc='upper center', 
          bbox_to_anchor=(0.5, -0.02), # 行列の底辺から少し離す
          ncol=1, 
          frameon=False, 
          fontsize=FONT_LEGEND)

# 余白調整: bottomを大きく取って凡例が見切れないようにする
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.show()
plt.savefig("sparse_matrix_simple_blue_large_legend.png", dpi=300, bbox_inches='tight')