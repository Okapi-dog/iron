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
ROWS = 12       # 行数
COLS = 16       # 元の行列の列数
WIDTH = 10      # 確保するデータ幅

# 転送設定
ELEMENTS_PER_CORE = 12  # 1つのコアに割り当てる非ゼロ要素数

# 色設定
C_NZ = '#4a90e2'   # 青 (非ゼロ)
C_EMPTY = '#f9f9f9' # (a)の背景（ほぼ白）

# コア割り当て用の色
CORE_COLORS = [
    '#FFB3BA', # Core 1 (赤系)
    '#BAFFC9', # Core 2 (緑系)
    '#BAE1FF', # Core 3 (青系)
    '#FFFFBA', # Core 4 (黄系)
    '#E2CBF7', # Core 5 (紫系)
    '#FFDFBA', # Core 6 (橙系)
    '#D0F0C0', # Core 7
]

# ★変更点: (b)のデータなし部分も白にする（(a)と統一）
C_NO_DATA = '#ffffff' 
C_GRID = '#dddddd' # グリッド線を少し濃くして、白背景でも枠が見えるようにする

# フォントサイズ
FONT_TITLE = 16
FONT_LABEL = 14
FONT_CELL = 12
FONT_ROW_INDEX = 11

# ==========================================
# 2. データ生成 (固定シード)
# ==========================================
np.random.seed(42) 

dense_map = {}      # 左図用
assign_map = {}     # 右図用

valid_element_counter = 0

for r in range(ROWS):
    n_items = np.random.randint(2, WIDTH + 1)
    col_indices = np.random.choice(range(COLS), size=n_items, replace=False)
    col_indices.sort()
    
    for w in range(WIDTH):
        if w < n_items:
            original_c = col_indices[w]
            dense_map[(r, original_c)] = C_NZ
            
            core_id = (valid_element_counter // ELEMENTS_PER_CORE) + 1
            valid_element_counter += 1
            
            assign_map[(r, w)] = {
                'core_id': core_id,
                'is_valid': True
            }
        else:
            assign_map[(r, w)] = {
                'core_id': None,
                'is_valid': False
            }

# ==========================================
# 3. 表データの集計 (横向き用に構造変更)
# ==========================================
core_stats = {} 

for (r, w), info in assign_map.items():
    if info['is_valid']:
        cid = info['core_id']
        if cid not in core_stats:
            core_stats[cid] = set()
        core_stats[cid].add(r)

sorted_core_ids = sorted(core_stats.keys())

# --- 横向き表のためのデータ構築 ---
# 列ラベル: Core 1, Core 2, ...
col_labels = [f"Core {cid}" for cid in sorted_core_ids]

# 行ラベル
row_labels = ["担当行数", "出力行番号"]

# データ部分 (2行 x コア数列)
row_counts = []
row_ranges = []

for cid in sorted_core_ids:
    rows = sorted(list(core_stats[cid]))
    
    # 担当行数
    row_counts.append(str(len(rows)))
    
    # 出力行番号
    if len(rows) > 0:
        if rows[0] == rows[-1]:
             row_ranges.append(f"{rows[0]}")
        else:
             row_ranges.append(f"{rows[0]}~{rows[-1]}")
    else:
        row_ranges.append("-")

cell_text = [row_counts, row_ranges]

# ==========================================
# 4. 描画関数群
# ==========================================

def draw_sparse_matrix(ax, rows, cols, data_map):
    ax.set_aspect('equal')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, facecolor=C_EMPTY))
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) in data_map:
                face = data_map[(r, c)]
                # グリッド線は白
                rect = patches.Rectangle((c, r), 1, 1, facecolor=face, edgecolor='white', lw=0.5)
                ax.add_patch(rect)
            else:
                rect = patches.Rectangle((c, r), 1, 1, fill=False, edgecolor='#e0e0e0', lw=0.5)
                ax.add_patch(rect)
                
    ax.add_patch(patches.Rectangle((0, 0), cols, rows, fill=False, edgecolor='black', lw=1.5))
    ax.set_title(f"(a) 疎行列\nサイズ: {rows}x{cols}", fontsize=FONT_TITLE)
    ax.set_xlabel("Matrix Columns", fontsize=FONT_LABEL)
    ax.set_ylabel("Matrix Rows", fontsize=FONT_LABEL)
    ax.set_xticks([])
    ax.set_yticks([])

def draw_core_assignment(ax, rows, width, data_map):
    ax.set_aspect('equal')
    ax.set_xlim(0, width)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    
    # 背景を白に設定
    ax.add_patch(patches.Rectangle((0, 0), width, rows, facecolor="white"))
    
    for r in range(rows):
        for c in range(width):
            info = data_map.get((r, c))
            is_valid = info['is_valid']
            core_id = info['core_id']
            
            if is_valid:
                color_idx = (core_id - 1) % len(CORE_COLORS)
                face_color = CORE_COLORS[color_idx]
                text_str = str(core_id)
                text_color = 'black'
                font_weight = 'bold'
                zorder = 2
                edge_color = 'white' # データ間の区切りは白で見やすく
            else:
                # ★変更: データなしも白にする
                face_color = C_NO_DATA
                text_str = "" 
                text_color = ''
                font_weight = 'normal'
                zorder = 1
                edge_color = C_GRID # 空き領域の枠線は薄いグレーで見せる
            
            rect = patches.Rectangle((c, r), 1, 1, facecolor=face_color, edgecolor=edge_color, lw=1.0, zorder=zorder)
            ax.add_patch(rect)
            
            if text_str:
                ax.text(c + 0.5, r + 0.5, text_str, ha='center', va='center',
                        fontsize=FONT_CELL, fontweight=font_weight, color=text_color, zorder=3)
            
            if c < width - 1:
                next_info = data_map.get((r, c+1))
                if is_valid and next_info['is_valid']:
                    if core_id != next_info['core_id']:
                        ax.plot([c+1, c+1], [r, r+1], color='black', lw=2.5, zorder=4)

    # 外枠
    ax.add_patch(patches.Rectangle((0, 0), width, rows, fill=False, edgecolor='black', lw=2.0, zorder=5))
    
    # 右側に行番号
    for r in range(rows):
        ax.text(width + 0.2, r + 0.5, str(r), 
                ha='left', va='center', 
                fontsize=FONT_ROW_INDEX, color='#333333', fontweight='bold')
    
    ax.text(width + 0.2, -0.8, "Row\nIndex", 
            ha='left', va='bottom', fontsize=10, color='#333333')

    ax.set_title(f"(b) コアへのデータ割り当て\n(非ゼロ要素{ELEMENTS_PER_CORE}個ごとにコアへ配分)", fontsize=FONT_TITLE)
    ax.set_xlabel("Data Columns", fontsize=FONT_LABEL)
    ax.set_ylabel("Matrix Rows", fontsize=FONT_LABEL)
    ax.set_xticks([])
    ax.set_yticks([])

# ==========================================
# 5. プロット実行
# ==========================================
fig = plt.figure(figsize=(14, 11)) 

# レイアウト: 下部の表エリアを少し狭くしても横向きなら入る
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.25], width_ratios=[COLS, WIDTH], hspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])

draw_sparse_matrix(ax1, ROWS, COLS, dense_map)
draw_core_assignment(ax2, ROWS, WIDTH, assign_map)

# 凡例
legend_patches = [
    patches.Patch(facecolor=C_NZ, label='非ゼロ要素'),
    patches.Patch(facecolor='white', alpha=0, label='        '), 
    patches.Patch(facecolor='white', edgecolor='black', label='数字: 割り当てコアID'),
    # データなしは白になったので、凡例からも「グレーの四角」を消すか、「白枠」として残すか。
    # ここでは混乱を避けるため「データなし」の凡例自体を削除し、図中の見た目で語らせるか、
    # あるいは枠線付きの白を見せるか。今回は枠線付き白を表示します。
    patches.Patch(facecolor='white', edgecolor=C_GRID, label='データなし'),
]

fig.legend(handles=legend_patches, 
           loc='center', 
           bbox_to_anchor=(0.5, 0.25), # 表の上
           ncol=4, 
           frameon=False, 
           fontsize=14)

# --- 表の描画 (横向き) ---
ax_table.axis('off')
table = ax_table.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 2.0) 

# ヘッダー(Core ID)の色付け
for j, cid in enumerate(sorted_core_ids):
    color_idx = (cid - 1) % len(CORE_COLORS)
    bg_color = CORE_COLORS[color_idx]
    
    # 列ヘッダー (行0, 列j) ※rowLabelsがあるため列インデックスはずれないが、
    # table.get_celld()のキーは (row_idx, col_idx) で、ヘッダーは row=0
    # データ部分は row=1~。
    
    # colLabelsのセル
    cell = table[(0, j)]
    cell.set_facecolor(bg_color)
    cell.set_text_props(weight='bold')

# 行ラベル(左端)の色調整
for i in range(len(row_labels)):
    # 行ヘッダーは列=-1
    cell = table[(i + 1, -1)]
    cell.set_facecolor('#f0f0f0')
    cell.set_text_props(weight='bold')

plt.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.05)
plt.show()
plt.savefig("sparse_core_assignment1.png", dpi=300, bbox_inches='tight')