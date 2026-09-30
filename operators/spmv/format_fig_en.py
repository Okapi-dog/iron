"""Illustrate sparse, ELL, SELL, and Blocked SELL-C-sigma in two figures.

Run with a Python environment containing matplotlib. The figure is schematic:
one colored cell represents a stored nonzero and a gray cell represents stored
padding. White space beyond the stair-step outline is not allocated.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Hiragino Maru Gothic Pro", "Yu Gothic", "Meiryo",
    "TakaoExGothic", "IPAPGothic", "VL PGothic",
    "Noto Sans CJK JP", "sans-serif",
]


WINDOW_ROWS = 16
BLOCK_HEIGHT = 2
BLOCK_WIDTH = 4
ROW_LENGTHS = (
    3, 10, 5, 12, 2, 8, 7, 11, 4, 9, 6, 10, 1, 8, 5, 3,
    4, 7, 2, 9, 5, 10, 3, 8, 1, 6, 4, 7, 2, 9, 5, 3,
)
ROWS = len(ROW_LENGTHS)
MATRIX_COLS = 32
ELL_WIDTH = max(ROW_LENGTHS)

# Keep the palette and line treatment of the original format_fig_en.py.
C_EMPTY = "#f0f0f0"
C_NZ = "#4a90e2"
C_MARK = "#ff9900"
C_GRID = "white"
INK = "black"
BLOCK_LINE_WIDTH = 3.5
FONT_SIZE = 22
LEGEND_FONT_SIZE = 16
WINDOW_FONT_SIZE = 17


def sorted_rows() -> list[int]:
    """Stable, descending NNZ order inside each independent window."""
    return [
        row
        for start in range(0, ROWS, WINDOW_ROWS)
        for row in sorted(
            range(start, start + WINDOW_ROWS),
            key=lambda r: -ROW_LENGTHS[r],
        )
    ]


def slice_widths(order: list[int], block_width: int) -> list[int]:
    widths = []
    for start in range(0, ROWS, BLOCK_HEIGHT):
        longest = max(ROW_LENGTHS[order[r]] for r in range(start, start + BLOCK_HEIGHT))
        widths.append(((longest + block_width - 1) // block_width) * block_width)
    return widths


def draw_cells(ax, order: list[int], widths: list[int], *, blocked: bool) -> None:
    for r, source_row in enumerate(order):
        width = widths[r // BLOCK_HEIGHT]
        nnz = ROW_LENGTHS[source_row]
        for c in range(width):
            ax.add_patch(patches.Rectangle(
                (c, r), 1, 1,
                facecolor=(C_MARK if c == 1 else C_NZ) if c < nnz else C_EMPTY,
                edgecolor=C_GRID, linewidth=0.5,
            ))

    if blocked:
        # Only actual 2x4 payload blocks are drawn. Completely empty blocks
        # do not exist and are therefore white, outside the stepped outline.
        for slice_index, width in enumerate(widths):
            y = BLOCK_HEIGHT * slice_index
            for x in range(0, width, BLOCK_WIDTH):
                ax.add_patch(patches.Rectangle(
                    (x, y), BLOCK_WIDTH, BLOCK_HEIGHT,
                    fill=False, edgecolor=INK, linewidth=1.2,
                ))


def draw_sparse_matrix(ax) -> None:
    """Use the same row NNZ counts as the three packed-format panels."""
    rng = random.Random(99)
    for r, nnz in enumerate(ROW_LENGTHS):
        nonzero_columns = sorted(rng.sample(range(MATRIX_COLS), nnz))
        colors = {c: C_MARK if item == 1 else C_NZ
                  for item, c in enumerate(nonzero_columns)}
        for c in range(MATRIX_COLS):
            ax.add_patch(patches.Rectangle(
                (c, r), 1, 1,
                facecolor=colors.get(c, C_EMPTY),
                edgecolor=C_GRID, linewidth=0.5,
            ))
    ax.add_patch(patches.Rectangle(
        (0, 0), MATRIX_COLS, ROWS,
        fill=False, edgecolor=INK, linewidth=1.5,
    ))


def draw_stepped_outline(ax, widths: list[int]) -> None:
    """Trace the allocated silhouette, including its inward/outward notches."""
    vertices = [(0, 0), (widths[0], 0)]
    for slice_index, width in enumerate(widths):
        bottom = (slice_index + 1) * BLOCK_HEIGHT
        vertices.append((width, bottom))
        if slice_index + 1 < len(widths):
            vertices.append((widths[slice_index + 1], bottom))
    vertices.extend([(0, ROWS), (0, 0)])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
    ax.add_patch(patches.PathPatch(
        MplPath(vertices, codes), fill=False,
        edgecolor=INK, linewidth=1.5,
        joinstyle="miter", capstyle="butt",
    ))


def prepare_axis(ax, title: str, cols: int = ELL_WIDTH) -> None:
    ax.set_xlim(-0.05, cols + 0.12)
    ax.set_ylim(ROWS + 0.15, -0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=FONT_SIZE, color=INK, pad=12)


def add_window_marks(ax, widths: list[int], *, right_label: bool = False) -> None:
    boundary_width = max(widths[WINDOW_ROWS // BLOCK_HEIGHT - 1],
                         widths[WINDOW_ROWS // BLOCK_HEIGHT])
    ax.plot([0, boundary_width], [WINDOW_ROWS, WINDOW_ROWS],
            color=INK, linewidth=BLOCK_LINE_WIDTH, zorder=10)
    if right_label:
        for number in (0, 1):
            top = number * WINDOW_ROWS
            x = ELL_WIDTH + 0.72
            ax.plot([x, x + 0.18], [top, top], color=INK, lw=1.5, clip_on=False)
            ax.plot([x + 0.18, x + 0.18], [top, top + WINDOW_ROWS],
                    color=INK, lw=1.5, clip_on=False)
            ax.plot([x, x + 0.18], [top + WINDOW_ROWS, top + WINDOW_ROWS],
                    color=INK, lw=1.5, clip_on=False)
            ax.text(x + 0.42, top + WINDOW_ROWS / 2,
                    f"window {number + 1}",
                    va="center", ha="left", color=INK,
                    fontsize=WINDOW_FONT_SIZE, clip_on=False)


def build_figures():
    original = list(range(ROWS))
    reordered = sorted_rows()
    ell_widths = [ELL_WIDTH] * (ROWS // BLOCK_HEIGHT)
    sell_widths = slice_widths(original, BLOCK_WIDTH)
    blocked_widths = slice_widths(reordered, BLOCK_WIDTH)

    fig1, axes1 = plt.subplots(
        1, 2, figsize=(14.5, 10.7), dpi=180,
        gridspec_kw={"width_ratios": [MATRIX_COLS, ELL_WIDTH]},
    )
    fig1.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.12, wspace=0.12)
    prepare_axis(axes1[0], "(a) Sparse Matrix", MATRIX_COLS)
    draw_sparse_matrix(axes1[0])
    prepare_axis(axes1[1], "(b) ELL")
    draw_cells(axes1[1], original, ell_widths, blocked=False)
    axes1[1].add_patch(patches.Rectangle((0, 0), ELL_WIDTH, ROWS,
                 fill=False, edgecolor=INK, linewidth=1.45))
    add_window_marks(axes1[1], ell_widths)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 13.2), dpi=180)
    fig2.subplots_adjust(left=0.07, right=0.84, top=0.94, bottom=0.09, wspace=0.22)
    prepare_axis(axes2[0], "(c) SELL")
    draw_cells(axes2[0], original, sell_widths, blocked=True)
    draw_stepped_outline(axes2[0], sell_widths)
    add_window_marks(axes2[0], sell_widths)

    prepare_axis(axes2[1], r"(d) Blocked SELL-C-$\sigma$")
    draw_cells(axes2[1], reordered, blocked_widths, blocked=True)
    draw_stepped_outline(axes2[1], blocked_widths)
    add_window_marks(axes2[1], blocked_widths, right_label=True)

    legend = [
        patches.Patch(facecolor=C_EMPTY, edgecolor=C_GRID, label="Zero / padding elements"),
        patches.Patch(facecolor=C_NZ, edgecolor=C_GRID, label="Nonzero elements"),
        patches.Patch(facecolor=C_MARK, edgecolor=C_GRID,
                      label="Second nonzero element in each row"),
    ]
    for fig in (fig1, fig2):
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.04),
                   ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE,
                   handlelength=1.6)

    return fig1, fig2, reordered, sell_widths, blocked_widths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("spmv_formats_fig_en.png"))
    args = parser.parse_args()

    fig1, fig2, order, sell_widths, blocked_widths = build_figures()
    for start in (0, WINDOW_ROWS):
        lengths = [ROW_LENGTHS[row] for row in order[start:start + WINDOW_ROWS]]
        assert lengths == sorted(lengths, reverse=True)
        assert set(order[start:start + WINDOW_ROWS]) == set(range(start, start + WINDOW_ROWS))
    assert all(width % BLOCK_WIDTH == 0 for width in sell_widths + blocked_widths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    second_output = args.output.with_name(args.output.stem + "_sell" + args.output.suffix)
    for fig, output in ((fig1, args.output), (fig2, second_output)):
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
