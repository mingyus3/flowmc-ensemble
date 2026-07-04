#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_method_figure.py — schematic of the ensemble pipeline and controls
(Figure referenced in Section 3.4). Pure matplotlib; colors match the paper."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")

C_MEM, C_ENS, C_POOL, C_SINGLE, C_GREY = "#56B4E9", "#0072B2", "#9467BD", "#009E73", "#777777"

def box(ax, x, y, w, h, text, fc, fontsize=8.5, tc="white", lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="none" if lw == 0 else "black", lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, zorder=3)

def arrow(ax, x0, y0, x1, y1, color=C_GREY, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=1.4,
                                 linestyle=ls, zorder=1))

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.6); ax.axis("off")

# target
box(ax, 0.15, 1.9, 1.15, 0.8, "Target\n$\\pi(x)$", "#333333", 10)

# five diverse members with tempering note
archs = ["(64,64)", "(96,96)", "(128,64)", "(64,128)", "(32,64,32)"]
for i, a in enumerate(archs):
    y = 3.9 - i * 0.72
    box(ax, 2.0, y, 2.05, 0.56,
        f"flowMC member {i+1}\nhidden {a} · 2,000 samples", C_MEM, 7.2)
    arrow(ax, 1.35, 2.3, 2.0, y + 0.28)
ax.text(3.02, 4.42, "five diverse members, run independently", fontsize=8, ha="center", color=C_GREY)
ax.text(3.02, 0.36, "adaptive tempering: member $m$ targets $\\pi^{1/T_m}$,\n"
                    "$T_m$ updated from the previous member's ESS", fontsize=7.5,
        ha="center", color=C_GREY, style="italic")

# pooled samples node
box(ax, 4.55, 1.9, 1.25, 0.8, "pooled\n5 × 2,000\n= 10,000", "#555555", 8.5)
for i in range(5):
    arrow(ax, 4.05, 4.18 - i * 0.72, 4.55, 2.3)

# branch 1: aggregation stack (the proposed method)
box(ax, 6.35, 3.0, 2.6, 1.25,
    "aggregation stack\nquality member weighting (Eq. 1)\n× coverage reweighting ($k$-means)\n→ resample with replacement", C_ENS, 7.6)
arrow(ax, 5.8, 2.55, 6.35, 3.55, color=C_ENS)
box(ax, 9.25, 3.25, 1.1, 0.75, "ensemble\noutput\n10,000", C_ENS, 7.6)
arrow(ax, 8.95, 3.62, 9.25, 3.62, color=C_ENS)

# branch 2: uniform pool control
box(ax, 6.35, 1.65, 2.6, 0.75,
    "uniform pool (control)\nconcatenate raw samples — no weights,\nno resampling", C_POOL, 7.6)
arrow(ax, 5.8, 2.2, 6.35, 2.02, color=C_POOL)
box(ax, 9.25, 1.65, 1.1, 0.75, "pool\noutput\n10,000", C_POOL, 7.6)
arrow(ax, 8.95, 2.02, 9.25, 2.02, color=C_POOL)

# control row: single chain matched budget
box(ax, 2.0, 0.62, 3.8, 0.62,
    "single flowMC chain @ 10,000 (matched-budget control)", C_SINGLE, 8)
arrow(ax, 1.35, 2.1, 2.0, 0.95, color=C_SINGLE)
box(ax, 9.25, 0.55, 1.1, 0.75, "single\noutput\n10,000", C_SINGLE, 7.6)
arrow(ax, 5.8, 0.93, 9.25, 0.9, color=C_SINGLE)

ax.set_title("The diverse flowMC ensemble, its aggregation stack, and the two controls "
             "compared throughout the paper", fontsize=10)
fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "fig_pipeline.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_pipeline.png"), dpi=200, bbox_inches="tight")
print("wrote fig_pipeline.pdf/.png")
