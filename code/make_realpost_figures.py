#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the (A)+(B) deep-revision experiments: real posteriors scored
with joint metrics on actual flowMC output. Writes to ../paper/figures."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")

df = pd.read_csv(f"{RES}/real_posteriors.csv")
ORDER = ["single@2000", "single@10000", "uniform_pool", "full_ensemble"]
LAB = {"single@2000": "single\n@2,000", "single@10000": "single\n@10,000",
       "uniform_pool": "uniform\npool", "full_ensemble": "full\nensemble"}
COL = {"single@2000": "#E8A33D", "single@10000": "#009E73",
       "uniform_pool": "#9467bd", "full_ensemble": "#1f77b4"}
TNAME = {"blr8": "Bayesian logistic regression (8-dim)",
         "hier": "Hierarchical variance model (10-dim, real funnel)"}

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- figure 10
# three metrics x two targets, grouped bars; the story is metric-dependent
metrics = [("marginal_JS", "Marginal JS distance\n(the paper's biased metric)"),
           ("energy_distance", "Energy distance\n(unbiased joint metric)"),
           ("mmd2", "MMD$^2$ (RBF)\n(unbiased joint metric)")]
targets = ["blr8", "hier"]
fig, axes = plt.subplots(len(targets), len(metrics),
                         figsize=(13, 6.4), squeeze=False)
for ti, tgt in enumerate(targets):
    d = df[df.target == tgt]
    for mi, (mk, mlab) in enumerate(metrics):
        ax = axes[ti][mi]
        g = d.groupby("variant")[mk].agg(["mean", "std"]).loc[ORDER]
        x = np.arange(len(ORDER))
        ax.bar(x, g["mean"], yerr=g["std"], capsize=3,
               color=[COL[v] for v in ORDER], width=0.66)
        for xi, v in enumerate(ORDER):
            ax.text(xi, g["mean"].iloc[xi] + g["std"].iloc[xi] + g["mean"].max() * 0.03,
                    f"{g['mean'].iloc[xi]:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([LAB[v] for v in ORDER], fontsize=7.8)
        ax.set_ylim(0, g["mean"].max() * 1.35)
        if mi == 0:
            ax.set_ylabel(f"{TNAME[tgt]}", fontsize=9)
        if ti == 0:
            ax.set_title(mlab, fontsize=9.5)
        # mark the single@10000 level for reference
        s10 = g.loc["single@10000", "mean"]
        ax.axhline(s10, ls="--", color=COL["single@10000"], lw=0.9, alpha=0.7)
fig.suptitle("Real posteriors, actual flowMC output: uniform pooling wins on every metric; "
             "the ensemble's disadvantage is confined to the biased marginal JS",
             fontsize=11, y=1.005)
fig.tight_layout()
save(fig, "fig_realpost_metrics")

# ---------------------------------------------------------------- figure 11
# "best variant" tally: which method is closest to the true posterior, by metric
recs = []
for (t, r), grp in df.groupby(["target", "rep"]):
    for mk, _ in metrics:
        best = grp.loc[grp[mk].idxmin(), "variant"]
        recs.append((mk, best))
rc = pd.DataFrame(recs, columns=["metric", "best"])
tab = pd.crosstab(rc["best"], rc["metric"]).reindex(index=ORDER,
                                                    columns=[m[0] for m in metrics]).fillna(0)
fig, ax = plt.subplots(figsize=(7.2, 4.0))
bottom = np.zeros(len(metrics))
mlabels = ["marginal JS\n(biased)", "energy distance\n(unbiased)", "MMD$^2$\n(unbiased)"]
xpos = np.arange(len(metrics))
for v in ORDER:
    vals = tab.loc[v].values
    ax.bar(xpos, vals, bottom=bottom, color=COL[v], width=0.6,
           label=v.replace("_", " ").replace("@", " @"))
    for xi in range(len(metrics)):
        if vals[xi] > 0:
            ax.text(xpos[xi], bottom[xi] + vals[xi] / 2, f"{int(vals[xi])}",
                    ha="center", va="center", fontsize=9,
                    color="white" if v in ("single@10000", "full_ensemble") else "black")
    bottom += vals
ax.set_xticks(xpos); ax.set_xticklabels(mlabels, fontsize=9)
ax.set_ylabel("times closest to the true posterior\n(out of 5 target$\\times$repetition runs)")
ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
ax.set_title("Which sampler is closest to the NUTS reference?\nThe answer flips "
             "between the biased and the unbiased metrics", fontsize=10.5)
save(fig, "fig_realpost_tally")

print("figures written to", FIG)
# echo the headline
piv = df.pivot_table(index=["target", "rep"], columns="variant", values="marginal_JS")
print("uniform_pool < single@10000 on marginal JS in",
      int(((piv["uniform_pool"] - piv["single@10000"]) < 0).sum()), "/ 5 runs")
for m in ["energy_distance", "mmd2"]:
    piv = df.pivot_table(index=["target", "rep"], columns="variant", values=m)
    print(f"uniform_pool < single@10000 on {m} in",
          int(((piv["uniform_pool"] - piv["single@10000"]) < 0).sum()), "/ 5 runs")
