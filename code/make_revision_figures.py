#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produce the new figures for the revised paper from the supplementary CSVs.
Outputs PDFs (and PNG previews) to ../paper/figures."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
M_REF = 20000.0

C = dict(floor="0.45", exact="#009E73", pool="#9467bd", resamp="#56B4E9",
         cover="#4C72B0", full="#1f77b4", meas="#B23A48", single2k="#E8A33D")

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------ figure 2'
fl = pd.read_csv(f"{RES}/supp_floor.csv")
cur = fl[fl.kind == "curve"].groupby("N")["JS"].mean()
x = 1.0 / cur.index.values + 1.0 / M_REF
c_fit = float(np.sum(x * cur.values**2) / np.sum(x * x))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
ax = axes[0]
Ngrid = np.geomspace(450, 22000, 300)
ax.plot(cur.index, cur.values, color=C["floor"], lw=2,
        label="perfect-sampler floor (measured)")
ax.plot(Ngrid, np.sqrt(c_fit * (1 / Ngrid + 1 / M_REF)), "--", color=C["meas"],
        lw=1.6, label=r"$\sqrt{c\,(1/N+1/M)}$, $c=%.1f$ nats" % c_fit)
ax.scatter([2000], [0.1038], color=C["single2k"], zorder=5, s=45,
           label="single @2,000")
ax.scatter([10000], [0.0649], color=C["exact"], zorder=5, s=45,
           label="single @10,000")
ax.scatter([10000], [0.0849], color=C["full"], zorder=5, s=55, marker="s",
           label="full ensemble (nominal 10,000)")
ax.annotate("ensemble sits far above the 10k floor:\nan aggregation bias, "
            "not a sample shortage\n(measured sample-set ESS 9,848)",
            xy=(10000, 0.0849), xytext=(1500, 0.115), fontsize=8.5,
            color=C["full"], arrowprops=dict(arrowstyle="-", color=C["full"], lw=0.8))
ax.set_xscale("log")
ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000])
ax.set_xticklabels(["500", "1k", "2k", "5k", "10k", "20k"])
ax.set_xlabel("number of samples $N$")
ax.set_ylabel("avg. marginal JS distance")
ax.set_title("(a) Floor vs. sample size and the $\\chi^2$ prediction", fontsize=10)
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
bins_df = fl[fl.kind == "bins"]
markers = {25: "o", 50: "s", 100: "^", 200: "D"}
for B, sub in bins_df.groupby("nbins"):
    g = sub.groupby("N")["JS"].mean()
    xb = 1.0 / g.index.values + 1.0 / M_REF
    cb = float(np.sum(xb * g.values**2) / np.sum(xb * xb))
    xs = np.linspace(0, xb.max() * 1.05, 50)
    ax.plot(xs * 1e3, cb * xs, "-", lw=1.1, alpha=0.7, color="0.55")
    ax.scatter(xb * 1e3, g.values**2, s=32, marker=markers[B],
               label=f"$B={B}$:  $c={cb:.1f}$", zorder=4)
ax.set_xlabel(r"$1/N + 1/M$  ($\times 10^{-3}$)")
ax.set_ylabel(r"$\widehat{\mathrm{JS}}^{\,2}$ (nats)")
ax.set_title("(b) $\\widehat{\\mathrm{JS}}^{\\,2}$ is linear in $1/N+1/M$ "
             "with slope $\\propto B$", fontsize=10)
ax.legend(fontsize=8, title="histogram bins", title_fontsize=8)
fig.suptitle("The marginal-JS noise floor of a perfect sampler is explained by "
             "finite-sample estimator bias", fontsize=11, y=1.02)
save(fig, "fig_floor_formula")

# ------------------------------------------------------------------ figure 7
dec = pd.read_csv(f"{RES}/supp_decomposition.csv")
pr = dec.groupby(["variant", "rep"])["JS"].mean().groupby("variant").agg(["mean", "std"])
order = ["exact_single_10000", "pool_uniform", "pool_resample_uniform",
         "pool_memberweight_resample", "pool_coverage_resample",
         "pool_full_aggregation"]
labels = ["Exact single\ndraw @10,000", "Uniform pool\n(5\u00d72,000 exact)",
          "+ resampling\n(uniform weights)", "+ member weighting\n+ resampling",
          "+ coverage reweighting\n+ resampling", "Full aggregation\nstack"]
colors = [C["exact"], C["pool"], C["resamp"], C["resamp"], C["cover"], C["full"]]
m = pr.loc[order, "mean"].values; s = pr.loc[order, "std"].values

fig, ax = plt.subplots(figsize=(9.2, 4.4))
bars = ax.bar(range(6), m, yerr=s, color=colors, capsize=3, width=0.62)
for i, v in enumerate(m):
    ax.text(i, v + 0.0022, f"{v:.4f}", ha="center", fontsize=9)
ax.axhline(0.0435, ls="--", color=C["floor"], lw=1.2)
ax.text(5.42, 0.0435, "floor @10k", fontsize=8, color=C["floor"], va="bottom",
        ha="right")
ax.axhline(0.0849, ls=":", color=C["meas"], lw=1.6)
ax.text(0.02, 0.0857, "flowMC ensemble, measured (0.0849)", fontsize=9,
        color=C["meas"])
ax.axhline(0.058, ls=":", color="0.6", lw=1.0)
ax.text(2.0, 0.0585, r"$\chi^2$ prediction for resampling (0.058)",
        fontsize=8, color="0.4")
ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=8.6)
ax.set_ylabel("Mean marginal JS distance over 11 configurations")
ax.set_ylim(0, 0.098)
ax.set_title("Sampler-free decomposition: the aggregation operators applied to "
             "exact samples reproduce ~90% of the ensemble's deficit",
             fontsize=10.5)
save(fig, "fig_decomposition")

# ------------------------------------------------------------------ figure 8
tp = pd.read_csv(f"{RES}/supp_tempering.csv")
g = tp.groupby(["target", "dim", "variant"])["JS"].agg(["mean", "std"])
meas = {("correlated_gaussian", 20): (0.0807, 0.0025),
        ("correlated_gaussian", 50): (0.0712, 0.0013),
        ("funnel", 20): (0.0736, 0.0017),
        ("funnel", 50): (0.0794, 0.0014)}
cfgs = [("correlated_gaussian", 20, "Correlated 20D"),
        ("correlated_gaussian", 50, "Correlated 50D"),
        ("funnel", 20, "Funnel 20D"), ("funnel", 50, "Funnel 50D")]
variants = [("pool_uniform_exact", "uniform pool, exact members", C["pool"]),
            ("pool_uniform_tempered", "uniform pool, tempered members",
             "#c49bd8"),
            ("tempered_full_aggregation", "tempered + full aggregation",
             C["full"])]
fig, ax = plt.subplots(figsize=(9.2, 4.2))
w = 0.19
for j, (vk, vl, col) in enumerate(variants):
    mm = [g.loc[(t, d, vk), "mean"] for t, d, _ in cfgs]
    ss = [g.loc[(t, d, vk), "std"] for t, d, _ in cfgs]
    ax.bar(np.arange(4) + (j - 1.5) * w, mm, width=w, yerr=ss, capsize=2,
           color=col, label=vl)
mm = [meas[(t, d)][0] for t, d, _ in cfgs]
ss = [meas[(t, d)][1] for t, d, _ in cfgs]
ax.bar(np.arange(4) + 1.5 * w, mm, width=w, yerr=ss, capsize=2,
       color=C["meas"], label="flowMC ensemble, measured (Table 1)")
ax.axhline(0.0435, ls="--", color=C["floor"], lw=1.2)
ax.text(3.62, 0.0442, "floor @10k", fontsize=8, color=C["floor"], ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels([c[2] for c in cfgs])
ax.set_ylabel("Marginal JS distance")
ax.set_ylim(0, 0.102)
ax.legend(fontsize=8.4, ncol=2)
ax.set_title("Exact tempered members + the aggregation stack reproduce the "
             "measured flowMC ensemble", fontsize=10.5)
save(fig, "fig_tempering")

# ------------------------------------------------------------------ figure 9
mb = pd.read_csv(f"{RES}/supp_members.csv")
g = mb.groupby(["variant", "M"])["JS"].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for vk, vl, col, mk in [("exact_single", "exact single draw of $2000M$",
                         C["exact"], "o"),
                        ("pool_uniform", "uniform pool of $M$ members",
                         C["pool"], "s"),
                        ("pool_full_aggregation", "full aggregation stack",
                         C["full"], "^")]:
    mm = g.loc[vk]
    ax.errorbar(mm.index, mm["mean"], yerr=mm["std"], marker=mk, color=col,
                lw=1.6, capsize=3, label=vl)
ax.annotate("aggregated pool at $M{=}20$ (40,000 samples)\n$\\approx$ exact "
            "draw at $M{=}5$ (10,000)", xy=(20, 0.0443), xytext=(6.3, 0.075),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20])
ax.set_xticklabels(["2", "5", "10", "20"])
ax.set_xlabel("number of ensemble members $M$ (2,000 samples each)")
ax.set_ylabel("Mean marginal JS distance (4 targets, 20D)")
ax.legend(fontsize=8.6)
ax.set_title("More members do not repair the aggregation:\nthe aggregated pool "
             "needs roughly $4\\times$ the samples of a plain pool",
             fontsize=10.5)
save(fig, "fig_members")

print("figures written:", FIG)
print("fitted c on curve:", round(c_fit, 2))
