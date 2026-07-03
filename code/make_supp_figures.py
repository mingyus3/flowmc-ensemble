#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_supp_figures.py — Generate the three supplementary figures added at revision:

  fig_ablation.{png,pdf}     (Sec 5.4)  one-at-a-time ablation, mean JS over 11 configs
  fig_equal_total.{png,pdf}  (Sec 5.6)  equal-total budget, per-config single vs ensemble
  fig_cost.{png,pdf}         (Sec 5.7)  accuracy-cost plane, measured runtimes only

Usage:
    python make_supp_figures.py --data_dir data --out_dir .

Inputs expected in --data_dir (released CSVs):
    results_equal_per_member.csv
    results_equal_per_member_no_cov.csv / _no_temp.csv / _no_w.csv
    results_single_10000.csv
    results_equal_total.csv
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORDER = [
    ("banana", 20), ("banana", 50),
    ("bimodal_gaussian", 20), ("bimodal_gaussian", 50),
    ("correlated_gaussian", 20), ("correlated_gaussian", 50),
    ("funnel", 20), ("funnel", 50),
    ("gaussian_ring", 20),
    ("rosenbrock", 20), ("rosenbrock", 50),
]
LABELS = ["Banana\n20D", "Banana\n50D", "Bimodal\n20D", "Bimodal\n50D",
          "Corr\n20D", "Corr\n50D", "Funnel\n20D", "Funnel\n50D",
          "Ring\n20D", "Rosen\n20D", "Rosen\n50D"]
FLOOR_2K, FLOOR_10K = 0.085, 0.043


def mean_by_config(df, method_substr):
    return np.array([df[(df.potential_fn == fn) & (df.dimension == dim) &
                        (df.method.str.contains(method_substr, regex=False))]
                     .JS_distance.mean() for fn, dim in ORDER])


def overall_mean(df, method_substr):
    return float(mean_by_config(df, method_substr).mean())


def fig_ablation(dd, out):
    epm = pd.read_csv(os.path.join(dd, "results_equal_per_member.csv"))
    s10 = pd.read_csv(os.path.join(dd, "results_single_10000.csv"))
    variants = [
        ("Single chain\n@2,000",     overall_mean(epm, "single"),   "#E69F00"),
        ("Full ensemble\n(all mechanisms)", overall_mean(epm, "ensemble"), "#0072B2"),
        ("No coverage\nreweighting", overall_mean(pd.read_csv(os.path.join(dd, "results_equal_per_member_no_cov.csv")), "ensemble"), "#56B4E9"),
        ("No adaptive\ntempering",   overall_mean(pd.read_csv(os.path.join(dd, "results_equal_per_member_no_temp.csv")), "ensemble"), "#56B4E9"),
        ("No member\nweighting",     overall_mean(pd.read_csv(os.path.join(dd, "results_equal_per_member_no_w.csv")), "ensemble"), "#56B4E9"),
        ("Single chain\n@10,000",    overall_mean(s10, "single@10000"), "#009E73"),
    ]
    names = [v[0] for v in variants]; vals = [v[1] for v in variants]; cols = [v[2] for v in variants]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(np.arange(len(vals)), vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
                ha="center", fontsize=8)
    ax.axhline(FLOOR_10K, ls="--", lw=1, color="grey")
    ax.text(len(vals) - 0.55, FLOOR_10K + 0.001, "floor @10k", fontsize=7, color="grey")
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Mean marginal JS distance over 11 configurations")
    ax.set_title("Removing any aggregation mechanism improves the ensemble;\n"
                 "no variant approaches the equal-budget single chain", fontsize=10)
    fig.tight_layout()
    _save(fig, out, "fig_ablation")


def fig_equal_total(dd, out):
    et = pd.read_csv(os.path.join(dd, "results_equal_total.csv"))
    single = mean_by_config(et, "flowMC-single")
    ens = mean_by_config(et, "equal_total")
    x = np.arange(len(ORDER)); w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w / 2, single, w, label="Single chain @ 2,000 (full budget)", color="#E69F00")
    ax.bar(x + w / 2, ens, w, label="Ensemble, equal-total (5 × 400 = 2,000)", color="#0072B2")
    ax.axhline(FLOOR_2K, ls="--", lw=1, color="grey")
    ax.text(len(ORDER) - 1.6, FLOOR_2K + 0.002, "floor @2k", fontsize=7, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel("Marginal JS distance (lower is better)")
    ax.set_title("Under an equal total budget of 2,000 samples, the ensemble is worse "
                 "on every configuration", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, out, "fig_equal_total")


def fig_cost(dd, out):
    epm = pd.read_csv(os.path.join(dd, "results_equal_per_member.csv"))
    s10 = pd.read_csv(os.path.join(dd, "results_single_10000.csv"))
    pts = [
        ("single @2,000",  epm[epm.method.str.contains("single")].runtime_sec.mean(),
         overall_mean(epm, "single"), "#E69F00", "o"),
        ("single @10,000", s10.runtime_sec.mean(),
         overall_mean(s10, "single@10000"), "#009E73", "o"),
        ("full ensemble (5×2,000)", epm[epm.method.str.contains("ensemble")].runtime_sec.mean(),
         overall_mean(epm, "ensemble"), "#0072B2", "s"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, rt, js, c, m in pts:
        ax.scatter(rt, js, s=90, color=c, marker=m, zorder=5, label=name)
        ax.annotate(f"{name}\n({rt:.0f} s, JS {js:.3f})", (rt, js),
                    textcoords="offset points", xytext=(10, 6), fontsize=8)
    ax.axhline(FLOOR_10K, ls="--", lw=1, color="grey")
    ax.text(80, FLOOR_10K + 0.001, "floor @10k", fontsize=7, color="grey")
    ax.set_xlabel("Mean wall-clock runtime per run (s)")
    ax.set_ylabel("Mean marginal JS distance over 11 configurations")
    ax.set_xlim(0, 105)
    ax.set_title("Accuracy versus cost: the equal-budget single chain is both the most\n"
                 "accurate and the cheapest; the ensemble is dominated on both axes", fontsize=10)
    fig.tight_layout()
    _save(fig, out, "fig_cost")


def _save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem + ".png/.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    fig_ablation(args.data_dir, args.out_dir)
    fig_equal_total(args.data_dir, args.out_dir)
    fig_cost(args.data_dir, args.out_dir)
    print("done.")
